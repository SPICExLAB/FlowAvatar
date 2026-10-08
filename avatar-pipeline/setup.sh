#!/usr/bin/env bash
# Avatar pipeline setup: third-party code at pinned versions plus our patches, and the
# publicly downloadable weights. Safe to re-run. Needs git, curl, conda and the two
# environments from envs/ (see README.md). The SMPL-X / SMPL body models are added
# afterwards with prepare_smplx.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="$ROOT/third_party"
WEIGHTS="$ROOT/weights"
mkdir -p "$TP" "$WEIGHTS"

for env in open-mmlab haha; do
    if ! conda run -n "$env" python -c "" >/dev/null 2>&1; then
        echo "Conda environment '$env' not found. Create it first: conda env create -f envs/$env.yml"
        exit 1
    fi
done

# --- third-party code -------------------------------------------------------------------
# clone NAME URL REF: shallow checkout of a commit or tag
clone() {
    local name=$1 url=$2 ref=$3 dir="$TP/$1"
    if [ ! -d "$dir/.git" ]; then
        git init -q "$dir"
        git -C "$dir" remote add origin "$url"
    fi
    if [ "$(git -C "$dir" rev-parse -q --verify HEAD 2>/dev/null)" = "" ]; then
        echo "  $name @ $ref"
        git -C "$dir" fetch -q --depth 1 origin "$ref"
        git -C "$dir" checkout -q FETCH_HEAD
    fi
}
# apply_patch NAME: apply patches/NAME.patch once
apply_patch() {
    local dir="$TP/$1" patch="$ROOT/patches/$1.patch"
    if git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
        return  # already applied
    fi
    git -C "$dir" apply "$patch"
    echo "  patched $1"
}

echo "== Third-party code (third_party/)"
clone HAHA https://github.com/david-svitov/HAHA.git 1fb63cf83bb3321e7bbb49efd5d55c0f26346fe6
apply_patch HAHA
clone SemanticGuidedHumanMatting https://github.com/cxgincsu/SemanticGuidedHumanMatting.git f116a2665e015bf84acfd865ae114ea96a1b23b9
apply_patch SemanticGuidedHumanMatting
clone STAR https://github.com/ZhenglinZhou/STAR.git 9b125749b0d35766ed83d047036d1aa5e384984c
clone mmpose https://github.com/open-mmlab/mmpose.git v0.28.1
apply_patch mmpose
clone mmhuman3d https://github.com/open-mmlab/mmhuman3d.git v0.11.0
apply_patch mmhuman3d
conda run --no-capture-output -n open-mmlab pip install -q --no-deps -e "$TP/mmhuman3d"
# mmhuman3d needs pytorch3d; built from source against the environment's PyTorch
# (this takes a while and needs the CUDA compiler from the environment)
if ! conda run -n open-mmlab python -c "import pytorch3d" >/dev/null 2>&1; then
    echo "  building pytorch3d for open-mmlab"
    FORCE_CUDA=1 conda run --no-capture-output -n open-mmlab pip install --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@42a4a7d432b03af90742db0be95a8f477f734cf9"
fi

# --- weights ----------------------------------------------------------------------------
# fetch URL FILE: download once
fetch() {
    [ -s "$2" ] && return
    echo "  $(basename "$2")"
    mkdir -p "$(dirname "$2")"
    curl -L --fail --retry 5 -C - -o "$2.part" "$1"  # resumes an interrupted download
    mv "$2.part" "$2"
}
# gdrive ID_OR_URL FILE_OR_DIR [--folder]: Google Drive download with gdown
gdrive() {
    [ -e "$2" ] && return
    echo "  $(basename "$2") (Google Drive)"
    if ! conda run --no-capture-output -n open-mmlab gdown ${3:-} "$1" -O "$2"; then
        echo "  Download failed. Get it manually from $1 and save it as $2"
    fi
}

echo "== PyMAF-X (third_party/mmhuman3d/data/)"
MMH="$TP/mmhuman3d/data"
fetch https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/pymaf_x/PyMAF-X_model_checkpoint.pth \
      "$MMH/pretrained_models/PyMAF-X_model_checkpoint.pth"
if [ ! -s "$MMH/body_models/smplx/smplx_to_smpl.npz" ]; then
    # mmhuman3d's data archive; only the files that are not SMPL / SMPL-X models are used
    fetch https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/mmhuman3d.7z "$WEIGHTS/mmhuman3d.7z"
    conda run --no-capture-output -n open-mmlab python "$ROOT/scripts/fetch_pymafx_data.py" "$WEIGHTS/mmhuman3d.7z" "$MMH"
    rm -f "$WEIGHTS/mmhuman3d.7z"
fi

echo "== Matting, face landmarks, upscaling (weights/, third_party/realesrgan/)"
gdrive https://drive.google.com/drive/folders/15mGzPJQFEchaZHt9vgbmyOy46XxWtEOZ "$WEIGHTS/SGHM" --folder
gdrive 1aOx0wYEZUfBndYy_8IYszLPG_D2fhxrT "$WEIGHTS/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl"
if [ ! -s "$WEIGHTS/shape_predictor_68_face_landmarks.dat" ]; then
    fetch http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "$WEIGHTS/shape_predictor_68_face_landmarks.dat.bz2"
    bunzip2 -f "$WEIGHTS/shape_predictor_68_face_landmarks.dat.bz2"
fi
if [ ! -x "$TP/realesrgan/realesrgan-ncnn-vulkan" ]; then
    fetch https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip \
          "$WEIGHTS/realesrgan.zip"
    python3 -m zipfile -e "$WEIGHTS/realesrgan.zip" "$TP/realesrgan"
    chmod +x "$TP/realesrgan/realesrgan-ncnn-vulkan"
    rm -f "$WEIGHTS/realesrgan.zip"
fi

echo
echo "Setup done. Next: conda run -n haha python prepare_smplx.py --smplx_model ... --unity_project ... --smpl_model ..."
echo "(see README.md, 'Body models')."
