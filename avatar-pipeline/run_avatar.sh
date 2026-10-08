#!/usr/bin/env bash
# Personalised avatar from one video:
#
#   ./run_avatar.sh /path/to/session/video.mp4
#
# Put each video in its own folder: all intermediate results and the outputs are
# written next to it. Outputs: texture.png (2048 x 2048, in the UV layout of the
# Unity avatar) and betas.json (ten SMPL-X shape parameters).
set -euo pipefail

if [ $# -ne 1 ]; then
    sed -n '2,8p' "$0"
    exit 1
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="$ROOT/third_party"
WEIGHTS="$ROOT/weights"
VIDEO="$(realpath "$1")"
DATA="$(dirname "$VIDEO")"

mmlab() { conda run --no-capture-output -n open-mmlab "$@"; }
haha() { conda run --no-capture-output -n haha "$@"; }

SGHM_WEIGHTS="$(find "$WEIGHTS" -name SGHM-ResNet50.pth 2>/dev/null | head -n 1)"
STAR_WEIGHTS="$WEIGHTS/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl"
DLIB_WEIGHTS="$WEIGHTS/shape_predictor_68_face_landmarks.dat"
for f in "$SGHM_WEIGHTS" "$STAR_WEIGHTS" "$DLIB_WEIGHTS" \
         "$ROOT/body_models/smplx/SMPLX_NEUTRAL.npz" "$ROOT/body_models/smplx/smplx-10.obj" \
         "$TP/HAHA/metadata/smplx_texture_f_alb_512.png" \
         "$TP/mmhuman3d/data/body_models/smpl/SMPL_NEUTRAL.pkl"; do
    if [ -z "$f" ] || [ ! -s "$f" ]; then
        echo "Missing ${f:-SGHM-ResNet50.pth}: run ./setup.sh and prepare_smplx.py (see README.md)"
        exit 1
    fi
done

echo "[1/10] Frames"
mmlab python "$ROOT/scripts/extract_frames.py" --video "$VIDEO"

echo "[2/10] 2D keypoints (mmpose)"
(cd "$TP/mmpose" && mmlab python demo/top_down_img_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --img-root "$VIDEO")

echo "[3/10] Person masks (Semantic Guided Human Matting)"
(cd "$TP/SemanticGuidedHumanMatting" && mmlab python test_image.py \
    --images-dir "$DATA/images/" --result-dir "$DATA/masks/" --pretrained-weight "$SGHM_WEIGHTS")

echo "[4/10] SMPL-X fit (PyMAF-X)"
(cd "$TP/mmhuman3d" && mmlab python demo/pymafx_estimate_smplx.py --vid_path "$VIDEO")

echo "[5/10] Reference render of the initial fit (HAHA)"
(cd "$TP/HAHA" && haha python main.py --base "$ROOT/configs/texture_init.yaml" --vid_path "$VIDEO")

echo "[6/10] Face alignment and head masks (STAR)"
mmlab python "$ROOT/scripts/align_faces.py" --video "$VIDEO" --star_dir "$TP/STAR" \
    --star_model "$STAR_WEIGHTS" --dlib_model "$DLIB_WEIGHTS"

echo "[7/10] Face, head and body textures (HAHA)"
for stage in texture_head texture_head_upper texture; do
    (cd "$TP/HAHA" && haha python main.py --base "$ROOT/configs/$stage.yaml" --vid_path "$VIDEO")
done

echo "[8/10] Blend the textures"
haha python "$ROOT/scripts/blend_textures.py" --video "$VIDEO"

echo "[9/10] Upscale to 2048 x 2048 (Real-ESRGAN)"
"$TP/realesrgan/realesrgan-ncnn-vulkan" -i "$DATA/texture_small.png" -o "$DATA/texture.png" \
    -n realesrgan-x4plus -m "$TP/realesrgan/models"

echo "[10/10] Shape parameters"
mmlab python "$ROOT/scripts/export_betas.py" --video "$VIDEO"

echo "Done: $DATA/texture.png and $DATA/betas.json"
