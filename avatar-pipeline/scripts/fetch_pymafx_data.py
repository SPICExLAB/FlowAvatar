"""Extract PyMAF-X's auxiliary data from mmhuman3d's data archive (used by setup.sh).

Only the files that are not SMPL / SMPL-X models are taken: the mesh downsampling
matrices, the partial-mesh vertex ids, J_regressor_extra, smpl_mean_params and
smplx_to_smpl. The body models themselves come from prepare_smplx.py.
"""
import os
import shutil
import sys
import tempfile

import py7zr

WANTED = {
    "data/smpl_downsampling.npz",
    "data/mano_downsampling.npz",
    "data/body_models/J_regressor_extra.npy",
    "data/body_models/smpl_mean_params.npz",
    "data/body_models/smplx/smplx_to_smpl.npz",
}


def main():
    archive, data_dir = sys.argv[1], sys.argv[2]
    with py7zr.SevenZipFile(archive) as z:
        names = []
        for name in z.getnames():
            rel = name[len("mmhuman3d/"):]
            if rel in WANTED or (rel.startswith("data/partial_mesh/") and rel.endswith(".npz")):
                names.append(name)
        with tempfile.TemporaryDirectory() as tmp:
            z.extract(path=tmp, targets=names)
            for name in names:
                dst = os.path.join(data_dir, name[len("mmhuman3d/data/"):])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(os.path.join(tmp, name), dst)
    print(f"  extracted {len(names)} files to {data_dir}")


if __name__ == "__main__":
    main()
