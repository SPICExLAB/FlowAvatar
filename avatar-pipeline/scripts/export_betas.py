"""Step 10: write the avatar's ten SMPL-X shape parameters for Unity.

PyMAF-X (step 4) stores the per-frame betas averaged over the video in
<video dir>/smplx/betas.npy; this writes them as a JSON list to <video dir>/betas.json
and prints them (FlowAvatarSMPLxOVR > Shape in the Unity client).
"""
import argparse
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    data = os.path.dirname(os.path.abspath(args.video))

    betas = np.load(os.path.join(data, "smplx", "betas.npy")).reshape(-1)[:10]
    out = os.path.join(data, "betas.json")
    with open(out, "w") as f:
        json.dump([round(float(b), 6) for b in betas], f)
    print("betas:", " ".join(f"{b:.4f}" for b in betas))
    print("wrote", out)


if __name__ == "__main__":
    main()
