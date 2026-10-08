"""Step 1: split the capture video into frames.

Writes every frame, upright and resized to 720 x 1280 (portrait), to
<video dir>/images/%06d.png. Phone cameras often store portrait clips as landscape
frames plus a rotation flag; the flag is applied here, as a video player would.
"""
import argparse
import os

import av
import imageio
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="capture video (portrait, full body in view)")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.video)), "images")
    os.makedirs(out_dir, exist_ok=True)

    with av.open(args.video) as container:
        # Display rotation in degrees, counter-clockwise like np.rot90
        rotation = round(container.streams.video[0].side_data.get("DISPLAYMATRIX", 0)) % 360
        for index, frame in enumerate(container.decode(video=0)):
            pixels = np.rot90(frame.to_ndarray(format="rgb24"), rotation // 90)
            image = Image.fromarray(pixels).resize((720, 1280))
            imageio.imwrite(os.path.join(out_dir, "%06d.png" % index), image)
    print(f"{index + 1} frames written to {out_dir}")


if __name__ == "__main__":
    main()
