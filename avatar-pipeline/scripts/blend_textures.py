"""Step 8: merge the three fitted textures into one 512 x 512 UV texture.

The face texture (texture_head/) is blended into the head texture
(texture_head_upper/) with assets/face_blend_mask.png, and the result into the body
texture (texture/), whose eyes are first replaced by assets/eye_texture.png, with
assets/body_blend_mask.png. Writes <video dir>/texture_small.png.
"""
import argparse
import os

import cv2
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")


def feather(src, dst, mask):
    """src where the mask is white, dst where it is black (mask: 8-bit image)."""
    alpha = mask / 255.0
    return (src * alpha + dst * (1 - alpha)).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    data = os.path.dirname(os.path.abspath(args.video))

    body = cv2.imread(os.path.join(data, "texture", "texture.png"))
    face = cv2.imread(os.path.join(data, "texture_head", "texture.png"))
    head = cv2.imread(os.path.join(data, "texture_head_upper", "texture.png"))
    size = body.shape[0]

    eyes = cv2.imread(os.path.join(ASSETS, "eye_texture.png"))
    eyes = cv2.resize(eyes, (size, size), interpolation=cv2.INTER_NEAREST)
    body = np.where(eyes > 0, eyes, body)

    face_mask = cv2.imread(os.path.join(ASSETS, "face_blend_mask.png"))
    body_mask = cv2.imread(os.path.join(ASSETS, "body_blend_mask.png"))
    texture = feather(feather(face, head, face_mask), body, body_mask)

    out = os.path.join(data, "texture_small.png")
    cv2.imwrite(out, texture)
    print("wrote", out)


if __name__ == "__main__":
    main()
