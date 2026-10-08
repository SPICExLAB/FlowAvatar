"""Step 6: align the face in each frame to the rendered SMPL-X face, build head masks.

Inputs in the video directory: images/, keypoints_2d.npy (step 2) and the stage-1
renders texture_init/val/{rasterization,rgb_image}/ (step 5). Facial landmarks come
from STAR (https://github.com/ZhenglinZhou/STAR), loaded from its clone in
third_party/STAR; the face box from dlib's 68-point shape predictor. Writes:

  align_head/                 frames whose face is affinely aligned to the render
  masks_aligned_head/         face box on the render, for the face texture (stage 2)
  masks_aligned_head_upper/   head region above the face box, for the head texture (stage 3)
  masks_head/                 everything except the face, for the body texture (stage 4)
"""
import argparse
import os
import sys

import cv2
import dlib
import numpy as np
import torch
from tqdm import tqdm

INPUT_SIZE = 256              # STAR network input
FACE_KEYPOINTS = (0, 1, 2)    # COCO nose, left eye, right eye
THRESH_ALIGN = 0.95           # face used for the face texture: nose and both eyes confident
THRESH_VISIBLE = 0.9          # face excluded from the body texture: any of them confident
OFFSET = 5                    # box margins (px)
OFFSET_BODY = 3


class FaceLandmarks:
    """98-point WFLW landmarks: dlib face box, then the STAR network on a 256 x 256 crop."""

    def __init__(self, star_dir, star_model, dlib_model, device_id=0):
        sys.path.insert(0, star_dir)
        from lib import utility  # STAR

        config = utility.get_config(argparse.Namespace(config_name="alignment"))
        config.device_id = device_id
        utility.set_environment(config)
        net = utility.get_net(config)
        checkpoint = torch.load(star_model, map_location=config.device)
        net.load_state_dict(checkpoint["net"])
        self.net = net.to(config.device).eval()
        self.device = config.device
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(dlib_model)

    @staticmethod
    def crop_matrix(scale, center_x, center_y):
        # Maps a face box of scale * 200 px around the centre onto the network input
        s = INPUT_SIZE / (scale * 200.0)
        c = (INPUT_SIZE - 1) / 2.0
        return np.array([[s, 0.0, c - s * center_x],
                         [0.0, s, c - s * center_y],
                         [0.0, 0.0, 1.0]], np.float32)

    def __call__(self, image):
        """Landmarks (98 x 2, image pixels) of the first detected face, or None."""
        faces = self.detector(image, 1)
        if len(faces) == 0:
            return None
        shape = self.shape_predictor(image, faces[0])
        points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        scale = float(min(x2 - x1, y2 - y1) / 200 * 1.05)
        matrix = self.crop_matrix(scale, float(x2 + x1) / 2, float(y2 + y1) / 2)

        crop = cv2.warpPerspective(image, matrix, (INPUT_SIZE, INPUT_SIZE),
                                   flags=cv2.INTER_LINEAR, borderValue=0)
        tensor = torch.from_numpy(crop[None]).float().permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        with torch.no_grad():
            landmarks = self.net(tensor.to(self.device))[-1][0]  # 98 x 2 in [-1, 1]
        # [-1, 1] -> crop pixels -> image pixels
        landmarks = ((landmarks + 1) * INPUT_SIZE - 1) / 2
        landmarks = landmarks.cpu().numpy()
        inverse = np.linalg.inv(matrix)
        return (landmarks @ inverse[:2, :2].T + inverse[:2, 2]).astype(np.float32)


def bbox(points):
    (xmin, ymin), (xmax, ymax) = points.min(axis=0), points.max(axis=0)
    return int(xmin - 5), int(xmax + 5), int(ymin - 5), int(ymax + 5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--star_dir", required=True, help="clone of the STAR repository")
    parser.add_argument("--star_model", required=True, help="WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl")
    parser.add_argument("--dlib_model", required=True, help="shape_predictor_68_face_landmarks.dat")
    args = parser.parse_args()

    data = os.path.dirname(os.path.abspath(args.video))
    render_dir = os.path.join(data, "texture_init", "val", "rasterization")
    rgb_dir = os.path.join(data, "texture_init", "val", "rgb_image")
    frames = sorted(f for f in os.listdir(render_dir) if f.endswith(".png"))
    keypoints = np.load(os.path.join(data, "keypoints_2d.npy"))
    out = {name: os.path.join(data, name) for name in
           ("align_head", "masks_head", "masks_aligned_head", "masks_aligned_head_upper")}
    for path in out.values():
        os.makedirs(path, exist_ok=True)

    landmarks_of = FaceLandmarks(args.star_dir, args.star_model, args.dlib_model)
    ymax = 0  # bottom of the last face box; carried over frames without a face
    for i in tqdm(range(keypoints.shape[0])):
        name = "%06d.png" % i
        rgb_image = cv2.imread(os.path.join(rgb_dir, frames[i]))
        mask_body = np.ones(rgb_image.shape) * 255
        mask_head_upper = np.ones(rgb_image.shape) * 255
        scores = keypoints[i, FACE_KEYPOINTS, 2]

        if np.all(scores > THRESH_ALIGN):
            render = cv2.imread(os.path.join(render_dir, frames[i]))
            render_points = landmarks_of(render)
            rgb_points = landmarks_of(rgb_image)
            if render_points is None or rgb_points is None:
                print(f"{i}: no face")
            else:
                # Align with the eye and mouth landmarks (60-97)
                affine, _ = cv2.estimateAffine2D(rgb_points[60:], render_points[60:], method=cv2.LMEDS)
                height, width = rgb_image.shape[:2]
                cv2.imwrite(os.path.join(out["align_head"], name), cv2.warpAffine(rgb_image, affine, (width, height)))

                # Box over eyebrows, eyes, nose and mouth (landmarks 33-97)
                xmin, xmax, ymin, ymax = bbox(rgb_points[33:])
                mask_body[ymin - OFFSET_BODY:ymax + OFFSET_BODY, xmin - OFFSET_BODY:xmax + OFFSET_BODY] = 0

                mask_face = np.zeros(rgb_image.shape)
                xmin, xmax, ymin, ymax = bbox(render_points[33:])
                mask_face[ymin - OFFSET:ymax + OFFSET, xmin - OFFSET:xmax + OFFSET] = 255
                cv2.imwrite(os.path.join(out["masks_aligned_head"], name), mask_face)

                mask_head_upper[ymin - OFFSET:ymax + OFFSET, xmin - OFFSET:xmax + OFFSET] = 0

        elif np.any(scores > THRESH_VISIBLE):
            rgb_points = landmarks_of(rgb_image)
            if rgb_points is None:
                print(f"{i}: no face")
            else:
                xmin, xmax, ymin, ymax = bbox(rgb_points[33:])
                mask_body[ymin - OFFSET_BODY:ymax + OFFSET_BODY, xmin - OFFSET_BODY:xmax + OFFSET_BODY] = 0
                mask_head_upper[ymin - OFFSET:ymax + OFFSET, xmin - OFFSET:xmax + OFFSET] = 0

        mask_head_upper[ymax + OFFSET:] = 0
        mask_body[:ymax] = 0
        cv2.imwrite(os.path.join(out["masks_head"], name), mask_body)
        cv2.imwrite(os.path.join(out["masks_aligned_head_upper"], name), mask_head_upper)


if __name__ == "__main__":
    main()
