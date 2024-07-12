import torch
from PIL import Image
import matplotlib.pyplot as plt
import os

from DeDoDe import dedode_detector_L, dedode_descriptor_B
from DeDoDe.utils import to_pixel_coords

from Steerers.utils import draw_img_match
from Steerers.matcher import matcher
from Steerers.steerers import Steerer

detector = dedode_detector_L(
    weights=torch.load(
        "model_weights/dedode_detector_L.pth", map_location=torch.device("cpu")
    )
)
descriptor = dedode_descriptor_B(
    weights=torch.load(
        "model_weights/dedode_descriptor_B.pth", map_location=torch.device("cpu")
    )
)
c4_steerer_weight = torch.load(
    "model_weights/B_C4_steerer_setting_A.pth", map_location=torch.device("cpu")
)
c4_steerer = Steerer(c4_steerer_weight)


def main(im1_path, im2_path, without_steerer_result, with_steerer_result):
    im_path_list = [im1_path, im2_path]
    im_pil_list = [Image.open(im_path) for im_path in im_path_list]
    w_list = [im_pil.size[0] for im_pil in im_pil_list]
    h_list = [im_pil.size[1] for im_pil in im_pil_list]

    detections_list = [
        detector.detect_from_path(im_path, num_keypoints=5_000)
        for im_path in im_path_list
    ]
    keypoints_list = [detections["keypoints"] for detections in detections_list]
    descriptions_list = [
        descriptor.describe_keypoints_from_path(im_path, keypoints)["descriptions"]
        for im_path, keypoints in zip(im_path_list, keypoints_list)
    ]

    for (steerer, steerer_order), result_img_path in [
        ((None, None), without_steerer_result),
        ((c4_steerer, 4), with_steerer_result),
    ]:
        matches1, matches2 = matcher(
            keypoints_list[0],
            descriptions_list[0],
            keypoints_list[1],
            descriptions_list[1],
            steerer=steerer,
            steerer_order=steerer_order,
        )

        matches_pixel_coords_list = [
            to_pixel_coords(matches, h, w)
            for matches, h, w in zip([matches1, matches2], h_list, w_list)
        ]

        print(len(matches_pixel_coords_list[0]))

        plt.figure(figsize=(10, 5), dpi=80)
        draw_img_match(
            im_path_list[0],
            im_path_list[1],
            matches_pixel_coords_list[0][:100],
            matches_pixel_coords_list[1][:100],
        )
        plt.axis("off")
        os.makedirs(os.path.dirname(result_img_path), exist_ok=True)
        plt.savefig(result_img_path)
        plt.clf()
        plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--im1", type=str, required=True)
    parser.add_argument("--im2", type=str, required=True)
    parser.add_argument("--wo", type=str, default="images/without_steerer_result.png")
    parser.add_argument("--w", type=str, default="images/with_steerer_result.png")
    args = parser.parse_args()
    main(args.im1, args.im2, args.wo, args.w)