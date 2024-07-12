import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Helper function for drawing matches
def draw_img_match(img1, img2, mkpts1, mkpts2, reverse_pair=True):
    if isinstance(img1, torch.Tensor):
        img1 = im_tensor_to_np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = im_tensor_to_np(img2)
    if isinstance(mkpts1, torch.Tensor):
        mkpts1 = mkpts1.detach().cpu().numpy()
    if isinstance(mkpts2, torch.Tensor):
        mkpts2 = mkpts2.detach().cpu().numpy()

    if isinstance(img1, np.ndarray):
        img1 = np.uint8(255 * img1)
    else:
        img1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2RGB)
    if isinstance(img2, np.ndarray):
        img2 = np.uint8(255 * img2)
    else:
        img2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2RGB)

    if reverse_pair:
        (
            img1,
            img2,
        ) = (
            img2,
            img1,
        )
        mkpts1, mkpts2 = mkpts2, mkpts1

    img = cv2.drawMatches(
        img1=img1,
        keypoints1=[cv2.KeyPoint(x=x, y=y, size=2) for x, y in mkpts1],
        img2=img2,
        keypoints2=[cv2.KeyPoint(x=x, y=y, size=2) for x, y in mkpts2],
        matches1to2=[
            cv2.DMatch(_trainIdx=i, _queryIdx=i, _distance=-1.0)
            for i in range(len(mkpts1))
        ],
        matchesThickness=2,
        outImg=None,
    )
    plt.imshow(img)


def im_tensor_to_np(x):
    return x[0].permute(1, 2, 0).detach().cpu().numpy()
