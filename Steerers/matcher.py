import torch
from DeDoDe.utils import dual_softmax_matcher


def matcher(
    keypoints_A,
    descriptions_A,
    keypoints_B,
    descriptions_B,
    steerer=None,
    steerer_order=None,
):
    normalize = True
    inv_temp = 20
    threshold = 0.01

    if steerer is not None and steerer_order is not None:
        best_num_matches = -1
        matches_A = matches_B = None
        for power in range(steerer_order):
            if power > 0:
                descriptions_A = steerer.forward(descriptions_A)
            P = dual_softmax_matcher(
                descriptions_A,
                descriptions_B,
                normalize=normalize,
                inv_temperature=inv_temp,
            )
            inds = torch.nonzero(
                (P == P.max(dim=-1, keepdim=True).values)
                * (P == P.max(dim=-2, keepdim=True).values)
                * (P > threshold)
            )
            batch_inds = inds[:, 0]
            num_matches = len(batch_inds)
            if num_matches > best_num_matches:
                matches_A = keypoints_A[batch_inds, inds[:, 1]]
                matches_B = keypoints_B[batch_inds, inds[:, 2]]
                best_num_matches = num_matches
        return matches_A, matches_B
    else:
        P = dual_softmax_matcher(
            descriptions_A,
            descriptions_B,
            normalize=normalize,
            inv_temperature=inv_temp,
        )
        inds = torch.nonzero(
            (P == P.max(dim=-1, keepdim=True).values)
            * (P == P.max(dim=-2, keepdim=True).values)
            * (P > threshold)
        )
        batch_inds = inds[:, 0]
        matches_A = keypoints_A[batch_inds, inds[:, 1]]
        matches_B = keypoints_B[batch_inds, inds[:, 2]]
        return matches_A, matches_B
