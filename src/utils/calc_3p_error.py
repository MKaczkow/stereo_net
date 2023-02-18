import torch
import numpy as np


def calc_3p_error(predicted: torch.Tensor, groundtruth: torch.Tensor) -> float:
    """
    Calculate 3-pixel-5%-error.

    Reference:  
    https://gist.github.com/MiaoDX/8d5f49c2ccb39d7f2cb8d4e57c3ab752

    :param predicted: predicted disparity
    :param groundtruth: groundtruth disparity
    :returns: 3-pixel-5%-error as scalar
    """

    predicted = predicted.to('cpu').detach().numpy()
    groundtruth = groundtruth.to('cpu').detach().numpy()
    
    not_empty = (predicted > 0) & \
                (~np.isnan(predicted)) & \
                (groundtruth > 0) & \
                (~np.isnan(groundtruth))

    # Why the heck 255?
    # Data is normalized to [0, 1], thus, to legitimately calculate pixel-wise values
    # it needs to be denormalized to standard pixel value range [0, 255]
    predicted_flatten = predicted[not_empty].flatten().astype(np.float32)*255
    groundtruth_flatten = groundtruth[not_empty].flatten().astype(np.float32)*255
    disp_diff_l = abs(predicted_flatten - groundtruth_flatten)
    accept_3p = (disp_diff_l <= 3) | (disp_diff_l <= groundtruth_flatten * 0.05)
    err_3p = 1 - np.count_nonzero(accept_3p) / (len(disp_diff_l))

    return err_3p
