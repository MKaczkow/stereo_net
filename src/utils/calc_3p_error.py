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

    # print(not_empty.shape)
    # print(type(not_empty))
    # print(not_empty.sum())
    # print(not_empty)

    #TODO: cleanup!

    # IMPORTANT!
    # Why the heck 255?
    # Data is normalized to [0, 1], thus, to legitimately calculate pixel-wise values
    # it needs to be denormalized to standard pixel value range [0, 255]
    predicted_flatten = predicted[not_empty].flatten().astype(np.float32)*255
    groundtruth_flatten = groundtruth[not_empty].flatten().astype(np.float32)*255

    # print(groundtruth_flatten.shape, predicted_flatten.shape)
    # print(groundtruth_flatten[0:10])
    # print(predicted_flatten[0:10])

    disp_diff_l = abs(predicted_flatten - groundtruth_flatten)

    # print('disp_diff_l')
    # print(disp_diff_l.shape)
    # print(disp_diff_l)
    
    accept_3p = (disp_diff_l <= 3) | (disp_diff_l <= groundtruth_flatten * 0.05)

    # print('accept_3p')
    # print(accept_3p.shape)
    # print(np.count_nonzero(accept_3p))
    # print(accept_3p)
    
    err_3p = 1 - np.count_nonzero(accept_3p) / (len(disp_diff_l))
    # print(predicted_flatten.shape, groundtruth_flatten.shape, accept_3p.shape, err_3p)

    return err_3p
