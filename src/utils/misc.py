from typing import Tuple

import torch
import torchvision.transforms as T


class SizeRequestedIsLargerThanImage(Exception):
    """
    One (or both) of the requested dimensions is larger than the cropped image.
    """

class CenterCrop():
    """
    """

    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, sample):
        height, width = sample['left'].size()[-2:]
        output_height = int(self.scale*height)
        output_width = int(self.scale*width)
        cropper = T.CenterCrop((output_height, output_width))
        for name, x in sample.items():  # pylint: disable=invalid-name
            sample[name] = cropper(x)
        return sample


class ToTensor():
    """
    Converts the left, right, and disparity maps into FloatTensors.
    Left and right uint8 images get rescaled to [0,1] floats.
    Disparities are already floats and just get turned into tensors.
    """

    @staticmethod
    def __call__(sample):
        torch_sample = {}
        for name, x in sample.items():  # pylint: disable=invalid-name
            if x.dtype == 'uint16':
                x = x / 256.0
                x = x.astype('uint8')
            torch_sample[name] = T.functional.to_tensor(x.copy())
        return torch_sample


class PadSampleToBatch():
    """
    Unsqueezes the first dimension to be 1 when loading in single image pairs.
    """

    @staticmethod
    def __call__(sample):
        for name, x in sample.items():  # pylint: disable=invalid-name
            sample[name] = torch.unsqueeze(x, dim=0)
        return sample


class Resize():
    """
    Resizes each of the images in a batch to a given height and width
    """

    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, sample):
        for name, x in sample.items():
            sample[name] = T.functional.resize(x, self.size)
        return sample


class Rescale():
    """
    Rescales the left and right image tensors (initially ranged between [0, 1]) and rescales them to be between [-1, 1].
    """

    @staticmethod
    def __call__(sample):
        for name in ['left', 'right']:
            sample[name] = (sample[name] - 0.5) * 2
        return sample
