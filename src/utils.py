"""
Helper functions for StereoNet training.

Includes a dataset object for the Scene Flow image and disparity dataset.
"""

from typing import Optional, Tuple, List
from pathlib import Path
from PIL import Image
import os
from datetime import datetime

import random
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as T
import matplotlib.pyplot as plt

import stereonet_types as st
import utils_io as utils_io


def image_loader(path: Path) -> npt.NDArray[np.uint8]:  # pylint: disable=missing-function-docstring
    img: npt.NDArray[np.uint8] = io.imread(path)
    # if img.shape[2] == 4:
    #     img_without_alpha = img[:,:,:3]
    #     # print(img_without_alpha.shape)
    #     # print(img_without_alpha.max(), img_without_alpha.min())
    #     # print(img_without_alpha.dtype)
    #     return img_without_alpha

    # print(img.shape)
    # print(img.max(), img.min())
    # print(img.dtype)

    return img

def webp_loader(target_path) -> torch.tensor:
    """Utility function to load images in .webp format."""
    # print(target_path)
    img = Image.open(target_path)
    img_array = np.array(img)
    # print(img_array.shape)
    # print(img_array.max(), img_array.min())
    # print(img_array.dtype)
    # img = Image.open(target_path).resize((960, 540, 3), Image.Resampling.BILINEAR)
    # image_data = np.ascontiguousarray(img)
    # image_data_copy = np.copy(image_data)   # workaround to remove WRITEABLE = False flag
    # image = torch.from_numpy(image_data_copy)
    # print(type(img), img.size)
    
    return img_array

def pfm_loader(path: Path) -> Tuple[npt.NDArray[np.float32], float]:  # pylint: disable=missing-function-docstring
    pfm: Tuple[npt.NDArray[np.float32], float] = utils_io.readPFM(path)
    return pfm

class SizeRequestedIsLargerThanImage(Exception):
    """
    One (or both) of the requested dimensions is larger than the cropped image.
    """


class Kitti2015Dataset(Dataset):  # type: ignore[type-arg]  # I don't know why this typing ignore is needed on the class level...
    """
    Download the RGB (cleanpass) PNG image and the Disparity files

    The train set includes FlyingThings3D Train folder and all files in Driving and Monkaa folders
    The test set includes FlyingThings3D Test folder
    """

    def __init__(self,
                 root_path: Path,
                 transforms: st.TorchTransformers,
                 string_exclude: Optional[str] = None,
                 string_include: Optional[str] = None
                 ):
        self.root_path = root_path
        self.string_exclude = string_exclude
        self.string_include = string_include

        if not isinstance(transforms, list):
            _transforms = [transforms]
        else:
            _transforms = transforms

        self.transforms = _transforms

        self.left_image_path, self.right_image_path, self.left_disp_path, self.right_disp_path = self.get_paths()

    def __len__(self) -> int:
        return len(self.left_image_path)

    def __getitem__(self, index: int) -> st.Sample_Torch:

        # print(self.left_image_path[0:5])
        left = image_loader(self.left_image_path[index])
        # print(type(left))
        right = image_loader(self.right_image_path[index])
        disp_left, _ = pfm_loader(self.left_disp_path[index])
        disp_right, _ = pfm_loader(self.right_disp_path[index])


        # disp_left, _ = pfm_loader(self.left_disp_path[index])
        # disp_left = disp_left[..., np.newaxis]
        # disp_left = np.ascontiguousarray(disp_left)

        # disp_right, _ = pfm_loader(self.right_disp_path[index])
        # disp_right = disp_right[..., np.newaxis]
        # disp_right = np.ascontiguousarray(disp_right)
        
        # disp_left, _ = pfm_loader(self.left_disp_path[index])
        # disp_left = disp_left[..., np.newaxis]
        # disp_left = np.ascontiguousarray(disp_left)

        # disp_right, _ = pfm_loader(self.right_disp_path[index])
        # disp_right = disp_right[..., np.newaxis]
        # disp_right = np.ascontiguousarray(disp_right)

        # I'm not sure why I need the following type ignore...
        sample: st.Sample_Numpy = {'left': left, 'right': right, 'disp_left': disp_left, 'disp_right': disp_right}  # type: ignore[assignment]
        # print(type(sample))
        # print(sample['left'].shape)
        # print(sample['right'].shape)
        # print(sample['disp_left'].shape)
        # print(sample['disp_right'].shape)

        # print('before')
        # print(sample['left'].shape, sample['left'].dtype, \
        #     sample['left'].max(), sample['left'].min())
        # print(sample['disp_left'].shape, sample['disp_left'].dtype, \
        #     sample['disp_left'].max(), sample['disp_left'].min())

        torch_sample = ToTensor()(sample)
        # print('intermediate')
        # print(torch_sample.shape)
        # print(torch_sample['left'].shape, torch_sample['left'].dtype, \
        #     torch_sample['left'].max(), torch_sample['left'].min())
        # print(torch_sample['disp_left'].shape, torch_sample['disp_left'].dtype, \
        #     torch_sample['disp_left'].max(), torch_sample['disp_left'].min())

        for transform in self.transforms:
            torch_sample = transform(torch_sample)

        # print('after')
        # print(torch_sample.shape)
        # print(torch_sample['left'].shape, torch_sample['left'].dtype, \
        #     torch_sample['left'].max(), torch_sample['left'].min())
        # print(torch_sample['disp_left'].shape, torch_sample['disp_left'].dtype, \
        #     torch_sample['disp_left'].max(), torch_sample['disp_left'].min())

        return torch_sample

    def get_paths(self) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
        """
        string_exclude: If this string appears in the parent path of an image, don't add them to the dataset (ie. 'TEST' will exclude any path with 'TEST' in Path.parts)
        string_include: If this string DOES NOT appear in the parent path of an image, don't add them to the dataset (ie. 'TEST' will require 'TEST' to be in the Path.parts)
        if shuffle is None, don't shuffle, else shuffle.
        """

        left_image_path = []
        right_image_path = []
        left_disp_path = []
        right_disp_path = []
   
        for root, _, files in os.walk(f'{self.root_path}/image_2'):
            for file in files:
                if file.endswith('_10.png'):
                    # print('left_image_path')
                    # print(f'{root}\{file}')
                    left_image_path.append(f'{root}\{file}')

        for root, _, files in os.walk(f'{self.root_path}/image_3'):
            for file in files:
                if file.endswith('_10.png'):
                    # print('right_image_path')
                    # print(f'{root}\{file}')
                    right_image_path.append(f'{root}\{file}')
        
        for root, _, files in os.walk(f'{self.root_path}/disp_noc_0'):
            for file in files:
                if file.endswith('.pfm'):
                    # print('left_disp_path')
                    # print(f'{root}\{file}')
                    left_disp_path.append(f'{root}\{file}')

        for root, _, files in os.walk(f'{self.root_path}/disp_noc_1'):
            for file in files:
                if file.endswith('.pfm'):
                    # print('right_disp_path')
                    # print(f'{root}\{file}')
                    right_disp_path.append(f'{root}\{file}')

        return (left_image_path, right_image_path, left_disp_path, right_disp_path)


class Kitti2012Dataset(Dataset):  # type: ignore[type-arg]  # I don't know why this typing ignore is needed on the class level...
    """
    Download the RGB (cleanpass) PNG image and the Disparity files

    The train set includes FlyingThings3D Train folder and all files in Driving and Monkaa folders
    The test set includes FlyingThings3D Test folder
    """

    def __init__(self,
                 root_path: Path,
                 transforms: st.TorchTransformers,
                 string_exclude: Optional[str] = None,
                 string_include: Optional[str] = None
                 ):
        self.root_path = root_path
        self.string_exclude = string_exclude
        self.string_include = string_include

        if not isinstance(transforms, list):
            _transforms = [transforms]
        else:
            _transforms = transforms

        self.transforms = _transforms

        self.left_image_path, self.right_image_path, self.left_disp_path = self.get_paths()

    def __len__(self) -> int:
        return len(self.left_image_path)

    def __getitem__(self, index: int) -> st.Sample_Torch:

        # print(self.left_image_path[0:5])
        left = image_loader(self.left_image_path[index])
        # print(type(left))
        right = image_loader(self.right_image_path[index])
        disp_left, _ = pfm_loader(self.left_disp_path[index])


        # disp_left, _ = pfm_loader(self.left_disp_path[index])
        # disp_left = disp_left[..., np.newaxis]
        # disp_left = np.ascontiguousarray(disp_left)

        # disp_right, _ = pfm_loader(self.right_disp_path[index])
        # disp_right = disp_right[..., np.newaxis]
        # disp_right = np.ascontiguousarray(disp_right)
        
        # disp_left, _ = pfm_loader(self.left_disp_path[index])
        # disp_left = disp_left[..., np.newaxis]
        # disp_left = np.ascontiguousarray(disp_left)

        # disp_right, _ = pfm_loader(self.right_disp_path[index])
        # disp_right = disp_right[..., np.newaxis]
        # disp_right = np.ascontiguousarray(disp_right)

        # I'm not sure why I need the following type ignore...
        sample: st.Sample_Numpy = {'left': left, 'right': right, 'disp_left': disp_left}  # type: ignore[assignment]
        # print(type(sample))
        # print(sample['left'].shape)
        # print(sample['right'].shape)
        # print(sample['disp_left'].shape)
        # print(sample['disp_right'].shape)

        # print('before')
        # print(sample['left'].shape, sample['left'].dtype, \
        #     sample['left'].max(), sample['left'].min())
        # print(sample['disp_left'].shape, sample['disp_left'].dtype, \
        #     sample['disp_left'].max(), sample['disp_left'].min())

        torch_sample = ToTensor()(sample)
        # print('intermediate')
        # print(torch_sample.shape)
        # print(torch_sample['left'].shape, torch_sample['left'].dtype, \
        #     torch_sample['left'].max(), torch_sample['left'].min())
        # print(torch_sample['disp_left'].shape, torch_sample['disp_left'].dtype, \
        #     torch_sample['disp_left'].max(), torch_sample['disp_left'].min())

        for transform in self.transforms:
            torch_sample = transform(torch_sample)

        # print('after')
        # print(torch_sample.shape)
        # print(torch_sample['left'].shape, torch_sample['left'].dtype, \
        #     torch_sample['left'].max(), torch_sample['left'].min())
        # print(torch_sample['disp_left'].shape, torch_sample['disp_left'].dtype, \
        #     torch_sample['disp_left'].max(), torch_sample['disp_left'].min())

        return torch_sample

    def get_paths(self) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
        """
        string_exclude: If this string appears in the parent path of an image, don't add them to the dataset (ie. 'TEST' will exclude any path with 'TEST' in Path.parts)
        string_include: If this string DOES NOT appear in the parent path of an image, don't add them to the dataset (ie. 'TEST' will require 'TEST' to be in the Path.parts)
        if shuffle is None, don't shuffle, else shuffle.
        """

        left_image_path = []
        right_image_path = []
        left_disp_path = []
   
        for root, _, files in os.walk(f'{self.root_path}/colored_0'):
            for file in files:
                if file.endswith('_10.png'):
                    # print('left_image_path')
                    # print(f'{root}\{file}')
                    left_image_path.append(f'{root}\{file}')

        for root, _, files in os.walk(f'{self.root_path}/colored_0'):
            for file in files:
                if file.endswith('_10.png'):
                    # print('right_image_path')
                    # print(f'{root}\{file}')
                    right_image_path.append(f'{root}\{file}')
        
        for root, _, files in os.walk(f'{self.root_path}/disp_noc'):
            for file in files:
                if file.endswith('.pfm'):
                    # print('left_disp_path')
                    # print(f'{root}\{file}')
                    left_disp_path.append(f'{root}\{file}')

        return (left_image_path, right_image_path, left_disp_path,)


class SceneflowDataset(Dataset):  # type: ignore[type-arg]  # I don't know why this typing ignore is needed on the class level...
    """
    Sceneflow dataset composed of FlyingThings3D, Driving, and Monkaa
    https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html

    Download the RGB (cleanpass) PNG image and the Disparity files

    The train set includes FlyingThings3D Train folder and all files in Driving and Monkaa folders
    The test set includes FlyingThings3D Test folder
    """

    def __init__(self,
                 root_path: Path,
                 transforms: st.TorchTransformers,
                 string_exclude: Optional[str] = None,
                 string_include: Optional[str] = None
                 ):
        self.root_path = root_path
        self.string_exclude = string_exclude
        self.string_include = string_include

        if not isinstance(transforms, list):
            _transforms = [transforms]
        else:
            _transforms = transforms

        self.transforms = _transforms

        self.left_image_path, self.right_image_path, self.left_disp_path, self.right_disp_path = self.get_paths()

    def __len__(self) -> int:
        return len(self.left_image_path)

    def __getitem__(self, index: int) -> st.Sample_Torch:
        # left = image_loader(self.left_image_path[index])
        # right = image_loader(self.right_image_path[index])
        left = webp_loader(self.left_image_path[index])
        right = webp_loader(self.right_image_path[index])

        disp_left, _ = pfm_loader(self.left_disp_path[index])
        disp_left = disp_left[..., np.newaxis]
        disp_left = np.ascontiguousarray(disp_left)

        disp_right, _ = pfm_loader(self.right_disp_path[index])
        disp_right = disp_right[..., np.newaxis]
        disp_right = np.ascontiguousarray(disp_right)

        # I'm not sure why I need the following type ignore...
        sample: st.Sample_Numpy = {'left': left, 'right': right, 'disp_left': disp_left, 'disp_right': disp_right}  # type: ignore[assignment]

        # print('before')
        # print(sample['left'].shape, sample['left'].dtype, \
        #     sample['left'].max(), sample['left'].min())
        # print(sample['disp_left'].shape, sample['disp_left'].dtype, \
        #     sample['disp_left'].max(), sample['disp_left'].min())

        torch_sample = ToTensor()(sample)
        # print('intermediate')
        # print(torch_sample.shape)
        # print(torch_sample['left'].shape, torch_sample['left'].dtype, \
        #     torch_sample['left'].max(), torch_sample['left'].min())
        # print(torch_sample['disp_left'].shape, torch_sample['disp_left'].dtype, \
        #     torch_sample['disp_left'].max(), torch_sample['disp_left'].min())

        for transform in self.transforms:
            torch_sample = transform(torch_sample)

        # print('after')
        # print(torch_sample.shape)
        # print(torch_sample['left'].shape, torch_sample['left'].dtype, \
        #     torch_sample['left'].max(), torch_sample['left'].min())
        # print(torch_sample['disp_left'].shape, torch_sample['disp_left'].dtype, \
        #     torch_sample['disp_left'].max(), torch_sample['disp_left'].min())

        return torch_sample

    def get_paths(self) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
        """
        string_exclude: If this string appears in the parent path of an image, don't add them to the dataset (ie. 'TEST' will exclude any path with 'TEST' in Path.parts)
        string_include: If this string DOES NOT appear in the parent path of an image, don't add them to the dataset (ie. 'TEST' will require 'TEST' to be in the Path.parts)
        if shuffle is None, don't shuffle, else shuffle.
        """

        left_image_path = []
        right_image_path = []
        left_disp_path = []
        right_disp_path = []

        for root, _, files in os.walk(f'{self.root_path}'):
            for file in files:
                if file.endswith('.webp') and root.endswith(r'\left'):
                    left_image_item = f'{root}\{file}'
                    # print(left_image_item)
                    left_image_path.append(left_image_item)

        for root, _, files in os.walk(f'{self.root_path}'):
            for file in files:
                if file.endswith('.webp') and root.endswith(r'\right'):
                    right_image_item = f'{root}\{file}'
                    # print(right_image_item)
                    right_image_path.append(right_image_item)
        
        left_disp_path = []
        for root, _, files in os.walk(f'{self.root_path}'):
            for file in files:
                if file.endswith('.pfm') and root.endswith(r'\left'):
                    left_disp_item = f'{root}\{file}'
                    # print(left_disp_item)
                    left_disp_path.append(left_disp_item)
        
        right_disp_path = []
        for root, _, files in os.walk(f'{self.root_path}'):
            for file in files:
                if file.endswith('.pfm') and root.endswith(r'\right'):
                    right_disp_item = f'{root}\{file}'
                    # print(right_disp_item)
                    right_disp_path.append(right_disp_item)

        # print(len(left_image_path), len(right_image_path), len(left_disp_path), len(right_disp_path))

        for i in list(range(100)):
            rng_idx = random.randint(0, len(left_image_path)-1)
            # print()
            # print(i)
            # print(left_image_path[rng_idx])
            # print(right_image_path[rng_idx])
            # print(left_disp_path[rng_idx])
            # print(right_disp_path[rng_idx])

        return (left_image_path, right_image_path, left_disp_path, right_disp_path)


class CenterCrop(st.TorchTransformer):
    """
    """

    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, sample: st.Sample_Torch) -> st.Sample_Torch:
        height, width = sample['left'].size()[-2:]
        output_height = int(self.scale*height)
        output_width = int(self.scale*width)
        cropper = T.CenterCrop((output_height, output_width))
        for name, x in sample.items():  # pylint: disable=invalid-name
            sample[name] = cropper(x)
        return sample


class ToTensor(st.NumpyToTorchTransformer):
    """
    Converts the left, right, and disparity maps into FloatTensors.
    Left and right uint8 images get rescaled to [0,1] floats.
    Disparities are already floats and just get turned into tensors.
    """

    @staticmethod
    def __call__(sample: st.Sample_Numpy) -> st.Sample_Torch:
        torch_sample: st.Sample_Torch = {}
        for name, x in sample.items():  # pylint: disable=invalid-name
            if x.dtype == 'uint16':
                x = x / 256.0
                x = x.astype('uint8')
            torch_sample[name] = T.functional.to_tensor(x.copy())
        return torch_sample


class PadSampleToBatch(st.TorchTransformer):
    """
    Unsqueezes the first dimension to be 1 when loading in single image pairs.
    """

    @staticmethod
    def __call__(sample: st.Sample_Torch) -> st.Sample_Torch:
        for name, x in sample.items():  # pylint: disable=invalid-name
            sample[name] = torch.unsqueeze(x, dim=0)
        return sample


class Resize(st.TorchTransformer):
    """
    Resizes each of the images in a batch to a given height and width
    """

    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, sample: st.Sample_Torch) -> st.Sample_Torch:
        for name, x in sample.items():
            sample[name] = T.functional.resize(x, self.size)
        return sample


class Rescale(st.TorchTransformer):
    """
    Rescales the left and right image tensors (initially ranged between [0, 1]) and rescales them to be between [-1, 1].
    """

    @staticmethod
    def __call__(sample: st.Sample_Torch) -> st.Sample_Torch:
        for name in ['left', 'right']:
            sample[name] = (sample[name] - 0.5) * 2
        return sample


def OLD_plot_figure(left: torch.Tensor, right: torch.Tensor, disp_gt: torch.Tensor, disp_pred: torch.Tensor) -> plt.figure:
    """
    Helper function to plot the left/right image pair from the dataset (ie. normalized between -1/+1 and c,h,w) and the
    ground truth disparity and the predicted disparity.  The disparities colour range between ground truth disparity min and max.
    """
    plt.close('all')
    fig, ax = plt.subplots(ncols=2, nrows=2)
    left = (torch.moveaxis(left, 0, 2) + 1) / 2
    right = (torch.moveaxis(right, 0, 2) + 1) / 2
    disp_gt = torch.moveaxis(disp_gt, 0, 2)
    disp_pred = torch.moveaxis(disp_pred, 0, 2)
    ax[0, 0].imshow(left, cmap='plasma')
    ax[0, 1].imshow(right, cmap='plasma')
    ax[1, 0].imshow(disp_gt, vmin=disp_gt.min(), vmax=disp_gt.max(), cmap='plasma')
    im = ax[1, 1].imshow(disp_pred, vmin=disp_gt.min(), vmax=disp_gt.max(), cmap='plasma')
    ax[0, 0].title.set_text('Left')
    ax[0, 1].title.set_text('Right')
    ax[1, 0].title.set_text('Ground truth disparity')
    ax[1, 1].title.set_text('Predicted disparity')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.27])
    fig.colorbar(im, cax=cbar_ax)
    return fig

def plot_figure(left: torch.Tensor, right: torch.Tensor, disp_gt, disp_pred) -> plt.figure:
    """
    Helper function to plot the left/right image pair from the dataset (ie. normalized between -1/+1 and c,h,w) and the
    ground truth disparity and the predicted disparity.  The disparities colour range between ground truth disparity min and max.
    """
    plt.close('all')
    fig, ax = plt.subplots(ncols=2, nrows=2)
    left = (left + 1) / 2   # rescale
    right = (right + 1) / 2   # rescale
    # disp_gt = torch.moveaxis(disp_gt, 0, 2)
    # disp_pred = torch.moveaxis(disp_pred, 0, 2)
    ax[0, 0].imshow(left.squeeze().moveaxis(0, 2), cmap='plasma')
    ax[0, 1].imshow(right.squeeze().moveaxis(0, 2), cmap='plasma')
    ax[1, 0].imshow(disp_gt, vmin=disp_gt.min(), vmax=disp_gt.max(), cmap='plasma')
    im = ax[1, 1].imshow(disp_pred, vmin=disp_gt.min(), vmax=disp_gt.max(), cmap='plasma')
    ax[0, 0].title.set_text('Left')
    ax[0, 1].title.set_text('Right')
    ax[1, 0].title.set_text('Ground truth disparity')
    ax[1, 1].title.set_text('Predicted disparity')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.27])
    fig.colorbar(im, cax=cbar_ax)
    return fig

def plot_and_save_figure(left: torch.Tensor, right: torch.Tensor, disp_gt, disp_pred) -> plt.figure:
    """
    Helper function to plot and save the left/right image pair from the dataset (ie. normalized between -1/+1 and c,h,w) and the
    ground truth disparity and the predicted disparity.  The disparities colour range between ground truth disparity min and max.
    """
    plt.close('all')
    fig, ax = plt.subplots(ncols=2, nrows=2)
    left = (left + 1) / 2   # rescale
    right = (right + 1) / 2   # rescale
    # disp_gt = torch.moveaxis(disp_gt, 0, 2)
    # disp_pred = torch.moveaxis(disp_pred, 0, 2)
    ax[0, 0].imshow(left.squeeze().moveaxis(0, 2), cmap='plasma')
    ax[0, 1].imshow(right.squeeze().moveaxis(0, 2), cmap='plasma')
    ax[1, 0].imshow(disp_gt, vmin=disp_gt.min(), vmax=disp_gt.max(), cmap='plasma')
    im = ax[1, 1].imshow(disp_pred, vmin=disp_gt.min(), vmax=disp_gt.max(), cmap='plasma')
    ax[0, 0].title.set_text('Left')
    ax[0, 1].title.set_text('Right')
    ax[1, 0].title.set_text('Ground truth disparity')
    ax[1, 1].title.set_text('Predicted disparity')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.27])
    fig.colorbar(im, cax=cbar_ax)

    IMG_DIR = r'D:\__repos\engineering_thesis\src\img'
    timestamp = datetime.now().strftime("%Y-%m-%d___%H%M__%S%f")[:-4]
    plt.savefig(f'{IMG_DIR}/results_{timestamp}.png', dpi=300)

    return fig