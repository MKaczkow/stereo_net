from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from skimage import io

from .utils_freiburg import readPFM


def png_loader(path):
    img = io.imread(path)
    return img


def webp_loader(target_path):
    """Utility function to load images in .webp format."""
    img = Image.open(target_path)
    img_array = np.array(img)    
    return img_array


def pfm_loader(path):
    pfm = readPFM(path)
    return pfm


def plot_figure(left: torch.Tensor, right: torch.Tensor, disp_gt, disp_pred) -> plt.figure:
    """
    Helper function to plot the left/right image pair from the dataset (ie. normalized between -1/+1 and c,h,w) and the
    ground truth disparity and the predicted disparity.  The disparities colour range between ground truth disparity min and max.
    """
    plt.close('all')
    fig, ax = plt.subplots(ncols=2, nrows=2)
    left = (left + 1) / 2   # rescale
    right = (right + 1) / 2   # rescale
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
