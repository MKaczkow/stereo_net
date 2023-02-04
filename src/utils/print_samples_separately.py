import numpy as np
import torch
# from model_kitti_2012 import StereoNet
# from model_kitti_2015 import StereoNet
from model_sceneflow import StereoNet
import utils as utils
import matplotlib.pyplot as plt


KITTI_2012_ROOT = r'D:\engineering_thesis_data\Kitti_2012\data_stereo_flow\training'
KITTI_2015_ROOT = r'D:\engineering_thesis_data\Kitti_2015\data_scene_flow\training'
SCENEFLOW_ROOT = r'D:\engineering_thesis_data\SceneFlow'
IMG_DIR = r'D:\__repos\engineering_thesis\src\img'


def main():

    kitti_2012_dataset = utils.Kitti2012Dataset(KITTI_2012_ROOT, transforms=[])
    kitti_2015_dataset = utils.Kitti2015Dataset(KITTI_2015_ROOT, transforms=[])
    sceneflow_dataset = utils.SceneflowDataset(SCENEFLOW_ROOT, transforms=[])

    kitti_2012_indexes = [49, 121]
    kitti_2015_indexes = [20, 64]
    sceneflow_indexes = [8078, 16134, 33599, 39610]


    for idx in kitti_2012_indexes:

        sample = kitti_2012_dataset[idx]

        transformers = [
            utils.PadSampleToBatch(),
            ]
            
        for transformer in transformers:
            sample = transformer(sample)

        model = StereoNet.load_from_checkpoint(
            r''
            )

        model.eval()
        with torch.no_grad():
            batched_prediction = model(sample)

        single_prediction = batched_prediction[0].numpy()
        single_prediction = np.moveaxis(single_prediction, 0, 2).squeeze() 
        single_prediction = np.where(sample['disp_left'].squeeze() > 0, single_prediction, 0)

        error_plane = np.abs(single_prediction - sample['disp_left'].squeeze().to('cpu').numpy())

        plt.imsave(f'{IMG_DIR}/new/kitti_2012/{idx}_left.png', sample['left'].squeeze().moveaxis(0, 2).cpu().numpy())
        plt.imsave(f'{IMG_DIR}/new/kitti_2012/{idx}_right.png', sample['right'].squeeze().moveaxis(0, 2).cpu().numpy())
        plt.imsave(f'{IMG_DIR}/new/kitti_2012/{idx}_gt.png', sample['disp_left'].squeeze(), vmin=sample['disp_left'].min(), vmax=sample['disp_left'].max(), cmap='plasma')
        plt.imsave(f'{IMG_DIR}/new/kitti_2012/{idx}_pred.png', single_prediction, vmin=sample['disp_left'].min(), vmax=sample['disp_left'].max(), cmap='plasma')
        plt.imsave(f'{IMG_DIR}/new/kitti_2012/{idx}_error.png', error_plane, vmin=sample['disp_left'].min(), vmax=sample['disp_left'].max(), cmap='plasma')



    for idx in kitti_2015_indexes:

        sample = kitti_2015_dataset[idx]

        transformers = [
            utils.PadSampleToBatch(),
            ]
            
        for transformer in transformers:
            sample = transformer(sample)

        model = StereoNet.load_from_checkpoint(
            r'D:\__repos\engineering_thesis\experiments\src\stereonet\lightning_logs\version_66\checkpoints\epoch=35-step=5760.ckpt'
            )

        model.eval()
        with torch.no_grad():
            batched_prediction = model(sample)

        single_prediction = batched_prediction[0].numpy()
        single_prediction = np.moveaxis(single_prediction, 0, 2).squeeze() 
        single_prediction = np.where(sample['disp_left'].squeeze() > 0, single_prediction, 0)

        error_plane = np.abs(single_prediction - sample['disp_left'].squeeze().to('cpu').numpy())

        plt.imsave(f'{IMG_DIR}/new/kitti_2015/{idx}_left.png', sample['left'].squeeze().moveaxis(0, 2).cpu().numpy())
        plt.imsave(f'{IMG_DIR}/new/kitti_2015/{idx}_right.png', sample['right'].squeeze().moveaxis(0, 2).cpu().numpy())
        plt.imsave(f'{IMG_DIR}/new/kitti_2015/{idx}_gt.png', sample['disp_left'].squeeze(), vmin=sample['disp_left'].min(), vmax=sample['disp_left'].max(), cmap='plasma')
        plt.imsave(f'{IMG_DIR}/new/kitti_2015/{idx}_pred.png', single_prediction, vmin=sample['disp_left'].min(), vmax=sample['disp_left'].max(), cmap='plasma')
        plt.imsave(f'{IMG_DIR}/new/kitti_2015/{idx}_error.png', error_plane, vmin=sample['disp_left'].min(), vmax=sample['disp_left'].max(), cmap='plasma')



    for idx in sceneflow_indexes:

        sample = sceneflow_dataset[idx]

        transformers = [
            utils.PadSampleToBatch(),
            ]
            
        for transformer in transformers:
            sample = transformer(sample)

        model = StereoNet.load_from_checkpoint(
            r'D:\__repos\engineering_thesis\experiments\src\stereonet\lightning_logs\version_38\checkpoints\epoch=2-step=864006.ckpt'
            )

        model.eval()
        with torch.no_grad():
            batched_prediction = model(sample)

        single_prediction = batched_prediction[0].numpy()
        single_prediction = np.moveaxis(single_prediction, 0, 2).squeeze() 

        error_plane = np.abs(single_prediction - sample['disp_left'].squeeze().to('cpu').numpy())

        plt.imsave(f'{IMG_DIR}/new/sceneflow/{idx}_left.png', sample['left'].squeeze().moveaxis(0, 2).cpu().numpy())
        plt.imsave(f'{IMG_DIR}/new/sceneflow/{idx}_right.png', sample['right'].squeeze().moveaxis(0, 2).cpu().numpy())
        plt.imsave(f'{IMG_DIR}/new/sceneflow/{idx}_gt.png', sample['disp_left'].squeeze(), vmin=sample['disp_left'].min(), vmax=sample['disp_left'].max(), cmap='plasma')
        plt.imsave(f'{IMG_DIR}/new/sceneflow/{idx}_pred.png', single_prediction, vmin=sample['disp_left'].min(), vmax=sample['disp_left'].max(), cmap='plasma')
        plt.imsave(f'{IMG_DIR}/new/sceneflow/{idx}_error.png', error_plane, vmin=sample['disp_left'].min(), vmax=sample['disp_left'].max(), cmap='plasma')


if __name__ == "__main__":
    main()
