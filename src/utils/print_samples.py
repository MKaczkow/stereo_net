import numpy as np
import torch
from model_kitti_2012 import StereoNet
import utils as utils


KITTI_2012_ROOT = r'D:\engineering_thesis_data\Kitti_2012\data_stereo_flow\training'
KITTI_2015_ROOT = r'D:\engineering_thesis_data\Kitti_2015\data_scene_flow\training'
SCENEFLOW_ROOT = r'D:\engineering_thesis_data\SceneFlow'


def main():
    kitti_2012_dataset = utils.Kitti2012Dataset(KITTI_2012_ROOT, transforms=[])
    kitti_2015_dataset = utils.Kitti2015Dataset(KITTI_2015_ROOT, transforms=[])
    sceneflow_dataset = utils.SceneflowDataset(SCENEFLOW_ROOT, transforms=[])
    
    for _ in range(50):
    # for _ in range(100):

        dataset = 'kitti_2012'
        idx = np.random.randint(0, len(kitti_2012_dataset))
        sample = kitti_2012_dataset[idx]
        
        # dataset = 'kitti_2015'
        # idx = np.random.randint(0, len(kitti_2015_dataset))
        # sample = kitti_2015_dataset[idx]

        # dataset = 'sceneflow'
        # idx = np.random.randint(0, len(sceneflow_dataset))
        # sample = sceneflow_dataset[idx]

        transformers = [
            utils.PadSampleToBatch(),
            ]
            
        for transformer in transformers:
            sample = transformer(sample)

        model = StereoNet.load_from_checkpoint(
            # BEST SCENE FLOW
            # r'D:\__repos\engineering_thesis\experiments\src\stereonet\lightning_logs\version_38\checkpoints\epoch=2-step=864006.ckpt'
            
            # BEST KITTI 2015
            # r'D:\__repos\engineering_thesis\experiments\src\stereonet\lightning_logs\version_66\checkpoints\epoch=35-step=5760.ckpt'

            # BEST KITTI 2012
            r'D:\__repos\engineering_thesis\experiments\src\stereonet\lightning_logs\version_67\checkpoints\epoch=46-step=7285.ckpt'
            )

        model.eval()
        with torch.no_grad():
            batched_prediction = model(sample)

        single_prediction = batched_prediction[0].numpy()
        single_prediction = np.moveaxis(single_prediction, 0, 2).squeeze() 
        if dataset == 'kitti_2012' or dataset == 'kitti_2015':
            single_prediction = np.where(sample['disp_left'].squeeze() > 0, single_prediction, 0)

        figure = utils.plot_and_save_example_figure(
            sample['left'], sample['right'], 
            sample['disp_left'].squeeze(), single_prediction.squeeze(),
            dataset, idx
            )

if __name__ == "__main__":
    main()
