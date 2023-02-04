import numpy as np
import torch
from model_sceneflow import StereoNet
import utils as utils
from datetime import datetime, timedelta


KITTI_2012_ROOT = r'D:\engineering_thesis_data\Kitti_2012\data_stereo_flow\training'
KITTI_2015_ROOT = r'D:\engineering_thesis_data\Kitti_2015\data_scene_flow\training'
SCENEFLOW_ROOT = r'D:\engineering_thesis_data\SceneFlow'


def main():
    kitti_2012_dataset = utils.Kitti2012Dataset(KITTI_2012_ROOT, transforms=[])
    kitti_2015_dataset = utils.Kitti2015Dataset(KITTI_2015_ROOT, transforms=[])
    sceneflow_dataset = utils.SceneflowDataset(SCENEFLOW_ROOT, transforms=[])
    
    # for _ in range(50):
    for _ in range(500):

        # dataset = 'kitti_2012'
        # idx = np.random.randint(0, len(kitti_2012_dataset))
        # sample = kitti_2012_dataset[idx]
        
        # dataset = 'kitti_2015'
        # idx = np.random.randint(0, len(kitti_2015_dataset))
        # sample = kitti_2015_dataset[idx]

        dataset = 'sceneflow'
        idx = np.random.randint(0, len(sceneflow_dataset))
        sample = sceneflow_dataset[idx]

        transformers = [
            utils.PadSampleToBatch(),
            ]
            
        for transformer in transformers:
            sample = transformer(sample)

        model = StereoNet.load_from_checkpoint(
            # BEST SCENE FLOW
            r'D:\__repos\engineering_thesis\experiments\src\stereonet\lightning_logs\version_38\checkpoints\epoch=2-step=864006.ckpt'
            
            # BEST KITTI 2015
            # r'D:\__repos\engineering_thesis\experiments\src\stereonet\lightning_logs\version_66\checkpoints\epoch=35-step=5760.ckpt'

            # BEST KITTI 2012
            # r'D:\__repos\engineering_thesis\experiments\src\stereonet\lightning_logs\version_67\checkpoints\epoch=46-step=7285.ckpt'
            )

        total_time = timedelta()

        model.eval()
        with torch.no_grad():
            t0 = datetime.now()
            batched_prediction = model(sample)
            t1 = datetime.now()

        total_time += (t1 - t0)
    
    # print(total_time / 50)
    print(total_time / 500)

    # KITTI 2012
    # 0:00:00.031000

    # KITTI 2015
    # 0:00:00.031659
    
    # SCENEFLOW
    # 0:00:00.003457
    
if __name__ == "__main__":
    main()
