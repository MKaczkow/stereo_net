import numpy as np
import torch
from datetime import datetime, timedelta

from misc import PadSampleToBatch

from ..datasets.SceneFlowDataset import SceneFlowDataset
from model.model_sceneflow import StereoNet

# from ..datasets.KITTI2012Dataset import KITTI2012Dataset
# from model.model_kitti_2015 import StereoNet

# from ..datasets.KITTI2015Dataset import KITTI2015Dataset
# from model.model_kitti_2012 import StereoNet


KITTI_2012_ROOT = r'D:\engineering_thesis_data\Kitti_2012\data_stereo_flow\training'
KITTI_2015_ROOT = r'D:\engineering_thesis_data\Kitti_2015\data_scene_flow\training'
SCENEFLOW_ROOT = r'D:\engineering_thesis_data\SceneFlow'


def main():
    # Choose proper dataset from the ones below
    # Uncomment respective lines
    
    # kitti_2012_dataset = Kitti2012Dataset(KITTI_2012_ROOT, transforms=[])
    # kitti_2015_dataset = Kitti2015Dataset(KITTI_2015_ROOT, transforms=[])
    sceneflow_dataset = SceneFlowDataset(SCENEFLOW_ROOT, transforms=[])
    
    for _ in range(500):

        # idx = np.random.randint(0, len(kitti_2012_dataset))
        # sample = kitti_2012_dataset[idx]
        
        # idx = np.random.randint(0, len(kitti_2015_dataset))
        # sample = kitti_2015_dataset[idx]

        idx = np.random.randint(0, len(sceneflow_dataset))
        sample = sceneflow_dataset[idx]

        transformers = [
            PadSampleToBatch(),
            ]
            
        for transformer in transformers:
            sample = transformer(sample)

        model = StereoNet.load_from_checkpoint(
            # BEST SCENE FLOW
            r'fill_in'
            
            # BEST KITTI 2015
            # r'fill_in'

            # BEST KITTI 2012
            # r'fill_in'
            )

        total_time = timedelta()

        model.eval()
        with torch.no_grad():
            t0 = datetime.now()
            _ = model(sample)
            t1 = datetime.now()

        total_time += (t1 - t0)
    
    print(total_time / 500)
    
    # Sample results
    # KITTI 2012
    # 0:00:00.031000

    # KITTI 2015
    # 0:00:00.031659
    
    # SCENEFLOW
    # 0:00:00.003457
    
if __name__ == "__main__":
    main()
