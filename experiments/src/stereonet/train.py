from pathlib import Path

from torch import Generator
from torch.utils.data import (
    random_split,
    DataLoader
)

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import(
     LearningRateMonitor,
     ModelCheckpoint
)

from utils import (
    Rescale, SceneflowDataset,
    Kitti2012Dataset, Kitti2015Dataset
)

from model_sceneflow import StereoNet
# from model_kitti_2015 import StereoNet
# from model_kitti_2012 import StereoNet


KITTI_2012_ROOT = \
    r'D:\engineering_thesis_data\Kitti_2012\data_stereo_flow\training'
KITTI_2015_ROOT = \
    r'D:\engineering_thesis_data\Kitti_2015\data_scene_flow\training'
SCENEFLOW_ROOT = r'D:\engineering_thesis_data\SceneFlow'
CHECKPOINT_PATH = \
    r'D:\__repos\engineering_thesis\experiments\src\stereonet\
    \lightning_logs\version_38\checkpoints\epoch=22-step=864006.ckpt'


def main(dataset):

    model = StereoNet()

    train_transforms = [
        Rescale()
    ]

    if dataset == 'kitti_2015':

        kitti_2015_dataset = Kitti2015Dataset(
            KITTI_2015_ROOT, transforms=train_transforms
            )
        train_len = int(0.8 * len(kitti_2015_dataset))
        val_len = len(kitti_2015_dataset) - train_len
        train_dataset, val_dataset = random_split(
            kitti_2015_dataset, 
            [train_len, val_len],
            Generator().manual_seed(420))
        val_transforms = [Rescale()]

    elif dataset == 'sceneflow':
        
        train_dataset = SceneflowDataset(
            SCENEFLOW_ROOT, string_exclude='TEST', 
            transforms=train_transforms
            )
        val_transforms = [Rescale()]
        val_dataset = SceneflowDataset(
            SCENEFLOW_ROOT, string_include='TEST', 
            transforms=val_transforms
            )
        print(len(train_dataset), len(val_dataset))
        
    elif dataset == 'kitti_2012':

        kitti_2012_dataset = Kitti2012Dataset(
            KITTI_2012_ROOT, transforms=train_transforms
            )
        train_len = int(0.8 * len(kitti_2012_dataset))
        val_len = len(kitti_2012_dataset) - train_len

        train_dataset, val_dataset = random_split(
            kitti_2012_dataset, 
            [train_len, val_len],
            Generator().manual_seed(420))
        val_transforms = [Rescale()]

    else:
        raise ValueError('Wrong dataset name, valid names: kitti, sceneflow')

    train_loader = DataLoader(
        train_dataset, batch_size=1, 
        shuffle=True, num_workers=4, drop_last=False
        )
    val_loader = DataLoader(
        val_dataset, batch_size=1, 
        shuffle=False, num_workers=4, drop_last=False
        )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss_epoch', save_top_k=-1, mode='min'
        )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = TensorBoardLogger(
        save_dir=str(Path.cwd()), name="lightning_logs"
        )
    trainer = Trainer(
        gpus=1, min_epochs=1, max_epochs=300, 
        logger=logger, callbacks=[lr_monitor, checkpoint_callback]
        )

    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader,
        ckpt_path=CHECKPOINT_PATH
        )


if __name__ == "__main__":
    # main('kitti_2015') 
    # main('kitti_2012') 
    main('sceneflow')
