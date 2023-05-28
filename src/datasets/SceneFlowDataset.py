import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from ..utils.utils_io import webp_loader, pfm_loader


class SceneFlowDataset(Dataset):

    def __init__(self,
                 root_path,
                 transforms,
                 string_exclude,
                 string_include
                 ):
        self.root_path = root_path
        self.string_exclude = string_exclude
        self.string_include = string_include
        self.transforms = transforms

        self.left_image_path, self.right_image_path, \
        self.left_disp_path, self.right_disp_path = \
            self.get_paths()

    def __len__(self):
        return len(self.left_image_path)

    def __getitem__(self, index: int):

        left = webp_loader(self.left_image_path[index])
        right = webp_loader(self.right_image_path[index])

        disp_left, _ = pfm_loader(self.left_disp_path[index])
        disp_left = disp_left[..., np.newaxis]
        disp_left = np.ascontiguousarray(disp_left)

        disp_right, _ = pfm_loader(self.right_disp_path[index])
        disp_right = disp_right[..., np.newaxis]
        disp_right = np.ascontiguousarray(disp_right)

        sample = {
            'left': left, 'right': right, 
            'disp_left': disp_left, 'disp_right': disp_right
            }

        torch_sample = ToTensor()(sample)
        for transform in self.transforms:
            torch_sample = transform(torch_sample)

        return torch_sample

    def get_paths(self):

        left_image_path = []
        right_image_path = []
        left_disp_path = []
        right_disp_path = []

        for root, _, files in os.walk(f'{self.root_path}'):
            for file in files:
                if file.endswith('.webp') and root.endswith(r'\left'):
                    left_image_item = f'{root}\{file}'
                    left_image_path.append(left_image_item)

        for root, _, files in os.walk(f'{self.root_path}'):
            for file in files:
                if file.endswith('.webp') and root.endswith(r'\right'):
                    right_image_item = f'{root}\{file}'
                    right_image_path.append(right_image_item)
        
        left_disp_path = []
        for root, _, files in os.walk(f'{self.root_path}'):
            for file in files:
                if file.endswith('.pfm') and root.endswith(r'\left'):
                    left_disp_item = f'{root}\{file}'
                    left_disp_path.append(left_disp_item)
        
        right_disp_path = []
        for root, _, files in os.walk(f'{self.root_path}'):
            for file in files:
                if file.endswith('.pfm') and root.endswith(r'\right'):
                    right_disp_item = f'{root}\{file}'
                    right_disp_path.append(right_disp_item)

        return (
            left_image_path, right_image_path, 
            left_disp_path, right_disp_path
            )