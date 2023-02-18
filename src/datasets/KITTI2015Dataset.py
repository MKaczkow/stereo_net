import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from ..utils.utils_io import pfm_loader, png_loader


class Kitti2015Dataset(Dataset):

    def __init__(self,
                 root_path,
                 transforms,
                 string_exclude = None,
                 string_include = None
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

    def __getitem__(self, index):

        left = png_loader(self.left_image_path[index])
        right = png_loader(self.right_image_path[index])
        disp_left, _ = pfm_loader(self.left_disp_path[index])
        disp_right, _ = pfm_loader(self.right_disp_path[index])

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
   
        for root, _, files in os.walk(f'{self.root_path}/image_2'):
            for file in files:
                if file.endswith('_10.png'):
                    left_image_path.append(f'{root}\{file}')

        for root, _, files in os.walk(f'{self.root_path}/image_3'):
            for file in files:
                if file.endswith('_10.png'):
                    right_image_path.append(f'{root}\{file}')
        
        for root, _, files in os.walk(f'{self.root_path}/disp_noc_0'):
            for file in files:
                if file.endswith('.pfm'):
                    left_disp_path.append(f'{root}\{file}')

        for root, _, files in os.walk(f'{self.root_path}/disp_noc_1'):
            for file in files:
                if file.endswith('.pfm'):
                    right_disp_path.append(f'{root}\{file}')

        return (
            left_image_path, right_image_path, 
            left_disp_path, right_disp_path
            )
