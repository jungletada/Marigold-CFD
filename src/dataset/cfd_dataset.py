import os
import random
import numpy as np
import torch
import tarfile
from PIL import Image
from torch.utils.data import Dataset

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode
from src.dataset.base_depth_dataset import DatasetMode
from src.util.depth_transform import DepthNormalizerBase


STAT_pressure={'min': -37.73662186, 'max': 57.6361618}
STAT_temperature={'min': 299.9764404, 'max':310.3595276}
STAT_velocity={'min': 0.0, 'max':0.3930110071636349}


class CFDDataset(Dataset):
    def __init__(self, 
                 dataset_dir, 
                 mode='train',
                 make_mask=True,
                ):
        
        super().__init__()
        self.dataset_dir = dataset_dir
        self.contour_dir = os.path.join(dataset_dir, 'contour')
        self.pressure_dir = os.path.join(dataset_dir, 'pressure')
        self.temperature_dir = os.path.join(dataset_dir, 'temperature')
        self.velocity_dir = os.path.join(dataset_dir, 'velocity')
        # self.filenames = sorted(os.listdir(self.contour_dir))
        self.target_names = []
        self.mode = mode
        self.make_mask = make_mask
        
        if mode == 'train':
            self.disp_name = 'CFD Train'
            with open(os.path.join(dataset_dir, 'list_train.txt'),'r') as f:
                filenames = f.readlines()
                self.filenames = [name.strip() + '.png' for name in filenames]
        else:
             self.disp_name = 'CFD Test'
             with open(os.path.join(dataset_dir, 'list_test.txt'),'r') as f:
                filenames = f.readlines()
                self.filenames = [name.strip() + '.png' for name in filenames]
                
        for base_dir in [self.pressure_dir, self.temperature_dir, self.velocity_dir]:
            for name in self.filenames:
                self.target_names.append(os.path.join(base_dir, name))

    def __len__(self):
        return len(self.target_names) # for three fieldss

    def __getitem__(self, index):
        file_name = self.target_names[index].split("/")[-1].replace('.png', '')
        rgb_rel_path = os.path.join(self.contour_dir, file_name + '.png')
        contour_norm, mask = self._load_rgb_data(rgb_rel_path=rgb_rel_path)
        field_norm = self._load_field_data(depth_rel_path=self.target_names[index])
        
        h, w = contour_norm.shape[1:]
        line_values = torch.linspace(1, 0, w, dtype=torch.float32)
        flow_norm = line_values.view(1, -1).expand(h, w).unsqueeze(0)

        if self.target_names[index].__contains__('pressure'):
            prompt = 'pressure field'
        elif self.target_names[index].__contains__('temperature'):
            prompt = 'temperature field'
        elif self.target_names[index].__contains__('velocity'):
            prompt = 'velocity field'
        else:
            raise NotImplementedError
        
        if self.mode == 'train':
            if random.random() < 0.5:
                contour_norm = contour_norm.flip(-1)
                field_norm = field_norm.flip(-1)
                flow_norm = flow_norm.flip(-1)
        
        outputs = {
            'inputs': contour_norm,
            'outputs': field_norm, 
            'flows': flow_norm,
            'prompt': prompt,
            'mask': mask,
            'file_name': file_name
        }
        return outputs 
    
    def _load_rgb_data(self, rgb_rel_path):
        rgb_data = Image.open(rgb_rel_path).convert('RGB')  # [H, W, 3]
        rgb_data = np.asarray(rgb_data)
        rgb_data = np.transpose(rgb_data, (2, 0, 1)).astype(int)  # [3, H, W]
        rgb_norm = rgb_data / 255.0 * 2.0 - 1.0      #  [0, 255] -> [-1, 1]
        rgb_norm = torch.from_numpy(rgb_norm).float()
        
        mask_data = Image.open(rgb_rel_path).convert('L')  # [H, W]
        mask_data = np.asarray(mask_data)
        mask_data = torch.from_numpy(mask_data / 255.0).float() 
        mask_data = mask_data.unsqueeze(0) > 0.5  # False: contour, True: field
        return rgb_norm, mask_data

    def _load_field_data(self, depth_rel_path):
        depth_data = Image.open(depth_rel_path).convert('L') # [H, W]
        depth_data = np.asarray(depth_data)
        depth_norm = depth_data / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        depth_norm = torch.from_numpy(depth_norm).float().unsqueeze(0)  # [1, H, W]
        return depth_norm

    