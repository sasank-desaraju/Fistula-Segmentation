import pytorch_lightning
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureType,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import random
import nibabel as nib
import numpy as np

class FistulaDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        """
        self.data = glob.glob(os.path.join(self.data_dir, '*.nii.gz'))
        self.data.sort()
        self.labels = glob.glob(os.path.join(self.data_dir, '*.nii.gz'))
        self.labels.sort()
        self.data = list(zip(self.data, self.labels))
        """

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        '''
        
        image = nib.load(self.data[idx]['image'])
        image = np.array(image.dataobj)
        label = nib.load(self.data[idx]['label'])
        label = np.array(label.dataobj)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample