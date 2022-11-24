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
import SimpleITK as sitk

class FistulaDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.resample = sitk.ResampleImageFilter()
        self.resample.SetSize((96, 512, 512))
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
        
        #image = nib.load(self.data[idx]['image'])
        image = sitk.ReadImage(self.data[idx]['image'], imageIO="NiftiImageIO")
        image = self.resample.Execute(image)
        image = sitk.GetArrayFromImage(image)       # This might be redundant. Maybe only either this or the next line is needed.
        image = np.array(image, dtype=np.float32)
        #image = np.array(image.dataobj, dtype=np.float32)

        #label = nib.load(self.data[idx]['label'])
        label = sitk.ReadImage(self.data[idx]['label'], imageIO="NiftiImageIO")
        label = self.resample.Execute(label)
        label = sitk.GetArrayFromImage(label)       # This might be redundant. Maybe only either this or the next line is needed.
        label = np.array(label, dtype=np.float32)
        #label = np.array(label.dataobj, dtype=np.float32)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample