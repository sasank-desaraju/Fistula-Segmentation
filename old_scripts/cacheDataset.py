"""
Sasank Desaraju
4/23/23

Using the MONAI Dataset class to cache data
"""

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
from math import floor, ceil
import pandas as pd

class CacheFistulaDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = pd.read_csv(os.path.join(self.config.etl['DATA_DIR'], self.config.dataset['DATA_NAME'], self.evaluation_type + '_' + self.config.dataset['DATA_NAME'] + '.csv'))
        self.images = []
        self.labels = []
        self.patient_ids = []
        for i in range(len(self.data)):
            image = sitk.ReadImage(os.path.join(self.config.dataset['IMAGE_ROOT'], self.data['image'][i]), imageIO='NiftiImageIO')
            label = sitk.ReadImage(os.path.join(self.config.dataset['IMAGE_ROOT'], self.data['label'][i]), imageIO='NiftiImageIO')
            patient_id = self.data['patient_id'][i]
            self.images.append(image)
            self.labels.append(label)
            self.patient_ids.append(patient_id)