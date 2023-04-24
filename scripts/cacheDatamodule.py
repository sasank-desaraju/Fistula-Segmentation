"""
Sasank Desaraju
4/23/23

Using the MONAI Dataset class to cache data
"""

import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import os
from skimage import io
import cv2

#from cacheDataset import CacheFistulaDataset
from monai.data import CacheDataset, list_data_collate, decollate_batch, ThreadDataLoader
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
    ResampleToMatchd,
    EnsureTyped,
    RandFlipd,
    RandRotate90d,
    DivisiblePadd,
    ToTensord
)


class CacheDatamodule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.batch_size = self.config.datamodule['BATCH_SIZE']
        #self.log(batch_size)
        self.num_workers = self.config.datamodule['NUM_WORKERS']
        self.pin_memory = self.config.datamodule['PIN_MEMORY']
        self.shuffle = self.config.datamodule['SHUFFLE']

        # TODO: check train dataset length and integrity
        # TODO: check val dataset length and integrity

        # Create train, val, test splits of the data
        self.train_data = pd.read_csv(os.path.join(self.config.etl['DATA_DIR'], self.config.dataset['DATA_NAME'], 'train' + '_' + self.config.dataset['DATA_NAME'] + '.csv'))
        self.val_data = pd.read_csv(os.path.join(self.config.etl['DATA_DIR'], self.config.dataset['DATA_NAME'], 'val' + '_' + self.config.dataset['DATA_NAME'] + '.csv'))
        self.test_data = pd.read_csv(os.path.join(self.config.etl['DATA_DIR'], self.config.dataset['DATA_NAME'], 'test' + '_' + self.config.dataset['DATA_NAME'] + '.csv'))

        resample_model = os.path.join(self.config.dataset['IMAGE_ROOT'], '7/7_image.nii.gz')

        # For each splits, create a list of dictionaries containing the image paths, label paths, and patient IDs
        # Have both image and label paths prepended by the self.config.dataset['IMAGE_ROOT']
        self.train_files = [{'image': os.path.join(self.config.dataset['IMAGE_ROOT'], self.train_data.iloc[i]['image']),
                        'label': os.path.join(self.config.dataset['IMAGE_ROOT'], self.train_data.iloc[i]['label']),
                        'resample_model': resample_model,
                        'patient_id': self.train_data.iloc[i]['patient_id']} for i in range(len(self.train_data))]
        self.val_files = [{'image': os.path.join(self.config.dataset['IMAGE_ROOT'], self.val_data.iloc[i]['image']),
                        'label': os.path.join(self.config.dataset['IMAGE_ROOT'], self.val_data.iloc[i]['label']),
                        'resample_model': resample_model,
                        'patient_id': self.val_data.iloc[i]['patient_id']} for i in range(len(self.val_data))]
        self.test_files = [{'image': os.path.join(self.config.dataset['IMAGE_ROOT'], self.test_data.iloc[i]['image']),
                        'label': os.path.join(self.config.dataset['IMAGE_ROOT'], self.test_data.iloc[i]['label']),
                        'resample_model': resample_model,
                        'patient_id': self.test_data.iloc[i]['patient_id']} for i in range(len(self.test_data))]

        # Set transformations
        self.train_transforms = Compose([
            LoadImaged(keys=['image', 'label', 'resample_model']),
            EnsureChannelFirstd(keys=['image', 'label', 'resample_model']),
            #Spacingd(keys=['image', 'label'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')),
            ResampleToMatchd(keys=['image', 'label'], key_dst='resample_model', mode=('bilinear', 'nearest')),
            # Use ResampleToMatchd to resample both the image and label to the same resolution since not all of the images are the same resolution

            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', spatial_size=(64, 64, 64), pos=1, neg=1, num_samples=4, image_key='image', image_threshold=0),
            RandFlipd(keys=['image', 'label'], prob=0.2, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.2, spatial_axis=1),
            RandFlipd(keys=['image', 'label'], prob=0.2, spatial_axis=2),
            RandRotate90d(keys=['image', 'label'], prob=0.2, max_k=3),
            DivisiblePadd(keys=['image', 'label'], k=16),
            EnsureTyped(keys=['image', 'label'])
        ])

        self.val_transforms = Compose([
            LoadImaged(keys=['image', 'label', 'resample_model']),
            EnsureChannelFirstd(keys=['image', 'label', 'resample_model']),
            #Spacingd(keys=['image', 'label'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')),
            ResampleToMatchd(keys=['image', 'label'], key_dst='resample_model', mode=('bilinear', 'nearest')),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            #RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', spatial_size=(64, 64, 64), pos=1, neg=1, num_samples=4, image_key='image', image_threshold=0),
            #RandFlipd(keys=['image', 'label'], prob=0.2, spatial_axis=0),
            #RandFlipd(keys=['image', 'label'], prob=0.2, spatial_axis=1),
            #RandFlipd(keys=['image', 'label'], prob=0.2, spatial_axis=2),
            #RandRotate90d(keys=['image', 'label'], prob=0.2, max_k=3),
            DivisiblePadd(keys=['image', 'label'], k=16),
            EnsureTyped(keys=['image', 'label'])
        ])
        
        self.test_transforms = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            Spacingd(keys=['image', 'label'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            EnsureTyped(keys=['image', 'label'])
        ])

    def setup(self, stage):

        if stage == 'fit':
            """
            self.training_set = CacheFistulaDataset(config=self.config,
                                                    evaluation_type='train')
            self.validation_set = CacheFistulaDataset(config=self.config,
                                                    evaluation_type='val')
            """
            self.training_set = CacheDataset(data=self.train_files,
                                            transform=self.train_transforms,
                                            cache_rate=1.0,
                                            cache_num=2,
                                            num_workers=self.num_workers)
            self.validation_set = CacheDataset(data=self.val_files,
                                            transform=self.val_transforms,
                                            cache_rate=1.0,
                                            cache_num=2,
                                            num_workers=self.num_workers)
        if stage == 'test':
            """
            self.test_set = CacheFistulaDataset(config=self.config,
                                                evaluation_type='test')
            """
            self.test_set = CacheDataset(data=self.test_files,
                                        transform=self.test_transforms,
                                        cache_rate=1.0,
                                        cache_num=2,
                                        num_workers=self.num_workers)

        return
    
    def train_dataloader(self):
        train_loader = ThreadDataLoader(dataset=self.training_set,
                                        batch_size=self.batch_size,
                                        shuffle=self.shuffle,
                                        num_workers=self.num_workers,
                                        pin_memory=self.pin_memory)
        return train_loader
    
    def val_dataloader(self):
        val_loader = ThreadDataLoader(dataset=self.validation_set,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=self.num_workers,
                                        pin_memory=self.pin_memory)
        return val_loader
    
    def test_dataloader(self):
        test_loader = ThreadDataLoader(dataset=self.test_set,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=self.num_workers,
                                        pin_memory=self.pin_memory)
        return test_loader