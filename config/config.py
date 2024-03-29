import torch
import torch.nn as nn
import albumentations as A
import numpy as np
import time
import os

"""
Fistula Segmentation
"""
class Configuration:
    def __init__(self):
        self.init = {
            'PROJECT_NAME': 'FistulaSegmentation',
            'MODEL_NAME': 'Development',
            'RUN_NAME': time.strftime('%Y-%m-%d-%H-%M-%S'),
            'WANDB_RUN_GROUP': 'Local',
            'FAST_DEV_RUN': True,  # Runs inputted batches (True->1) and disables logging and some callbacks
            'MAX_EPOCHS': 3,
            'MAX_STEPS': -1,    # -1 means it will do all steps and be limited by epochs
            'STRATEGY': 'auto'    # This is the training strategy. Should be 'ddp' for multi-GPU (like HPG)
        }
        self.etl = {
            'DATA_DIR': "data",
            # Lol what is this?
            # HHG2TG lol; deterministic to aid reproducibility
            'RANDOM_STATE': 42,
        }

        self.dataset = {
            'DATA_NAME': 'BaseSplit',
            'USE_TRANSFORMS': False,
            'IMAGE_ROOT': '/home/sasank/Dropbox (UFL)/FistulaData/Segmentations/',
            #'IMAGE_ROOT': '/media/sasank/LinuxStorage/Dropbox (UFL)/FistulaData/Segmentations/',
            #'IMAGE_SIZE': (512, 512, 96)
            'IMAGE_SIZE': (256, 128, 256)
        }

        self.datamodule = {
            #'CKPT_FILE': '/media/sasank/LinuxStorage/Dropbox (UFL)/FistulaData/checkpoints/100Epoch87IOU.ckpt',
            'CKPT_FILE': '/home/sasank/Dropbox (UFL)/FistulaData/checkpoints/100Epoch87IOU.ckpt',
            'BATCH_SIZE': 1,
            'FIT_CACHE_NUM': 2,
            'SHUFFLE': True,        # Only for training; for test and val this is set in the datamodule script to False
            'NUM_WORKERS': 2,
            'PIN_MEMORY': False
        }


        # hyperparameters for training
        self.hparams = {
            'LOAD_FROM_CHECKPOINT': False,
            'learning_rate': 1e-3
        }

        #self.transform = None
        self.transform = \
        A.Compose([
            # Let's do only rigid transformations for now
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.2),
            A.Transpose(p=0.2),
        ],
        p=1.0)
