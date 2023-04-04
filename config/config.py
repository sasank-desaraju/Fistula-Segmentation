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
            'MAX_EPOCHS': 1,
            'MAX_STEPS': -1,    # -1 means it will do all steps and be limited by epochs
            'STRATEGY': None    # This is the training strategy. Should be 'ddp' for multi-GPU (like HPG)
        }
        self.etl = {
            'DATA_DIR': "data",
            # Lol what is this?
            # HHG2TG lol; deterministic to aid reproducibility
            'RANDOM_STATE': 42,
        }

        # TODO: Must be figured out and changed, obviously
        self.dataset = {
            'DATA_NAME': 'Project_Split',
            'USE_ALBUMENTATIONS': False,
            'NUM_CLASSES': 10,
            'NUM_CHANNELS': 3,
            'IMAGE_HEIGHT': 300,
            'IMAGE_WIDTH': 300,
            'IMAGE_FILE': '/home/sasank/Documents/FML/FML-Bucket/data/data_train.npy',
            'LABEL_FILE': '/home/sasank/Documents/FML/FML-Bucket/data/labels_train.npy'
        }

        self.datamodule = {
            'NPY_DIRECTORY': '/data/',
            'CKPT_FILE': '/home/sasank/Documents/FML/FML-Bucket/checkpoints/epoch=0-step=2377.ckpt',
            'BATCH_SIZE': 1,
            'SHUFFLE': True,        # Only for training; for test and val this is set in the datamodule script to False
            'NUM_WORKERS': 2,
            'PIN_MEMORY': False
            #'SUBSET_PIXELS': True - this is now in dataset
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
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        p=1.0)
