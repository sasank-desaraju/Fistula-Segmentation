"""
Sasank Desaraju
4/4/2023
"""

from datetime import datetime
from importlib import import_module
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.accelerators import find_usable_cuda_devices
from net import SegmentationNet
#from datamodule import SegmentationDataModule
from cacheDatamodule import CacheDatamodule
import sys
import os
import time
import wandb


def main(config, wandb_logger):

    #data_module = SegmentationDataModule(config=config)
    data_module = CacheDatamodule(config=config)

    model = SegmentationNet(config=config)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        monitor='val/loss',
        filename=wandb_logger.name + 'lowest_val_loss',
        save_top_k=1,
        mode='min'
    )
    earlystopping_callback = EarlyStopping(
        monitor='val/loss',
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='min'
    )

    # Our trainer object contains a lot of important info.
    trainer = pl.Trainer(
        accelerator='cuda',
        devices=find_usable_cuda_devices(-1),
        #devices=-1,     # use all available devices (GPUs)
        #auto_select_gpus=True,  # helps use all GPUs, not quite understood...      # Deprecated
        logger=wandb_logger,   # tried to use a WandbLogger object. Hasn't worked...
        default_root_dir=os.getcwd(),
        #callbacks=[JTMLCallback(config, wandb_run)],    # pass in the callbacks we want
        #callbacks=[JTMLCallback(config, wandb_run), save_best_val_checkpoint_callback],
        #callbacks=[checkpoint_callback, earlystopping_callback],
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
        fast_dev_run=config.init['FAST_DEV_RUN'],
        max_epochs=config.init['MAX_EPOCHS'],
        max_steps=config.init['MAX_STEPS'],
        strategy=config.init['STRATEGY'])

    trainer.fit(model, data_module)

    # TODO: Are the trainer and Wandb doing the same thing/overwriting the checkpoint?
    #Save model using .ckpt file format. This includes .pth info and other (hparams) info.
    trainer.save_checkpoint(CKPT_DIR + config.init['WANDB_RUN_GROUP'] + config.init['MODEL_NAME'] + '.ckpt')
    
    # Save model using Wandb
    wandb.save(CKPT_DIR + config.init['WANDB_RUN_GROUP'] + '/' + config.init['MODEL_NAME'] + '.ckpt')

if __name__ == '__main__':

    CONFIG_DIR = os.getcwd() + '/config/'

    sys.path.append(CONFIG_DIR)
    config_module = import_module(sys.argv[1])

    # Instantiating the config file
    config = config_module.Configuration()

    # Setting the checkpoint directory
    CKPT_DIR = os.getcwd() + '/checkpoints/'

    wandb_logger = WandbLogger(
        project=config.init['PROJECT_NAME'],
        name=config.init['RUN_NAME'],
        log_model=True,
        #prefix='fit_',
        group=config.init['WANDB_RUN_GROUP'],
        job_type='fit',
        save_dir='logs/'
    )

    # List the names of all of the available CUDA devices
    for device in find_usable_cuda_devices(-1):
        #print(device)
        # Print the device name
        #print(torch.cuda.get_device_name(device))
        print(f'Device {device} is a {torch.cuda.get_device_name(device)} with {torch.cuda.get_device_capability(device)} cuda capability.')

    # If using tensor cores (such as the A100s on HPG), then set precision to 16-bit
    if config.init['WANDB_RUN_GROUP'] == 'HiPerGator':
        #torch.set_float32_matmul_precision('medium')
        torch.set_float32_matmul_precision('high')
    
    torch.multiprocessing.set_sharing_strategy('file_system')

    wandb_logger.log_hyperparams(params=config.init|config.etl|config.dataset|config.datamodule|config.hparams)
    # Print the wandb logger hyperparameters
    print(wandb_logger.experiment.config)

    main(config, wandb_logger)

    # Sync and close the Wandb logging. Good to have for DDP, I believe.
    wandb.finish()
