import pytorch_lightning
from pytorch_lightning.strategies.ddp import DDPStrategy
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
import argparse

from net import Net

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, required=False,
                    default='/media/sasank/LinuxStorage/Dropbox (UFL)/FistulaData/Segmentations/',
                    help='Directory for where images are stored')
parser.add_argument('--strategy', type=str, required=False,
                    default='/media/sasank/LinuxStorage/Dropbox (UFL)/FistulaData/Segmentations/',
                    help='Directory for where images are stored')
args = parser.parse_args()

data_dir = args.dataroot
#data_dir = '/media/sasank/LinuxStorage/Dropbox (UFL)/FistulaData/Segmentations/'

# initialise the LightningModule
net = Net(data_dir)

root_dir = os.getcwd()

# set up loggers and checkpoints
log_dir = os.path.join(root_dir, "logs")
tb_logger = pytorch_lightning.loggers.TensorBoardLogger(
    save_dir=log_dir
)

#ddp = DDPStrategy(find_unused_parameters=False)



"""
# initialise Lightning's trainer.
trainer = pytorch_lightning.Trainer(
    fast_dev_run=False,
    strategy=args.strategy,
    #gpus=[0],
    accelerator='gpu',
    devices=-1,        # this is the number of gpus to use, right?
    auto_select_gpus=True,
    #max_epochs=100,
    max_epochs=3,
    logger=tb_logger,
    enable_checkpointing=True,
    num_sanity_val_steps=1,
    log_every_n_steps=16
)

# train
trainer.fit(net)

print(
    f"train completed, best_metric: {net.best_val_dice:.4f} "
    f"at epoch {net.best_val_epoch}"
    )

# Saving model
torch.save(net.state_dict(), 'checkpoints/latest_model.pth')
"""

# Load model
net.load_state_dict(torch.load('checkpoints/latest_model.pth'))


# Evaluating the model on the validation set.
# Is this legit to evaluate on the validation set?
net.eval()
device = torch.device("cuda:0")
net.to(device)
with torch.no_grad():
    for i, val_data in enumerate(net.test_dataloader()):
        #roi_size = (160, 160, 160)
        # Using the same roi_size as in the training set.
        roi_size = (64, 64, 64)
        # Bigger batch size, though. I guess it's fine...
        sw_batch_size = 4
        # Unsqueezing the channel dimension *rolls eyes*.
        val_data["image"] = val_data["image"].unsqueeze(1)
        val_data["label"] = val_data["label"].unsqueeze(1)
        #print('val image shape is ', val_data["image"].shape)
        #print('val label shape is ', val_data["label"].shape)
        val_outputs = sliding_window_inference(
            val_data["image"].to(device), roi_size, sw_batch_size, net
        )
        # plot the slice [:, :, 80]
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image {i}")
        plt.imshow(val_data["image"][0, 0, :, :, 80], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label {i}")
        plt.imshow(val_data["label"][0, 0, :, :, 80])
        plt.subplot(1, 3, 3)
        plt.title(f"output {i}")
        plt.imshow(torch.argmax(
            val_outputs, dim=1).detach().cpu()[0, :, :, 80])
        #plt.show()
        plt.savefig('test_plots/latest_run/val_image_{}.png'.format(i))
