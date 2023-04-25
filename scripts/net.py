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
from monai.networks.nets import UNet, SwinUNETR
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, compute_dice, compute_iou
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader
from monai.config import print_config
from monai.apps import download_and_extract
from monai.transforms import SaveImage          # TODO: Use this to save test output images for comparison
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import tempfile
import shutil
import os
import glob
import random
from torch.profiler import profile, record_function, ProfilerActivity

from dataset import FistulaDataset

class SegmentationNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        )
        """
        self._model = SwinUNETR(
            img_size=config.dataset['IMAGE_SIZE'],
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            depths=(2, 2, 2, 2)
        )
        """
        # * Send the model to GPU
        self._model.cuda()
        # * Assert that the model is on GPU
        assert next(self._model.parameters()).is_cuda, 'Model is not on GPU but rather on ' + str(next(self._model.parameters()).device)

        # TODO: Should we use 2 out_channels and softmax or 1 out_channel and sigmoid?
        self.loss_function = DiceLoss(to_onehot_y=False, softmax=False)
        # Make loss function for 2 classes
        #self.loss_function = DiceLoss(to_onehot_y=False, softmax=False)
        # TODO: Do we want to use to_onehot_y?
        # It seems like some other people are indeed using it.
        #self.loss_function = DiceLoss(to_onehot_y=True, sigmoid=True)
        # TODO: What are post_pred and post_label???
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])

        #self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.dice_metric = compute_dice
        self.iou_metric = compute_iou
        self.best_val_dice = 0
        self.best_val_epoch = 0
        #self.prepare_data()

    def forward(self, x):
        """
        with profile(activities=[ProfilerActivity.CPU],
                profile_memory=True, record_shapes=True) as prof:
            # FIXED: We're getting an OOM error here. How big is the model?
            # This was fixed for now by only running it on HPG.
            # How big are the loaded nifti images?
        """
        output = self._model(x)
            #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        return output

    def prepare_data(self):

        """
        # set deterministic training for reproducibility
        set_determinism(seed=42)        # HHG2G reference lol
        random.seed(42)

        # set up the correct data path
        #images = sorted(glob.glob(os.path.join(self.data_dir, "*_image.nii.gz"), recursive=True))
        images = []
        labels = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith("_image.nii.gz"):
                    images.append(os.path.join(root, file))
                if file.endswith("_label.nii.gz"):
                    labels.append(os.path.join(root, file))
        images = sorted(images)
        labels = sorted(labels)
        #labels = sorted(glob.glob(os.path.join(self.data_dir, "*_label.nii.gz"), recursive=True))
        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(images, labels)
        ]
        random.shuffle(data_dicts)
        assert len(data_dicts) == 50, f'data_dicts is not 50 long but rather {len(data_dicts)}'
        train_files, val_files, test_files = data_dicts[:35], data_dicts[35:45], data_dicts[45:]
        """


        # define the data transforms
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    # TODO: Check these, including the a_min and a_max.
                    keys=["image"], a_min=-57, a_max=164,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                # randomly crop out patch samples from
                # big image based on pos / neg ratio
                # the image centers of negative samples
                # must be in valid image area
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                # user can also add other random transforms
                #                 RandAffined(
                #                     keys=['image', 'label'],
                #                     mode=('bilinear', 'nearest'),
                #                     prob=1.0,
                #                     spatial_size=(96, 96, 96),
                #                     rotate_range=(0, 0, np.pi/15),
                #                     scale_range=(0.1, 0.1, 0.1)),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )

        # we use cached datasets - these are 10x faster than regular datasets
        """
        self.train_ds = CacheDataset(
            #data=train_files, transform=train_transforms,
            data=train_files, transform=None,
            cache_rate=1.0, num_workers=4,
        )
        """

        #self.train_dataset = FistulaDataset(data=train_files, transform=None)

        """
        self.val_ds = CacheDataset(
            #data=val_files, transform=val_transforms,
            data=val_files, transform=None,
            cache_rate=1.0, num_workers=4,
        )
        """

        #self.val_dataset = FistulaDataset(data=val_files, transform=None)
#         self.train_ds = monai.data.Dataset(
#             data=train_files, transform=train_transforms)
#         self.val_ds = monai.data.Dataset(
#             data=val_files, transform=val_transforms)

        #self.test_dataset = FistulaDataset(data=test_files, transform=None)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        # Make the labels binary two-class
        labels = torch.where(labels > 0, torch.ones_like(labels), torch.zeros_like(labels))
        labels = torch.stack((labels, 1 - labels), dim=1)
        labels = torch.squeeze(labels, dim=2)

        preds = self(images)
        #print(f'Train preds shape: {preds.shape} and labels shape: {labels.shape}')
        loss = self.loss_function(preds, labels)
        dice_score = self.dice_metric(y_pred=preds, y=labels).mean()            # Average across batch
        iou_score = self.iou_metric(y_pred=preds, y=labels).mean()              # Average across batch
        self.log_dict(
            {
                "train/loss": loss,
                "train/dice": dice_score,
                "train/IOU": iou_score,
                "train/epoch": self.current_epoch
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        # Make the labels binary two-class
        labels = torch.where(labels > 0, torch.ones_like(labels), torch.zeros_like(labels))
        labels = torch.stack((labels, 1 - labels), dim=1)
        labels = torch.squeeze(labels, dim=2)

        preds = self(images)
        #print(f'Preds shape: {preds.shape} and labels shape: {labels.shape}')
        loss = self.loss_function(preds, labels)
        dice_score = self.dice_metric(y_pred=preds, y=labels).mean()            # Average across batch
        iou_score = self.iou_metric(y_pred=preds, y=labels).mean()              # Average across batch
        self.log_dict(
            {
                "val/loss": loss,
                "val/dice": dice_score,
                "val/IOU": iou_score,
                "val/epoch": self.current_epoch
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        # I'm doing this for the callbacks to work. Idk how it works/conflicts with the log_dict above.
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        #outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        #labels = [self.post_label(i) for i in decollate_batch(labels)]
        
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        
        # Make the labels binary two-class
        labels = torch.where(labels > 0, torch.ones_like(labels), torch.zeros_like(labels))
        labels = torch.stack((labels, 1 - labels), dim=1)
        labels = torch.squeeze(labels, dim=2)
        
        preds = self(images)

        loss = self.loss_function(preds, labels)
        dice_score = self.dice_metric(y_pred=preds, y=labels).mean()            # Average across batch
        iou_score = self.iou_metric(y_pred=preds, y=labels).mean()              # Average across batch
        self.log_dict(
            {
                "test/loss": loss,
                "test/dice": dice_score,
                "test/IOU": iou_score,
                "test/epoch": self.current_epoch        # Idk what this even means lol
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        # * Save the predictions in .nii.gz format
        for i in range(len(preds)):
            break
            pred = preds[i]
            label = labels[i]
            image = images[i]
            pred = torch.argmax(pred, dim=0)
            pred = pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            image = image.detach().cpu().numpy()
            pred = nib.Nifti1Image(pred, np.eye(4))
            label = nib.Nifti1Image(label, np.eye(4))
            image = nib.Nifti1Image(image, np.eye(4))
            nib.save(pred, f"pred_{i}.nii.gz")
            nib.save(label, f"label_{i}.nii.gz")
            nib.save(image, f"image_{i}.nii.gz")

        return loss

    def old_validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        return {"log": tensorboard_logs}
