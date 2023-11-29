from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    ResampleToMatchd,
    EnsureType,
    EnsureTyped,
    RandFlipd,
    RandRotate90d,
    DivisiblePadd,
    ToTensord,
    Invertd,
    SaveImaged
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
from monai.inferers import sliding_window_inference
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        self.config = config
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
            out_channels=2,
            depths=(2, 2, 2, 2),
            num_heads=(3, 5, 12, 24)
        )
        """
        # * Send the model to GPU
        self._model.cuda()
        # * Assert that the model is on GPU
        assert next(self._model.parameters()).is_cuda, 'Model is not on GPU but rather on ' + str(next(self._model.parameters()).device)

        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        # Make loss function for 2 classes
        #self.loss_function = DiceLoss(to_onehot_y=False, softmax=False)
        # TODO: Do we want to use to_onehot_y?
        # It seems like some other people are indeed using it.
        #self.loss_function = DiceLoss(to_onehot_y=True, sigmoid=True)
        # TODO: What are post_pred and post_label???
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        #self.dice_metric = compute_dice
        self.iou_metric = compute_iou
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []
        self.test_step_outputs = []
        #self.prepare_data()

    def forward(self, x):
        return self._model(x)

    #def prepare_data(self):
    def setup(self, stage):
        """
        The goal of this function is to create the train, val, and test datasets.
        These are set as self.train_ds, self.val_ds, and self.test_ds, respectively.
        Each of these should be a monai.data.Dataset object or some more advanced wrapper such as a monai.data.CacheDataset.
        This means that all other variables need not be fields of the SegmentationNet LightningModule class.
        """

        # config is accessible as self.config
        batch_size = self.config.datamodule['BATCH_SIZE']
        #self.log(batch_size)
        num_workers = self.config.datamodule['NUM_WORKERS']
        pin_memory = self.config.datamodule['PIN_MEMORY']
        shuffle = self.config.datamodule['SHUFFLE']

        # TODO: check train dataset length and integrity
        # TODO: check val dataset length and integrity

        # * Create pd.dataframes of train, val, test splits of the data
        # Reading the image and label paths relative to the DATA_DIR to a pd dataframe
        train_data = pd.read_csv(os.path.join(self.config.etl['DATA_DIR'], self.config.dataset['DATA_NAME'], 'train' + '_' + self.config.dataset['DATA_NAME'] + '.csv'))
        val_data = pd.read_csv(os.path.join(self.config.etl['DATA_DIR'], self.config.dataset['DATA_NAME'], 'val' + '_' + self.config.dataset['DATA_NAME'] + '.csv'))
        test_data = pd.read_csv(os.path.join(self.config.etl['DATA_DIR'], self.config.dataset['DATA_NAME'], 'test' + '_' + self.config.dataset['DATA_NAME'] + '.csv'))

        # * Create dictionaries of the data
        IMAGE_ROOT = self.config.dataset['IMAGE_ROOT']
        train_files = [
            {
                # Prepend each image and label path by the IMAGE_ROOT since the dataframe is of paths relative to the DATA_DIR
                "image": os.path.join(IMAGE_ROOT, image_name),
                "label": os.path.join(IMAGE_ROOT, label_name)
            } for image_name, label_name in zip(train_data['image'], train_data['label'])
        ]
        val_files = [
            {
                "image": os.path.join(IMAGE_ROOT, image_name),
                "label": os.path.join(IMAGE_ROOT, label_name)
            } for image_name, label_name in zip(val_data['image'], val_data['label'])
        ]
        test_files = [
            {
                "image": os.path.join(IMAGE_ROOT, image_name),
                "label": os.path.join(IMAGE_ROOT, label_name)
            } for image_name, label_name in zip(test_data['image'], test_data['label'])
        ]
        # The above are dictionaries of the form {"image": image_name, "label": label_name} for each image_name, label_name pair in the train, val, and test splits
        # TODO: check that the data is being loaded correctly
        # Each of the fields in the dictionary is a path to the image or label

        # TODO: Check that determinism has been set. I think it is set with Lightning?


        # TODO: Use a transform to resample all the images to the same size.
        # Maybe SpatialResample, ResampleToMatch, or Resize (the dictionary versions, of course)
        # Need to do this for train, val, and test

        # * Train transforms
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Resized(
                    keys=["image", "label"],
                    spatial_size=(240, 110, 200),
                    mode=("nearest", "nearest"),
                ),

                # '''
                # Spacingd(
                #     keys=["image", "label"],
                #     pixdim=(1.5, 1.5, 2.0),
                #     mode=("bilinear", "nearest"),
                # ),
                # '''

                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),


                # randomly crop out patch samples from
                # big image based on pos / neg ratio
                # the image centers of negative samples
                # must be in valid image area
                # RandCropByPosNegLabeld(
                #     keys=["image", "label"],
                #     label_key="label",
                #     spatial_size=(96, 96, 96),
                #     pos=1,
                #     neg=1,
                #     num_samples=4,
                #     image_key="image",
                #     image_threshold=0,
                # ),
                        # Apparently randCropByPosNegLabeld changes the dataset to a list of dictionaries
                        # Bruh.



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
                Resized(
                    keys=["image", "label"],
                    spatial_size=(240, 110, 200),
                    mode=("nearest", "nearest"),
                ),
                # """
                # Spacingd(
                #     keys=["image", "label"],
                #     pixdim=(1.5, 1.5, 2.0),
                #     mode=("bilinear", "nearest"),
                # ),
                # """
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )

        # From the base PyTorch tutorial
        # Spacingd, Orientationd, and CropForegroundd are all transforms that only take the `image` as input
        # TODO: Use a transform to resample all the images to the same size.
        # Maybe SpatialResample, ResampleToMatch, or Resize (the dictionary versions, of course)
        # Need to do this for train, val, and test
        test_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                # Here the Spacingd only takes the `image` as input
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
                # Here the Orientationd only takes the `image` as input
                Resized(
                    keys=["image", "label"],
                    spatial_size=(240, 110, 200),
                    mode=("nearest", "nearest"),
                ),
                #       Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # Here the CropForegroundd only takes the `image` as input
                CropForegroundd(keys=["image"], source_key="image"),
            ]
        )

        # self.test_ds = Dataset(data=test_files, transform=test_transforms)
        # test_dataloader = DataLoader(self.test_ds, batch_size=1, num_workers=4)

        # From the base PyTorch tutorial
        # This is for use in the test_step function
        self.post_test_transforms = Compose(
            [
                Invertd(
                    keys="pred",
                    transform=test_transforms,
                    orig_keys="image",
                    meta_keys="pred_meta_dict",
                    orig_meta_keys="image_meta_dict",
                    meta_key_postfix="meta_dict",
                    nearest_interp=False,
                    to_tensor=True,
                ),
                AsDiscreted(keys="pred", argmax=True, to_onehot=2),
                AsDiscreted(keys="label", to_onehot=2),
            ]
        )

        # * Create the datasets
        self.train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=1.0,
            num_workers=4,
        )
        self.val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=1.0,
            num_workers=4,
        )
        self.test_ds = CacheDataset(
            data=test_files,
            transform=test_transforms,
            cache_rate=1.0,
            num_workers=4,
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.config.datamodule['BATCH_SIZE'],
            shuffle=self.config.datamodule['SHUFFLE'],
            num_workers=self.config.datamodule['NUM_WORKERS'],
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.config.datamodule['BATCH_SIZE'],
            shuffle=False,
            num_workers=self.config.datamodule['NUM_WORKERS'],
            collate_fn=list_data_collate,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_ds,
            batch_size=2,
            shuffle=False,
            num_workers=4,
            collate_fn=list_data_collate,
        )
        # If batch_size=1, it's automatically squeezing the batch dimension.
        # To fix that, I will unsqueeze the batch dimension if batch_size=1.
        if test_loader.batch_size == 1:
            print("Unsqueezing batch dimension")
            for batch in test_loader.dataset:
                batch["image"] = torch.unsqueeze(batch["image"],0)
                batch["label"] = torch.unsqueeze(batch["label"],0)
                batch["foreground_start_coord"] = np.expand_dims(batch["foreground_start_coord"],0)
                batch["foreground_end_coord"] = np.expand_dims(batch["foreground_end_coord"],0)
            #torch.unsqueeze(test_loader.dataset,0)
        return test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-4)
        return optimizer

    # The below train step, val step, and val epoch end hook are all from the Lightning tutorial
    # I need to modify them.
    # At least to remove the Tensorboard logging since I am not using that.
    # Maybe leave the Tensorboard logging and see if Wandb can just use it like that?
    def training_step(self, batch, batch_idx):
        print("Started training step")
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        #tensorboard_logs = {"train_loss": loss.item()}
        #return {"loss": loss, "log": tensorboard_logs}
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        print("Started validation step")
        images, labels = batch["image"], batch["label"]
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        loss = self.loss_function(outputs, labels)
        self.log('val_loss', loss)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        return d

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
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
        self.validation_step_outputs.clear()  # free memory
        return {"log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        print("Started test step")
        images, labels = batch["image"], batch["label"]
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        loss = self.loss_function(outputs, labels)
        #self.log('val_loss', loss)
        self.log('test_loss', loss)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        d = {"test_loss": loss, "test_number": len(outputs)}
        self.test_step_outputs.append(d)
        return d

    def on_test_epoch_end(self):
        test_loss, num_items = 0, 0
        for output in self.test_step_outputs:
            test_loss += output["test_loss"].sum().item()
            num_items += output["test_number"]
        mean_test_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_test_loss = torch.tensor(test_loss / num_items)
        tensorboard_logs = {
            "test_dice": mean_test_dice,
            "test_loss": mean_test_loss,
        }
        self.log("mean_test_dice", mean_test_dice)
        self.log("mean_test_loss", mean_test_loss)
        # if mean_val_dice > self.best_val_dice:
        #     self.best_val_dice = mean_val_dice
        #     self.best_val_epoch = self.current_epoch
        print(
            # f"current epoch: {self.current_epoch} "
            f"mean test dice: {mean_test_dice:.4f}"
            # f"\nbest mean dice: {self.best_val_dice:.4f} "
            # f"at epoch: {self.best_val_epoch}"
        )
        self.test_step_outputs.clear()  # free memory
        return {"log": tensorboard_logs}

























    def foobar(self):

        self.test_transforms = Compose([
            LoadImaged(keys=['image', 'label', 'resample_model']),
            EnsureChannelFirstd(keys=['image', 'label', 'resample_model']),
            #Spacingd(keys=['image', 'label'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')),
            ResampleToMatchd(keys=['image', 'label'], key_dst='resample_model', mode=('bilinear', 'nearest')),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            DivisiblePadd(keys=['image', 'label'], k=16),
            EnsureTyped(keys=['image', 'label'])
        ])

        """
        Invertd(
            keys=['pred'],
            transform=self.test_transforms,
            orig_keys='image',
            meta_keys='pred_meta_dict',
            orig_meta_keys='image_meta_dict',
            meta_key_postfix='meta_dict',
            nearest_interp=True,
            to_tensor=True,
        ),
        AsDiscrete(
            keys=['pred', 'label'],
            argmax=(True, False),
            to_onehot=2,        # 2 classes
            n_classes=2,
        ),
        """

        self.post_test_transforms = Compose([
            

            SaveImaged(
                keys=['image', 'pred', 'label'],
                meta_keys=['image_meta_dict', 'pred_meta_dict', 'label_meta_dict'],
                #meta_key_postfix='meta_dict',
                output_dir='./output',
                output_postfix='test',
                output_ext='.nii.gz',
            ),
        ])

        return -1

    def old_forward(self, x):
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

    def old_prepare_data(self):

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
        """

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


    def old_configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-4)
        return optimizer

    def old_training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        # Make the labels binary two-class
        # TODO: This is wrong. The labels should be one-class and either 1 or 0
        labels = torch.where(labels > 0, torch.ones_like(labels), torch.zeros_like(labels))
        # labels = torch.stack((labels, 1 - labels), dim=1)
        # labels = torch.squeeze(labels, dim=2)

        print(f'Train images shape: {images.shape} and labels shape: {labels.shape}')
        preds = self(images)
        print(f'Train images shape: {images.shape} and labels shape: {labels.shape}')
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

    def old_validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        # Make the labels binary two-class
        labels = torch.where(labels > 0, torch.ones_like(labels), torch.zeros_like(labels))
        labels = torch.stack((labels, 1 - labels), dim=1)
        labels = torch.squeeze(labels, dim=2)

        print(f'Val images shape: {images.shape} and labels shape: {labels.shape}')
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

    def old_test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        
        # Make the labels binary two-class
        labels = torch.where(labels > 0, torch.ones_like(labels), torch.zeros_like(labels))
        labels = torch.stack((labels, 1 - labels), dim=1)
        labels = torch.squeeze(labels, dim=2)
        
        preds = self(images)

        roi_size = (160, 160, 160)
        sw_batch_size = 4
        #batch["pred"] = sliding_window_inference(batch["image"], roi_size, sw_batch_size, self._model)
        batch["pred"] = self._model(batch["image"])
        #batch = [self.post_test_transforms(i) for i in decollate_batch(batch)]

        # Check if labels and preds are the same shape
        #print(f'Preds shape: {preds.shape} and labels shape: {labels.shape}')
        print(f'Preds shape: {batch["pred"].shape} and labels shape: {batch["label"].shape}')
        # Image shape
        #print(f'Image shape: {images.shape}')
        print(f'Image shape: {batch["image"].shape}')

        # Save the predictions in .nii.gz format using MONAI's SaveImageD without using post_test_transforms
        #batch["pred"] = torch.argmax(batch["pred"], dim=1, keepdim=True)
        #batch["pred"] = [self.post_pred(i) for i in decollate_batch(batch["pred"])]
        #batch["label"] = [self.post_label(i) for i in decollate_batch(batch["label"])]
        #batch = [self.post_test_transforms(i) for i in decollate_batch(batch)]
        # Send the predictions and label to just one channel, with the other channel being 1 - the first channel
        batch["pred"] = torch.argmax(batch["pred"], dim=1, keepdim=True)
        #batch["pred"] = torch.stack((batch["pred"], 1 - batch["pred"]), dim=1)
        #batch["pred"] = torch.squeeze(batch["pred"], dim=1)

        #batch["label"] = torch.stack((batch["label"], 1 - batch["label"]), dim=1)
        #batch["label"] = torch.squeeze(batch["label"], dim=2)
        print(f'After proc: Preds shape: {batch["pred"].shape} and labels shape: {batch["label"].shape}')
        # Print the pred_meta_dict and label_meta_dict
        #print(f'Pred meta dict: {batch["pred_meta_dict"]}')
        #print(f'Label meta dict: {batch["label_meta_dict"]}')

        SaveImaged(
            keys=['image'],
            meta_keys=['image_meta_dict'],
            output_dir='./output',
            output_postfix='blah',
            output_ext='.nii.gz',
        )(batch)

        print("Saved the image")

        SaveImaged(
            keys=['pred', 'label'],
            #meta_keys=['pred_meta_dict', 'label_meta_dict'],
            output_dir='./output',
            output_postfix='blah',
            output_ext='.nii.gz',
            # Make the file end in .nii.gz
        )(batch)

        print("Saved the label and pred images")

        SaveImaged(
            keys=['image', 'pred', 'label'],
            meta_keys=['image_meta_dict', 'pred_meta_dict', 'label_meta_dict'],
            output_dir='./output'
            #output_postfix='test',
            #output_ext='.nii.gz',
            # Make the file end in .nii.gz
            #output_postfix='.nii.gz',
        )(batch)

        

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
