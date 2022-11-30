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
from torch.profiler import profile, record_function, ProfilerActivity

from dataset import FistulaDataset

#print_config()

class Net(pytorch_lightning.LightningModule):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        )
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.prepare_data()

    def forward(self, x):
        """
        with profile(activities=[ProfilerActivity.CPU],
                profile_memory=True, record_shapes=True) as prof:
            # TODO: We're getting an OOM error here. How big is the model?
            # How big are the loaded nifi images?
        """
        output = self._model(x)
            #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        return output

    def prepare_data(self):

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

        self.train_dataset = FistulaDataset(data=train_files, transform=None)

        """
        self.val_ds = CacheDataset(
            #data=val_files, transform=val_transforms,
            data=val_files, transform=None,
            cache_rate=1.0, num_workers=4,
        )
        """

        self.val_dataset = FistulaDataset(data=val_files, transform=None)
#         self.train_ds = monai.data.Dataset(
#             data=train_files, transform=train_transforms)
#         self.val_ds = monai.data.Dataset(
#             data=val_files, transform=val_transforms)

    def train_dataloader(self):
        """
        old_train_loader = DataLoader(
            self.train_ds, batch_size=2, shuffle=True,
            num_workers=4, collate_fn=list_data_collate
        )
        """

        train_loader = DataLoader(self.train_dataset, batch_size=5, shuffle=True, num_workers=4)
        return train_loader

    def val_dataloader(self):
        """
        old_val_loader = DataLoader(
        self.val_ds, batch_size=1, num_workers=4)
        """

        val_loader = DataLoader(self.val_dataset, batch_size=1, num_workers=4)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        images = images[:, None, :, :, :]
        labels = labels[:, None, :, :, :]
        #print('images is')
        #print(images)
        #print(f'images shape: {images.shape}')
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        output = self.forward(images)
            #with record_function("model_inference"):
            #    logits = self._model(images)
        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}


        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        # TODO: What is roi_size? And sw_batch_size?
        #roi_size = (160, 160, 160)
        # Let's make roi_size smaller in all dimensions than the actual image (512,512,96)
        #roi_size = (64, 64, 64)
        roi_size = (64, 64, -1)
        sw_batch_size = 1
        # TODO: What is sliding_window_inference?
        # TODO: okay, sliding_window_inference ends up calling fall_back_tuple
        # , which seems to think that image_size_ is 2-dimensional, not 3-dimensional
        # Where is "image_size_" even coming from???
        # Found "image_size_" in the source code for sliding_window_inference.
        # Okay, sliding_window_inference assumes the input is batchxchannelxspatial.
        # I think what happened to us is our channel dimension was implicitly squeezed since it's 1.
        # Thus, we need to add a channel dimension to our input.
        # Thus, we need to unsqueeze our input at index=1.
        # If we added it at index=0, we would nuke ourselves if we changed our batch size from 1 lol.

        # Artificially putting a batch dimension in the image size to fix the problem.
        # This actually worked smh.
        images = images.unsqueeze(1)
        labels = labels.unsqueeze(1)
        print('inputs.shape is ' + str(images.shape))
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward)
        print('outputs.shape is ' + str(outputs.shape))
        print('labels.shape is ' + str(labels.shape))
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        return {"val_loss": loss, "val_number": len(outputs)}

    def validation_epoch_end(self, outputs):
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
