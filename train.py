import numpy as np
import os
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import io
import fsspec
from PIL import Image
from torch.utils.data import Dataset

### New imports for Lightning
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, BackboneFinetuning
torch.set_float32_matmul_precision('medium')

### Configure the training job
# All hyperparameters will be set here, in one convenient place
# This part is the same as the "vanilla" Pytorch version
config = {
    "initial_epochs": 5,
    "total_epochs": 20,
    "patience": 5,
    "batch_size": 32,
    "lr": 1e-4,
    "fine_tune_lr": 1e-6,
    "model_architecture": "MobileNetV2",
    "dropout_probability": 0.5,
    "random_horizontal_flip": 0.5,
    "random_rotation": 15,
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
    "color_jitter_saturation": 0.2,
    "color_jitter_hue": 0.1
}

### Prepare data loaders
# This part is the same as the "vanilla" Pytorch version

# Get bucket from environment variable, default to 'data'
s3_bucket = os.getenv("S3_DATA_BUCKET", "data")
s3_prefix = os.getenv("S3_DATA_PREFIX", "Food-11")

# Define transforms for training data augmentation
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=config["random_horizontal_flip"]),
    transforms.RandomRotation(config["random_rotation"]),
    transforms.ColorJitter(
        brightness=config["color_jitter_brightness"],
        contrast=config["color_jitter_contrast"],
        saturation=config["color_jitter_saturation"],
        hue=config["color_jitter_hue"]
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define streaming dataset for MinIO using fsspec (like data lab)
fs_kwargs = {}
endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
if endpoint_url:
    fs_kwargs['client_kwargs'] = {'endpoint_url': endpoint_url}

def get_samples(bucket, prefix, split):
    fs = fsspec.filesystem('s3', **fs_kwargs)
    base = f"{bucket}/{prefix}/{split}"
    pattern = f"{base}/class_*/*"
    paths = fs.glob(pattern)
    paths = [p for p in paths if not p.endswith('/')]
    paths.sort()

    samples = []
    for p in paths:
        parts = p.split('/')
        try:
            cls = next(seg for seg in parts if seg.startswith('class_'))
            label = int(cls.split('_')[1])
        except Exception:
            continue
        samples.append({'path': p, 'label': label})
    return samples

class RemoteImageDataset(Dataset):
    def __init__(self, samples, fs_kwargs, transform=None):
        self.samples = samples
        self.fs_kwargs = fs_kwargs
        self.transform = transform
        self._fs = None

    def _get_fs(self):
        if self._fs is None:
            self._fs = fsspec.filesystem('s3', **self.fs_kwargs)
        return self._fs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        fs = self._get_fs()
        with fs.open(s['path'], 'rb') as f:
            b = f.read()

        img = Image.open(io.BytesIO(b)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, int(s['label'])

# Load datasets
train_samples = get_samples(s3_bucket, s3_prefix, "training")
val_samples = get_samples(s3_bucket, s3_prefix, "validation")
test_samples = get_samples(s3_bucket, s3_prefix, "evaluation")

train_dataset = RemoteImageDataset(train_samples, fs_kwargs, transform=train_transform)
val_dataset = RemoteImageDataset(val_samples, fs_kwargs, transform=val_test_transform)
test_dataset = RemoteImageDataset(test_samples, fs_kwargs, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)


### Define training and validation/test functions
### Define the model

# We create a class LightningFood11Model that inherits the Pytorch Lightning LightningModule
# The Pytorch "boilerplate" has moved inside it:
#  - the model defintion is now inside init
#  - we are going to use Lightning's convenient BackboneFinetuning callback, so we also define the part of the model that is the backbone
#  - the forward pass from the train and validate functions are now inside the forward method
#  - the backward pass from the train and validate functions are now inside the training_step, validation_step, and test_step methods
#  - the optimizer configuration is now inside configure_optimizers

class LightningFood11Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
        num_ftrs = self.model.last_channel
        self.model.classifier = nn.Sequential(
            nn.Dropout(config["dropout_probability"]),
            nn.Linear(num_ftrs, 11)
        )
        self.criterion = nn.CrossEntropyLoss()

    @property
    def backbone(self):
        """Expose the backbone for BackboneFinetuning callback."""
        return self.model.features

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        # update loss and accuracy in progress bar every epoch
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('train_accuracy', acc, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return {"loss": loss, "train_accuracy": acc}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        # need to set val_loss so that callbacks can use it
        # also update loss and accuracy in progress bar every epoch
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val_accuracy', acc, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_accuracy": acc}

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=config["lr"])
        return optimizer

### Lightning callbacks
# Many of the things we hand-coded in Pytorch are available "out of the box" in Pytorch Lightning
# - saving model when vaidation loss improves: use ModelCheckpoint
# - early stopping: use EarlyStopping
# - un-freeze backbone/base model after a few epochs, and continue training with a small learning rate: BackboneFinetuning

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",  # where to save the model
    filename="food11",  # model name
    monitor="val_loss",  # watch validation loss
    mode="min",  # save the model with the lowest validation loss
    save_top_k=1  # keep only the best model
)

early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    patience=config["patience"],
    mode="min"
)

backbone_finetuning_callback = BackboneFinetuning(
    unfreeze_backbone_at_epoch=config["initial_epochs"],
    backbone_initial_lr = config["fine_tune_lr"],  # Sets initial learning rate for finetuning
    should_align=True
)


### Training loop
# The training loop in "vanilla" Pytorch is completely replaced with a Lightning Trainer
# it also includes baked-in support for distributed training across GPUs
# we set devices="auto" and let it figure out by itself how many GPUs are available, and how to use them

lightning_food11_model = LightningFood11Model()

trainer = Trainer(
    max_epochs=config["total_epochs"],
    accelerator="gpu",
    devices="auto",
    callbacks=[checkpoint_callback, early_stopping_callback, backbone_finetuning_callback]
)

trainer.fit(lightning_food11_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

### Evaluate on test set
trainer.test(lightning_food11_model, dataloaders=test_loader)
