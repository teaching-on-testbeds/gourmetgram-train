import numpy as np
import os
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

### New imports for Lightning
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, BackboneFinetuning
torch.set_float32_matmul_precision('medium')


## New imports for Ray
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

## New imports for Ray Tune
from ray import tune
from ray.tune.schedulers import ASHAScheduler


### Configure the training job 
# All hyperparameters will be set here, in one convenient place
# This part is the same as the "vanilla" Pytorch version
config = {
    "initial_epochs": 5,
    "total_epochs": 20,
    "patience": 5,
    "batch_size": tune.choice([32, 64]),
    "lr": 1e-4,
    "fine_tune_lr": 1e-6,
    "model_architecture": "MobileNetV2",
    "dropout_probability": tune.uniform(0.1, 0.8),
    "random_horizontal_flip": 0.5,
    "random_rotation": 15,
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
    "color_jitter_saturation": 0.2,
    "color_jitter_hue": 0.1
}


### New for Ray Tune - wrap all the Lightning code in a function
def train_func(config):

    ### Prepare data loaders
    # This part is the same as the "vanilla" Pytorch version

    # Get data directory from environment variable, if set
    food_11_data_dir = os.getenv("FOOD11_DATA_DIR", "Food-11")

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

    # Load datasets
    train_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'training'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'validation'), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'evaluation'), transform=val_test_transform)

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
    # Tune will launch one trial per resource bundle we request
    # so inside each trial we use a single device and report validation metrics back to Tune

    lightning_food11_model = LightningFood11Model()
        
    tune_report_callback = TuneReportCheckpointCallback(
        metrics={
            "ptl/val_accuracy": "val_accuracy",
            "ptl/val_loss": "val_loss",
        },
        filename="checkpoint",
        on="validation_end",
    )

    trainer = Trainer(
        max_epochs=config["total_epochs"],
        devices=1,
        accelerator="auto",
        callbacks=[early_stopping_callback, backbone_finetuning_callback, tune_report_callback]
    )

    trainer.fit(lightning_food11_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    ### Evaluate on test set
    trainer.test(lightning_food11_model, dataloaders=test_loader)

### New for Ray Tune
def tune_asha(num_samples):
    scheduler = ASHAScheduler(max_t=config["total_epochs"], grace_period=1, reduction_factor=2)
    trainable = tune.with_resources(train_func, resources={"CPU": 4, "GPU": 0.5})
    tuner = tune.Tuner(
        trainable=trainable,
        param_space=config,
        tune_config=tune.TuneConfig(
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
            max_concurrent_trials=4,
        ),
        run_config=tune.RunConfig(storage_path="s3://ray"),
    )
    return tuner.fit()

results = tune_asha(num_samples=16)
