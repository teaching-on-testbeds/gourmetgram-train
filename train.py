import numpy as np
import os
import subprocess
import time
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models
import torchvision.transforms as transforms

### Configure the training job 
# All hyperparameters will be set here, in one convenient place
config = {
    "initial_epochs": 5,
    "total_epochs": 20,
    "patience": 5,
    "batch_size": 32,
    "lr": 1e-4,
    "fine_tune_lr": 1e-5,
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

# Get data directory from environment variable, if set
# otherwise, assume data is in a directory named "Food-11"
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

# Create data loaders
train_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'training'), transform=train_transform)
val_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'validation'), transform=val_test_transform)
test_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'evaluation'), transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

### Define training and validation/test functions
# This is Pytorch boilerplate

# training function - one epoch
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

# validate function - one epoch
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

### Define the model


# Define model
food11_model = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
num_ftrs = food11_model.last_channel
food11_model.classifier = nn.Sequential(
    nn.Dropout(config["dropout_probability"]),
    nn.Linear(num_ftrs, 11)
)

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
food11_model = food11_model.to(device)

# Initial training: only the classification head, freeze the backbone/base model
for param in food11_model.features.parameters():
    param.requires_grad = False

trainable_params  = sum(p.numel() for p in food11_model.parameters() if p.requires_grad)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(food11_model.classifier.parameters(), lr=config["lr"])

### Training loop for initial training

best_val_loss = float('inf')

# train new classification head on pre-trained model for a few epochs
for epoch in range(config["initial_epochs"]):
    epoch_start_time = time.time()
    train_loss, train_acc = train(food11_model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(food11_model, val_loader, criterion, device)
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Time: {epoch_time:.2f}s")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(food11_model, "food11.pth")
        print("  Validation loss improved. Model saved.")

### Un-freeze backbone/base model and keep training with smaller learning rate

# unfreeze to fine-tune the entire model
for param in food11_model.features.parameters():
    param.requires_grad = True

trainable_params  = sum(p.numel() for p in food11_model.parameters() if p.requires_grad)

# optimizer for the entire model with a smaller learning rate for fine-tuning
optimizer = optim.Adam(food11_model.parameters(), lr=config["fine_tune_lr"])

patience_counter = 0

# Fine-tune entire model for the remaining epochs
for epoch in range(config["initial_epochs"], config["total_epochs"]):

    epoch_start_time = time.time()
    train_loss, train_acc = train(food11_model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(food11_model, val_loader, criterion, device)
    epoch_time = time.time() - epoch_start_time

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Time: {epoch_time:.2f}s")

    # Check for improvement in validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(food11_model, "food11.pth")
        print("  Validation loss improved. Model saved.")

    else:
        patience_counter += 1
        print(f"  No improvement in validation loss. Patience counter: {patience_counter}")

    if patience_counter >= config["patience"]:
        print("  Early stopping triggered.")
        break


### Evaluate on test set
test_loss, test_acc = validate(food11_model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")