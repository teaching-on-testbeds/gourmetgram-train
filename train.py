import numpy as np
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()

# Local path to copy dataset into
local_data_dir = '/tmp/Food-11'
os.makedirs(local_data_dir, exist_ok=True)

# Copy dataset from GCS to local VM disk
os.system(f'gsutil -m cp -r {args.data_dir}/* {local_data_dir}/')
food_11_data_dir = local_data_dir

config = {
    "initial_epochs": 5,
    "total_epochs": 20,
    "patience": 5,
    "batch_size": 32,
    "lr": 1e-4,
    "fine_tune_lr": 1e-5,
    "dropout_probability": 0.5,
    "random_horizontal_flip": 0.5,
    "random_rotation": 15,
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
    "color_jitter_saturation": 0.2,
    "color_jitter_hue": 0.1
}

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

train_loader = DataLoader(
    datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'training'), transform=train_transform),
    batch_size=config["batch_size"], shuffle=True
)

val_loader = DataLoader(
    datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'validation'), transform=val_test_transform),
    batch_size=config["batch_size"], shuffle=False
)

test_loader = DataLoader(
    datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'evaluation'), transform=val_test_transform),
    batch_size=config["batch_size"], shuffle=False
)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
    return running_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(dataloader), correct / total

food11_model = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
num_ftrs = food11_model.last_channel
food11_model.classifier = nn.Sequential(
    nn.Dropout(config["dropout_probability"]),
    nn.Linear(num_ftrs, 11)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
food11_model = food11_model.to(device)

for param in food11_model.features.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(food11_model.classifier.parameters(), lr=config["lr"])

best_val_loss = float('inf')

# Save path (local file) for state_dict only
local_model_path = os.path.join("/tmp", "food11.pth")

for epoch in range(config["initial_epochs"]):
    t_loss, t_acc = train(food11_model, train_loader, criterion, optimizer, device)
    v_loss, v_acc = validate(food11_model, val_loader, criterion, device)
    print(f"[Initial] Epoch {epoch+1}: Train Loss={t_loss:.4f}, Val Loss={v_loss:.4f}")
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        torch.save(food11_model.state_dict(), local_model_path)
        print("  Model state_dict saved.")

for param in food11_model.features.parameters():
    param.requires_grad = True

optimizer = optim.Adam(food11_model.parameters(), lr=config["fine_tune_lr"])
patience_counter = 0

for epoch in range(config["initial_epochs"], config["total_epochs"]):
    t_loss, t_acc = train(food11_model, train_loader, criterion, optimizer, device)
    v_loss, v_acc = validate(food11_model, val_loader, criterion, device)
    print(f"[Fine-tune] Epoch {epoch+1}: Train Loss={t_loss:.4f}, Val Loss={v_loss:.4f}")
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        patience_counter = 0
        torch.save(food11_model.state_dict(), local_model_path)
        print("  Model state_dict saved.")
    else:
        patience_counter += 1
        if patience_counter >= config["patience"]:
            print("Early stopping triggered.")
            break

test_loss, test_acc = validate(food11_model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Upload state_dict file to GCS
os.system(f'gsutil cp {local_model_path} {args.output_dir}/food11.pth')
print(f"Model uploaded to {args.output_dir}/food11.pth")