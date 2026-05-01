import os
import argparse
import time
import random
from google.cloud import storage
from google.cloud import aiplatform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()


PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("CLOUD_ML_PROJECT_ID")
REGION = os.getenv("REGION", os.getenv("CLOUD_ML_REGION", "us-central1"))
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")

if PROJECT_ID and EXPERIMENT_NAME:
    aiplatform.init(project=PROJECT_ID, location=REGION, experiment=EXPERIMENT_NAME)
    run_name = f"gourmetgram-run-{int(time.time())}-{random.randint(1000, 9999)}"
    run_ctx = aiplatform.start_run(run=run_name)
    run_ctx.__enter__()
else:
    run_ctx = None

# Stream dataset directly from GCS (no full local copy)
def _parse_gs(uri):
    if not uri.startswith('gs://'):
        raise ValueError(f'Expected gs:// URI for --data_dir, got {uri}')
    rest = uri[5:]
    b, _, pref = rest.partition('/')
    return b, pref.rstrip('/')

class GCSImageDataset(Dataset):
    def __init__(self, bucket_name, samples, class_to_id, transform):
        self.bucket_name = bucket_name
        self.samples = samples
        self.class_to_id = class_to_id
        self.transform = transform
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        blob_name, class_name = self.samples[idx]
        blob = self.bucket.blob(blob_name)
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(blob.download_as_bytes())).convert('RGB')
        x = self.transform(img)
        y = self.class_to_id[class_name]
        return x, y

def _collect_split_samples(bucket_name, root_prefix, split, class_to_id=None):
    prefix = f"{root_prefix}/{split}/"
    by_class = {}
    client = storage.Client()
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        if blob.name.endswith('/'):
            continue
        rel = blob.name[len(prefix):]
        parts = rel.split('/')
        if len(parts) < 2:
            continue
        class_name = parts[0]
        if class_to_id is not None and class_name not in class_to_id:
            continue
        by_class.setdefault(class_name, []).append(blob.name)

    if class_to_id is None:
        class_names = sorted(by_class.keys())
        class_to_id = {c: i for i, c in enumerate(class_names)}

    samples = []
    for c in sorted(by_class.keys()):
        samples.extend((name, c) for name in sorted(by_class[c]))

    return samples, class_to_id

bucket_name, root_prefix = _parse_gs(args.data_dir)

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
    "color_jitter_hue": 0.1,
    "num_workers": 4
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

train_samples, class_to_id = _collect_split_samples(bucket_name, root_prefix, 'training')
val_samples, _ = _collect_split_samples(bucket_name, root_prefix, 'validation', class_to_id=class_to_id)
test_samples, _ = _collect_split_samples(bucket_name, root_prefix, 'evaluation', class_to_id=class_to_id)

if not train_samples or not val_samples or not test_samples:
    raise RuntimeError('Expected non-empty training/validation/evaluation splits under --data_dir')

train_loader = DataLoader(
    GCSImageDataset(bucket_name, train_samples, class_to_id, train_transform),
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(config["num_workers"] > 0),
)

val_loader = DataLoader(
    GCSImageDataset(bucket_name, val_samples, class_to_id, val_test_transform),
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(config["num_workers"] > 0),
)

test_loader = DataLoader(
    GCSImageDataset(bucket_name, test_samples, class_to_id, val_test_transform),
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(config["num_workers"] > 0),
)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for i, (inputs, labels) in enumerate(dataloader, start=1):
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
        if i % 20 == 0:
            print(f"  train batch {i}/{len(dataloader)}", flush=True)
    return running_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader, start=1):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if i % 20 == 0:
                print(f"  val/test batch {i}/{len(dataloader)}", flush=True)
    return running_loss / len(dataloader), correct / total

food11_model = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
num_ftrs = food11_model.last_channel
food11_model.classifier = nn.Sequential(
    nn.Dropout(config["dropout_probability"]),
    nn.Linear(num_ftrs, len(class_to_id))
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
food11_model = food11_model.to(device)

for param in food11_model.features.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(food11_model.classifier.parameters(), lr=config["lr"])

best_val_loss = float('inf')

if run_ctx:
    aiplatform.log_params(config)
    aiplatform.log_params({"data_dir": args.data_dir, "output_dir": args.output_dir})

for epoch in range(config["initial_epochs"]):
    t_loss, t_acc = train(food11_model, train_loader, criterion, optimizer, device)
    v_loss, v_acc = validate(food11_model, val_loader, criterion, device)
    print(f"[Initial] Epoch {epoch+1}: Train Loss={t_loss:.4f}, Val Loss={v_loss:.4f}")
    if run_ctx:
        aiplatform.log_metrics({"epoch": epoch+1, "train_loss": t_loss, "train_acc": t_acc, "val_loss": v_loss, "val_acc": v_acc})
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        torch.save(food11_model.state_dict(), "food11.pth")
        print("  Model saved.")

for param in food11_model.features.parameters():
    param.requires_grad = True

optimizer = optim.Adam(food11_model.parameters(), lr=config["fine_tune_lr"])
patience_counter = 0

for epoch in range(config["initial_epochs"], config["total_epochs"]):
    t_loss, t_acc = train(food11_model, train_loader, criterion, optimizer, device)
    v_loss, v_acc = validate(food11_model, val_loader, criterion, device)
    print(f"[Fine-tune] Epoch {epoch+1}: Train Loss={t_loss:.4f}, Val Loss={v_loss:.4f}")
    if run_ctx:
        aiplatform.log_metrics({"epoch": epoch+1, "train_loss": t_loss, "train_acc": t_acc, "val_loss": v_loss, "val_acc": v_acc})
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        patience_counter = 0
        torch.save(food11_model.state_dict(), "food11.pth")
        print("  Model saved.")
    else:
        patience_counter += 1
        if patience_counter >= config["patience"]:
            print("Early stopping triggered.")
            break

test_loss, test_acc = validate(food11_model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

if run_ctx:
    aiplatform.log_metrics({"test_loss": test_loss, "test_acc": test_acc})

### Upload model to GCS
out_bucket, out_prefix = _parse_gs(args.output_dir)
out_blob = f"{out_prefix}/food11.pth" if out_prefix else "food11.pth"
storage.Client().bucket(out_bucket).blob(out_blob).upload_from_filename("food11.pth")
print(f"Model uploaded to gs://{out_bucket}/{out_blob}")

if run_ctx:
    run_ctx.__exit__(None, None, None)
