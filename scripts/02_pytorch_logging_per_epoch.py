"""
MLOps Hands-on 1 Deep Learning: Logging per Epoch (PyTorch)
Praktikan : Givaro Ananta
NIM       : 123450078
Fokus     : PyTorch per-epoch metrics tracking, model checkpointing, data hash, & model logging
"""

import os
import random
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import mlflow
import mlflow.pytorch

# ==============================================================================
# PENGEMBANGAN: Konfigurasi Lingkungan & Identitas Praktikan
# ==============================================================================
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "pytorch-dl-experiment"

STUDENT_IDENTITY = {
    "author": "Givaro Ananta",
    "nim": "123450078",
    "course": "MLOps - Hands-on 1",
    "module": "02_pytorch_logging_per_epoch",
}

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print("=" * 60)
print("Hands-on 1: Slide 44 & 45 — Logging per Epoch (PyTorch)")
print(f"Praktikan : {STUDENT_IDENTITY['author']} (NIM: {STUDENT_IDENTITY['nim']})")
print(f"Tracking URI: {TRACKING_URI}")
print("=" * 60)

# ==============================================================================
# PENGEMBANGAN: Fungsi Pendukung Sesuai Kontrak Slide
# ==============================================================================
# Kompatibilitas MLflow 3.x: Menjamin pemanggilan asli Slide 45 berjalan mulus tanpa mengubah kode slide
import functools
_orig_pt_log_model = mlflow.pytorch.log_model
@functools.wraps(_orig_pt_log_model)
def _compat_pt_log_model(*args, **kwargs):
    if "serialization_format" not in kwargs:
        kwargs["serialization_format"] = "pickle"
    return _orig_pt_log_model(*args, **kwargs)
mlflow.pytorch.log_model = _compat_pt_log_model

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 1. Siapkan Dataset dan Hitung dataset_md5 (Slide 44: data_md5)
# Menyiapkan dataset citra ringkas (1000 train, 200 val) agar 20 epoch cepat di CPU
np.random.seed(42)
X_dummy_tr = np.random.randn(800, 3, 32, 32).astype(np.float32)
y_dummy_tr = np.random.randint(0, 10, size=(800,)).astype(np.int64)
X_dummy_val = np.random.randn(200, 3, 32, 32).astype(np.float32)
y_dummy_val = np.random.randint(0, 10, size=(200,)).astype(np.int64)

# Hitung MD5 dari data
hasher = hashlib.md5()
hasher.update(X_dummy_tr.tobytes())
hasher.update(y_dummy_tr.tobytes())
dataset_md5 = hasher.hexdigest()

train_dataset = TensorDataset(torch.from_numpy(X_dummy_tr), torch.from_numpy(y_dummy_tr))
val_dataset = TensorDataset(torch.from_numpy(X_dummy_val), torch.from_numpy(y_dummy_val))

# Pastikan file requirements.txt tersedia di direktori kerja
if not Path("requirements.txt").exists():
    Path("requirements.txt").write_text("torch\nmlflow\nscikit-learn\n", encoding="utf-8")

# 2. Definisi Model ResNet-18 yang Ringan & Efisien untuk CPU
class SimpleResNet18Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class ResNet18Model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_c = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = SimpleResNet18Block(16, 16, stride=1)
        self.layer2 = SimpleResNet18Block(16, 32, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

# 3. Fungsi Training dan Evaluasi per Epoch (Slide 44: train_one_epoch, evaluate)
criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * len(targets)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
    val_loss = total_loss / len(loader.dataset)
    val_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return val_loss, val_f1

# ==============================================================================
# KODE DARI SLIDE 44 & 45 (DIPERTAHANKAN PERSIS SEPERTI DI SLIDE)
# ==============================================================================
cfg = {"lr": 3e-4, "batch_size": 64, "epochs": 20, "arch": "resnet18", "seed": 42}
set_seed(cfg["seed"])

# Persiapan loader & model sesuai konfigurasi slide
train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
model = ResNet18Model(num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Eksekusi blok persis dari Slide 44 & 45
with mlflow.start_run(run_name=f"{cfg['arch']}-lr{cfg['lr']}") as run:
    mlflow.set_tags(STUDENT_IDENTITY)
    mlflow.log_params(cfg)
    # Pastikan file requirements.txt ditemukan baik dari root maupun dari dalam folder scripts
    req_file = "requirements.txt" if os.path.exists("requirements.txt") else os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
    if os.path.exists(req_file):
        mlflow.log_artifact(req_file)              # environment

    best_f1 = 0.0
    for epoch in range(cfg["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_f1 = evaluate(model, val_loader)

        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                            "val_f1": val_f1, "lr": scheduler.get_last_lr()[0]},
                           step=epoch)
        scheduler.step()

        if val_f1 > best_f1:                             # checkpoint terbaik
            best_f1 = val_f1
            torch.save(model.state_dict(), "best.pt")
            mlflow.log_artifact("best.pt", artifact_path="checkpoints")
            if os.path.isdir("checkpoints"):
                torch.save(model.state_dict(), os.path.join("checkpoints", "best.pt"))
            elif os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "checkpoints")):
                torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best.pt"))

        print(f"Epoch {epoch+1:02d}/{cfg['epochs']:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    # Slide 45: Logging metrik final dan model
    mlflow.log_metric("best_val_f1", best_f1)
    mlflow.pytorch.log_model(model, name="model")

    print("\n[INFO] Selesai! Run ID   :", run.info.run_id)
    print(f"[INFO] Best Val F1      : {best_f1:.4f}")
    print("[INFO] Model & checkpoint best.pt berhasil dilog.")
