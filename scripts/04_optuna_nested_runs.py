"""
MLOps Hands-on 1 Hyperparameter Tuning dengan Nested Runs (Optuna + MLflow)
Praktikan : Givaro Ananta
NIM       : 123450078
Fokus     : Optuna study dengan Parent Run & Child Runs (nested=True)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import mlflow
import optuna

# ==============================================================================
# PENGEMBANGAN: Konfigurasi Lingkungan & Identitas Praktikan
# ==============================================================================
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "optuna-hyperparameter-tuning"
N_TRIALS = int(os.getenv("N_TRIALS", "20"))  # Default 20 trial agar cepat namun representatif

STUDENT_IDENTITY = {
    "author": "Givaro Ananta",
    "nim": "123450078",
    "course": "MLOps - Hands-on 1",
    "module": "04_optuna_nested_runs",
}

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print("=" * 60)
print("Hands-on 1: Slide 47 & 48 — Nested Runs (Optuna + MLflow)")
print(f"Praktikan : {STUDENT_IDENTITY['author']} (NIM: {STUDENT_IDENTITY['nim']})")
print(f"Jumlah Trial: {N_TRIALS}")
print(f"Tracking URI: {TRACKING_URI}")
print("=" * 60)

# ==============================================================================
# PENGEMBANGAN: Dataset & Implementasi train_and_eval(params)
# ==============================================================================
# Menyiapkan data tabular breast cancer yang distandarisasi untuk MLP
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)

X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_tr_t, y_tr_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

class FastMLP(nn.Module):
    def __init__(self, in_features, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

def train_and_eval(params):
    """Fungsi pelengkap yang dipanggil oleh objective(trial) di Slide 47."""
    loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    model = FastMLP(in_features=30, dropout=params["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.CrossEntropyLoss()

    # Training 10 epoch cepat
    model.train()
    for _ in range(10):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    # Evaluasi F1
    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).argmax(dim=1).numpy()
    score = float(f1_score(y_val, preds, average="macro", zero_division=0))
    return score

# ==============================================================================
# KODE DARI SLIDE 47 (DIPERTAHANKAN PERSIS SEPERTI DI SLIDE)
# ==============================================================================
def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
    }
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.set_tags(STUDENT_IDENTITY)
        mlflow.log_params(params)
        val_f1 = train_and_eval(params)
        mlflow.log_metric("val_f1", val_f1)
    return val_f1

# ==============================================================================
# KODE DARI SLIDE 48 (DIPERTAHANKAN PERSIS SEPERTI DI SLIDE)
# ==============================================================================
optuna.logging.set_verbosity(optuna.logging.WARNING)

with mlflow.start_run(run_name="optuna-sweep-01") as parent_run:              # parent run
    mlflow.set_tags(STUDENT_IDENTITY)
    mlflow.set_tag("study_name", "optuna-sweep-01")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_val_f1", study.best_value)

    print("\n[INFO] Parent Run ID :", parent_run.info.run_id)
    print(f"[INFO] Best Trial    : #{study.best_trial.number}")
    print(f"[INFO] Best Val F1   : {study.best_value:.4f}")
    print("[INFO] Best Params   :", study.best_params)
    print(f"[INFO] Semua {N_TRIALS} child runs berhasil dicatat dalam hierarki nested run!")
