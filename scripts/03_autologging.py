"""
MLOps Hands-on 1 Autologging
Praktikan : Givaro Ananta
NIM       : 123450078
Fokus     : Penggunaan mlflow.autolog() & mlflow.sklearn.autolog()
"""

import os
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==============================================================================
# PENGEMBANGAN: Konfigurasi Lingkungan & Identitas Praktikan
# ==============================================================================
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "autologging-demo"

STUDENT_IDENTITY = {
    "author": "Givaro Ananta",
    "nim": "123450078",
    "course": "MLOps - Hands-on 1",
    "module": "03_autologging",
}

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print("=" * 60)
print("Hands-on 1: Slide 46 — Autologging")
print(f"Praktikan : {STUDENT_IDENTITY['author']} (NIM: {STUDENT_IDENTITY['nim']})")
print(f"Tracking URI: {TRACKING_URI}")
print("=" * 60)

# ==============================================================================
# KODE DARI SLIDE 46 (DIPERTAHANKAN PERSIS SEPERTI DI SLIDE)
# ==============================================================================
# Satu baris untuk mencatat params, metrics, dan model secara otomatis:
mlflow.autolog()                        # mendeteksi library yang dipakai

# atau spesifik framework:
mlflow.sklearn.autolog()
# mlflow.tensorflow.autolog()             # Keras: model.fit(...) → loss/metrics per epoch
# mlflow.pytorch.autolog()                # PyTorch Lightning Trainer
# mlflow.xgboost.autolog()

# ==============================================================================
# PENGEMBANGAN: Demonstrasi Eksekusi Nyata Autologging
# ==============================================================================
print("\n[INFO] Menjalankan pelatihan model dengan Autologging aktif...")
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

params = {"n_estimators": 100, "max_depth": 6, "random_state": 42}

with mlflow.start_run(run_name="autolog-rf-demo") as run:
    # Praktik terbaik dari Slide 46: autolog + tag manual untuk konteks bisnis/identitas
    mlflow.set_tags(STUDENT_IDENTITY)
    mlflow.set_tag("dataset", "Breast Cancer Wisconsin")

    # Fit model (MLflow autolog otomatis mencatat parameter, skor train, metrik, dan model)
    model = RandomForestClassifier(**params)
    model.fit(X_tr, y_tr)

    # Evaluasi (Autolog otomatis mencatat metrik evaluasi)
    score = model.score(X_val, y_val)

    print("[INFO] Model fit selesai!")
    print("[INFO] Run ID          :", run.info.run_id)
    print(f"[INFO] Akurasi Validasi: {score:.4f}")
    print("[INFO] Semua hyperparameter, skor akurasi, dan model artifact tercatat otomatis.")
