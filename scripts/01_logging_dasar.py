"""
MLOps Hands-on 1 Logging Dasar
Praktikan : Givaro Ananta
NIM       : 123450078
Fokus     : MLflow Tracking Dasar dengan scikit-learn (RandomForestClassifier)
"""

import os
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

# ==============================================================================
# PENGEMBANGAN: Konfigurasi Identitas & Server
# ==============================================================================
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
STUDENT_IDENTITY = {
    "author_student": "Givaro Ananta",
    "nim": "123450078",
    "course": "MLOps - Hands-on 1",
    "module": "01_logging_dasar",
}

print("=" * 60)
print("Hands-on 1: Slide 42 & 43 — Logging Dasar")
print(f"Praktikan : {STUDENT_IDENTITY['author_student']} (NIM: {STUDENT_IDENTITY['nim']})")
print(f"Tracking URI: {TRACKING_URI}")
print("=" * 60)

# ==============================================================================
# KODE DARI SLIDE 42 (DIPERTAHANKAN PERSIS SEPERTI DI SLIDE)
# ==============================================================================
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("breast-cancer-clf")

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

params = {"n_estimators": 200, "max_depth": 8, "random_state": 42}

# ==============================================================================
# KODE DARI SLIDE 43 (DIPERTAHANKAN PERSIS SEPERTI DI SLIDE)
# ==============================================================================
with mlflow.start_run(run_name="rf-baseline") as run:
    # Kode dari slide:
    mlflow.set_tags({"stage": "baseline", "author": "luky"})
    # Pengayaan tag identitas praktikan:
    mlflow.set_tags(STUDENT_IDENTITY)
    mlflow.log_params(params)

    model = RandomForestClassifier(**params).fit(X_tr, y_tr)
    proba = model.predict_proba(X_val)[:, 1]

    val_f1 = f1_score(y_val, proba > 0.5)
    val_auc = roc_auc_score(y_val, proba)

    mlflow.log_metric("val_f1", val_f1)
    mlflow.log_metric("val_auc", val_auc)
    mlflow.sklearn.log_model(model, name="model", input_example=X_val[:5],
                             skops_trusted_types=["sklearn.tree._tree.Tree"])   # RF (MLflow 3)

    # Output informasi untuk praktikan
    print("\n[INFO] Run ID      :", run.info.run_id)
    print(f"[INFO] Metric val_f1 : {val_f1:.4f}")
    print(f"[INFO] Metric val_auc: {val_auc:.4f}")
    print("[INFO] Model berhasil dilog ke MLflow!")
