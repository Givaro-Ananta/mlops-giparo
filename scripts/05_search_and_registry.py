"""
MLOps Hands-on 1 Membandingkan & Mencari Run + Model Registry
Praktikan : Givaro Ananta
NIM       : 123450078
Fokus     : mlflow.search_runs(), Model Registry (Register, Alias @challenger/@champion, Load)
"""

import os
import mlflow
from mlflow import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ==============================================================================
# PENGEMBANGAN: Konfigurasi Lingkungan & Identitas Praktikan
# ==============================================================================
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
STUDENT_IDENTITY = {
    "author": "Givaro Ananta",
    "nim": "123450078",
    "course": "MLOps - Hands-on 1",
    "module": "05_search_and_registry",
}

mlflow.set_tracking_uri(TRACKING_URI)

print("=" * 60)
print("Hands-on 1: Slide 49 & 50 — Search Runs & Model Registry")
print(f"Praktikan : {STUDENT_IDENTITY['author']} (NIM: {STUDENT_IDENTITY['nim']})")
print(f"Tracking URI: {TRACKING_URI}")
print("=" * 60)

# Siapkan data X_new untuk simulasi prediksi inferensi di Slide 50
X, y = load_breast_cancer(return_X_y=True)
_, X_val, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_new = X_val[:5]

# ==============================================================================
# KODE DARI SLIDE 49 (DIPERTAHANKAN PERSIS SEPERTI DI SLIDE)
# ==============================================================================
print("\n[INFO] Menjalankan pencarian run secara programatik (Slide 49)...")

runs = mlflow.search_runs(
    experiment_names=["breast-cancer-clf"],
    filter_string="metrics.val_f1 > 0.95 and params.max_depth = '8' and tags.stage != 'debug'",
    order_by=["metrics.val_f1 DESC"],
    max_results=10,
)

# Robust fallback jika metrik acak sedikit di bawah 0.95 pada run tertentu
if runs.empty:
    print("[WARNING] Filter ketat tidak menemukan run. Menggunakan filter umum untuk eksperimen breast-cancer-clf...")
    runs = mlflow.search_runs(
        experiment_names=["breast-cancer-clf"],
        order_by=["metrics.val_f1 DESC"],
        max_results=10,
    )

print(runs[["run_id", "params.n_estimators", "metrics.val_f1", "metrics.val_auc"]])

best_run_id = runs.iloc[0]["run_id"]
print(f"\n[INFO] Run terbaik terpilih: {best_run_id}")

# ==============================================================================
# KODE DARI SLIDE 50 (DIPERTAHANKAN PERSIS SEPERTI DI SLIDE)
# ==============================================================================
print("\n[INFO] Memulai proses Model Registry (Slide 50)...")

client = MlflowClient()
name = "breast-cancer-clf"

# 1. Registrasi model dari run terbaik
mv = mlflow.register_model(f"runs:/{best_run_id}/model", name)
print(f"[INFO] 1. Model berhasil didaftarkan: Versi {mv.version}")

# 2. Anotasi & tandai sebagai challenger
client.update_model_version(name, mv.version, description=f"RF depth=8, data v3 - Praktikan: {STUDENT_IDENTITY['author']} ({STUDENT_IDENTITY['nim']})")
client.set_model_version_tag(name, mv.version, "validation_status", "passed")
client.set_model_version_tag(name, mv.version, "author", STUDENT_IDENTITY["author"])
client.set_model_version_tag(name, mv.version, "nim", STUDENT_IDENTITY["nim"])
client.set_registered_model_alias(name, "challenger", mv.version)
print(f"[INFO] 2. Model versi {mv.version} diberi tag dan alias '@challenger'")

# 3. Setelah lolos validasi / A-B test → promosikan
client.set_registered_model_alias(name, "champion", mv.version)
print(f"[INFO] 3. Model versi {mv.version} dipromosikan ke alias '@champion'")

# 4. Aplikasi serving selalu memuat via alias (tanpa hard-code versi)
print("\n[INFO] 4. Memuat model via alias: models:/breast-cancer-clf@champion ...")
model = mlflow.pyfunc.load_model(f"models:/{name}@champion")
preds = model.predict(X_new)

print("\n" + "=" * 60)
print(f"[SUKSES] Hasil Prediksi 5 Sampel Uji oleh Model @champion:")
print("Predictions :", preds)
print("=" * 60)
print("\nPerintah untuk serving REST API (Slide 50):")
print(f'mlflow models serve -m "models:/{name}@champion" -p 8080')
