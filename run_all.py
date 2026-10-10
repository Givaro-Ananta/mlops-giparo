"""
Hands-on 1 — Master Runner
Praktikan : Givaro Ananta
NIM       : 123450078
Prodi     : Sains Data 
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

STUDENT_NAME = "Givaro Ananta"
STUDENT_NIM = "123450078"
BASE_DIR = Path(__file__).resolve().parent

def check_server(tracking_uri: str) -> bool:
    try:
        req = urllib.request.urlopen(tracking_uri, timeout=3)
        return req.getcode() == 200
    except Exception:
        return False

def run_script(script_name: str):
    print("\n" + "=" * 70)
    print(f"▶ MENJALANKAN: {script_name}")
    # Mendukung eksekusi jika skrip berada di folder scripts/ maupun root
    script_path = BASE_DIR / "scripts" / script_name
    if not script_path.exists():
        script_path = BASE_DIR / script_name
    res = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR))
    if res.returncode != 0:
        print(f"\n[ERROR] Skrip {script_name} keluar dengan kode {res.returncode}")
        sys.exit(res.returncode)

def main():
    print("*" * 70)
    print("PRAKTIKUM MLOPS — HANDS-ON 1: MLFLOW EXPERIMENT TRACKING & REGISTRY")
    print(f"Nama : {STUDENT_NAME}")
    print(f"NIM  : {STUDENT_NIM}")
    print("*" * 70)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    print(f"\n[1/6] Memeriksa MLflow Tracking Server di {tracking_uri}...")
    if check_server(tracking_uri):
        print("  ✓ Server aktif dan merespons (HTTP 200).")
    else:
        print("  ! Server tidak merespons di port 5000.")
        print("  Silakan jalankan server di terminal terpisah:")
        print("  ./start_server.sh")
        print("  Atau biarkan skrip menggunakan mode backend SQLite lokal.")

    # Eksekusi berurutan modul Hands-on 1
    run_script("01_logging_dasar.py")
    run_script("02_pytorch_logging_per_epoch.py")
    run_script("03_autologging.py")
    run_script("04_optuna_nested_runs.py")
    run_script("05_search_and_registry.py")

    print("\n" + "*" * 70)
    print(f"SELURUH MODUL HANDS-ON 1 BERHASIL DIJALANKAN DENGAN SUKSES!")
    print(f"Praktikan : {STUDENT_NAME} (NIM: {STUDENT_NIM})")
    print("Buka UI MLflow di: http://127.0.0.1:5000 untuk melihat:")
    print("  1. Experiment 'breast-cancer-clf' (Logging Dasar & Model Registry)")
    print("  2. Experiment 'pytorch-dl-experiment' (Logging per Epoch 20 Epoch)")
    print("  3. Experiment 'autologging-demo' (Autologging scikit-learn)")
    print("  4. Experiment 'optuna-hyperparameter-tuning' (Parent Run + 20 Child Runs)")
    print("  5. Model Registry 'breast-cancer-clf' versi 1 dengan alias @champion")
    print("*" * 70)

if __name__ == "__main__":
    main()
