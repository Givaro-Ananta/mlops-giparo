# Praktikum MLOps — Hands-on 1: MLflow Experiment Tracking & Registry

**Nama:** Givaro Ananta
**NIM:** 123450078
**Program Studi:** Sains Data
**Mata Kuliah:** MLOps (Minggu 6)

---

## 📌 Ringkasan Tugas

Tugas ini mengimplementasikan dan mengembangkan seluruh materi kode pada **Hands-on 1: MLflow** dari materi perkuliahan Minggu 6 (Slide 39 hingga 52). Seluruh sintaks dan struktur kode asli yang diberikan di slide dipertahankan secara utuh tanpa modifikasi, lalu dikembangkan dengan lingkungan eksekusi lengkap, dataset nyata, kompatibilitas framework MLflow 3.x.

---

## 📁 Struktur Direktori `mlops giparo`

```text
mlops giparo/
├── README.md                      # Panduan lengkap, arsitektur, dan instruksi eksekusi
├── LAPORAN_HANDSON_1.md           # Laporan praktikum detail mencakup hasil nyata & analisis
├── requirements.txt               # Daftar pustaka pendukung (mlflow, torch, scikit-learn, optuna)
├── start_server.sh                # Skrip peluncuran MLflow Tracking Server (Slide 41)
├── 01_logging_dasar.py            # Logging Dasar scikit-learn (Random Forest)
├── 02_pytorch_logging_per_epoch.py# Deep Learning PyTorch per-epoch tracking & best checkpoint
├── 03_autologging.py              # Autologging scikit-learn & framework support
├── 04_optuna_nested_runs.py       # Hyperparameter Tuning Optuna dengan Nested Runs
├── 05_search_and_registry.py      # Pencarian run programatik & Model Registry (@champion)
└── run_all.py                     # Master runner untuk menjalankan seluruh alur otomatis
```

---

## 🔬 Pemetaan Modul ke Slide Perkuliahan

| Modul                                         | Slide Acuan             | Fokus & Kode yang Diterapkan                                                                                                                                                                                                                                                                           |
| --------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`start_server.sh`**                 | **Slide 41**      | Menjalankan MLflow Tracking Server dengan backend database SQLite (`sqlite:///mlflow.db`) dan artifact root `./mlartifacts` di port 5000.                                                                                                                                                          |
| **`01_logging_dasar.py`**             | **Slide 42 & 43** | Experiment`breast-cancer-clf`, logging parameter RandomForest, metrik `val_f1` dan `val_auc`, serta packaging model sklearn dengan `skops_trusted_types`.                                                                                                                                      |
| **`02_pytorch_logging_per_epoch.py`** | **Slide 44 & 45** | Arsitektur PyTorch ResNet, logging parameter (`lr`, `batch_size`, `epochs`, `arch`, `seed`), hash dataset (`data_md5`), environment `requirements.txt`, loop 20 epoch logging metrik per-epoch (`step=epoch`), model checkpointing `best.pt`, dan `mlflow.pytorch.log_model`.      |
| **`03_autologging.py`**               | **Slide 46**      | Demonstrasi`mlflow.autolog()` dan `mlflow.sklearn.autolog()` untuk mencatat otomatis seluruh parameter, kurva evaluasi, dan model tanpa kode logging manual.                                                                                                                                       |
| **`04_optuna_nested_runs.py`**        | **Slide 47 & 48** | Pencarian hyperparameter (`lr`, `batch_size`, `dropout`) dengan Optuna study. Struktur hierarki: 1 Parent Run (`optuna-sweep-01`) menampung 20 Child Runs (`trial-0` s.d. `trial-19`) dengan `nested=True`.                                                                              |
| **`05_search_and_registry.py`**       | **Slide 49 & 50** | Query programatik via`mlflow.search_runs(filter_string=...)`, seleksi `best_run_id`, pendaftaran model ke Model Registry, penetapan tag validasi, pemberian alias `@challenger`, promosi ke `@champion`, serta loading model via `models:/breast-cancer-clf@champion` untuk inferensi batch. |

---

## Langkah-langkah Menjalankan

### 1. Prasyarat Lingkungan

Pastikan dependensi terpasang:

```bash
cd "mlops-giparo"
pip install -r requirements.txt
```

### 2. Menjalankan MLflow Tracking Server (Terminal 1)

Jalankan server tracking lokal sesuai spesifikasi Slide 41:

```bash
./start_server.sh
```

*Server akan aktif di `http://127.0.0.1:5000`.*

### 3. Menjalankan Eksperimen (Terminal 2)

Anda dapat menjalankan seluruh alur Hands-on 1 sekaligus menggunakan master runner:

```bash
python run_all.py
```

Atau menjalankan modul secara terpisah satu per satu:

```bash
# Modul 1: Logging Dasar scikit-learn
python 01_logging_dasar.py

# Modul 2: PyTorch Deep Learning Logging per Epoch
python 02_pytorch_logging_per_epoch.py

# Modul 3: Autologging Demo
python 03_autologging.py

# Modul 4: Optuna Nested Runs
python 04_optuna_nested_runs.py

# Modul 5: Search Runs Programatik & Model Registry
python 05_search_and_registry.py
```

### 4. Meninjau Hasil & Analisis di MLflow UI

Buka peramban (browser) ke: **`http://127.0.0.1:5000`**

#### A. Eksperimen & Eksplorasi Visual (Compare Runs)

1. **Pilih beberapa run → Klik Compare:**
   - **Parallel Coordinates Plot:** Pada eksperimen `optuna-hyperparameter-tuning`, bandingkan 20 child runs untuk memetakan bagaimana kombinasi hyperparameter multi-dimensi (`lr`, `batch_size`, `dropout`) memengaruhi metrik `val_f1` (menunjukkan konfigurasi optimum F1 = 0.9719).
   - **Scatter Plot & Contour Plot:** Memetakan distribusi metrik terhadap pasangan parameter secara visual untuk melihat lanskap pencarian hyperparameter dan area konvergensi terbaik.
2. **Overlay Learning Curves Antar Run:**
   - Pada eksperimen `pytorch-dl-experiment`, manfaatkan fitur *overlay* untuk menumpuk kurva metrik per-epoch (`train_loss`, `val_loss`, dan `val_f1`) selama 20 epoch antar run guna mendeteksi kestabilan pelatihan serta potensi *overfitting*.
3. **Inspeksi Artifacts & Diff Parameter:**
   - **Tab Artifacts:** Memeriksa direktori artefak yang tersimpan pada setiap run, meliputi bobot model biner (`model/`), berkas dependensi lingkungan (`requirements.txt`, `MLmodel`), matriks konfusi hasil autologging, dan checkpoint PyTorch `best.pt`.
   - **Fitur Diff Parameter:** Membandingkan perbedaan konfigurasi parameter antar run secara otomatis tanpa pemeriksaan manual satu per satu.

#### B. Model Registry

- Buka menu **Models** di antarmuka web UI (`http://127.0.0.1:5000/#/models`).
- Model `breast-cancer-clf` (Version 1) telah terdaftar lengkap dengan tag identitas praktikan **Givaro Ananta (123450078)**, status validasi, dan alias aktif **`@champion`** yang siap dilayani melalui REST API:
  ```bash
  mlflow models serve -m "models:/breast-cancer-clf@champion" -p 8080
  ```

---

## 📊 Hasil Eksekusi Praktikum (Hands-on 1)

| Modul        | Eksperimen                       | Run Name              | Run ID / Versi                       | Metrik Kunci                                                          | Status    |
| ------------ | -------------------------------- | --------------------- | ------------------------------------ | --------------------------------------------------------------------- | --------- |
| **01** | `breast-cancer-clf`            | `rf-baseline`       | `21e232098d8a49e89f2ff97293c17cf7` | `val_f1 = 0.9655`, `val_auc = 0.9940`                             | Selesai   |
| **02** | `pytorch-dl-experiment`        | `resnet18-lr0.0003` | `968a74a7df4443d7975974a2ea45ec4d` | Train Loss:`2.3447` $\rightarrow$ `2.1910`, Best F1: `0.0518` | Selesai   |
| **03** | `autologging-demo`             | `autolog-rf-demo`   | `07083129b745435b8a5fac30ad8d5ccb` | Akurasi Validasi =`0.9474` (94.74%)                                 | Selesai   |
| **04** | `optuna-hyperparameter-tuning` | `optuna-sweep-01`   | `e831a60f14e6430b836811ee487a2b7d` | Best Trial#10 (`val_f1 = 0.9719`), 20 Child Runs                    | Selesai   |
| **05** | Model Registry                   | `breast-cancer-clf` | Versi 1 (`@champion`)              | Prediksi Inferensi:`[0, 1, 0, 0, 0]`                                | Terdaftar |
