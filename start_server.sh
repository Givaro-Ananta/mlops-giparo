#!/usr/bin/env bash
# Menjalankan MLflow Tracking Server
# Identitas: Givaro Ananta (NIM: 123450078)

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "=========================================================="
echo "Memulai MLflow Tracking Server (Hands-on 1)"
echo "Praktikan: Givaro Ananta (NIM: 123450078)"
echo "Backend Store : sqlite:///mlflow.db"
echo "Artifact Root : ./mlartifacts"
echo "URL           : http://127.0.0.1:5000"
echo "=========================================================="

mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000
