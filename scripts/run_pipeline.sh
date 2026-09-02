#!/usr/bin/env bash
# TransactionGuard — End-to-end training pipeline
# Run from project root: bash scripts/run_pipeline.sh

set -e

# Activate virtual environment
source .venv/bin/activate

echo "======================================"
echo " TransactionGuard Training Pipeline"
echo "======================================"

echo ""
echo "--- Phase 1: Data Generation & EDA ---"
python src/data/eda.py

echo ""
echo "--- Phase 2: Baseline Tabular Model ---"
python src/models/train_baseline.py

echo ""
echo "--- Phase 3: Graph-Augmented Model ---"
python src/models/train_graph.py

echo ""
echo "--- Phase 4: SHAP Explainability ---"
python src/explainability/shap_explainer.py

echo ""
echo "======================================"
echo " Pipeline complete."
echo " Artifacts saved to ./artifacts/"
echo " Metrics saved to ./metrics/"
echo ""
echo " To start backend:"
echo "   source .venv/bin/activate && uvicorn backend.main:app --reload --port 8000"
echo ""
echo " To start frontend:"
echo "   cd frontend && npm run dev"
echo "======================================"
