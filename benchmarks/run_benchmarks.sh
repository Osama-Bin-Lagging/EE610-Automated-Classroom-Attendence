#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

echo "===== Starting benchmark queue: $(date) ====="

echo -e "\n[1/4] Hard Negatives..."
python bench_hard_negatives.py 2>&1 | tee results/log_hard_negatives.txt

echo -e "\n[2/4] Augmentation (--fast)..."
python bench_augmentation.py --fast 2>&1 | tee results/log_augmentation.txt

echo -e "\n[3/4] TTA..."
python bench_tta.py 2>&1 | tee results/log_tta.txt

echo -e "\n[4/4] Ensemble..."
python bench_ensemble.py 2>&1 | tee results/log_ensemble.txt

echo -e "\n===== All benchmarks done: $(date) ====="
