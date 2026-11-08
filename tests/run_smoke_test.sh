#!/usr/bin/env bash
# Offline end-to-end test of the whole pipeline (tiny random models + synthetic data).
# Numbers produced here are meaningless; the test only proves the code runs and that
# the leakage / determinism checks hold.  Run from the repository root.
set -euo pipefail
python tests/make_tiny_fixtures.py
FX=tests/fixtures
COMMON=(--config configs/default.yaml
  --set data.csv_path=$FX/synthetic.csv --set data.pairs_path=$FX/pairs.csv --set data.expected_pairs=3
  --set data.min_source_issues=50 --set data.min_target_test_issues=10
  --set embeddings.bert_model=$FX/tiny_bert --set embeddings.sbert_model=$FX/tiny_sbert
  --set embeddings.cache_dir=tests/smoke_cache --set output.results_dir=tests/smoke_results
  --set train.max_epochs=4 --set train.patience=2 --set 'train.grid={lr: [0.001], dropout: [0.2, 0.4]}'
  --set text.max_tokens=32)
rm -rf tests/smoke_cache tests/smoke_results
python -m src.run_experiment "${COMMON[@]}" --check-pairs
python -m src.run_experiment "${COMMON[@]}" --modes combined bert sbert
# determinism: re-running one pair with --force must reproduce identical numbers
cp tests/smoke_results/combined/pairs/pair_01.json /tmp/pair01_a.json
python -m src.run_experiment "${COMMON[@]}" --pairs 1 --modes combined --force > /dev/null
python - <<'EOF'
import json
a = json.load(open("/tmp/pair01_a.json")); b = json.load(open("tests/smoke_results/combined/pairs/pair_01.json"))
assert a["test"] == b["test"], (a["test"], b["test"])
assert a["cv"]["fold_MAE"] == b["cv"]["fold_MAE"]
print("determinism check passed:", a["test"])
EOF
python -m src.make_report --config configs/default.yaml --modes combined bert sbert --allow-smoke --results-dir tests/smoke_results \
  --out tests/smoke_results/smoke_report.pdf
echo "SMOKE TEST FINISHED"
