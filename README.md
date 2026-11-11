# Combined BERT + SBERT Embeddings for Cross-Project Story Point Estimation (Random Split)

**NLP project, Problem Statement 11** — combined word and sentence embeddings, cross-project setting, random split.
**Group 14** - Nidhi Binu, Nandini M Prakash, Tarun Pradeep, Nepala Chiruhhasan, Tej Pratap Singh

## 1. Problem statement
Evaluate whether combined word (BERT) and sentence (SBERT) embeddings improve story point estimation when generalising from a source project to a different target project (guideline project 11).

## 2. Objectives
1. Predict the story points of an issue from its title and description.
2. Compare three representations under identical data partitions: BERT word-level, SBERT sentence-level, combined BERT + SBERT.
3. Evaluate on the 39 assigned (source → target) project pairs with MAE and MdAE; report 5-fold validation results separately from final held-out test results.

## 3. Dataset
TAWOS (Tawosi et al.), https://doi.org/10.5522/04/21308124. Issues with a numeric story point > 0 and non-empty text are kept. Text = title + description with `{code}` blocks, URLs and Jira markup removed. Issues with story point > 500 (7 issues, obvious data-entry errors) are removed; no other outlier removal. The data is **not** included in this repository (size/licence): download the TAWOS MySQL dump from the DOI above, load it into a local MySQL 8 database named `tawos`, and export the labelled issues with `src/export_tawos.py` (step 0 in section 7).

## 4. Method
| Part | Choice |
|---|---|
| Word-level | frozen `bert-base-uncased`, last hidden layer, token vectors, max 128 tokens |
| Sentence-level | frozen `sentence-transformers/all-MiniLM-L6-v2` (one vector / issue) |
| Fusion (combined) | BERT tokens → 3×[Conv1d(k=3)+ReLU+Dropout] → masked global max-pool; SBERT → Linear+ReLU; **concatenate** → Linear+ReLU+Dropout → Linear(1) |
| BERT-only / SBERT-only | same head, only the corresponding branch (SBERT-only has no CNN since a single vector has no sequence axis) |
| Training | AdamW, Huber loss, batch 32, ≤30 epochs, early stopping (patience 5) on validation MAE, predictions clipped to [0, max train SP], output bias initialised to the training median |
| Grid | lr ∈ {1e-3, 3e-4} × dropout ∈ {0.2, 0.4}, chosen by mean 5-fold validation MAE |
| Seeds | master seed 42; split seed = 42 + pair id |

All values live in `configs/default.yaml`.

## 5. Protocol (guideline 3.2)
For each pair:
1. **Development data** = random 80% of the labelled *source* issues. **Test data** = random 20% of the labelled *target* issues. Nothing else is used.
2. The development data is divided into 5 random folds; each fold is the validation fold once.
3. The configuration with the lowest mean validation MAE is selected (development data only).
4. It is retrained on all development data (median best epoch of the 5 folds) and evaluated **once** on the target test set.
5. The same partitions are used for BERT, SBERT and combined runs (paired comparison).
6. Reference baseline: always predict the median story point of the source development data.
7. Pairs with fewer than 50 source-dev issues or fewer than 10 target test issues are skipped and listed in the report.

Code asserts that no issue is shared between source and target, that train/validation folds are disjoint and that the folds cover the development set.

## 6. Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.10+. A GPU speeds up embedding extraction a lot but is not required.

## 7. Reproducing the experiments
```bash
# 0. Load the TAWOS dump into MySQL (database `tawos`), then export the labelled issues
python -m src.export_tawos --mysql --db tawos --user root --password XXXX   # -> data/tawos_story_points.csv

# 1. Check that all 78 project names in the 39 pairs match TAWOS
python -m src.check_pairs

# 2. Run (resumable; finished pairs are skipped)
python -m src.run_experiment --modes combined bert sbert

# 3. Summary tables, figures and a PDF overview from the saved results
python -m src.make_report --modes combined bert sbert
```
Outputs: `results/<mode>/pairs/pair_XX.json` (all fold metrics, grid, test metrics, config, versions), `results/<mode>/predictions/*.csv`, `results/summary_<mode>.csv`, `results/figures/*.png`; `make_report` also writes an overview PDF to `report/` (not tracked).

### Project name mapping
Three project names of the assignment sheet are mapped manually in `configs/default.yaml` (`data.project_aliases`): C++ driver (MongoDB) -> The Mongo C++ driver; Evergreen -> The MongoDB Engineering (Jira key EVG); C++ driver (Appcelerator) -> Aptana Studio (assigned by elimination: the only unmatched project in the Appcelerator repository).

### Colab notebooks
`notebooks/nlp1.ipynb` (combined and SBERT runs, story-point cap, reruns) and `notebooks/nlp2.ipynb` (BERT run, final checks, report build) are the Colab notebooks with which the reported results were produced.

## 8. Offline self-test
`bash tests/run_smoke_test.sh` runs the whole pipeline on synthetic data with tiny random models (no internet). Its numbers are meaningless; it only checks that the code runs, the split guards hold and results are deterministic. `make_report` refuses results not produced with `embeddings.backend: hf`.

## 9. Repository layout
```
configs/default.yaml     all settings      src/data.py        cleaning, splits
data/pairs.csv           39 pairs          src/embeddings.py  BERT / SBERT extraction + cache
src/export_tawos.py      DB -> CSV         src/model.py       CNN + fusion network
src/run_experiment.py    experiments       src/train.py       training, metrics
src/make_report.py       tables/plots/PDF  src/check_pairs.py  project-name check
notebooks/               Colab runs        tests/             offline smoke test
```
