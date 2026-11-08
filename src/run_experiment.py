"""Run the cross-project / random-split experiments.

Examples
--------
python -m src.run_experiment --check-pairs              # verify the 39 pairs match TAWOS
python -m src.run_experiment --list-projects            # projects + number of labelled issues
python -m src.run_experiment                            # all 39 pairs, mode from config
python -m src.run_experiment --pairs 1 2 --modes bert sbert combined
python -m src.run_experiment --set embeddings.backend=hash --set train.max_epochs=3   # smoke test
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
import yaml

from . import data as D
from .embeddings import get_embeddings
from .train import Block, fit, metrics, predict, set_seed


def load_cfg(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def grid_points(grid: dict):
    keys = sorted(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]


def run_pair(cfg, pair, issues, available, device, force=False):
    mode = cfg["model"]["mode"]
    out_dir = os.path.join(cfg["output"]["results_dir"], mode)
    os.makedirs(os.path.join(out_dir, "pairs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    pid = int(pair["pair_id"])
    out_json = os.path.join(out_dir, "pairs", f"pair_{pid:02d}.json")
    if os.path.exists(out_json) and not force:
        print(f"[pair {pid}] {mode}: already done, skipping (use --force to redo)")
        return

    t0 = time.time()
    aliases = cfg["data"].get("project_aliases", {})
    src_name = D.resolve_project(pair["train_project"], available, aliases)
    tgt_name = D.resolve_project(pair["test_project"], available, aliases)
    src = issues[issues.project == src_name].reset_index(drop=True)
    tgt = issues[issues.project == tgt_name].reset_index(drop=True)
    rec = {"pair_id": pid, "train_project": pair["train_project"], "test_project": pair["test_project"],
           "train_project_tawos": src_name, "test_project_tawos": tgt_name, "mode": mode,
           "backend": cfg["embeddings"]["backend"], "n_source": len(src), "n_target_total": len(tgt)}

    dev_idx, folds, test_idx = D.make_split(len(src), len(tgt), cfg, pid)
    rec["n_source_dev"] = int(len(dev_idx))
    rec["n_target_test"] = int(len(test_idx))
    if len(dev_idx) < cfg["data"]["min_source_issues"] or len(test_idx) < cfg["data"]["min_target_test_issues"]:
        rec["status"] = "skipped"
        rec["reason"] = (f"too few labelled issues (source dev={len(dev_idx)}, target test={len(test_idx)}; "
                         f"minimums {cfg['data']['min_source_issues']}/{cfg['data']['min_target_test_issues']})")
        json.dump(rec, open(out_json, "w"), indent=2)
        print(f"[pair {pid}] SKIPPED: {rec['reason']}")
        return

    # leakage guards: target issues are never part of the source development data
    assert set(src.issue_id).isdisjoint(set(tgt.issue_id)), "source/target issue overlap"
    for tr, va in folds:
        assert set(tr).isdisjoint(set(va))
    assert sorted(np.concatenate([va for _, va in folds]).tolist()) == dev_idx.tolist()
    for tr, va in folds:
        assert set(tr) | set(va) == set(dev_idx.tolist())

    s_tok, s_mask, s_sent = get_embeddings(cfg, src_name, src.issue_id.values, src.text.values)
    t_tok, t_mask, t_sent = get_embeddings(cfg, tgt_name, tgt.issue_id.values, tgt.text.values)
    y_src, y_tgt = src.sp.values.astype(np.float32), tgt.sp.values.astype(np.float32)
    seed = cfg["seed"] + pid

    # ---- model selection on development data only (5-fold CV on the source project)
    grid_results = []
    for hp in grid_points(cfg["train"]["grid"]):
        fold_m, best_eps = [], []
        for k, (tr, va) in enumerate(folds):
            model, ep = fit(cfg, hp, Block(s_tok, s_mask, s_sent, y_src, tr),
                            Block(s_tok, s_mask, s_sent, y_src, va), seed + 100 * k, device)
            m = metrics(y_src[va], predict(model, Block(s_tok, s_mask, s_sent, y_src, va), device,
                                           clip_max=float(y_src[tr].max())))
            fold_m.append(m)
            best_eps.append(ep)
        mae = [m["MAE"] for m in fold_m]
        mdae = [m["MdAE"] for m in fold_m]
        grid_results.append({"hp": hp, "fold_MAE": mae, "fold_MdAE": mdae,
                             "MAE_mean": float(np.mean(mae)), "MAE_std": float(np.std(mae, ddof=1)),
                             "MdAE_mean": float(np.mean(mdae)), "MdAE_std": float(np.std(mdae, ddof=1)),
                             "best_epochs": best_eps})
        print(f"[pair {pid}] {mode} hp={hp} val MAE={np.mean(mae):.3f}±{np.std(mae, ddof=1):.3f}")
    best = min(grid_results, key=lambda r: (r["MAE_mean"], r["MdAE_mean"]))
    rec.update(grid=grid_results, selected_hp=best["hp"], cv=best)

    # ---- final model: refit on ALL development data, evaluate once on the target test set
    n_epochs = max(1, int(round(float(np.median(best["best_epochs"])))))
    model, _ = fit(cfg, best["hp"], Block(s_tok, s_mask, s_sent, y_src, dev_idx),
                   None, seed + 999, device, fixed_epochs=n_epochs)
    test_blk = Block(t_tok, t_mask, t_sent, y_tgt, test_idx)
    pred = predict(model, test_blk, device, clip_max=model.clip_max)
    rec["final_epochs"] = n_epochs
    rec["test"] = metrics(y_tgt[test_idx], pred)
    base = float(np.median(y_src[dev_idx]))  # reference: always predict the source-project median
    rec["baseline_median_test"] = metrics(y_tgt[test_idx], np.full(len(test_idx), base))
    rec["source_sp_median"] = base
    rec["target_sp_median"] = float(np.median(y_tgt[test_idx]))
    rec["status"] = "done"
    rec["runtime_sec"] = round(time.time() - t0, 1)
    rec["env"] = {"torch": torch.__version__, "python": platform.python_version(), "device": str(device)}
    rec["config"] = cfg
    json.dump(rec, open(out_json, "w"), indent=2, default=str)
    pd.DataFrame({"issue_id": tgt.issue_id.values[test_idx], "y_true": y_tgt[test_idx],
                  "y_pred": pred}).to_csv(os.path.join(out_dir, "predictions", f"pair_{pid:02d}.csv"), index=False)
    print(f"[pair {pid}] {mode} TEST MAE={rec['test']['MAE']:.3f} MdAE={rec['test']['MdAE']:.3f} "
          f"(median baseline MAE={rec['baseline_median_test']['MAE']:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", help="override, e.g. --set train.max_epochs=3")
    ap.add_argument("--pairs", nargs="*", type=int, help="pair ids to run (default: all 39)")
    ap.add_argument("--modes", nargs="*", help="bert sbert combined (default: config model.mode)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list-projects", action="store_true")
    ap.add_argument("--check-pairs", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.set)

    issues = D.load_issues(cfg)
    counts = issues.groupby("project").size().sort_values(ascending=False)
    if a.list_projects:
        print(counts.to_string())
        return
    available = sorted(counts.index)
    pairs = D.load_pairs(cfg)
    if a.check_pairs:
        bad = 0
        for _, r in pairs.iterrows():
            row = [r.pair_id]
            for col in ("train_project", "test_project"):
                try:
                    n = D.resolve_project(r[col], available, cfg["data"].get("project_aliases", {}))
                    row.append(f"{r[col]} -> {n} ({counts[n]})")
                except ValueError as e:
                    row.append(f"{r[col]} -> !! {e}")
                    bad += 1
            print(*row, sep=" | ")
        print(f"\n{bad} unresolved names" if bad else "\nAll names resolved.")
        return

    if a.pairs:
        pairs = pairs[pairs.pair_id.isin(a.pairs)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  backend={cfg['embeddings']['backend']}  pairs={len(pairs)}")
    for mode in (a.modes or [cfg["model"]["mode"]]):
        c = json.loads(json.dumps(cfg))
        c["model"]["mode"] = mode
        for _, pair in pairs.iterrows():
            run_pair(c, pair, issues, available, device, a.force)


if __name__ == "__main__":
    main()
