from __future__ import annotations

import difflib
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


_CODE_BLOCK = re.compile(r"\{(code|noformat)[^}]*\}.*?\{\1\}", re.S | re.I)
_URL = re.compile(r"https?://\S+")
_JIRA_MARKUP = re.compile(r"\{[^}]{1,40}\}|\[~[^\]]*\]|\[\^[^\]]*\]|!\S+!")
_WS = re.compile(r"\s+")


def clean_text(title, description) -> str:
    title = "" if pd.isna(title) else str(title)
    description = "" if pd.isna(description) else str(description)
    description = _CODE_BLOCK.sub(" ", description)
    text = f"{title}. {description}"
    text = _URL.sub(" ", text)
    text = _JIRA_MARKUP.sub(" ", text)
    return _WS.sub(" ", text).strip()


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9+]+", " ", str(s).lower()).strip()


def resolve_project(short_name: str, available: list[str], aliases: dict) -> str:
    if short_name in aliases:
        target = aliases[short_name]
        hits = [a for a in available if norm_name(a) == norm_name(target)]
        if len(hits) != 1:
            raise ValueError(f"Alias '{short_name}' -> '{target}' is not a TAWOS project name. "
                             f"Available: {available}")
        return hits[0]
    n = norm_name(short_name)
    exact = [a for a in available if norm_name(a) == n]
    if len(exact) == 1:
        return exact[0]
    contain = [a for a in available if n in norm_name(a) or norm_name(a) in n]
    if len(contain) == 1:
        return contain[0]
    cands = contain or difflib.get_close_matches(short_name, available, n=5, cutoff=0.4)
    raise ValueError(
        f"Cannot uniquely match project '{short_name}'. Candidates: {cands}. "
        "Add an entry to data.project_aliases in the config."
    )


def load_issues(cfg: dict) -> pd.DataFrame:
    c = cfg["data"]["columns"]
    raw = pd.read_csv(cfg["data"]["csv_path"])
    missing = [v for v in c.values() if v not in raw.columns]
    if missing:
        raise KeyError(f"Columns {missing} not in CSV. Available: {list(raw.columns)}")
    df = pd.DataFrame(
        {
            "issue_id": raw[c["id"]],
            "project": raw[c["project"]].astype(str),
            "title": raw[c["title"]],
            "description": raw[c["description"]],
            "created": pd.to_datetime(raw[c["created"]], errors="coerce"),
            "sp": pd.to_numeric(raw[c["story_point"]], errors="coerce"),
        }
    )
    df = df.drop_duplicates("issue_id")
    df = df[df["sp"].notna() & (df["sp"] > 0)].copy()
    cap = cfg["data"].get("max_story_point")
    removed = df.iloc[0:0]
    if cap is not None:
        removed = df[df["sp"] > cap]
        df = df[df["sp"] <= cap].copy()
    df["text"] = [clean_text(t, d) for t, d in zip(df["title"], df["description"])]
    df = df[df["text"].str.len() > 3].reset_index(drop=True)
    df.attrs["removed_over_cap"] = removed[["project", "issue_id", "sp"]].copy()
    return df


def load_pairs(cfg: dict) -> pd.DataFrame:
    pairs = pd.read_csv(cfg["data"]["pairs_path"])
    expected = cfg["data"].get("expected_pairs", 39)
    assert len(pairs) == expected, f"expected {expected} pairs, found {len(pairs)}"
    return pairs


def make_split(n_source: int, n_target: int, cfg: dict, pair_id: int):
    seed = cfg["seed"] + pair_id
    rng = np.random.RandomState(seed)
    n_dev = min(max(int(round(cfg["split"]["source_dev_fraction"] * n_source)), 1), n_source)
    dev_idx = np.sort(rng.choice(n_source, size=n_dev, replace=False))
    kf = KFold(n_splits=cfg["split"]["n_folds"], shuffle=True, random_state=seed)
    folds = [(dev_idx[tr], dev_idx[va]) for tr, va in kf.split(dev_idx)]
    k = int(round(cfg["split"]["target_test_fraction"] * n_target))
    k = min(max(k, 1), n_target)
    target_test = np.sort(rng.choice(n_target, size=k, replace=False))
    return dev_idx, folds, target_test
