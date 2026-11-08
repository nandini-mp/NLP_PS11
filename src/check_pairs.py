import sys

import pandas as pd
import yaml

from . import data as D


def main(config="configs/default.yaml"):
    cfg = yaml.safe_load(open(config))
    issues = D.load_issues(cfg)
    rem = issues.attrs.get("removed_over_cap")
    if rem is not None and len(rem):
        print(f"\nCLEANING RULE: story point > {cfg['data']['max_story_point']} removed ({len(rem)} issues):")
        print(rem.groupby("project")["sp"].agg(["count", "max"]).to_string())
    counts = issues.groupby("project").size().sort_values(ascending=False)
    print(f"\n{len(issues)} usable issues in {len(counts)} projects (story point > 0, non-empty text)\n")
    print("ALL TAWOS PROJECTS WITH STORY POINTS (usable issues):")
    print(counts.to_string())
    available = sorted(counts.index)
    pairs = D.load_pairs(cfg)
    aliases = cfg["data"].get("project_aliases", {})
    sp = cfg["split"]
    print("\n\nPAIR CHECK  (dev = source issues used for 5-fold; test = target issues in final test)\n")
    bad, small = 0, []
    for _, r in pairs.iterrows():
        res = []
        for col in ("train_project", "test_project"):
            try:
                res.append(D.resolve_project(r[col], available, aliases))
            except ValueError as e:
                res.append(None)
                print(f"pair {r.pair_id:>2}  !! {r[col]}: {e}")
                bad += 1
        if None in res:
            continue
        n_dev = round(sp["source_dev_fraction"] * counts[res[0]])
        n_test = round(sp["target_test_fraction"] * counts[res[1]])
        skip = n_dev < cfg["data"]["min_source_issues"] or n_test < cfg["data"]["min_target_test_issues"]
        if skip:
            small.append(int(r.pair_id))
        print(f"pair {r.pair_id:>2}  {r.train_project!r} -> {res[0]!r} (dev {n_dev})  |  "
              f"{r.test_project!r} -> {res[1]!r} (test {n_test})" + ("   << WOULD BE SKIPPED" if skip else ""))
    print()
    print(f"{bad} unresolved names" if bad else "All 78 names resolved.")
    print(f"Pairs that would be skipped for being too small: {small if small else 'none'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
