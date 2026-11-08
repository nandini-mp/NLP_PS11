"""Aggregate results, make tables/figures, and build the final PDF report.

    python -m src.make_report                       # uses results/ and report/discussion.md
    python -m src.make_report --modes combined bert sbert

Everything numeric in the PDF is read from results/*.json. The written discussion
comes from report/discussion.md (you write it after seeing the numbers).
Results produced with the 'hash' smoke-test backend are rejected.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NAMES = {"bert": "BERT word embeddings", "sbert": "SBERT sentence embeddings",
         "combined": "Combined BERT + SBERT"}


def load_mode(results_dir, mode, allow_smoke=False):
    rows = []
    for f in sorted(glob.glob(os.path.join(results_dir, mode, "pairs", "pair_*.json"))):
        r = json.load(open(f))
        if r.get("backend") != "hf" and not allow_smoke:
            raise SystemExit(f"{f} was produced with backend '{r.get('backend')}' (smoke test). "
                             "Re-run with embeddings.backend=hf before building the report.")
        row = {k: r.get(k) for k in ("pair_id", "train_project", "test_project", "n_source",
                                     "n_target_total", "n_target_test", "status", "reason")}
        if r["status"] == "done":
            row.update(cv_MAE_mean=r["cv"]["MAE_mean"], cv_MAE_std=r["cv"]["MAE_std"],
                       cv_MdAE_mean=r["cv"]["MdAE_mean"], cv_MdAE_std=r["cv"]["MdAE_std"],
                       test_MAE=r["test"]["MAE"], test_MdAE=r["test"]["MdAE"],
                       base_MAE=r["baseline_median_test"]["MAE"], base_MdAE=r["baseline_median_test"]["MdAE"],
                       lr=r["selected_hp"]["lr"], dropout=r["selected_hp"]["dropout"],
                       epochs=r["final_epochs"], config=r["config"])
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(x, d=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def figures(dfs, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    paths = []
    main_mode = next(iter(dfs))
    d = dfs[main_mode]
    d = d[d.status == "done"].sort_values("pair_id")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(11, 4))
    w = 0.8 / (len(dfs) + 1)
    ax.bar(x - 0.4 + w / 2, d.base_MAE, w, label="Median baseline", color="#bbbbbb")
    for i, (m, df) in enumerate(dfs.items()):
        dd = df[df.status == "done"].set_index("pair_id").reindex(d.pair_id)
        ax.bar(x - 0.4 + w * (i + 1.5), dd.test_MAE.values, w, label=NAMES[m])
    ax.set_xticks(x)
    ax.set_xticklabels(d.pair_id.astype(int), fontsize=7)
    ax.set_xlabel("Cross-project pair id")
    ax.set_ylabel("Test MAE (story points)")
    ax.set_title("Held-out test MAE per pair")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(fig_dir, "test_mae_per_pair.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(("Test MAE per pair, compared with the median baseline.", p))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(d.cv_MAE_mean, d.test_MAE, s=18)
    lim = max(d.cv_MAE_mean.max(), d.test_MAE.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlabel("Mean 5-fold validation MAE (source project)")
    ax.set_ylabel("Test MAE (target project)")
    ax.set_title(f"{NAMES[main_mode]}: validation vs. cross-project test")
    fig.tight_layout()
    p = os.path.join(fig_dir, "val_vs_test.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(("Within-source validation MAE versus cross-project test MAE (dashed: equality).", p))
    return paths


def stats_sentences(dfs):
    out = []
    for m, df in dfs.items():
        d = df[df.status == "done"]
        better = int((d.test_MAE < d.base_MAE).sum())
        out.append(f"{NAMES[m]}: test MAE below the median baseline in {better} of {len(d)} pairs; "
                   f"mean test MAE {d.test_MAE.mean():.3f}, median across pairs {d.test_MAE.median():.3f}; "
                   f"baseline mean {d.base_MAE.mean():.3f}.")
        if len(d) >= 6:
            try:
                p = wilcoxon(d.test_MAE, d.base_MAE).pvalue
                out.append(f"Wilcoxon signed-rank test (per-pair test MAE vs. baseline), {NAMES[m]}: p = {p:.4f}.")
            except ValueError:
                pass
    ms = list(dfs)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a = dfs[ms[i]].set_index("pair_id")
            b = dfs[ms[j]].set_index("pair_id")
            ids = [k for k in a.index if k in b.index and a.status[k] == "done" and b.status[k] == "done"]
            if len(ids) >= 6:
                try:
                    p = wilcoxon(a.test_MAE[ids], b.test_MAE[ids]).pvalue
                    out.append(f"Wilcoxon signed-rank test on test MAE, {NAMES[ms[i]]} vs {NAMES[ms[j]]} "
                               f"({len(ids)} pairs): p = {p:.4f}.")
                except ValueError:
                    pass
    return out


def md_to_flowables(text, styles):
    fl = []
    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if block.startswith("## "):
            fl.append(Paragraph(escape(block[3:]), styles["H2"]))
        elif block.startswith("# "):
            continue
        elif all(l.strip().startswith("- ") for l in block.splitlines()):
            for l in block.splitlines():
                fl.append(Paragraph("&bull; " + escape(l.strip()[2:]), styles["Body"]))
        else:
            fl.append(Paragraph(escape(" ".join(block.split())), styles["Body"]))
    return fl


def table(data, col_widths=None, font=7.5):
    cell = ParagraphStyle("cell", fontSize=font, leading=font + 2)
    hcell = ParagraphStyle("hcell", fontSize=font, leading=font + 2, fontName="Helvetica-Bold")
    data = [[(Paragraph(escape(str(c)), hcell if i == 0 else cell) if isinstance(c, str) else c) for c in row]
            for i, row in enumerate(data)]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), font), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]))
    return t


def build_pdf(cfg, dfs, fig_paths, stat_lines, out_pdf, discussion):
    ss = getSampleStyleSheet()
    styles = {"Title": ParagraphStyle("T", parent=ss["Title"], fontSize=17),
              "H1": ParagraphStyle("H1", parent=ss["Heading1"], fontSize=13, spaceBefore=10),
              "H2": ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11),
              "Body": ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13)}
    rc = cfg.get("report", {})
    title = rc.get("title", "Story point estimation: results")
    main = next(iter(dfs))
    c = cfg
    doc = SimpleDocTemplate(out_pdf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm, title=title)
    S = []
    S += [Paragraph(escape(title), styles["Title"]),
          Paragraph("<b>Assigned problem statement (Project 11):</b> Evaluate whether combined word and sentence "
                    "embeddings improve story point estimation when generalising from source projects to a "
                    "different target project (cross-project, random split).", styles["Body"]), Spacer(1, 8)]
    S.append(Paragraph("1. Problem and objective", styles["H1"]))
    S += md_to_flowables(
        "Story point estimation predicts the relative effort of an issue from its text. We study whether "
        "concatenating BERT token-level representations with SBERT sentence-level representations helps a "
        "regression model trained on one TAWOS project predict story points of issues in a different project. "
        "Performance is measured with Mean Absolute Error (MAE) and Median Absolute Error (MdAE).", styles)
    S.append(Paragraph("2. Data", styles["H1"]))
    d = dfs[main]
    S += md_to_flowables(
        f"Dataset: TAWOS (https://doi.org/10.5522/04/21308124). Issues with a numeric story point greater than 0 and "
        f"non-empty text were kept"
        + (f"; issues with story point above {c['data']['max_story_point']} (obvious data-entry errors) were removed" if c["data"].get("max_story_point") is not None else "")
        + f". Text = title + description with code blocks, URLs and Jira markup removed. "
        f"The 39 assigned source/target pairs were used; {int((d.status == 'done').sum())} pairs were evaluated and "
        f"{int((d.status != 'done').sum())} were skipped (listed in Section 5).", styles)
    S.append(Paragraph("3. Method", styles["H1"]))
    e, m, t, sp = c["embeddings"], c["model"], c["train"], c["split"]
    S += md_to_flowables(
        f"- Word-level representation: frozen {e['bert_model']}, hidden layer {e['bert_layer']}, contextual token vectors, "
        f"max {c['text']['max_tokens']} tokens.\n\n"
        f"- Sentence-level representation: frozen {e['sbert_model']} (one vector per issue).\n\n"
        f"- Fusion (late concatenation): token vectors pass through a three-layer 1-D CNN (channels {m['conv_channels']}, "
        f"kernel {m['kernel_size']}, ReLU, dropout) and a masked global max-pool; the SBERT vector passes through a "
        f"linear layer ({m['sbert_proj_dim']} units, ReLU); the two are concatenated and fed to an MLP head "
        f"({m['head_hidden']} units) with a single output.\n\n"
        f"- Training: AdamW, Huber loss (delta {t['huber_delta']}), batch {t['batch_size']}, up to {t['max_epochs']} "
        f"epochs with early stopping (patience {t['patience']}) on validation MAE; predictions clipped to [0, max training SP]. "
        f"Output bias initialised to the training median.\n\n"
        f"- Hyper-parameter grid: {t['grid']}; seed {c['seed']} (+ pair id for splits).", styles)
    S.append(Paragraph("4. Experimental protocol", styles["H1"]))
    S += md_to_flowables(
        f"Development data = a random {int(sp['source_dev_fraction'] * 100)}% of the labelled issues of the source project, split into {sp['n_folds']} random folds "
        f"(each fold is the validation fold once). The configuration with the lowest mean validation MAE is selected "
        f"using development data only, retrained on all development data for the median best epoch, and evaluated "
        f"once on the final test set: a random {int(sp['target_test_fraction'] * 100)}% sample of the target "
        f"project's labelled issues. No target data is used for training or model selection. The same partitions "
        f"are used for every representation. As a reference, a trivial baseline always predicts the median "
        f"story point of the source project.", styles)

    S.append(PageBreak())
    S.append(Paragraph("5. Results", styles["H1"]))
    # summary table (required format)
    S.append(Paragraph("5.1 Summary across evaluated pairs (mean over pairs)", styles["H2"]))
    head = ["Representation", "Model configuration", "Test MAE", "Test MdAE", "Val MAE (5-fold mean)", "Val MdAE (5-fold mean)"]
    rows = [head]
    for mo, df in dfs.items():
        dd = df[df.status == "done"]
        cfgm = f"{e['bert_model'] if mo != 'sbert' else ''}{' + ' if mo == 'combined' else ''}{e['sbert_model'] if mo != 'bert' else ''}"
        rows.append([NAMES[mo], cfgm, fmt(dd.test_MAE.mean()), fmt(dd.test_MdAE.mean()),
                     fmt(dd.cv_MAE_mean.mean()), fmt(dd.cv_MdAE_mean.mean())])
    ex = next(iter(dfs.values()))
    ex = ex[ex.status == "done"]
    rows.append(["Median baseline", "predict source-project median", fmt(ex.base_MAE.mean()), fmt(ex.base_MdAE.mean()), "-", "-"])
    S.append(table(rows, [3.2 * cm, 5.2 * cm, 1.8 * cm, 1.8 * cm, 2.6 * cm, 2.6 * cm], 7.5))
    S.append(Spacer(1, 6))
    for line in stat_lines:
        S.append(Paragraph("&bull; " + escape(line), styles["Body"]))

    for mo, df in dfs.items():
        S.append(Paragraph(f"5.2 Per-pair results: {NAMES[mo]}", styles["H2"]))
        rows = [["Pair", "Source -> Target", "n src", "n test", "Val MAE (mean±sd)", "Val MdAE (mean±sd)",
                 "Test MAE", "Test MdAE", "Base MAE", "Base MdAE"]]
        for _, r in df.sort_values("pair_id").iterrows():
            lab = Paragraph(escape(f"{r.train_project} -> {r.test_project}"), ParagraphStyle("c", fontSize=6.5, leading=8))
            if r.status == "done":
                rows.append([int(r.pair_id), lab, int(r.n_source), int(r.n_target_test),
                             f"{fmt(r.cv_MAE_mean)}±{fmt(r.cv_MAE_std)}", f"{fmt(r.cv_MdAE_mean)}±{fmt(r.cv_MdAE_std)}",
                             fmt(r.test_MAE), fmt(r.test_MdAE), fmt(r.base_MAE), fmt(r.base_MdAE)])
            else:
                rows.append([int(r.pair_id), lab, r.n_source, r.n_target_test, "skipped", "skipped", "-", "-", "-", "-"])
        S.append(table(rows, [0.9 * cm, 4.2 * cm, 1.1 * cm, 1.1 * cm, 2.3 * cm, 2.3 * cm, 1.4 * cm, 1.4 * cm, 1.3 * cm, 1.3 * cm], 6.5))
        skipped = df[df.status != "done"]
        if len(skipped):
            S.append(Paragraph("Skipped pairs: " + escape("; ".join(f"pair {int(r.pair_id)} ({r.reason})" for _, r in skipped.iterrows())), styles["Body"]))
    S.append(PageBreak())
    S.append(Paragraph("5.3 Figures", styles["H1"]))
    for cap, p in fig_paths:
        S.append(Image(p, width=15 * cm, height=15 * cm * (4 / 11 if "per_pair" in p else 4.5 / 5), kind="proportional"))
        S.append(Paragraph(escape(cap), styles["Body"]))
    if discussion.strip():
        S.append(Paragraph("6. Discussion, limitations and conclusions", styles["H1"]))
        S += md_to_flowables(discussion, styles)
    S.append(Paragraph("Reproducibility", styles["H1"]))
    S += md_to_flowables("All code, configuration (configs/default.yaml), per-pair JSON results and predictions are in the "
                         "repository. Follow README.md to reproduce.", styles)
    doc.build(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--modes", nargs="*", default=["combined"])
    ap.add_argument("--discussion", default="report/discussion.md")
    ap.add_argument("--out", default="report/final_report.pdf")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--allow-smoke", action="store_true", help="testing only")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    rd = a.results_dir or cfg["output"]["results_dir"]
    dfs = {}
    for mo in a.modes:
        df = load_mode(rd, mo, a.allow_smoke)
        if len(df):
            dfs[mo] = df
            df.drop(columns=["config"], errors="ignore").to_csv(os.path.join(rd, f"summary_{mo}.csv"), index=False)
    if not dfs:
        raise SystemExit("no results found; run src.run_experiment first")
    # Describe the configuration that was actually run (stored inside each result file).
    done = [d[d.status == "done"] for d in dfs.values() if (d.status == "done").any()]
    if not done:
        raise SystemExit("no finished pairs in the results")
    report_section = cfg.get("report", {})
    cfg = done[0].iloc[0]["config"]
    cfg["report"] = report_section
    paths = figures(dfs, os.path.join(rd, "figures"))
    stat_lines = stats_sentences(dfs)
    disc = open(a.discussion).read() if os.path.exists(a.discussion) else ""
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    build_pdf(cfg, dfs, paths, stat_lines, a.out, disc)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
