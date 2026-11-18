"""Report figures and tables generated from the saved experiment results.

Every figure is drawn from files written by the experiment scripts. Nothing is
typed in by hand, and figures whose inputs do not exist yet are skipped.

Outputs: results/figures/*.png and results/tables/*.md|csv

Usage:  python scripts/make_report_figures.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathsnap import config  # noqa: E402

FIG = config.RESULTS_DIR / "figures"
TAB = config.RESULTS_DIR / "tables"
MODELS = config.RESULTS_DIR / "models"
# Categorical slots in fixed order (validated palette, light mode); grey = non-learned baselines.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
GREY = "#a3a29c"
INK, INK2 = "#0b0b0b", "#52514e"
PRETTY = {
    "baseline_nearest_arc": "Baseline: nearest arc", "baseline_distance_heading": "Baseline: distance + heading",
    "logistic_regression": "Logistic regression", "gaussian_nb": "Gaussian NB", "knn": "KNN",
    "random_forest": "Random forest", "lightgbm": "LightGBM", "mlp": "MLP",
    "lightgbm_tuned": "LightGBM (tuned)", "random_forest_tuned": "Random forest (tuned)",
    "heuristic": "Heuristic (no model)",
}


def style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "regular", "axes.labelcolor": INK2,
        "axes.edgecolor": "#c9c8c2", "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": "#e7e6e1", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })


def name(m: str) -> str:
    return PRETTY.get(m, m)


def save(fig, fname: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / fname)
    plt.close(fig)
    print("wrote", FIG / fname)


def hbar(ax, labels, values, errors=None, colors=None, fmt="{:.3f}", xlim=None):
    y = np.arange(len(labels))
    ax.barh(y, values, xerr=errors, color=colors, height=0.62, edgecolor="white", linewidth=2,
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for yi, v, e in zip(y, values, errors if errors is not None else [0] * len(values)):
        if np.isfinite(v):
            ax.text(v + (e if np.isfinite(e) else 0) + (xlim[1] - xlim[0]) * 0.01 if xlim else v, yi,
                    fmt.format(v), va="center", fontsize=9, color=INK)
    if xlim:
        ax.set_xlim(*xlim)


def model_figures():
    path = MODELS / "model_comparison.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df.sort_values("cv_top1_mean", ascending=False).reset_index(drop=True)
    colors = [GREY if m.startswith("baseline") else C1 for m in df["model"]]
    labels = [name(m) for m in df["model"]]

    for metric, title, fname in [
        ("cv_top1", "Cross-validated Top-1 accuracy (5 trip-grouped folds, train)", "cv_top1.png"),
        ("cv_macro_f1", "Cross-validated macro F1 (5 trip-grouped folds, train)", "cv_macro_f1.png"),
    ]:
        lo = max(0.0, np.nanmin(df[f"{metric}_mean"]) - 0.08)
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(df) + 1.2))
        hbar(ax, labels, df[f"{metric}_mean"].to_numpy(), df[f"{metric}_std"].to_numpy(), colors,
             xlim=(lo, 1.0))
        ax.set_title(title + "\nbars = mean, whiskers = std across folds; grey = non-learned baselines",
                     loc="left", color=INK)
        save(fig, fname)

    # student_test: Top-1 and macro F1 side by side (two series -> legend)
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(df) + 1.4))
    y = np.arange(len(df))
    h = 0.36
    ax.barh(y - h / 2, df["test_top1"], height=h, color=C1, edgecolor="white", linewidth=2, label="Top-1 accuracy")
    ax.barh(y + h / 2, df["test_macro_f1"], height=h, color=C2, edgecolor="white", linewidth=2, label="Macro F1")
    for yi, a, b in zip(y, df["test_top1"], df["test_macro_f1"]):
        ax.text(a + 0.003, yi - h / 2, f"{a:.3f}", va="center", fontsize=8, color=INK)
        ax.text(b + 0.003, yi + h / 2, f"{b:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(max(0, min(df["test_top1"].min(), df["test_macro_f1"].min()) - 0.08), 1.0)
    ax.legend(loc="lower right")
    ax.set_title("Student-test performance (10 held-out development trips)", loc="left", color=INK)
    save(fig, "student_test_top1_macro_f1.png")

    # log loss (probabilistic models only)
    ll = df[df["test_log_loss"].notna()].sort_values("test_log_loss")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(ll) + 1.3))
    y = np.arange(len(ll))
    ax.barh(y - h / 2, ll["cv_log_loss_mean"], height=h, color=C1, edgecolor="white", linewidth=2, label="CV (train)")
    ax.barh(y + h / 2, ll["test_log_loss"], height=h, color=C2, edgecolor="white", linewidth=2, label="Student test")
    for yi, a, b in zip(y, ll["cv_log_loss_mean"], ll["test_log_loss"]):
        ax.text(a, yi - h / 2, f" {a:.3f}", va="center", fontsize=8, color=INK)
        ax.text(b, yi + h / 2, f" {b:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [name(m) for m in ll["model"]])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Log loss of candidate probabilities (lower is better)", loc="left", color=INK)
    save(fig, "log_loss.png")

    # fold stability
    folds = pd.read_csv(MODELS / "cv_folds.csv")
    order = list(df["model"])
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.2))
    for i, m in enumerate(order):
        v = folds.loc[folds["model"] == m, "top1"].to_numpy()
        ax.scatter(v, np.full(len(v), i), s=36, color=GREY if m.startswith("baseline") else C1,
                   edgecolor="white", linewidth=1.5, zorder=3)
    ax.set_yticks(range(len(order)), [name(m) for m in order])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Top-1 accuracy of each validation fold")
    ax.set_title("Stability across the 5 trip-grouped folds (one dot per fold)", loc="left", color=INK)
    save(fig, "cv_fold_stability.png")

    # computational cost: one measure per chart, no dual axis
    learned = df[~df["model"].str.startswith("baseline")].sort_values("infer_s_student_test")
    for col, title, fname in [
        ("infer_s_student_test", "Inference time on all student-test candidates (s, log scale)", "inference_time.png"),
        ("fit_s", "Training time on the 80 training trips (s, log scale)", "training_time.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(learned) + 1.1))
        y = np.arange(len(learned))
        ax.barh(y, learned[col], color=C1, height=0.6, edgecolor="white", linewidth=2)
        for yi, v in zip(y, learned[col]):
            ax.text(v * 1.08, yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
        ax.set_xscale("log")
        ax.set_yticks(y, [name(m) for m in learned["model"]])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", color=INK)
        save(fig, fname)

    # confusion matrices on student_test (row-normalised)
    runs = {m: json.loads((MODELS / "runs" / f"{m}.json").read_text()) for m in order}
    n = len(order)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.9 * rows))
    for ax, m in zip(axes.ravel(), order):
        s = runs[m]["student_test"]
        cm = np.array([[s["tn"], s["fp"]], [s["fn"], s["tp"]]], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j]):,})", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.grid(False)
        ax.set_title(name(m), fontsize=9, color=INK)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Student-test confusion matrices (threshold 0.5, rows normalised)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    save(fig, "confusion_matrices.png")


def curve_figures():
    curves = sorted((MODELS / "curves").glob("*_loss_curve.csv")) if (MODELS / "curves").exists() else []
    if not curves:
        return
    fig, axes = plt.subplots(1, len(curves), figsize=(5.2 * len(curves), 3.8), squeeze=False)
    for ax, f in zip(axes[0], curves):
        df = pd.read_csv(f)
        x = df.columns[0]
        ax.plot(df[x], df["train_loss"], color=C1, lw=2, label="training loss")
        if "val_loss" in df:
            ax.plot(df[x], df["val_loss"], color=C2, lw=2, label="validation loss")
            best = int(df["val_loss"].idxmin())
            ax.scatter([df[x][best]], [df["val_loss"][best]], s=40, color=C2, edgecolor="white", zorder=3)
        ax.set_xlabel(x)
        ax.set_ylabel("binary log loss")
        ax.set_title(name(f.stem.replace("_loss_curve", "")), loc="left", color=INK)
        ax.legend()
    fig.suptitle("Training curves (internal trip-grouped 90/10 split of the training data)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    save(fig, "loss_curves.png")


def tuned_figure():
    runs = MODELS / "runs"
    tuned = sorted(runs.glob("*_tuned.json")) if runs.exists() else []
    for t in tuned:
        base = t.stem.replace("_tuned", "")
        if not (runs / f"{base}.json").exists():
            continue
        a, b = json.loads((runs / f"{base}.json").read_text()), json.loads(t.read_text())
        rows = [
            ("CV macro F1", a["cv"]["macro_f1"][0], b["cv"]["macro_f1"][0]),
            ("CV Top-1", a["cv"]["top1"][0], b["cv"]["top1"][0]),
            ("Student-test Top-1", a["student_test"]["top1"], b["student_test"]["top1"]),
            ("Student-test macro F1", a["student_test"]["macro_f1"], b["student_test"]["macro_f1"]),
        ]
        table = pd.DataFrame(rows, columns=["metric", "untuned", "tuned"])
        extra = pd.DataFrame([
            ("Student-test log loss", a["student_test"]["log_loss"], b["student_test"]["log_loss"]),
            ("CV log loss", a["cv"]["log_loss"][0], b["cv"]["log_loss"][0]),
            ("Inference time (s, student test)", a["inference_seconds_student_test"], b["inference_seconds_student_test"]),
            ("Training time (s, 80 trips)", a["fit_seconds_full_train"], b["fit_seconds_full_train"]),
        ], columns=["metric", "untuned", "tuned"])
        TAB.mkdir(parents=True, exist_ok=True)
        pd.concat([table, extra]).to_csv(TAB / f"{base}_tuned_vs_untuned.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 3.6))
        y = np.arange(len(table))
        h = 0.36
        ax.barh(y - h / 2, table["untuned"], height=h, color=C1, edgecolor="white", linewidth=2, label="untuned")
        ax.barh(y + h / 2, table["tuned"], height=h, color=C2, edgecolor="white", linewidth=2, label="tuned (Optuna)")
        for yi, u, v in zip(y, table["untuned"], table["tuned"]):
            ax.text(u + 0.002, yi - h / 2, f"{u:.4f}", va="center", fontsize=8, color=INK)
            ax.text(v + 0.002, yi + h / 2, f"{v:.4f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y, table["metric"])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(max(0, table[["untuned", "tuned"]].min().min() - 0.05), 1.0)
        ax.legend(loc="lower right")
        ax.set_title(f"{name(base)}: untuned vs Bayesian-tuned (same folds and partitions)", loc="left", color=INK)
        save(fig, f"{base}_tuned_vs_untuned.png")


def hmm_figures():
    """Classifier-only vs classifier + HMM on student_test (results/hmm/sequence_eval.csv)."""
    path = config.RESULTS_DIR / "hmm" / "sequence_eval.csv"
    if not path.exists():
        return
    s = pd.read_csv(path)
    wide = s.pivot(index="model", columns="mode")
    order = wide[("accuracy", "classifier_hmm")].sort_values(ascending=False).index
    labels = [name("heuristic" if m.startswith("heuristic") else m) for m in order]
    for metric, title, fname, fmt in [
        ("accuracy", "Pointwise accuracy on student test: classifier only vs classifier + HMM", "hmm_accuracy.png", "{:.3f}"),
        ("continuity", "Route continuity on student test (share of connected consecutive arcs)", "hmm_continuity.png", "{:.4f}"),
        ("route_f1", "Route F1 (length-weighted overlap with the reference route), student test", "hmm_route_f1.png", "{:.3f}"),
    ]:
        a = wide.loc[order, (metric, "classifier_only")].to_numpy()
        b = wide.loc[order, (metric, "classifier_hmm")].to_numpy()
        fig, ax = plt.subplots(figsize=(9, 0.55 * len(order) + 1.3))
        y = np.arange(len(order))
        h = 0.36
        ax.barh(y - h / 2, a, height=h, color=C1, edgecolor="white", linewidth=2, label="classifier only (argmax)")
        ax.barh(y + h / 2, b, height=h, color=C2, edgecolor="white", linewidth=2, label="classifier + HMM/Viterbi")
        for yi, u, v in zip(y, a, b):
            ax.text(u, yi - h / 2, " " + fmt.format(u), va="center", fontsize=8, color=INK)
            ax.text(v, yi + h / 2, " " + fmt.format(v), va="center", fontsize=8, color=INK)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(max(0.0, min(a.min(), b.min()) - 0.05), 1.01)
        ax.legend(loc="lower right")
        ax.set_title(title, loc="left", color=INK)
        save(fig, fname)

    per_trip = config.RESULTS_DIR / "hmm" / "sequence_eval_per_trip.csv"
    if per_trip.exists():
        p = pd.read_csv(per_trip)
        p = p[p["model"] == "lightgbm"]
        if len(p):
            w = p.pivot(index="trip_id", columns="mode", values="accuracy").sort_values("classifier_hmm")
            fig, ax = plt.subplots(figsize=(9, 4))
            x = np.arange(len(w))
            ax.plot(x, w["classifier_only"], "o", color=C1, ms=7, label="classifier only")
            ax.plot(x, w["classifier_hmm"], "o", color=C2, ms=7, label="classifier + HMM")
            for xi, u, v in zip(x, w["classifier_only"], w["classifier_hmm"]):
                ax.plot([xi, xi], [u, v], color="#c9c8c2", lw=1.5, zorder=0)
            ax.set_xticks(x, [f"{int(t):08d}" if str(t).isdigit() else str(t) for t in w.index], rotation=45)
            ax.set_ylabel("pointwise accuracy")
            ax.legend(loc="lower right")
            ax.set_title("LightGBM per student-test trip: where HMM decoding helps", loc="left", color=INK)
            save(fig, "hmm_per_trip_lightgbm.png")


def markdown_table(df: pd.DataFrame) -> str:
    cells = [[str(c) for c in df.columns]] + [[str(v).replace("|", "\\|") for v in row] for row in df.to_numpy()]
    lines = ["| " + " | ".join(cells[0]) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(r) + " |" for r in cells[1:]]
    return "\n".join(lines) + "\n"


def comparison_table():
    """Consolidated comparison table in the report's column order (Deliverables 3.1)."""
    path = MODELS / "model_comparison.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    cont = {}
    seq = config.RESULTS_DIR / "hmm" / "sequence_eval.csv"
    if seq.exists():
        s = pd.read_csv(seq)
        mode_col = next((c for c in s.columns if c.lower() in ("mode", "decoder")), None)
        cont_col = next((c for c in s.columns if "continuity" in c.lower()), None)
        if mode_col and cont_col and "model" in s.columns:
            clf = s[s[mode_col].astype(str).str.contains("classifier|clf|argmax", case=False)
                    & ~s[mode_col].astype(str).str.contains("hmm", case=False)]
            cont = dict(zip(clf["model"], clf[cont_col]))
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Model": name(r["model"]), "Algorithm family": r["family"],
            "Key configuration": r["key_config"] if r["key_config"] != "rule-based" else "rule-based (no training)",
            "CV macro F1": f"{r['cv_macro_f1_mean']:.4f} ± {r['cv_macro_f1_std']:.4f}",
            "Top-1 accuracy (student test)": f"{r['test_top1']:.4f}",
            "Route continuity (classifier only, student test)": (f"{cont[r['model']]:.4f}" if r["model"] in cont
                                                                  else "see results/hmm"),
            "Student-test macro F1": f"{r['test_macro_f1']:.4f}",
            "Log loss (student test)": ("N/A (hard rule, no probabilities)" if pd.isna(r["test_log_loss"])
                                        else f"{r['test_log_loss']:.4f}"),
            "Inference time (s, 201,662 candidate rows)": f"{r['infer_s_student_test']:.3f}",
        })
    out = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    out.to_csv(TAB / "model_comparison_report.csv", index=False)
    (TAB / "model_comparison_report.md").write_text(markdown_table(out), encoding="utf-8")
    print("wrote", TAB / "model_comparison_report.csv")


def main():
    style()
    model_figures()
    curve_figures()
    tuned_figure()
    hmm_figures()
    comparison_table()


if __name__ == "__main__":
    main()
