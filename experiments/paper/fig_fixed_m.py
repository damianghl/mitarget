"""
Fixed-smoothing sweep, from ``experiments/fixed_m_sweep.py``.

fig_fixed_m.pdf   (a) oracle pools at nbar = 2: eCE of the m-estimate with m = c * s-hat minus
                  the baseline's, as a share of the oracle's gain, against log2(m / s-hat);
                  (b) bakeoff: held-out log-loss of the same encoders minus the baseline's, as a
                  share of the baseline's gain (median and interquartile range over the fixed set
                  of 20 tasks), with the common default m = 20 per task
tab_fixed_m.tex   per bakeoff task: median s-hat and CV m over the 25 training folds, and the
                  held-out loss of m = 20 and m = 1 relative to the baseline, in milli-nats and
                  as a share of the baseline's gain

    python fig_fixed_m.py --pools sweep/pools --tasks sweep/tasks --meta table_tasks.csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from _style import COLORS, plt
from fig_oracle import POOLS
from tab_bakeoff import tex_name

REL = [1 / 16, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 16.0]
LOW_GAIN = {"churn/state", "churn/pseudo_id", "airlines/Flight", "click_small/user_id"}


def rel_name(c):
    return "baseline" if c == 1.0 else f"rel_{c:g}"


def pool_curves(path, nbar):
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(path, "*.csv"))])
    d = d[d.nbar == nbar]
    k = ["pool", "shuffled", "rep"]
    w = d.pivot_table(index=k, columns="method", values="eCE")
    gain = w["marginal"] - w["oracle"]
    out = {}
    for c in REL:
        out[c] = ((w[rel_name(c)] - w["baseline"]) / gain).groupby(level=[0, 1]).mean()
    return pd.DataFrame(out)


def load_tasks(path):
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(path, "*.csv"))])
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from realdata_tables import SHORT
        d["task"] = d.task.replace(SHORT)
    except ImportError:
        pass
    w = d.pivot_table(index=["task", "rep", "fold"], columns="method", values="logloss")
    sm = d.groupby(["task", "rep", "fold"])[["s_hat", "m_cv"]].first()
    return w, sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--meta", required=True, help="table_tasks.csv with H_Y per task")
    ap.add_argument("--logloss", required=True, help="table_logloss.csv (prior-only log-loss per task)")
    ap.add_argument("--nbar", type=float, default=2.0)
    ap.add_argument("--fig", default="fig_fixed_m.pdf")
    ap.add_argument("--tab", default="tab_fixed_m.tex")
    a = ap.parse_args()

    pc = pool_curves(a.pools, a.nbar)
    w, sm = load_tasks(a.tasks)
    ll = pd.read_csv(a.logloss)
    prior = ll[ll.method == "Prior only"].set_index("task").logloss
    base = w["baseline"].groupby(level=0).mean()
    gain = prior.reindex(base.index) - base
    diff = w.sub(w["baseline"], axis=0).groupby(level=0).mean()
    share = diff.div(gain, axis=0)
    fixed = [t for t in share.index if t not in LOW_GAIN]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    x = np.log2(REL)
    cmap = plt.get_cmap("tab10")
    for i, (feat, lab) in enumerate(POOLS):
        for shuf, ls in ((False, "-"), (True, ":")):
            if (feat, shuf) in pc.index:
                axes[0].plot(x, 100 * pc.loc[(feat, shuf)].values, ls=ls, color=cmap(i), marker="o", ms=2.5,
                             label=lab if not shuf else None)
    axes[0].set_xlabel(r"$\log_2(m/\hat s)$")
    axes[0].set_ylabel("extra loss, % of oracle's gain")
    axes[0].set_title(rf"(a) oracle pools, $\bar n={a.nbar:g}$ (dotted: shuffled)")
    axes[0].set_ylim(-10, 120)
    axes[0].legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)

    rel = share.loc[fixed, [rel_name(c) for c in REL]] * 100
    med, q1, q3 = rel.median(), rel.quantile(0.25), rel.quantile(0.75)
    axes[1].fill_between(x, q1.values, q3.values, color=COLORS["baseline"], alpha=0.2, lw=0)
    axes[1].plot(x, med.values, color=COLORS["baseline"], marker="o", ms=2.5,
                 label=r"$m=c\,\hat s$ (median, IQR)")
    s_med = sm.groupby(level=0).s_hat.median()
    for col, lab, mk in (("fixed_m20", r"$m=20$", "^"), ("fixed_m1", r"$m=1$", "v")):
        axes[1].scatter(np.log2(float(col[7:]) / s_med.loc[fixed]), 100 * share.loc[fixed, col],
                        color=COLORS["fixed_m"] if col == "fixed_m20" else COLORS["ref"], marker=mk, s=9,
                        label=lab, zorder=3)
    axes[1].set_xlabel(r"$\log_2(m/\hat s)$")
    axes[1].set_ylabel("extra loss, % of baseline's gain")
    axes[1].set_title("(b) bakeoff, 20 tasks")
    axes[1].set_ylim(-10, 120)
    axes[1].legend(frameon=False, fontsize=5.5, loc="upper center")
    fig.tight_layout()
    fig.savefig(a.fig)

    cv_med = sm.groupby(level=0).m_cv.median()
    lines = [r"\begin{tabular}{l r r r r r r}", r"\toprule",
             r"feature & $\hat s$ & CV $m$ & $m=20$ (mnats) & $m=20$ (\%) & $m=1$ (mnats) & $m=1$ (\%)\\",
             r"\midrule"]
    for t in sorted(share.index):
        cells = [tex_name(t), f"{s_med[t]:.3g}", f"{cv_med[t]:.3g}"]
        for col in ("fixed_m20", "fixed_m1"):
            cells += [f"{1e3 * diff.loc[t, col]:+.2f}",
                      "--" if t in LOW_GAIN else f"{100 * share.loc[t, col]:+.0f}"]
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(a.tab, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(pd.DataFrame({"s_hat": s_med, "m_cv": cv_med, "m20": share["fixed_m20"], "m1": share["fixed_m1"],
                        "rel2": share["rel_2"], "rel0.5": share["rel_0.5"], "rel4": share["rel_4"],
                        "cv": share["cv_m"]}).round(3).to_string())


if __name__ == "__main__":
    main()
