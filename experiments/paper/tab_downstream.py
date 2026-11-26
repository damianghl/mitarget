"""
Appendix table tab:downstream: Amazon employee access, encoders inside a downstream logistic regression
and gradient-boosting model, log-loss and its difference to the baseline (milli-nats; * p < 0.05,
Nadeau-Bengio corrected t-test).

Reads the three ``downstream_amazon_summary.csv`` files of ``experiments/downstream_amazon.py`` (main run;
``count_glmm_mean/`` for the GLMM at its posterior mean; ``count_loo/`` for the "+ count" encoders):

    python tab_downstream.py results/downstream_amazon --tab tab_downstream.tex

The boosting column on the raw columns shows only the baseline and the "+ count" rows: without the count
no encoder differs from the baseline there.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

COLS = [("raw", "logreg"), ("raw", "gbm"), ("pairs", "logreg"), ("pairs", "gbm")]
ROWS = [("baseline", "baseline", "main"), ("npmle", "NPMLE", "main"), ("freq_prior", r"freq.\ prior", "main"),
        ("glmm", "GLMM mode", "main"), ("glmm_mean", "GLMM mean", "mean"),
        ("fixed_m20", "fixed $m=20$", "main"), ("catboost_ts", "$m{=}1$ (CatBoost)", "main")]
COUNT_ROWS = [("baseline_count", "baseline + count"), ("npmle_count", "NPMLE + count")]
HIDDEN = {("raw", "gbm")}   # encoders without the count are shown only for the baseline in this column


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--tab", default="tab_downstream.tex")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    src = {"main": a.folder / "downstream_amazon_summary.csv",
           "mean": a.folder / "count_glmm_mean" / "downstream_amazon_summary.csv",
           "loo": a.folder / "count_loo" / "downstream_amazon_summary.csv"}
    d = {k: pd.read_csv(v).set_index(["feature_set", "model", "encoder"]) for k, v in src.items()}

    def get(which, fs, model, enc):
        key = (fs, model, enc)
        return d[which].loc[key] if key in d[which].index else None

    rows = [(lab, which, enc) for enc, lab, which in ROWS] + [(lab, "loo", enc) for enc, lab in COUNT_ROWS]
    best = {}
    for col in COLS:
        vals = [get(w, *col, e).log_loss for _, w, e in rows if get(w, *col, e) is not None
                and (col not in HIDDEN or e == "baseline" or e.endswith("_count"))]
        best[col] = min(vals)

    def cells(which, enc):
        out = []
        for col in COLS:
            r = get(which, *col, enc)
            if r is None or (col in HIDDEN and enc != "baseline" and not enc.endswith("_count")):
                out += ["", ""]
                continue
            ll = f"{r.log_loss:.4f}"
            if enc.endswith("_count") and r.log_loss == best[col]:
                ll = rf"\textbf{{{ll}}}"
            if enc == "baseline":
                out += [ll, ""]
                continue
            star = "^{*}" if r.p_log_loss < a.alpha else ""
            out += [ll, f"${1000 * r.d_log_loss:+.1f}{star}$"]
        return out

    def line(label, cs):
        return re.sub(" {2,}", " ", label + " & " + " & ".join(cs) + r" \\")

    L = [r"% experiments/downstream_amazon.py; ``+ count'' rows from the rerun with the leave-one-out count "
         r"for training rows.",
         r"\begin{tabular}{l rr rr rr rr}", r"\toprule",
         r" & \multicolumn{4}{c}{raw (8 columns)} & \multicolumn{4}{c}{raw + pairs (36 columns)} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r" & \multicolumn{2}{c}{logistic} & \multicolumn{2}{c}{boosting} & \multicolumn{2}{c}{logistic} & "
         r"\multicolumn{2}{c}{boosting} \\",
         r"encoder & log-loss & $\Delta$ & log-loss & $\Delta$ & "
         r"log-loss & $\Delta$ & log-loss & $\Delta$ \\",
         r"\midrule"]
    L += [line(lab, cells(w, e)) for lab, w, e in rows[:len(ROWS)]]
    L.append(r"\midrule")
    L += [line(lab, cells(w, e)) for lab, w, e in rows[len(ROWS):]]
    L.append(r"\midrule")
    onehot = d["main"].loc[("raw", "logreg", "onehot")].log_loss
    native = d["main"].loc[("raw", "catboost", "catboost_native")].log_loss
    L.append(line("one-hot logistic", [f"{onehot:.4f}"] + [""] * 7))
    L.append(rf"native CatBoost & \multicolumn{{8}}{{l}}{{{native:.4f} "
             r"(raw columns; builds its own combinations)} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    Path(a.tab).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
