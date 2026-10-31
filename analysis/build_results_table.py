"""Generate a publication-ready LaTeX table from all completed experiment results."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib     import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common import (
    load_all_runs,
    SCHEME_ORDER, SCHEME_LABELS,
    RULE_ORDER, RULE_LABELS,
)


def build_results_table(
    results_dir: str = "results/raw",
    metric:      str = "test_acc",
    output:      str = "results/aggregated/results_table.tex",
) -> None:
    """Build the table of one metric over all finished runs and write it as LaTeX and CSV."""
    import csv, statistics

    runs = load_all_runs(results_dir)
    if not runs:
        print("No runs found.")
        return

    cell_data: dict[tuple, list[float]] = defaultdict(list)
    datasets_seen:  set[str] = set()
    learners_seen:  set[str] = set()
    schemes_seen:   set[str] = set()

    for r in runs:
        val = r.get(metric)
        if val is None:
            continue
        ds  = r.get("dataset",  "?")
        lr  = r.get("learning", "?")
        cod = r.get("coding",   "?")
        cell_data[(ds, lr, cod)].append(float(val))
        datasets_seen.add(ds)
        learners_seen.add(lr)
        schemes_seen.add(cod)

    if not cell_data:
        print(f"No runs have the metric '{metric}'.")
        return

    ds_order  = sorted(datasets_seen)
    lr_order  = [r for r in RULE_ORDER if r in learners_seen] + \
                sorted(l for l in learners_seen if l not in RULE_ORDER)
    cod_order = [c for c in SCHEME_ORDER if c in schemes_seen] + \
                sorted(c for c in schemes_seen if c not in SCHEME_ORDER)

    scale = 100.0 if "acc" in metric else 1.0

    tex_lines = _build_latex(cell_data, ds_order, lr_order, cod_order,
                              metric, scale)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"LaTeX table written: {out_path}")

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        cols = ["dataset", "learning_rule"] + cod_order
        w    = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for ds in ds_order:
            for lr in lr_order:
                row_dict = {"dataset": ds, "learning_rule": lr}
                for cod in cod_order:
                    vals = cell_data.get((ds, lr, cod), [])
                    if vals:
                        m = statistics.mean(vals) * scale
                        s = statistics.stdev(vals) * scale if len(vals) > 1 else 0.0
                        row_dict[cod] = f"{m:.2f}±{s:.2f}" if s > 0 else f"{m:.2f}"
                    else:
                        row_dict[cod] = "—"
                w.writerow(row_dict)
    print(f"CSV summary written:  {csv_path}")

    print()
    _print_text_table(cell_data, ds_order, lr_order, cod_order, metric, scale)


def _build_latex(
    cell_data, ds_order, lr_order, cod_order, metric, scale
) -> list[str]:
    import statistics

    n_cols     = len(cod_order)
    col_spec   = "ll|" + "c" * n_cols
    col_header = " & ".join(
        r"\textbf{" + SCHEME_LABELS.get(c, c) + "}" for c in cod_order
    )
    metric_disp = metric.replace("_", " ").title()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Test Accuracy (\%) --- {metric_disp}}}",
        rf"\label{{tab:results_{metric}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"\textbf{{Dataset}} & \textbf{{Rule}} & {col_header} \\",
        r"\midrule",
    ]

    prev_ds = None
    for ds in ds_order:
        for lr in lr_order:
            row_vals = {}
            for cod in cod_order:
                v = cell_data.get((ds, lr, cod), [])
                if v:
                    row_vals[cod] = statistics.mean(v) * scale
                else:
                    row_vals[cod] = None

            valid = {c: v for c, v in row_vals.items() if v is not None}
            if valid:
                best_val = max(valid.values()) if "acc" in metric else min(valid.values())
            else:
                best_val = None

            ds_cell = rf"\multirow{{*}}{{*}}{{{ds.upper()}}}" if ds != prev_ds else ""
            lr_cell = RULE_LABELS.get(lr, lr)

            cells = []
            for cod in cod_order:
                v = row_vals.get(cod)
                if v is None:
                    cells.append("—")
                else:
                    std_vals = cell_data.get((ds, lr, cod), [])
                    s = statistics.stdev(std_vals) * scale if len(std_vals) > 1 else 0.0
                    text = f"{v:.1f}" + (rf"$\pm${s:.1f}" if s > 0 else "")
                    if v == best_val:
                        text = rf"\textbf{{{text}}}"
                    cells.append(text)

            lines.append(f"{ds_cell} & {lr_cell} & " + " & ".join(cells) + r" \\")
            prev_ds = ds

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    return lines


def _print_text_table(cell_data, ds_order, lr_order, cod_order, metric, scale):
    import statistics

    col_w = max(len(RULE_LABELS.get(lr, lr)) for lr in lr_order)
    ds_w  = max(len(ds) for ds in ds_order)
    cell_w = 10

    header = (f"{'Dataset':>{ds_w}}  {'Rule':<{col_w}}"
              + "".join(f"  {SCHEME_LABELS.get(c, c):>{cell_w}}" for c in cod_order))
    print(header)
    print("-" * len(header))

    for ds in ds_order:
        for lr in lr_order:
            cells = []
            for cod in cod_order:
                vals = cell_data.get((ds, lr, cod), [])
                if vals:
                    m = statistics.mean(vals) * scale
                    cells.append(f"{m:{cell_w}.2f}")
                else:
                    cells.append(f"{'—':>{cell_w}}")
            lr_label = RULE_LABELS.get(lr, lr)
            print(f"{ds:>{ds_w}}  {lr_label:<{col_w}}" + "".join(f"  {c}" for c in cells))
        print()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate LaTeX results table")
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--metric",      default="test_acc",
                   choices=["test_acc", "test_loss", "best_val_acc"])
    p.add_argument("--output",      default="results/aggregated/results_table.tex")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    build_results_table(
        results_dir = args.results_dir,
        metric      = args.metric,
        output      = args.output,
    )


if __name__ == "__main__":
    main()
