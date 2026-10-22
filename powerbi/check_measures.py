"""Recompute the report's headline numbers from powerbi/data/ with pandas.

Run:  python powerbi/check_measures.py            (needs pandas; exit code 1 on any failure)

These are the values the Power BI cards should show. Each function mirrors one DAX
measure in measures.dax and names it. Every number is printed with its result_basis,
because the same model has different values under different bases (for example Random
Forest noisy accuracy is 97.74, 97.97 and 97.63 in three experiments) and the report
never puts two bases on one page.

The values come from the result tables in this repo's README.md (the raw result CSVs
are gitignored), so this script checks the pack against the data files, and
test_pack.py checks the data files against README.md. It does not re-run the experiments.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data"
REPO_ROOT = HERE.parent

# Every result row must cite one of these sources and one of these bases.
# Add a value only when a number genuinely comes from it (keep in step with build_data.py).
ALLOWED_SOURCES = ("README.md",)
ALLOWED_BASES = (
    "3-seed mean, primary split",
    "stage 09 representation test",
    "stage 09 window sweep",
    "stage 08 shuffle test",
    "stage 10 split variance",
)
RESULT_TABLES = ("fact_wifi_model_results", "fact_wifi_decomposition", "fact_wifi_window_sweep",
                 "fact_wifi_shuffle", "fact_wifi_variance")


def load(data_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [t for t in RESULT_TABLES if not (data_dir / f"{t}.csv").exists()]
    if missing:
        sys.exit(f"Missing CSV(s) in {data_dir}: {', '.join(m + '.csv' for m in missing)}")
    return {t: pd.read_csv(data_dir / f"{t}.csv") for t in RESULT_TABLES}


def dax_measures(path: Path = HERE / "measures.dax") -> dict[str, str]:
    """Measure name -> formula (whitespace collapsed) parsed from measures.dax.
    A measure starts at column 0 as `Name = ...`; indented lines continue it; `--` lines are comments;
    the DimDate calendar table is not a measure."""
    out: dict[str, str] = {}
    cur = None
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip() or line.lstrip().startswith("--"):
            continue
        if line[0] not in " \t":
            name, _, expr = line.partition(" = ")
            name = name.strip().rstrip("=").strip()
            if "=" in line and name != "DimDate":
                cur = name
                out[cur] = expr.strip()
            else:
                cur = None
        elif cur:
            out[cur] += " " + line.strip()
    return {k: re.sub(r"\s+", " ", v).strip() for k, v in out.items()}


def basis_of(df: pd.DataFrame) -> str:
    vals = sorted(df["result_basis"].unique())
    return vals[0] if len(vals) == 1 else "MIXED: " + " + ".join(vals)


# --- one function per DAX measure ------------------------------------------------
def best_noisy_accuracy(t) -> float:
    # Best Noisy Accuracy = MAX ( fact_wifi_model_results[noisy_accuracy] )
    return t["fact_wifi_model_results"]["noisy_accuracy"].max()


def context_gain_matched(t) -> float:
    # Context Gain (matched models, pts): AVERAGE of 10-packet rows for the models that also have
    # a Single-packet row, minus AVERAGE of the Single-packet rows.
    m = t["fact_wifi_model_results"]
    single = m[m["context"] == "Single-packet"]
    ten = m[(m["context"] == "10-packet") & m["model"].isin(single["model"])]
    return ten["noisy_accuracy"].mean() - single["noisy_accuracy"].mean()


def _decomp(t, name: str) -> float:
    d = t["fact_wifi_decomposition"]
    return d.loc[d["representation"] == name, "noisy_accuracy"].max()


def aggregation_gain(t) -> float:
    # Aggregation Gain (pts) = Mean-pooled window - Single packet
    return _decomp(t, "Mean-pooled window") - _decomp(t, "Single packet")


def beyond_average_gain(t) -> float:
    # Beyond-Average Gain (pts) = Flattened window - Mean-pooled window
    return _decomp(t, "Flattened window") - _decomp(t, "Mean-pooled window")


def shuffle_delta(t, model: str) -> float:
    # Shuffle Delta RF / Transformer (pts) = SUM ( delta ) for that model
    s = t["fact_wifi_shuffle"]
    return s.loc[s["model"] == model, "delta"].sum()


def split_gap(t) -> float:
    # Split Gap (pts) = Random Forest - Transformer, noisy accuracy in the variance table
    v = t["fact_wifi_variance"]
    pick = lambda m: v.loc[v["model"] == m, "noisy_accuracy"].max()
    return pick("Random Forest") - pick("Transformer")


def headline(t) -> list[tuple[str, float, str]]:
    """The five numbers the Power BI cards should match: (measure, value, basis)."""
    return [
        ("Best Noisy Accuracy", best_noisy_accuracy(t), basis_of(t["fact_wifi_model_results"])),
        ("Context Gain (matched models, pts)", context_gain_matched(t), basis_of(t["fact_wifi_model_results"])),
        ("Aggregation Gain (pts)", aggregation_gain(t), basis_of(t["fact_wifi_decomposition"])),
        ("Beyond-Average Gain (pts)", beyond_average_gain(t), basis_of(t["fact_wifi_decomposition"])),
        ("Shuffle Delta Transformer (pts)", shuffle_delta(t, "Transformer"), basis_of(t["fact_wifi_shuffle"])),
    ]


# --- checks ----------------------------------------------------------------------
def label_check(t) -> bool:
    """Every row's source and result_basis must be in the allowed lists, and the source must exist."""
    ok = True
    print("\nSource and basis labels (allowed lists in this file)")
    for name, df in t.items():
        for col, allowed in (("source", ALLOWED_SOURCES), ("result_basis", ALLOWED_BASES)):
            bad = sorted(set(df[col].astype(str)) - set(allowed))
            if bad:
                print(f"  FAIL: {name}.{col} has value(s) not in the allowed list: {bad}")
                ok = False
        for src in set(df["source"]):
            if not (REPO_ROOT / src).exists():
                print(f"  FAIL: {name} cites {src!r}, which does not exist in the repo")
                ok = False
    if ok:
        print(f"  ok: {sum(len(d) for d in t.values())} rows in {len(t)} tables")
    return ok


def one_basis_check(t) -> bool:
    """Each result table must hold exactly one basis, so a page built on one table never mixes bases."""
    ok = True
    print("\nOne basis per table")
    for name, df in t.items():
        n = df["result_basis"].nunique()
        print(f"  {'ok  ' if n == 1 else 'FAIL'} {name}: {basis_of(df)}")
        ok = ok and n == 1
    return ok


def arithmetic_check(t) -> bool:
    """Stored deltas and gaps must follow from the stored accuracies."""
    ok = True
    print("\nArithmetic inside the data")
    for r in t["fact_wifi_shuffle"].itertuples():
        good = abs((r.shuffled - r.ordered) - r.delta) <= 0.011
        ok = ok and good
        print(f"  {'ok  ' if good else 'FAIL'} shuffle {r.model}: shuffled - ordered = {r.shuffled - r.ordered:+.2f}, stored delta {r.delta:+.2f}")
    return ok


def cross_basis_note(t) -> None:
    """Never a failure: show where one model/setting has different values under different bases."""
    m, d, s, sh, v = (t[k] for k in RESULT_TABLES)
    rf10 = m[(m["model"] == "Random Forest") & (m["context"] == "10-packet")]["noisy_accuracy"].iloc[0]
    rf1 = m[(m["model"] == "Random Forest") & (m["context"] == "Single-packet")]["noisy_accuracy"].iloc[0]
    print("\nSame setting, different bases (these are different experiments; do not chart them together)")
    print(f"  RF 10-packet:   {rf10:.2f} ({basis_of(m)}) | {_decomp(t, 'Flattened window'):.2f} ({basis_of(d)})"
          f" | {v.loc[v['model'] == 'Random Forest', 'noisy_accuracy'].iloc[0]:.2f} ({basis_of(v)})")
    print(f"  RF single:      {rf1:.2f} ({basis_of(m)}) | {_decomp(t, 'Single packet'):.2f} ({basis_of(d)})"
          f" | {s.loc[s['window_size'] == 1, 'noisy_accuracy'].iloc[0]:.2f} ({basis_of(s)}, W=1)")
    tr = m[m["model"].str.startswith("Transformer")]["noisy_accuracy"].iloc[0]
    print(f"  Transformer:    {tr:.2f} ({basis_of(m)}) | {sh.loc[sh['model'] == 'Transformer', 'ordered'].iloc[0]:.2f} ({basis_of(sh)}, ordered)"
          f" | {v.loc[v['model'] == 'Transformer', 'noisy_accuracy'].iloc[0]:.2f} ({basis_of(v)})")


def freshness(data_dir: Path) -> None:
    p = data_dir / "export_info.json"
    if not p.exists():
        print("\nFreshness: export_info.json not found; run powerbi/build_data.py")
        return
    info = json.loads(p.read_text(encoding="utf-8"))
    print(f"\nLast exported {info['exported_on']} from {info['git_ref']} at {info['git_ref_sha'][:7]}"
          f" (latest commit {info['latest_commit']}, {info['latest_commit_date'][:10]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="folder with the CSVs")
    args = ap.parse_args()
    t = load(args.data)

    print("Headline numbers - the Power BI cards should show these (the five cards matched in Power BI Desktop on 2026-10-22)")
    for name, val, basis in headline(t):
        print(f"  {name:<36} {val:>8.2f}   basis: {basis}")

    ok = label_check(t)
    ok = one_basis_check(t) and ok
    ok = arithmetic_check(t) and ok
    cross_basis_note(t)
    freshness(args.data)
    print("\nALL CHECKS PASSED" if ok else "\nCHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
