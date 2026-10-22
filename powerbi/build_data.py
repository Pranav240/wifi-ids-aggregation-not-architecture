"""Regenerate the CSVs in powerbi/data/ from this repo. Do not hand-edit the CSVs.

Two kinds of table, two sources:

  README.md        fact_wifi_model_results, fact_wifi_decomposition,
                   fact_wifi_window_sweep, fact_wifi_shuffle, fact_wifi_variance.
                   The result tables and the window-sweep sentence in README.md are
                   parsed at --ref. The raw result CSVs (data/results/) are gitignored
                   and are NOT in this repo, so README.md is the only source.
                   If the README wording or table layout changes, this script
                   fails instead of guessing.

  git history      dim_repo, fact_commits, fact_repo_language.
                   Commits that only touch powerbi/ are left out, and so are the
                   powerbi/ files in the line counts, so the report's own files
                   do not inflate the project's size.

Every result row carries a result_basis: the experiment the README says the number
comes from. The same model appears under different bases with different values
(for example Random Forest noisy accuracy is 97.74, 97.97 and 97.63 in three
experiments). The report never puts two bases on one page.

Usage:
    python powerbi/build_data.py                         # refresh powerbi/data/ from origin/main
    python powerbi/build_data.py --ref a5fd24a --out /tmp/check
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPO_SLUG = "wifi-ids-aggregation-not-architecture"
EXCLUDE = "powerbi"

# result_basis values. Each names the README passage the numbers come from.
B_PRIMARY = "3-seed mean, primary split"      # "Results summary" table
B_DECOMP = "stage 09 representation test"      # "Where the robustness actually comes from" table
B_SWEEP = "stage 09 window sweep"              # window-size sweep bullet
B_SHUFFLE = "stage 08 shuffle test"            # "Ordering contributes nothing" table
B_SPLITS = "stage 10 split variance"           # "Uncertainty" table
ALLOWED_BASES = (B_PRIMARY, B_DECOMP, B_SWEEP, B_SHUFFLE, B_SPLITS)
ALLOWED_SOURCES = ("README.md",)

EXT = {"py": "Python", "md": "Markdown", "txt": "Text", "pptx": "PowerPoint", "json": "JSON"}


def git(*args: str, binary: bool = False):
    out = subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True)
    return out.stdout if binary else out.stdout.decode("utf-8")


# --- README parsing ------------------------------------------------------------
def num(cell: str) -> float:
    """'**97.74%**' -> 97.74, '−0.47' -> -0.47, '+1.76' -> 1.76. Raises on anything else."""
    s = cell.replace("**", "").replace("%", "").replace("−", "-").replace("+", "").strip()
    return float(s)


def table_after(text: str, anchor: str) -> list[list[str]]:
    """Rows (header excluded) of the first markdown table after the line containing anchor."""
    lines = text.split("\n")
    hits = [i for i, l in enumerate(lines) if anchor in l]
    if len(hits) != 1:
        raise SystemExit(f"README.md: expected exactly one line containing {anchor!r}, found {len(hits)}")
    i = hits[0] + 1
    while i < len(lines) and not lines[i].startswith("|"):
        i += 1
    rows = []
    while i < len(lines) and lines[i].startswith("|"):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if not set("".join(cells)) <= set("-: "):
            rows.append(cells)
        i += 1
    return rows[1:]


def r2(x: float) -> float:
    return round(x, 3)


def readme_tables(readme: str) -> dict[str, tuple[list[dict], list[str]]]:
    seeds = re.search(r"averaged over (\d+) model seeds", readme)
    if not seeds:
        raise SystemExit("README.md: 'averaged over N model seeds' sentence not found")
    n_seeds = int(seeds.group(1))

    model_rows = []
    for cells in table_after(readme, "averaged over"):
        model = cells[0]
        for context, cell in (("Single-packet", cells[1]), ("10-packet", cells[2])):
            if cell.strip("* ") in ("—", "-", ""):
                continue  # no value reported for this context (Transformer single-packet)
            model_rows.append(dict(model=model, context=context, noisy_accuracy=r2(num(cell)),
                                   seeds=n_seeds, result_basis=B_PRIMARY, source="README.md"))

    decomp_rows = []
    for order, cells in enumerate(table_after(readme, "Decomposed via mean-pooling"), start=1):
        decomp_rows.append(dict(representation=cells[0], input_dim=int(cells[1]),
                                noisy_accuracy=r2(num(cells[2])), step_order=order,
                                result_basis=B_DECOMP, source="README.md"))

    sweep = re.search(r"W\s*=\s*([\d,\s]+)→\s*([\d.,%\s]+)\)", readme)
    if not sweep:
        raise SystemExit("README.md: window-size sweep sentence not found")
    ws = [int(x) for x in re.findall(r"\d+", sweep.group(1))]
    accs = [float(x) for x in re.findall(r"\d+\.\d+", sweep.group(2))]
    if len(ws) != len(accs):
        raise SystemExit(f"README.md: sweep has {len(ws)} window sizes but {len(accs)} accuracies")
    sweep_rows = [dict(window_size=w, noisy_accuracy=a, result_basis=B_SWEEP, source="README.md")
                  for w, a in zip(ws, accs)]

    shuffle_rows = []
    for cells in table_after(readme, "Shuffling packet order"):
        shuffle_rows.append(dict(model=cells[0], ordered=num(cells[1]), shuffled=num(cells[2]),
                                 delta=num(cells[3]), result_basis=B_SHUFFLE, source="README.md"))

    var_rows = []
    for cells in table_after(readme, "Split variance exceeds model-seed variance"):
        var_rows.append(dict(model=cells[0], noisy_accuracy=num(cells[1]), model_seed_std=num(cells[2]),
                             split_std=num(cells[3]), result_basis=B_SPLITS, source="README.md"))

    def pack(rows):
        return rows, list(rows[0])

    return {
        "fact_wifi_model_results": pack(model_rows),
        "fact_wifi_decomposition": pack(decomp_rows),
        "fact_wifi_window_sweep": pack(sweep_rows),
        "fact_wifi_shuffle": pack(shuffle_rows),
        "fact_wifi_variance": pack(var_rows),
    }


# --- git history ---------------------------------------------------------------
def line_count(blob: bytes) -> int:
    return blob.count(b"\n") + (1 if blob and not blob.endswith(b"\n") else 0)


def git_tables(ref: str) -> dict[str, tuple[list[dict], list[str]]]:
    spec = ["--no-merges", ref, "--", ".", f":(exclude){EXCLUDE}"]
    stamps = git("log", "--format=%aI", *spec).split()
    parsed = sorted(dt.datetime.fromisoformat(s) for s in stamps)
    by_dt = {dt.datetime.fromisoformat(s): s for s in stamps}
    first, last = by_dt[parsed[0]][:10], by_dt[parsed[-1]][:10]

    files = [f for f in git("ls-tree", "-r", "--name-only", ref).split("\n")
             if f and not f.startswith(EXCLUDE + "/")]
    lang: Counter = Counter()
    for fn in files:
        e = fn.rsplit(".", 1)[-1].lower() if "." in fn else ""
        if e in ("py", "md", "txt"):
            lang[EXT[e]] += line_count(git("cat-file", "blob", f"{ref}:{fn}", binary=True))
    total = sum(lang.values())

    commits: list[dict] = []
    raw = git("log", "--abbrev=7", "--shortstat", "--pretty=format:\x1e%h\x1f%an\x1f%aI\x1f%s", *spec)
    cur = None
    for line in raw.split("\n"):
        if line.startswith("\x1e"):
            sha, au, iso, msg = line[1:].split("\x1f", 3)
            cur = dict(repo=REPO_SLUG, sha=sha, author=au, committed_at=iso, commit_date=iso[:10],
                       message=msg, files_changed=0, insertions=0, deletions=0)
            commits.append(cur)
        elif "changed" in line and cur is not None:
            for v, k in re.findall(r"(\d+) (file|insertion|deletion)", line):
                cur[{"file": "files_changed", "insertion": "insertions", "deletion": "deletions"}[k]] = int(v)

    repo_row = dict(repo=REPO_SLUG, project_name="Wi-Fi Intrusion Detection — Noise Robustness Study",
                    category="Research", url=f"https://github.com/Pranav240/{REPO_SLUG}",
                    first_commit=first, last_commit=last, commits_in_clone=len(stamps),
                    tracked_files=len(files), lines_total=total)
    return {
        "dim_repo": ([repo_row], list(repo_row)),
        "fact_commits": (commits, list(commits[0])),
        "fact_repo_language": ([dict(repo=REPO_SLUG, language=k, lines=v) for k, v in sorted(lang.items())],
                               ["repo", "language", "lines"]),
    }


def write_csv(out: Path, name: str, rows: list[dict], cols: list[str]) -> int:
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default="origin/main", help="git ref to read README.md and history from")
    ap.add_argument("--out", type=Path, default=REPO / "powerbi" / "data", help="output folder")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    readme = git("show", f"{args.ref}:README.md")
    tables = {**readme_tables(readme), **git_tables(args.ref)}
    for name, (rows, cols) in tables.items():
        print(f"{name:<26} {write_csv(args.out, name, rows, cols):>3} rows")
    sha = git("rev-parse", args.ref).strip()
    latest = git("log", args.ref, "-1", "--abbrev=7", "--format=%h|%aI").strip().split("|")
    info = {
        "exported_on": dt.date.today().isoformat(),
        "git_ref": args.ref,
        "git_ref_sha": sha,
        "latest_commit": latest[0],
        "latest_commit_date": latest[1],
        "row_counts": {name: len(rows) for name, (rows, _) in tables.items()},
    }
    (args.out / "export_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"git ref: {args.ref} ({sha[:7]}), exported_on {info['exported_on']}")


if __name__ == "__main__":
    main()
