"""Fails if the Power BI pack and the repo disagree.  Run: python -m pytest powerbi/test_pack.py

Needs pandas and pytest. Nothing here checks Power BI itself; see "Headline numbers" in README.md.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import check_measures as cm  # noqa: E402
import build_data as bd  # noqa: E402

DATA = HERE / "data"
README = (HERE / "README.md").read_text(encoding="utf-8")
REPO_README = (REPO / "README.md").read_text(encoding="utf-8")
T = cm.load(DATA)


def block(text: str, name: str) -> list[list[str]]:
    m = re.search(rf"<!-- {name}:start -->\n(.*?)\n<!-- {name}:end -->", text, re.S)
    assert m, f"README.md has no <!-- {name}:start --> block"
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in m.group(1).split("\n") if l.startswith("|")]
    return rows[2:]  # drop header and separator


def test_readme_headline_numbers_match_data():
    stated = {r[0]: (r[1], r[2]) for r in block(README, "headline")}
    computed = cm.headline(T)
    assert set(stated) == {n for n, _, _ in computed}
    for name, val, basis in computed:
        assert stated[name] == (f"{val:.2f}", basis), f"README says {stated[name]}, data gives {val:.2f} ({basis})"


def test_readme_row_counts_match_csvs():
    for name, n in block(README, "rows"):
        assert len(pd.read_csv(DATA / f"{name}.csv")) == int(n), name


def test_readme_dax_table_matches_measures_dax():
    formulas = cm.dax_measures()
    in_readme = {}
    for line in README.split("\n"):
        m = re.match(r"^\| (.+?) \| .+ \| `(.+)` \|$", line)
        if m and m.group(1) in formulas:
            in_readme[m.group(1)] = m.group(2)
    assert set(in_readme) == set(formulas), set(formulas) ^ set(in_readme)
    for name, f in formulas.items():
        assert in_readme[name] == f, name


def fmt(x: float) -> str:
    return f"{x:.2f}"


def test_every_result_number_appears_in_repo_readme():
    plain = REPO_README.replace("−", "-")
    nums = set()
    for tab in ("fact_wifi_model_results", "fact_wifi_decomposition", "fact_wifi_shuffle", "fact_wifi_variance"):
        df = T[tab]
        for col in df.select_dtypes("number").columns:
            if col in ("seeds", "input_dim", "step_order"):
                continue
            nums.update(abs(v) for v in df[col])
    nums.update(T["fact_wifi_window_sweep"]["noisy_accuracy"])
    for v in sorted(nums):
        pat = re.compile(rf"(?<![\d.]){re.escape(f'{v:.2f}')}(?!\d)|(?<![\d.]){re.escape(f'{v:g}')}(?!\d)")
        assert pat.search(plain), f"{v} is in the CSVs but not in README.md"


def test_labels_are_allowed_and_one_basis_per_table():
    assert cm.ALLOWED_BASES == bd.ALLOWED_BASES and cm.ALLOWED_SOURCES == bd.ALLOWED_SOURCES
    assert cm.label_check(T) and cm.one_basis_check(T) and cm.arithmetic_check(T)


def test_a_bad_label_is_caught():
    bad = {k: v.copy() for k, v in T.items()}
    bad["fact_wifi_shuffle"].loc[0, "source"] = "docs/engineering-log.md"
    bad["fact_wifi_variance"].loc[0, "result_basis"] = "stage 99"
    assert not cm.label_check(bad)


def test_csvs_rebuild_byte_for_byte(tmp_path):
    info = json.loads((DATA / "export_info.json").read_text(encoding="utf-8"))
    sha = info["git_ref_sha"]
    if subprocess.run(["git", "cat-file", "-e", sha], cwd=REPO, capture_output=True).returncode:
        pytest.skip(f"commit {sha[:7]} is not in this clone (shallow clone?)")
    subprocess.run([sys.executable, str(HERE / "build_data.py"), "--ref", sha, "--out", str(tmp_path)],
                   check=True, capture_output=True)
    for f in sorted(DATA.glob("*.csv")):
        assert (tmp_path / f.name).read_bytes() == f.read_bytes(), f"{f.name} differs from a fresh build"
