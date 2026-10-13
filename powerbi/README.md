# Power BI report: Wi-Fi IDS noise robustness

Report pack for the ICCSIT 2026 paper repo. It answers three questions with the repo's own numbers:
(1) which models hold up best on noisy Wi-Fi traffic, (2) how much of the gain comes from aggregating packets
rather than from the model, and (3) whether packet order matters.

Everything here is built from files in this repo. Nothing is typed in by hand: `build_data.py` writes the CSVs,
`check_measures.py` recomputes the headline numbers, and `test_pack.py` fails if this README disagrees with the data.

## Purpose

The paper's claim is that multi-packet aggregation, not model architecture and not packet order, drives noise robustness.
The report shows that claim as charts a reader can check against the repo's `README.md`.

## Data

The raw experiment outputs (`data/results/`) are gitignored and are **not** in this repo. The only place the results
exist in the repo is the `README.md` result tables, so the five result CSVs are parsed from those tables by
`build_data.py`. The report therefore shows what the README states; it does not re-run or independently verify the experiments.

| File | Source | Rows | Result basis | Date range |
|---|---|---|---|---|
| `fact_wifi_model_results.csv` | `README.md` "Results summary" table | 11 | 3-seed mean, primary split | not dated |
| `fact_wifi_decomposition.csv` | `README.md` "Where the robustness actually comes from" table | 3 | stage 09 representation test | not dated |
| `fact_wifi_window_sweep.csv` | `README.md` window-size sweep bullet | 4 | stage 09 window sweep | not dated |
| `fact_wifi_shuffle.csv` | `README.md` "Ordering contributes nothing" table | 2 | stage 08 shuffle test | not dated |
| `fact_wifi_variance.csv` | `README.md` "Uncertainty" table | 2 | stage 10 split variance | not dated |
| `dim_repo.csv` | `git log` / `git ls-tree` | 1 | - | 2026-10-08 to 2026-10-08 |
| `fact_commits.csv` | `git log` (merge commits and commits touching only `powerbi/` left out) | 3 | - | 2026-10-08 to 2026-10-08 |
| `fact_repo_language.csv` | `git ls-tree` line counts (`powerbi/` left out) | 3 | - | snapshot at export |

Row counts (checked by `test_pack.py`):

<!-- rows:start -->
| File | Rows |
|---|---|
| fact_wifi_model_results | 11 |
| fact_wifi_decomposition | 3 |
| fact_wifi_window_sweep | 4 |
| fact_wifi_shuffle | 2 |
| fact_wifi_variance | 2 |
| dim_repo | 1 |
| fact_commits | 3 |
| fact_repo_language | 3 |
<!-- rows:end -->

The three git tables describe a repo with 3 commits on a single day (2026-10-08). No page uses them
(see "Report pages"). They are kept so the original queries and measures keep working.

### Result basis

The same model and setting has different values in different experiments, and the README does not say why. For example, Random
Forest 10-packet noisy accuracy is 97.74 (3-seed mean), 97.97 (stage 09) and 97.63 (stage 10). Each result row therefore carries a
`result_basis` naming the README passage it came from, **every visual title states the basis, and no page uses two bases**.

| result_basis | Table | README passage |
|---|---|---|
| 3-seed mean, primary split | `fact_wifi_model_results` | "Noisy-test accuracy, averaged over 3 model seeds on the primary split" |
| stage 09 representation test | `fact_wifi_decomposition` | "Decomposed via mean-pooling (stage 09)" |
| stage 09 window sweep | `fact_wifi_window_sweep` | window-size sweep bullet (W=1: 79.3, W=2: 84.6, W=5: 91.5, W=10: 92.8) |
| stage 08 shuffle test | `fact_wifi_shuffle` | "Shuffling packet order within each window (stage 08)" |
| stage 10 split variance | `fact_wifi_variance` | "Split variance exceeds model-seed variance (stage 10)" |

`check_measures.py` fails if a row has a `source` or `result_basis` outside its allowed lists, or if a table holds more than one basis.

## Power Query

One blank query per block in `queries.pq`, all reading `DataFolder & "\<file>.csv"`:

- `dim_repo`: one row for the repo (name, URL, first/last commit, file and line counts).
- `fact_commits`: one row per commit, plus a `commit_kind` column (Fix / Test / Docs / Other) from the first 12 characters of the message.
- `fact_repo_language`: lines of code per language.
- `fact_wifi_model_results`: model x context noisy accuracy (3-seed mean), typed, with `result_basis`.
- `fact_wifi_decomposition`: single packet / mean-pooled / flattened accuracy, typed, with `step_order` and `result_basis`.
- `fact_wifi_window_sweep`: accuracy by window size, typed, with `result_basis`.
- `fact_wifi_shuffle`: ordered vs shuffled accuracy and delta per model, typed, with `result_basis`.
- `fact_wifi_variance`: accuracy, model-seed std and split std per model, typed, with `result_basis`.

## DAX measures

Formulas are copied from `measures.dax` (`test_pack.py` checks every one appears there unchanged).

| Measure | Plain-English meaning | Formula |
|---|---|---|
| Commits | Number of commits in the export. | `COUNTROWS ( fact_commits )` |
| Active Days | Distinct days with at least one commit. | `DISTINCTCOUNT ( fact_commits[commit_date] )` |
| Commits / Active Day | Commits divided by active days. | `DIVIDE ( [Commits], [Active Days] )` |
| Lines Added | Sum of inserted lines. | `SUM ( fact_commits[insertions] )` |
| Lines Deleted | Sum of deleted lines. | `SUM ( fact_commits[deletions] )` |
| Churn Ratio | Deleted lines divided by added lines. | `DIVIDE ( [Lines Deleted], [Lines Added] )` |
| Commits Last 30d | Commits in the 30 days up to the last date on the calendar. | `CALCULATE ( [Commits], DATESINPERIOD ( DimDate[Date], MAX ( DimDate[Date] ), -30, DAY ) )` |
| Repo Lines (current) | Total lines of Python, Markdown and text in the repo at the export ref. | `SUM ( fact_repo_language[lines] )` |
| Basis (primary comparison) | Text for chart titles: the result_basis of the rows in the filter context, or MIXED BASES if more than one. | `SELECTEDVALUE ( fact_wifi_model_results[result_basis], "MIXED BASES - do not read" )` |
| Basis (representation test) | Same, for the representation-test table. | `SELECTEDVALUE ( fact_wifi_decomposition[result_basis], "MIXED BASES - do not read" )` |
| Basis (window sweep) | Same, for the window-sweep table. | `SELECTEDVALUE ( fact_wifi_window_sweep[result_basis], "MIXED BASES - do not read" )` |
| Basis (shuffle test) | Same, for the shuffle-test table. | `SELECTEDVALUE ( fact_wifi_shuffle[result_basis], "MIXED BASES - do not read" )` |
| Basis (split variance) | Same, for the split-variance table. | `SELECTEDVALUE ( fact_wifi_variance[result_basis], "MIXED BASES - do not read" )` |
| Best Noisy Accuracy | Highest noisy-test accuracy among the rows in view (97.74, Random Forest 10-packet, with no slicer). | `MAX ( fact_wifi_model_results[noisy_accuracy] )` |
| Context Gain (matched models, pts) | Average 10-packet accuracy minus average single-packet accuracy, using only models that have both rows (the Transformer has no single-packet row). | `VAR SingleModels = CALCULATETABLE ( VALUES ( fact_wifi_model_results[model] ), fact_wifi_model_results[context] = "Single-packet" ) RETURN CALCULATE ( AVERAGE ( fact_wifi_model_results[noisy_accuracy] ), fact_wifi_model_results[context] = "10-packet", KEEPFILTERS ( fact_wifi_model_results[model] IN SingleModels ) ) - CALCULATE ( AVERAGE ( fact_wifi_model_results[noisy_accuracy] ), fact_wifi_model_results[context] = "Single-packet" )` |
| Aggregation Gain (pts) | Mean-pooled window accuracy minus single-packet accuracy: what averaging the packets in a window adds. | `CALCULATE ( MAX ( fact_wifi_decomposition[noisy_accuracy] ), fact_wifi_decomposition[representation] = "Mean-pooled window" ) - CALCULATE ( MAX ( fact_wifi_decomposition[noisy_accuracy] ), fact_wifi_decomposition[representation] = "Single packet" )` |
| Beyond-Average Gain (pts) | Flattened window accuracy minus mean-pooled accuracy: what keeping each packet adds beyond their average. | `CALCULATE ( MAX ( fact_wifi_decomposition[noisy_accuracy] ), fact_wifi_decomposition[representation] = "Flattened window" ) - CALCULATE ( MAX ( fact_wifi_decomposition[noisy_accuracy] ), fact_wifi_decomposition[representation] = "Mean-pooled window" )` |
| Shuffle Delta RF (pts) | Random Forest accuracy with window order shuffled minus ordered. Negative means shuffling hurt. | `CALCULATE ( SUM ( fact_wifi_shuffle[delta] ), fact_wifi_shuffle[model] = "Random Forest" )` |
| Shuffle Delta Transformer (pts) | Same for the Transformer. Positive means it did better with the order destroyed. | `CALCULATE ( SUM ( fact_wifi_shuffle[delta] ), fact_wifi_shuffle[model] = "Transformer" )` |
| Seed Std | Standard deviation across model seeds. Only meaningful with model on the axis. | `SUM ( fact_wifi_variance[model_seed_std] )` |
| Split Std | Standard deviation across data splits. Only meaningful with model on the axis. | `SUM ( fact_wifi_variance[split_std] )` |
| Split Gap (pts) | Random Forest minus Transformer noisy accuracy in the split-variance table. | `CALCULATE ( MAX ( fact_wifi_variance[noisy_accuracy] ), fact_wifi_variance[model] = "Random Forest" ) - CALCULATE ( MAX ( fact_wifi_variance[noisy_accuracy] ), fact_wifi_variance[model] = "Transformer" )` |

### Needs my review

- **Context Gain (matched models, pts)**: DAX not run (no Power BI in this environment). The pandas version in check_measures.py is the intended value; if the card differs, this measure is the suspect.
- **Seed Std**: Meaningless as a single card because it sums over models.
- **Split Std**: Same: only valid split by model.

### Headline numbers

`check_measures.py` prints these. They are the values the Power BI cards should show.
**Not yet confirmed against Power BI.** Whoever builds the report should compare each card and note any mismatch here.

<!-- headline:start -->
| Measure | Value | Basis |
|---|---|---|
| Best Noisy Accuracy | 97.74 | 3-seed mean, primary split |
| Context Gain (matched models, pts) | 17.03 | 3-seed mean, primary split |
| Aggregation Gain (pts) | 11.00 | stage 09 representation test |
| Beyond-Average Gain (pts) | 4.99 | stage 09 representation test |
| Shuffle Delta Transformer (pts) | 1.76 | stage 08 shuffle test |
<!-- headline:end -->

Note on Beyond-Average Gain: the repo `README.md` says "+5.0 pts" for this step. The two accuracies it comes from (97.97 and 92.98)
differ by 4.99, so the data gives 4.99, not 5.0. The report shows 4.99.

## Report pages

One question per page. Each page uses one result table, so it has one basis, and the basis is in every visual title
using the matching `Basis (...)` measure.

| # | Page title | Question | Visuals | Basis |
|---|---|---|---|---|
| 1 | Which models hold up on noisy traffic? | Does a 10-packet window beat a single packet, and for which models? | Clustered bar: `model` on the axis, `noisy_accuracy` as values, `context` as legend. Cards: Best Noisy Accuracy, Context Gain (matched models, pts). Slicer on `context`. | 3-seed mean, primary split |
| 2 | Where does the gain come from? | How much comes from averaging packets and how much from keeping them? | Column chart of `fact_wifi_decomposition` by `step_order`. Cards: Aggregation Gain (pts), Beyond-Average Gain (pts). | stage 09 representation test |
| 3 | Does packet order matter? | Does shuffling the packets inside a window change accuracy? | Clustered bar: `ordered` and `shuffled` per model. Cards: Shuffle Delta RF (pts), Shuffle Delta Transformer (pts). | stage 08 shuffle test |

Optional pages, each on its own basis. They are real findings from the paper, kept off the main three to respect "2 or 3 pages":

| # | Page title | Visuals | Basis |
|---|---|---|---|
| 4 | How fast does accuracy grow with window size? | Line chart of `fact_wifi_window_sweep` (`window_size` x `noisy_accuracy`). | stage 09 window sweep |
| 5 | How uncertain is the gap? | Clustered bar of `model_seed_std` and `split_std` per model; card Split Gap (pts). | stage 10 split variance |

The repo-activity tables are not on any page: 3 commits on one day is not a trend.

Example visual title: `Noisy accuracy by model and context | basis: ` + `[Basis (primary comparison)]`.

## How to rebuild in Power BI Desktop (Windows)

1. Clone the repo and check out the branch. Install Python with pandas to run the checks.
2. Run `python powerbi/check_measures.py` and keep the output open; you will compare the cards to it.
3. Power BI Desktop > Home > Transform data > Manage Parameters > New. Name `DataFolder`, type Text, current value the full path to `powerbi\data`, for example `C:\Users\you\wifi-ids-aggregation-not-architecture\powerbi\data`.
4. For each block in `queries.pq`: New Source > Blank Query > Advanced Editor > delete the template text > paste the block (from `let` to the final `in ...`) > Done > rename the query to the name in the `// ----` comment line.
5. Close & Apply.
6. Modeling > New table: paste the `DimDate` line from `measures.dax`. Mark it as a date table on `Date`. (Only needed for the activity measures.)
7. For each row of the DAX table above: Modeling > New measure > paste the formula, name it exactly as in the Measure column.
8. View > Themes > Browse for themes > `theme.json`.
9. Model view: relationships `dim_repo[repo]` to `fact_commits[repo]` and to `fact_repo_language[repo]` (one to many); `DimDate[Date]` to `fact_commits[commit_date]`. The five `fact_wifi_*` tables have no relationships and stay stand-alone. That is deliberate: it stops a filter on one basis reaching another.
10. Build pages 1 to 3 as in "Report pages". Put the `Basis (...)` measure in each visual's title (Format > Title > fx).
11. Compare each card to the output of step 2. Write down any mismatch in the Headline numbers section.

## Screenshots

Save to `powerbi/screenshots/`. These are the shots to capture (1600 px wide, light theme).

| File | Page | The shot must show |
|---|---|---|
| `01-primary-comparison.png` | 1 | Bar chart of all 11 model/context rows; cards showing 97.74 and 17.03; the basis text "3-seed mean, primary split" in the title. |
| `02-aggregation-vs-model.png` | 2 | Three columns (single packet, mean-pooled window, flattened window); cards showing 11.00 and 4.99; the basis text "stage 09 representation test". |
| `03-packet-order.png` | 3 | Ordered vs shuffled bars for Random Forest and Transformer; cards showing -0.47 and 1.76; the basis text "stage 08 shuffle test". |
| `04-window-sweep.png` | 4 (optional) | Line through W=1, 2, 5, 10; basis "stage 09 window sweep". |
| `05-uncertainty.png` | 5 (optional) | Seed std vs split std bars for both models; card showing 3.50; basis "stage 10 split variance". |

## Last export

Data last exported **2026-10-13** from `origin/main` at `e7ce6d7`
(latest commit `e7ce6d7`, 2026-10-08). Also recorded in `data/export_info.json`.

To refresh after the README results change or new commits land:

```
git fetch origin
python powerbi/build_data.py            # rewrites powerbi/data/ from origin/main
python powerbi/check_measures.py        # prints the numbers the cards should show
python -m pytest powerbi/test_pack.py   # fails if this README no longer matches the data
```

If `build_data.py` stops with an error, the README table layout changed; it will not guess.

## Tests

`python -m pytest powerbi/test_pack.py` (needs pandas and pytest) checks that:

1. the headline table above equals what `check_measures.py` computes from the CSVs;
2. the row-count table above equals the CSVs;
3. every formula in the DAX table equals the one in `measures.dax`, and every measure there is in the table;
4. every result number in the CSVs appears in the repo `README.md`;
5. `source` and `result_basis` values are in their allowed lists, and each table holds one basis;
6. rebuilding the CSVs from the recorded git ref reproduces the committed files (line endings ignored).

## Known caveats

- The results come from README tables, not from re-running the experiments, and the raw result files are not in the repo.
- The README states the same setting under different bases without explaining the differences (RF single packet: 78.89 / 81.98 / 79.30; Transformer: 94.88 / 93.32 / 94.13). Do not put them on one axis.
- The window sweep at W=10 (92.8) is not the same number as the mean-pooled window in the representation test (92.98), though both are described in the same README section.
- The noise is synthetic, and the repo README lists a possible protocol shortcut (`arp.opcode`) and two thin classes. State these on the report.
- The Context Gain measure was rewritten (see below) and its DAX has not been run in Power BI.

## Changes from the first version of this pack

- Added `result_basis` to the five result tables and queries; added the `Basis (...)` measures.
- Added `build_data.py`, `check_measures.py`, `test_pack.py`.
- Replaced `Context Gain (pts)` (averaged 10-packet over six models against single-packet over five, which mixes model sets) with `Context Gain (matched models, pts)`.
- Removed `Avg Noisy Accuracy` (averaged across both contexts, so it mixed single-packet and 10-packet rows).
- Replaced `Shuffle Delta (pts)` (a sum across two models, which is meaningless) with one measure per model.
- Added `Split Gap (pts)`.
