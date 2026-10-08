# Power BI report: Wi-Fi IDS noise robustness

Place this folder at `powerbi/` in the repo.

## Setup (Power BI Desktop, Windows)
1. Transform data > Manage Parameters > New: `DataFolder` (Text) = full path to `powerbi\data`.
2. For each block in `queries.pq`: New Source > Blank Query > Advanced Editor > paste > rename to the block name.
3. Close & Apply. Add `DimDate` and the measures from `measures.dax` (mark DimDate as date table).
4. View > Themes > Browse > `theme.json`.
5. Relationships (1 -> *): `dim_repo[repo]` to `fact_commits[repo]` and `fact_repo_language[repo]`; `DimDate[Date]` to `fact_commits[commit_date]`. The `fact_wifi_*` tables stand alone.

## Pages
1. **Primary comparison:** clustered bar `model` x `noisy_accuracy`, legend = `context`. Headline: Random Forest 10-packet 97.74% vs Transformer 94.88%.
2. **Where robustness comes from:** column chart of `fact_wifi_decomposition` ordered by `step_order` (81.98 > 92.98 > 97.97); cards Aggregation Gain and Beyond-Average Gain; line of `fact_wifi_window_sweep`.
3. **Ordering test:** clustered bar ordered vs shuffled per model; card Shuffle Delta.
4. **Uncertainty:** bars of `model_seed_std` vs `split_std` per model.
5. **Repo activity:** commits by month, languages.

## Caveats
- All numbers are transcribed from the README. The raw result CSVs are gitignored, so nothing here is recomputed. If you want the report to track reruns, load `data/results/*` from your local run instead.
- Random Forest noisy accuracy appears as 97.74 (3 seeds), 97.97 (stage 09/08) and 97.63 (across splits). These come from different experiments; do not put them on one axis as if comparable.
- The window sweep (W=1: 79.3, W=10: 92.8) differs from the decomposition table (single packet 81.98, mean-pooled window 92.98) even though they are described as the same setup. Check the stage 09 output before presenting both together.
- Synthetic noise, possible protocol shortcut (`arp.opcode` ~43% of importance) and thin classes are listed in the README; mention them on the report.
- Data snapshot: 2026-10-08.
