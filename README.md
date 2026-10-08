# Wi-Fi Intrusion Detection: Context vs. Architecture

Companion code for an ablation study on noise-robust Wi-Fi intrusion detection
using the AWID3 dataset.

**Headline finding:** multi-packet aggregation — not model architecture, and not
packet ordering — drives noise robustness. A Random Forest on flattened
10-packet windows outperforms a hyperparameter-tuned Transformer whose
positional encoding is demonstrably inert on this data.

---

## Results summary

Noisy-test accuracy, averaged over 3 model seeds on the primary split:

| Model | Single-packet | 10-packet context |
|---|---|---|
| Decision Tree | 68.50% | 72.81% |
| CNN | 71.71% | 95.21% |
| LSTM | 72.43% | 93.91% |
| XGBoost | 74.20% | 91.22% |
| Random Forest | 78.89% | **97.74%** |
| Transformer (BBOT-tuned) | — | 94.88% |

Clean-test accuracy is 98.7–99.7% for every model in every condition, so the
noisy condition is where any meaningful separation appears.

### Where the robustness actually comes from

Decomposed via mean-pooling (stage 09), Random Forest, noisy test:

| Representation | Input dim | Noisy accuracy |
|---|---|---|
| Single packet | 18 | 81.98% |
| Mean-pooled window | 18 | 92.98% |
| Flattened window | 180 | 97.97% |

- **+11.0 pts** from aggregation alone (noise averaging over redundant packets)
- **+5.0 pts** from retaining individual packets beyond their average
- Window-size sweep (W = 1, 2, 5, 10 → 79.3%, 84.6%, 91.5%, 92.8%) shows the
  decelerating curve a √W noise-reduction account predicts

### Ordering contributes nothing

Shuffling packet order *within* each window (stage 08), noisy accuracy:

| Model | Ordered | Shuffled | Δ |
|---|---|---|---|
| Random Forest | 97.97% | 97.50% | −0.47 |
| Transformer | 93.32% | 95.08% | **+1.76** |

The Transformer performs *better* with order destroyed. Combined with the
within/between-window variance ratio of 0.058 (packets inside a window are
~17× more similar to each other than windows are to one another), the
remaining +5.0 pts above is distributional information, not sequential.

### Uncertainty

Split variance exceeds model-seed variance (stage 10), so error bars must be
reported across data splits, not just model initialisations:

| Model | Noisy acc | Model-seed std | Split std |
|---|---|---|---|
| Random Forest | 97.63% | 0.152 | 0.576 |
| Transformer | 94.13% | 1.193 | 1.335 |

The 3.50-point gap is ~2.4× the combined across-split std, and Random Forest
wins on all three splits individually.

---

## Setup

```bash
python -m venv wifi_ids_env

# Windows
.\wifi_ids_env\Scripts\Activate.ps1
# Linux / macOS
source wifi_ids_env/bin/activate

pip install -r requirements.txt
```

### Raw data

Download AWID3 and point the pipeline at the directory containing the 13
per-attack-category folders (`1.Deauth`, `2.Disas`, … each holding multiple
CSV files):

```powershell
$env:WIFI_IDS_RAW_DIR = "D:\awid3_attacks"      # PowerShell
```
```bash
export WIFI_IDS_RAW_DIR=/data/awid3_attacks      # bash
```

Or edit `RAW_DIR` in `config.py`. Verify with:

```bash
python config.py
```

which prints the active configuration and validates it. Note that ~44 GB of
raw captures are needed for stages 01–02; every later stage reads only from
`data/prepared/`.

---

## Pipeline

Run in order. Stages 03–10 depend only on `data/prepared/`, so once 01–02b
have run you can re-run any downstream stage independently.

| Stage | Script | Purpose | Approx. runtime |
|---|---|---|---|
| 01 | `01_select_features.py` | Scans raw captures for attack rows; selects 18 features via missingness/variance filtering, XGBoost importance, and correlation pruning | ~10 min |
| 02 | `02_prepare_data.py` | Session-aware (MAC-grouped) non-overlapping 10-packet windowing, capped stratified sampling, train/val/test split, scaling | ~20 min |
| 02b | `02b_add_test_noise.py` | Builds the noisy test set using a corruption process deliberately distinct from training augmentation | < 1 min |
| 03 | `03_train_baselines.py` | DT / RF / XGBoost / CNN / LSTM on single packets — the deployment-realistic primary comparison | ~20 min |
| 04 | `04_train_transformer.py` | Optuna (TPE) hyperparameter search, then tuned and default Transformers on windowed input | ~45 min |
| 05 | `05_train_flattened_ablation.py` | Same five baselines given the Transformer's 10-packet context — isolates architecture from context | ~15 min |
| 06 | `06_multiseed_eval.py` | 3 model seeds for the four closest-performing models | ~15 min |
| 07 | `07_generate_figures.py` | Publication figures from saved result CSVs | < 1 min |
| 08 | `08_shuffle_test.py` | **Decisive:** shuffles packet order within windows to test whether ordering matters at all | ~20 min |
| 09 | `09_mean_pooling.py` | Mean-pooling baseline and window-size sweep — decomposes the aggregation effect | ~5 min |
| 10 | `10_split_variance.py` | Re-splits with 3 split seeds to measure split-induced variance | ~15 min |

Runtimes are for CPU on a typical laptop; TensorFlow is CPU-only on native
Windows.

---

## Outputs

```
data/prepared/          windowed / flattened / single-packet .npz, scaler,
                        label encoder, selected features, config snapshot
data/results/
  baselines/            single-packet results, per-class reports, confusion matrices
  transformer/          Optuna trials, best params, tuned + default results
  flattened_ablation/   context-equalized results
  multiseed/            per-seed raw + mean±std summary
  shuffle_test/         ordered vs shuffled + verdict JSON
  mean_pooling/         representation comparison + window-size sweep
  split_variance/       variance decomposition + gap analysis
figures/                publication PNGs (300 DPI)
```

Every model directory contains per-class classification reports and confusion
matrices, not just aggregate accuracy.

---

## Design decisions worth knowing

**Train and test noise are deliberately different.** Training augmentation uses
Gaussian noise (0.10) and missing values (0.08). Test-time evaluation uses a
higher magnitude (0.25), a higher missing rate (0.15), *and* a sign-flip
corruption absent from training. `config.validate_config()` fails loudly if
these ever converge — evaluating on noise drawn from the training distribution
measures memorisation, not generalisation.

**Windows are non-overlapping and split at the window level.** Overlapping
windows would place near-duplicate samples on both sides of the train/test
boundary.

**Session grouping uses `wlan.sa`**, falling back to `wlan.ta` / `wlan.bssid`,
so a window never spans unrelated device conversations.

**Capture-session artefacts are excluded outright** (hardware clocks, fixed
radio channel/frequency). These can fingerprint which capture file a row came
from rather than encoding attack behaviour.

**Labels are derived from folder names**, not the raw `Label` column, because
the raw data contains casing and spelling drift (`Kr00k` vs `Kr00K`,
`Rogue_AP` vs `RogueAP`) that would otherwise fragment classes.

---

## Known limitations

- **Noise is synthetic and uncalibrated.** The corruption process is not derived
  from measured wireless interference; sign-flipping in particular has no direct
  physical analogue. Claims should be read as robustness to synthetic
  feature-space perturbation, not to characterised RF conditions.
- **Possible protocol shortcut.** `arp.opcode` alone carries ~43% of feature
  importance, and attack categories in AWID3 map closely onto distinct
  protocols. Models may partly be identifying *which protocol is in use* rather
  than malicious behaviour. This has not been quantified — dropping the top
  protocol-identifying features and re-measuring is the obvious next test.
- **Two thin classes.** Rogue AP (130 windows) and SQL Injection (262) are
  limited by the raw data, not by sampling. Rogue AP has ~20 test windows, so
  its per-class metrics carry wide uncertainty.
- **Split variance is a lower bound.** Stage 10 re-splits the already-windowed,
  already-scaled pool rather than re-running stage 02 per split, so a small
  amount of scaler information crosses splits.
- **Window labels use majority vote**, so windows straddling a normal→attack
  transition are assigned to whichever dominates.

---

## Citation

If you use this code, please cite the accompanying paper (ICCSIT 2026
submission, Paper ID ZX1060).
