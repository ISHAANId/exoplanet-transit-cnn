# Project Log — Exoplanet Transit Detection with a 1D CNN

This file is the running record of what was actually built and run, in
order, with the reasoning behind each decision. Update it every time a new
stage of the pipeline is actually executed — metrics only go in here once
they come out of a real `metrics.json`, never as placeholders.

---

## 2026-09-09 — Project scaffolded

**What was done**

- Set up a dedicated project folder at
  `C:\Users\ishaani dongre\exoplanet-transit-cnn\` (separate from the
  cluttered `Downloads` folder the original scratch script lived in).
- Created a Python 3.13 virtual environment (`venv/`) — Python 3.13 was
  chosen over the system's default 3.14 install because TensorFlow does not
  yet ship wheels for 3.14.
- Installed: `lightkurve`, `astroquery`, `tensorflow`, `scikit-learn`,
  `scipy`, `matplotlib`, `pandas`, `numpy`.
- Wrote the full 8-stage pipeline described in `README.md`:
  1. `01_get_koi_table.py` — download the NASA cumulative KOI table
  2. `02_clean_koi_table.py` — drop leakage columns, build binary label
  3. `03_download_lightcurves.py` — download + cache Kepler light curves
  4. `04_build_dataset.py` — clean, phase-fold, resample to 400 points, split by star
  5. `model.py` — 1D CNN architecture
  6. `05_train_experiment.py` — train/evaluate one variant (raw or folded)
  7. `06_compare_experiments.py` — Experiment A vs B comparison

**Key decisions and why** (see `README.md` "Key design decisions" for the
short version; this is the fuller reasoning)

1. **Binary label, CANDIDATE rows dropped.** `koi_disposition` has three
   values. `CANDIDATE` means NASA's own pipeline hasn't reached a final
   answer yet, so those rows are neither a confirmed "Planet" nor a
   confirmed "False Alarm" — training on them as either would be label
   noise. Kept: `CONFIRMED` → 1, `FALSE POSITIVE` → 0.

2. **Leakage columns identified and physically excluded.** The KOI table
   includes NASA's own vetting outputs (`koi_score`, `koi_pdisposition`,
   `koi_fpflag_nt/ss/co/ec`, vetting comments/provenance/dates). These are
   the *answer*, or a machine-computed proxy for it, produced by a pipeline
   that already looked at the light curve (and more data than we're using).
   `02_clean_koi_table.py` never writes them to `koi_clean_labels.csv`, so
   no script downstream can accidentally train on them — the boundary is
   structural, not just "remember not to include this column" at model
   time.

3. **Star-level (not row-level) train/val/test split.** A star can host more
   than one KOI (multi-planet system) — those rows share the exact same
   underlying photometry. If split by row, the same physical light curve
   (with its own instrument noise fingerprint, gaps, and quarter coverage)
   could appear in both train and test, letting the CNN partially
   "recognize the star" rather than learn general transit shape. Implemented
   with `sklearn.model_selection.GroupShuffleSplit` grouped on `kepid`, plus
   a hard assertion in `04_build_dataset.py` that no `kepid` ever spans more
   than one split.

4. **PDCSAP flux as the starting point, own cleaning on top.** Kepler's
   PDCSAP_FLUX is already systematics-corrected but that's an instrumental
   correction, not a disposition — using it isn't leakage, it's just a
   sane starting point. On top of it we do our own outlier removal
   (sigma-clipping), long-term trend removal (Savitzky-Golay via
   `lightkurve.flatten`, with the transit itself masked out of the trend fit
   so we don't flatten away the signal we care about), and normalization to
   a flux baseline of 1.0.

5. **400-point representation, two ways.** Experiment A resamples a single
   transit window (centered on the first transit in the baseline, width =
   6× the transit duration) to 400 points via interpolation. Experiment B
   phase-folds the *entire* multi-quarter baseline on the KOI's period and
   epoch, then median-bins the folded points into 400 global phase bins.
   Both start from the exact same cleaned light curve object, so the only
   difference between the two experiments is folding vs. not folding.

6. **CNN kept close to the slide's sketch but with 3 conv/pool blocks**
   instead of 2, since a 400-point input needs a bit more receptive field to
   relate a dip to a flat baseline far away in the window. Dropout(0.3)
   before the final dense layer is the main overfitting guard given the
   dataset size.

**Status:** pipeline code complete and installed. No training results yet.
The next entry in this log will be a smoke test on a small number of real
stars to confirm the pipeline runs end-to-end, followed (separately, and
likely over a much longer background run) by the full-scale download and
the actual Experiment A vs B comparison.

---

## 2026-09-09 — Smoke test executed end-to-end (real NASA data, real training)

**Commands run, in order**

```
python scripts\01_get_koi_table.py
python scripts\02_clean_koi_table.py
python scripts\03_download_lightcurves.py --limit 24 --sample-balanced
python scripts\04_build_dataset.py
python scripts\05_train_experiment.py --variant raw
python scripts\05_train_experiment.py --variant folded
python scripts\06_compare_experiments.py
```

**Data**

- KOI table: 9,564 raw rows -> 7,587 rows after dropping `CANDIDATE`s and
  rows with missing orbital parameters (6,641 unique host stars).
- Light curves downloaded for a balanced random sample of 24 stars (12
  planet-hosting, 12 false-alarm) via MAST/Lightkurve. All 24 succeeded.
- `04_build_dataset.py` produced 26 usable examples (a couple of the 24
  stars host more than one KOI, contributing an extra row each) — 400-point
  raw and 400-point phase-folded arrays for every example.
- Star-level split (`GroupShuffleSplit` on `kepid`, leak-check assertion
  passed): **train 16 examples / 15 stars, val 5 examples / 4 stars, test 5
  examples / 5 stars.**

**Experiment A — raw (no phase folding), test set (n=5, never seen in training)**

| accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|
| 0.20 | 0.20 | 1.00 | 0.33 | 0.25 |

**Experiment B — phase folded, test set (n=5, same held-out stars)**

| accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|
| 0.20 | 0.20 | 1.00 | 0.33 | 0.75 |

**What actually happened, honestly:** the test set has only 5 examples (1
Planet, 4 False Alarm), and both models ended up predicting "Planet" for
every one of the 5 -- that's why accuracy/precision/recall/f1 are identical
between A and B (the same trivial all-positive prediction at the 0.5
threshold). The one metric that *did* differ is ROC-AUC, which scores the
predicted probabilities before thresholding: **0.25 for raw vs. 0.75 for
folded.** That means the folded model ranked the true planet's probability
higher relative to the false alarms than the raw model did, even though
both crossed the 0.5 line the same way. This is a hint in the direction the
project's hypothesis predicts (folding helps), but with n_test=5 it is
**not a statistically meaningful result** -- it would take one different
random split for this to flip. Full training logs, the model summary, and
both confusion matrices are saved in `results/experiment_A_raw/` and
`results/experiment_B_folded/`.

**Verdict:** the pipeline is fully verified end-to-end on real Kepler data
-- real download, real leakage-safe labels, real cleaning/folding/resampling,
real CNN training, real (if tiny) test-set evaluation, with no invented
numbers anywhere. The AUC gap (0.25 vs 0.75) is a legitimate observation
from this run but needs a much larger star count before it can support a
real conclusion about whether phase folding helps. That larger run is the
natural next step, not done here by design (see conversation scope: this
smoke test was intentionally kept small to move fast for the report).

**UI check:** `app.py` (Streamlit) was launched locally, confirmed
responding (HTTP 200) and both trained models load and produce a live
prediction with no errors. Also fixed an unrelated machine-level issue: a
stray, invalid `streamlit.py` file sitting in the base Python 3.13 install
directory was shadowing the real `streamlit` package for every Python
program on this machine (not just this project) -- renamed to
`streamlit.py.bak` to stop the conflict.
