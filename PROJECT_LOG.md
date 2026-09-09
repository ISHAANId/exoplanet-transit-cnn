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

## (next entry goes here once the smoke test / full run happens)

Template for a results entry:

```
## YYYY-MM-DD — <what was run>

Command(s) run:
    ...

Data: N stars downloaded, N examples built, train/val/test = _/_/_ stars

Experiment A (raw, no fold):
    accuracy=  precision=  recall=  f1=  roc_auc=

Experiment B (phase folded):
    accuracy=  precision=  recall=  f1=  roc_auc=

Verdict: <what the numbers actually showed, including if the result was
noisy/inconclusive because of small sample size -- do not oversell a small
smoke-test result>
```
