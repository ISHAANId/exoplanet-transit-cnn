# Exoplanet Transit Detection with a 1D CNN

**Research question:** Does phase-folding a Kepler light curve on its
candidate orbital period improve a 1D CNN's ability to tell a real transiting
planet apart from a false positive, compared to feeding the CNN a raw
(cleaned but unfolded) light curve?

## Pipeline

| # | Script | What it does |
|---|--------|---------------|
| 1a | `scripts/01_get_koi_table.py` | Downloads the NASA cumulative KOI table (via astroquery / NASA Exoplanet Archive) |
| 1b | `scripts/02_clean_koi_table.py` | Builds a **leakage-safe** label table: binary label + orbital params only, all vetting/disposition columns dropped |
| 1c | `scripts/03_download_lightcurves.py` | Downloads and caches each star's Kepler light curve with Lightkurve |
| 2/3/4/5 | `scripts/04_build_dataset.py` | Cleans each light curve, builds the **raw** (Experiment A) and **phase-folded** (Experiment B) 400-point views, splits by star into train/val/test |
| 6 | `scripts/model.py` | The 1D CNN architecture (shared by both experiments) |
| 7 | `scripts/05_train_experiment.py` | Trains + evaluates on one dataset (`--variant raw` or `--variant folded`) |
| 8 | `scripts/06_compare_experiments.py` | Loads both experiments' metrics and produces the A-vs-B comparison table/chart |

`config.py` holds every shared constant (paths, `N_POINTS = 400`, the random
seed, the list of leakage columns, split fractions) so all scripts agree with
each other.

`PROJECT_LOG.md` is the running log of what was actually done, the
methodology, and honest results (filled in as each stage is actually run —
never with placeholder numbers).

## How to run it

```powershell
# 1. Create/activate the virtual environment (already created for you)
.\venv\Scripts\Activate.ps1

# 2. Get the KOI table and build the leakage-safe label table
python scripts\01_get_koi_table.py
python scripts\02_clean_koi_table.py

# 3. Download light curves. This is the slow step -- start with a small
#    sample to make sure everything works before committing hours to the
#    full ~8700-star table.
python scripts\03_download_lightcurves.py --limit 40 --sample-balanced
python scripts\04_build_dataset.py

# 4. Train both experiments
python scripts\05_train_experiment.py --variant raw
python scripts\05_train_experiment.py --variant folded

# 5. Compare them
python scripts\06_compare_experiments.py
```

When you're ready for the real run (not just a smoke test), drop `--limit`
from step 3 and let it run in the background — downloading light curves for
thousands of stars from MAST will take a long time.

## Key design decisions (the "why", not just the "what")

- **Leakage columns removed:** `koi_disposition`, `koi_pdisposition`,
  `koi_score`, `koi_fpflag_nt/ss/co/ec`, and the vetting-provenance columns
  are dropped before anything is saved to `koi_clean_labels.csv` — see
  `config.LEAKAGE_COLUMNS` for the full list with reasoning per column. The
  CNN's *only* input is the flux array; none of these ever reach the model.
- **Split by star, not by row:** a star with multiple KOIs (a multi-planet
  system) has all of its rows forced into the same split, and the split
  itself is computed with `GroupShuffleSplit` grouped on `kepid`. Otherwise
  the same physical light curve (with its specific noise fingerprint) could
  appear in both train and test.
- **Both experiments share one cleaned light curve per star.** Cleaning
  (outlier removal, transit-aware detrending, normalization) happens once;
  Experiment A and B only differ in the final resampling step (single
  transit window vs. globally phase-folded, both to 400 points). This keeps
  the A/B comparison about folding specifically, not about two different
  cleaning pipelines.
- **Transit-aware flattening:** the long-term trend fit excludes the transit
  itself (via `create_transit_mask`), otherwise a naive Savitzky-Golay
  detrend partially fits away the very dip we want the CNN to learn.
- **No numbers are invented.** `05_train_experiment.py` refuses to report
  results if there isn't enough real data to train on, and every metric in
  `PROJECT_LOG.md` is copied from an actual `metrics.json` produced by a run.
