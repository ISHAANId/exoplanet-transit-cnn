"""
Step 2 + 3 + 4 (star-level split) + 5 (resample to 400 points), all in one
script, because they have to be applied together per star to avoid subtle
leakage bugs (e.g. cleaning a star once and reusing it correctly, rather than
accidentally re-downloading/re-cleaning slightly differently for train vs
test).

For every KOI row that has a cached light curve (from 03_download_lightcurves.py):
  1. load + clean the star's light curve (noise/outliers/trend/normalize)
  2. build the Experiment A input: one raw transit window, 400 points
  3. build the Experiment B input: the full phase-folded global view, 400 points
  4. assign the star (kepid) to train/val/test

Splitting BY STAR (not by row) is the leakage control described in the
project brief: if the same star ended up in both train and test, the model
could partly "recognize the star" (its specific noise pattern, gaps, depth)
rather than learning the general shape of a transit vs. a false alarm --
and a multi-planet-candidate star would otherwise leak its own light curve
across the split via its other KOI rows.

Run:
    python scripts/04_build_dataset.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    KOI_CLEAN_TABLE_PATH, LIGHTCURVE_CACHE_DIR, RAW_DATASET_PATH, FOLDED_DATASET_PATH,
    N_POINTS, RANDOM_SEED, TEST_FRACTION, VAL_FRACTION,
)
from lightcurve_utils import clean_lightcurve, make_raw_view, make_folded_view  # noqa: E402


def star_level_split(kepids: np.ndarray) -> np.ndarray:
    """Assign every example a split label ('train'/'val'/'test') such that
    all examples sharing a kepid land in the same split."""
    groups = kepids
    n = len(groups)

    gss_test = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=RANDOM_SEED)
    trainval_idx, test_idx = next(gss_test.split(np.zeros(n), groups=groups))

    gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_SEED)
    train_idx_rel, val_idx_rel = next(
        gss_val.split(np.zeros(len(trainval_idx)), groups=groups[trainval_idx])
    )
    train_idx = trainval_idx[train_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    split = np.array(["?"] * n, dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    return split


def main():
    from lightkurve import LightCurve

    labels = pd.read_csv(KOI_CLEAN_TABLE_PATH)

    raw_X, folded_X, y, kepids, koi_names = [], [], [], [], []
    cleaned_cache = {}  # kepid -> cleaned LightCurve, so a multi-KOI star isn't re-cleaned per row

    n_ok, n_no_cache, n_failed = 0, 0, 0
    for i, row in enumerate(labels.itertuples(), 1):
        kepid = int(row.kepid)
        cache_file = LIGHTCURVE_CACHE_DIR / f"KIC_{kepid}.npz"
        if not cache_file.exists():
            n_no_cache += 1
            continue

        try:
            if kepid not in cleaned_cache:
                d = np.load(cache_file)
                lc = LightCurve(time=d["time"], flux=d["flux"], flux_err=d["flux_err"])
                cleaned_cache[kepid] = clean_lightcurve(
                    lc, row.koi_period, row.koi_time0bk, row.koi_duration
                )
            lc_clean = cleaned_cache[kepid]

            raw_view = make_raw_view(lc_clean, row.koi_period, row.koi_time0bk,
                                      row.koi_duration, N_POINTS)
            folded_view = make_folded_view(lc_clean, row.koi_period, row.koi_time0bk, N_POINTS)

            if np.isnan(raw_view).any() or np.isnan(folded_view).any():
                n_failed += 1
                continue

            raw_X.append(raw_view)
            folded_X.append(folded_view)
            y.append(int(row.label))
            kepids.append(kepid)
            koi_names.append(row.kepoi_name)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            print(f"  [{i}] KIC {kepid} ({row.kepoi_name}) failed: {e}")

    print(f"\nBuilt {n_ok} examples. no_cached_lightcurve={n_no_cache} failed_processing={n_failed}")

    if n_ok == 0:
        print("Nothing to save -- run 03_download_lightcurves.py first.")
        return

    y = np.array(y, dtype=np.int64)
    kepids = np.array(kepids, dtype=np.int64)
    koi_names = np.array(koi_names, dtype=object)
    raw_X = np.stack(raw_X).astype(np.float32)
    folded_X = np.stack(folded_X).astype(np.float32)

    split = star_level_split(kepids)

    # Sanity check: no star should ever appear in more than one split.
    df_check = pd.DataFrame({"kepid": kepids, "split": split})
    offenders = df_check.groupby("kepid")["split"].nunique()
    assert (offenders == 1).all(), "Leakage bug: a star appears in more than one split!"

    for name, X in [("raw", raw_X), ("folded", folded_X)]:
        path = RAW_DATASET_PATH if name == "raw" else FOLDED_DATASET_PATH
        np.savez(path, X=X, y=y, kepid=kepids, koi_name=koi_names, split=split)
        print(f"Saved {name} dataset -> {path}  X.shape={X.shape}")

    print("\nSplit sizes (examples / unique stars):")
    for s in ["train", "val", "test"]:
        m = split == s
        print(f"  {s:5s}: {m.sum():4d} examples, {len(set(kepids[m]))} stars, "
              f"planet-rate={y[m].mean():.2f}")


if __name__ == "__main__":
    main()
