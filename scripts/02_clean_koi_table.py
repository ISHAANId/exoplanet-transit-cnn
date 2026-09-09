"""
Step 1b / Step 4 (leakage prevention, tabular side): turn the raw KOI table
into a small, leakage-safe table of exactly what later steps are allowed to
use: an identifier, the physical parameters needed for phase-folding, and a
binary label.

Why this is its own script (and its own saved CSV):
    Keeping the "label / metadata" table separate from the "raw archive
    table" makes the leakage boundary explicit and inspectable. Every later
    script reads ONLY koi_clean_labels.csv, which physically cannot contain
    koi_score, the fpflags, koi_pdisposition, etc. because we never wrote
    them into it. See config.LEAKAGE_COLUMNS for the full list and the
    reasoning for each one.

Label design:
    koi_disposition has three values: CONFIRMED, CANDIDATE, FALSE POSITIVE.
    CANDIDATE means "not yet vetted to a final answer" -- it is neither a
    confirmed planet nor a confirmed false alarm, so training a binary
    Planet/False-Alarm classifier on it would inject label noise. We drop
    CANDIDATE rows and keep a clean binary problem:
        CONFIRMED       -> label 1 ("Planet")
        FALSE POSITIVE  -> label 0 ("False Alarm")

Run:
    python scripts/02_clean_koi_table.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import KOI_RAW_TABLE_PATH, KOI_CLEAN_TABLE_PATH, KEEP_COLUMNS  # noqa: E402


def main():
    df = pd.read_csv(KOI_RAW_TABLE_PATH)
    print(f"Loaded {len(df)} raw KOI rows.")

    # Binary label, built ONCE, here, from the disposition column -- which is
    # then thrown away and never written to the clean table.
    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
    df["label"] = (df["koi_disposition"] == "CONFIRMED").astype(int)

    keep = KEEP_COLUMNS + ["label"]
    clean = df[keep].dropna(subset=["kepid", "koi_period", "koi_time0bk", "koi_duration"])

    print(f"Kept {len(clean)} rows after dropping CANDIDATEs and rows with missing "
          f"orbital parameters.")
    print(f"Planet (1) vs False Alarm (0):\n{clean['label'].value_counts()}")
    print(f"Unique host stars (kepid): {clean['kepid'].nunique()}")

    clean.to_csv(KOI_CLEAN_TABLE_PATH, index=False)
    print(f"Saved leakage-safe label table -> {KOI_CLEAN_TABLE_PATH}")
    print(f"Columns in this file: {list(clean.columns)}")
    print("(Notice koi_score, koi_pdisposition, koi_fpflag_* etc. are NOT in this list.)")


if __name__ == "__main__":
    main()
