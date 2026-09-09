"""
Step 1a: Download the NASA Kepler KOI (Kepler Object of Interest) cumulative
table.

This table has one row per KOI (a "planet candidate slot" around a star) and
includes:
  - the disposition (CONFIRMED planet / CANDIDATE / FALSE POSITIVE)
  - orbital parameters (period, transit epoch, duration, depth)
  - the vetting flags NASA's pipeline used to reach that disposition

We query it straight from the NASA Exoplanet Archive's TAP service via
astroquery, so the data is always the current official table (no manually
downloaded CSV to go stale).

Run:
    python scripts/01_get_koi_table.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import KOI_RAW_TABLE_PATH  # noqa: E402


def main():
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    print("Querying NASA Exoplanet Archive: cumulative KOI table ...")
    table = NasaExoplanetArchive.query_criteria(table="cumulative", select="*")
    df = table.to_pandas()

    print(f"Downloaded {len(df)} KOI rows, {df.shape[1]} columns.")
    print("Disposition counts:")
    print(df["koi_disposition"].value_counts())

    df.to_csv(KOI_RAW_TABLE_PATH, index=False)
    print(f"Saved raw KOI table -> {KOI_RAW_TABLE_PATH}")


if __name__ == "__main__":
    main()
