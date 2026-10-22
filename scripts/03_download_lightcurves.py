"""
Step 1c: Download the actual Kepler light curves (brightness over time) for
every star (kepid) in the leakage-safe label table, using Lightkurve.

For each star:
  - search all available long-cadence quarters
  - download them
  - stitch them into one continuous light curve
  - cache the result to disk as a small .npz (time, flux, flux_err) so we
    never have to hit MAST again for that star

This is the slow, network-bound step of the whole project (Kepler quarters
are tens of MB each and MAST can be slow), so it is written to be safely
re-run: it skips any star that is already cached.

Run (small smoke test):
    python scripts/03_download_lightcurves.py --limit 10

Run (full dataset -- can take hours, run it in the background):
    python scripts/03_download_lightcurves.py
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import KOI_CLEAN_TABLE_PATH, LIGHTCURVE_CACHE_DIR, RANDOM_SEED  # noqa: E402

# A dropped MAST connection can hang the underlying socket call forever without
# raising (observed in practice: a run stalled 20+ hours on one star with no
# error). Bounding each star to a daemon thread with a join timeout means a
# hung star gets skipped instead of freezing the entire multi-hour download.
STAR_TIMEOUT_SECONDS = 300


def cache_path(kepid: int) -> Path:
    return LIGHTCURVE_CACHE_DIR / f"KIC_{kepid}.npz"


def download_one_star(kepid: int):
    """Search, download, and stitch every long-cadence quarter for one star.

    We use PDCSAP_FLUX (Pre-search Data Conditioning SAP flux): this is
    Kepler's own systematics-corrected flux, already partially cleaned of
    instrumental artifacts. Starting from PDCSAP_FLUX instead of raw SAP_FLUX
    is standard practice (Shallue & Vanderburg 2018) and it does NOT touch
    the astrophysical transit signal, so it isn't a form of leakage -- it's
    just a better-behaved "brightness over time" starting point for our own
    cleaning in the next step.
    """
    import lightkurve as lk

    search = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", author="Kepler",
                                   cadence="long")
    if len(search) == 0:
        return None

    lc_collection = search.download_all()
    if lc_collection is None or len(lc_collection) == 0:
        return None

    lc = lc_collection.stitch()  # normalizes each quarter and concatenates them
    lc = lc.remove_nans()

    if len(lc) == 0:
        return None

    return lc


def download_with_timeout(kepid: int, timeout: float = STAR_TIMEOUT_SECONDS):
    """Run download_one_star on a daemon thread; give up after `timeout` seconds
    instead of blocking forever on a hung connection. The thread is left to die
    on its own (it's a daemon, so it won't keep the process alive) rather than
    forcibly killed, which Python threads don't support."""
    outcome: "queue.Queue" = queue.Queue(maxsize=1)

    def worker():
        try:
            outcome.put(("ok", download_one_star(kepid)))
        except Exception as e:  # noqa: BLE001 -- MAST/network errors are varied and non-fatal
            outcome.put(("error", e))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return "timeout", None
    return outcome.get()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process this many stars (for a quick smoke test).")
    parser.add_argument("--sample-balanced", action="store_true",
                         help="When used with --limit, sample roughly half planet / "
                              "half false-alarm stars instead of taking the first N rows.")
    args = parser.parse_args()

    labels = pd.read_csv(KOI_CLEAN_TABLE_PATH)
    stars = labels.drop_duplicates(subset="kepid")[["kepid", "label"]]

    if args.limit is not None:
        if args.sample_balanced:
            n_each = args.limit // 2
            planets = stars[stars.label == 1].sample(n=n_each, random_state=RANDOM_SEED)
            falses = stars[stars.label == 0].sample(n=args.limit - n_each,
                                                      random_state=RANDOM_SEED)
            stars = pd.concat([planets, falses]).sample(frac=1, random_state=RANDOM_SEED)
        else:
            stars = stars.head(args.limit)

    print(f"Downloading light curves for {len(stars)} stars ...")

    ok, skipped_cached, failed = 0, 0, 0
    for i, row in enumerate(stars.itertuples(), 1):
        kepid = int(row.kepid)
        out_path = cache_path(kepid)
        if out_path.exists():
            skipped_cached += 1
            continue

        print(f"[{i}/{len(stars)}] KIC {kepid} ...", end=" ", flush=True)
        status, value = download_with_timeout(kepid)

        if status == "timeout":
            failed += 1
            print(f"TIMED OUT after {STAR_TIMEOUT_SECONDS}s, skipping.")
            time.sleep(2)
            continue
        if status == "error":
            failed += 1
            print(f"FAILED: {value}")
            time.sleep(2)  # be polite to MAST after an error before trying the next star
            continue

        lc = value
        if lc is None:
            print("no data available, skipping.")
            failed += 1
            continue

        try:
            np.savez(
                out_path,
                time=np.asarray(lc.time.value, dtype=np.float64),
                flux=np.asarray(lc.flux.value, dtype=np.float64),
                flux_err=np.asarray(lc.flux_err.value, dtype=np.float64),
            )
            ok += 1
            print(f"ok ({len(lc)} cadences).")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAILED to save: {e}")

    print(f"\nDone. downloaded={ok} already_cached={skipped_cached} failed={failed}")
    print(f"Cache directory: {LIGHTCURVE_CACHE_DIR}")


if __name__ == "__main__":
    main()
