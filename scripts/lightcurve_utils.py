"""
Shared light-curve processing functions used by 04_build_dataset.py.

Kept in its own module (rather than copy-pasted) because BOTH experiments
(A: raw, B: phase-folded) must start from the exact same cleaned light
curve -- that is what makes the A-vs-B comparison in step 8 a fair test of
"does phase-folding help", rather than a test of two different cleaning
pipelines.
"""

import numpy as np

from config import OUTLIER_SIGMA, FLATTEN_WINDOW, RAW_WINDOW_IN_DURATIONS


def clean_lightcurve(lc, period_days: float, epoch_bkjd: float, duration_hours: float):
    """Step 2: noise / outliers / trend / normalization.

    Order matters:
      1. remove outliers (cosmic rays, sudden pixel sensitivity dropouts)
         BEFORE flattening, so a single huge spike can't distort the
         trend fit.
      2. mask out the transits themselves before flattening. If we don't,
         the long-term trend fitter (Savitzky-Golay) partially fits AWAY
         the transit dips as if they were part of the trend, weakening
         exactly the signal we want the CNN to learn from.
      3. normalize AFTER flattening, so the final flux values are centered
         on 1.0 regardless of the star's absolute brightness -- this is
         what makes light curves from different stars comparable.
    """
    lc = lc.remove_outliers(sigma=OUTLIER_SIGMA)

    try:
        transit_mask = lc.create_transit_mask(
            period=period_days, transit_time=epoch_bkjd, duration=duration_hours / 24.0
        )
        lc = lc.flatten(window_length=FLATTEN_WINDOW, mask=transit_mask)
    except Exception:
        # Fall back to an unmasked flatten if the mask can't be built (e.g.
        # too few points) -- better than crashing the whole pipeline on one star.
        lc = lc.flatten(window_length=FLATTEN_WINDOW)

    lc = lc.normalize()
    return lc


def _resample_uniform(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Interpolate (x, y) onto x_grid, filling any point outside the observed
    range with the median flux (i.e. "nothing unusual happened here")."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    fill = float(np.nanmedian(y))
    return np.interp(x_grid, x, y, left=fill, right=fill)


def make_raw_view(lc, period_days: float, epoch_bkjd: float, duration_hours: float,
                   n_points: int) -> np.ndarray:
    """Experiment A input: a single transit, NOT folded.

    We center a window on the first transit inside the observing baseline
    and resample that window to n_points. The window width is a multiple of
    the transit duration so the dip is visible but the model only ever sees
    ONE transit event -- exactly the "raw/cleaned, no folding" condition the
    experiment is supposed to test.
    """
    time = lc.time.value
    flux = lc.flux.value

    t0 = time.min()
    n = np.floor((t0 - epoch_bkjd) / period_days)
    first_transit_time = epoch_bkjd + n * period_days
    if first_transit_time < t0:
        first_transit_time += period_days

    half_window = (RAW_WINDOW_IN_DURATIONS * duration_hours / 24.0) / 2.0
    grid = np.linspace(-half_window, half_window, n_points)
    return _resample_uniform(time - first_transit_time, flux, grid)


def make_folded_view(lc, period_days: float, epoch_bkjd: float,
                      n_points: int) -> np.ndarray:
    """Experiment B input: phase-fold on the known period/epoch, then collapse
    the WHOLE baseline into one period-length "global view" of n_points bins.

    Folding maps every cadence's time to phase = ((t - epoch) mod period),
    recentered to [-period/2, period/2). Every transit in the baseline lands
    on top of every other transit near phase 0, so binning across all of them
    (rather than resampling a single transit) is what averages down noise and
    is the entire point of folding.

    We use median binning (not plain linear interpolation) because it is
    robust to the occasional leftover outlier and naturally handles many
    points landing in the same phase bin, which is the normal case once a
    multi-quarter baseline is folded on a short period.
    """
    time = lc.time.value
    flux = lc.flux.value

    phase = np.mod(time - epoch_bkjd, period_days) / period_days  # in [0, 1)
    phase = np.where(phase > 0.5, phase - 1.0, phase)              # in [-0.5, 0.5)

    bin_edges = np.linspace(-0.5, 0.5, n_points + 1)
    bin_idx = np.digitize(phase, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_points - 1)

    binned = np.full(n_points, np.nan)
    for b in range(n_points):
        vals = flux[bin_idx == b]
        if len(vals) > 0:
            binned[b] = np.median(vals)

    # Fill any empty bins (gaps in coverage) by interpolating from neighbors.
    if np.isnan(binned).any():
        good = ~np.isnan(binned)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        binned[~good] = np.interp(bin_centers[~good], bin_centers[good], binned[good])

    return binned
