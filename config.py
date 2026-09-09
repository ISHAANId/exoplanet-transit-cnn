"""
Shared configuration for the exoplanet transit-detection CNN project.

Every script in scripts/ imports from here so that paths, constants, and the
random seed are defined in exactly one place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LIGHTCURVE_CACHE_DIR = DATA_DIR / "lightcurves"      # raw downloaded FITS, one folder per star
PROCESSED_DIR = DATA_DIR / "processed"                # cleaned + resampled numpy arrays
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

KOI_RAW_TABLE_PATH = DATA_DIR / "koi_raw.csv"
KOI_CLEAN_TABLE_PATH = DATA_DIR / "koi_clean_labels.csv"

RAW_DATASET_PATH = PROCESSED_DIR / "dataset_raw.npz"        # Experiment A input (no phase fold)
FOLDED_DATASET_PATH = PROCESSED_DIR / "dataset_folded.npz"  # Experiment B input (phase folded)

for d in (DATA_DIR, LIGHTCURVE_CACHE_DIR, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Light-curve processing
# ---------------------------------------------------------------------------
N_POINTS = 400          # every light curve is resampled to exactly this many points
OUTLIER_SIGMA = 5        # sigma-clipping threshold for outlier removal
FLATTEN_WINDOW = 401     # Savitzky-Golay window (in cadences) used to remove long-term trends

# For the *unfolded* ("raw") representation used in Experiment A, we still
# need a fixed-length window. We center it on the known transit epoch of the
# first transit in the observing baseline and take N_POINTS cadences around
# it. This keeps Experiment A "fair": it sees the same transit event(s) that
# Experiment B folds on, it just doesn't get the benefit of stacking every
# transit in the baseline on top of each other.
RAW_WINDOW_IN_DURATIONS = 6  # window width = 6x the transit duration, centered on one transit

# ---------------------------------------------------------------------------
# Data-leakage-prone columns in the NASA cumulative KOI table
# ---------------------------------------------------------------------------
# These columns are all direct outputs of NASA's own vetting/robovetting
# pipeline (human review + the "Robovetter" software). They are effectively
# a restatement of the label, or a strong proxy for it, computed by looking
# at the SAME transit signal (and more) that we are trying to classify from
# scratch. If any of these leak into training, the model can reach high
# accuracy without learning anything about the light curve shape.
#
# koi_disposition   -- the label itself (CONFIRMED / CANDIDATE / FALSE POSITIVE)
# koi_pdisposition  -- Kepler pipeline's own disposition (near-duplicate of the label)
# koi_score         -- the pipeline's own confidence that the KOI is a planet
# koi_fpflag_nt     -- "not transit-like" vetting flag
# koi_fpflag_ss     -- "stellar eclipse" vetting flag
# koi_fpflag_co     -- "centroid offset" vetting flag
# koi_fpflag_ec     -- "ephemeris match" (contamination) vetting flag
# koi_comment       -- free-text vetting notes, often literally says "FALSE POSITIVE"
# koi_disp_prov     -- provenance of the disposition decision
# koi_vet_stat      -- vetting status
# koi_vet_date      -- vetting date
# koi_datalink_dvr / koi_datalink_dvs -- links to the vetting report itself
LEAKAGE_COLUMNS = [
    "koi_disposition",
    "koi_pdisposition",
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_comment",
    "koi_disp_prov",
    "koi_vet_stat",
    "koi_vet_date",
    "koi_datalink_dvr",
    "koi_datalink_dvs",
]

# Columns we DO need, but only as physical parameters for phase-folding and
# for identifying/downloading the star -- never as CNN input features.
KEEP_COLUMNS = [
    "kepid",          # star ID (used for the star-level train/test split)
    "kepoi_name",      # KOI identifier (one row per candidate, a star can have several)
    "koi_period",      # orbital period, days -- needed to phase-fold
    "koi_time0bk",     # transit epoch (BKJD) -- needed to phase-fold
    "koi_duration",    # transit duration, hours -- needed to size the window / mask
]

# ---------------------------------------------------------------------------
# Train / validation / test split (by STAR, not by row -- see 04_build_dataset.py)
# ---------------------------------------------------------------------------
TEST_FRACTION = 0.2
VAL_FRACTION = 0.2  # fraction of the remaining train+val pool
