"""
Step 10: repeat the whole comparison over many random star-level splits.

One train/test split can be lucky or unlucky. Here we re-split the stars N times
(by star, so no star is ever in two splits), retrain every model each time, and
report the average and spread. This tells us which differences are real and
which are noise, including the main A (raw) vs B (folded) claim.

Writes results/repeated_splits/metrics.json

Run:
    python scripts/08_repeated_splits.py            (10 splits)
    python scripts/08_repeated_splits.py --seeds 3  (quick check)
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from config import RAW_DATASET_PATH, FOLDED_DATASET_PATH, RESULTS_DIR, N_POINTS  # noqa: E402
from model import build_cnn, prepare  # noqa: E402

spec = importlib.util.spec_from_file_location("zoo", HERE / "07_model_zoo.py")
zoo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zoo)


def split_by_star(kepid, seed, test_frac=0.2, val_frac=0.2):
    from sklearn.model_selection import GroupShuffleSplit
    n = len(kepid)
    tv, te = next(GroupShuffleSplit(1, test_size=test_frac, random_state=seed).split(np.zeros(n), groups=kepid))
    tr_rel, va_rel = next(GroupShuffleSplit(1, test_size=val_frac, random_state=seed).split(np.zeros(len(tv)), groups=kepid[tv]))
    return tv[tr_rel], tv[va_rel], te


def fit_predict_keras(build, Xtr, ytr, Xva, yva, Xte, cw):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    m = build()
    zoo.fit_keras(m, Xtr[..., None], ytr, Xva[..., None], yva, cw)
    return m.predict(Xte[..., None], verbose=0).ravel()


def main():
    import tensorflow as tf
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=str(RESULTS_DIR / "repeated_splits"))
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    fo = np.load(FOLDED_DATASET_PATH, allow_pickle=True)
    ra = np.load(RAW_DATASET_PATH, allow_pickle=True)
    assert (fo["koi_name"] == ra["koi_name"]).all(), "raw and folded rows are not aligned"
    y, kep = fo["y"], fo["kepid"]
    Pf, Pr = prepare(fo["X"]), prepare(ra["X"])

    builders = {
        "ExoFinder CNN (folded)": lambda: build_cnn(N_POINTS),
        "Deeper CNN": zoo.build_deeper_cnn,
        "GRU": zoo.build_gru,
    }
    PL = prepare(fo["X_local"]) if "X_local" in fo.files else None
    names = ["ExoFinder CNN (raw)", *builders, *(["Two-view CNN"] if PL is not None else []),
             "Random Forest", "Logistic Regression"]
    acc = {n: [] for n in names}
    auc = {n: [] for n in names}

    from sklearn.metrics import accuracy_score, roc_auc_score
    for seed in range(args.seeds):
        tf.random.set_seed(seed); np.random.seed(seed)
        tr, va, te = split_by_star(kep, seed)
        ytr, yva, yte = y[tr], y[va], y[te]
        cw = {0: len(ytr) / (2 * (ytr == 0).sum()), 1: len(ytr) / (2 * (ytr == 1).sum())}
        probs = {}
        probs["ExoFinder CNN (raw)"] = fit_predict_keras(lambda: build_cnn(N_POINTS), Pr[tr], ytr, Pr[va], yva, Pr[te], cw)
        for n, b in builders.items():
            probs[n] = fit_predict_keras(b, Pf[tr], ytr, Pf[va], yva, Pf[te], cw)
        if PL is not None:
            import tensorflow as tf2
            tf2.keras.backend.clear_session()
            tv = zoo.build_two_view_cnn(PL.shape[1])
            zoo.fit_keras(tv, [Pf[tr][..., None], PL[tr][..., None]], ytr, [Pf[va][..., None], PL[va][..., None]], yva, cw)
            probs["Two-view CNN"] = tv.predict([Pf[te][..., None], PL[te][..., None]], verbose=0).ravel()
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1).fit(Pf[tr], ytr)
        probs["Random Forest"] = rf.predict_proba(Pf[te])[:, 1]
        lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.05).fit(Pf[tr], ytr)
        probs["Logistic Regression"] = lr.predict_proba(Pf[te])[:, 1]
        for n in names:
            acc[n].append(float(accuracy_score(yte, probs[n] >= 0.5)))
            auc[n].append(float(roc_auc_score(yte, probs[n])))
        print(f"split {seed + 1}/{args.seeds}: " + ", ".join(f"{n} {acc[n][-1]:.3f}" for n in names), flush=True)

    summary = {"n_splits": args.seeds, "n_examples": int(len(y)), "n_stars": int(len(set(map(int, kep)))), "models": {}}
    for n in names:
        a, u = np.array(acc[n]), np.array(auc[n])
        summary["models"][n] = {
            "accuracy_mean": float(a.mean()), "accuracy_std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "accuracy_min": float(a.min()), "accuracy_max": float(a.max()),
            "auc_mean": float(u.mean()), "auc_std": float(u.std(ddof=1)) if len(u) > 1 else 0.0,
            "accuracy_per_split": acc[n], "auc_per_split": auc[n],
        }
    # Paired comparisons: how often does the folded CNN beat raw, and the deeper CNN beat the original?
    def wins(a, b):
        d = np.array(acc[a]) - np.array(acc[b])
        return {"mean_diff": float(d.mean()), "splits_a_better": int((d > 0).sum()), "of": int(len(d))}
    summary["folded_vs_raw"] = wins("ExoFinder CNN (folded)", "ExoFinder CNN (raw)")
    summary["deeper_vs_original"] = wins("Deeper CNN", "ExoFinder CNN (folded)")
    if PL is not None:
        summary["twoview_vs_original"] = wins("Two-view CNN", "ExoFinder CNN (folded)")
    json.dump(summary, open(out_dir / "metrics.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "models"}, indent=2))
    for n in names:
        m = summary["models"][n]
        print(f"{n:26s} acc {m['accuracy_mean']:.3f} +/- {m['accuracy_std']:.3f}   auc {m['auc_mean']:.3f} +/- {m['auc_std']:.3f}")


if __name__ == "__main__":
    main()
