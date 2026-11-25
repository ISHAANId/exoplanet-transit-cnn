"""
Step 9: compare five different model types on the same folded light curves, and
break each one's accuracy down by the host star's type (M, K, G, F).

All five see the same train/val/test split as the main experiment, so the
numbers are directly comparable. Writes results/model_zoo/metrics.json, which the
app reads.

Run:
    python scripts/07_model_zoo.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FOLDED_DATASET_PATH, MODELS_DIR, RESULTS_DIR, RANDOM_SEED, N_POINTS  # noqa: E402
from model import build_cnn, prepare  # noqa: E402
from planet_info import star_class, load_koi_table  # noqa: E402

KOI_RAW = Path(__file__).resolve().parent.parent / "data" / "koi_raw.csv"
MIN_GROUP = 20     # star-type groups smaller than this are flagged as unreliable

MODEL_INFO = {
    "ExoFinder CNN": "Our 3-layer 1D convolutional network. The model used everywhere else in the app.",
    "Deeper CNN": "Five convolution layers, pooled to the strongest dip anywhere in the curve. More capacity.",
    "GRU": "A recurrent network that reads the curve left to right, like a sequence.",
    "Random Forest": "300 decision trees voting on the curve's values. No deep learning.",
    "Two-view CNN": "Reads two pictures of each star: the whole orbit AND a zoomed-in look at just the dip, so it can judge the dip's true shape.",
    "Logistic Regression": "The simplest baseline: one weighted sum of the curve's values.",
}


def build_deeper_cnn():
    from tensorflow import keras
    from tensorflow.keras import layers
    x_in = keras.Input(shape=(N_POINTS, 1))
    x = x_in
    for filters in (16, 32, 64, 64, 128):
        x = layers.Conv1D(filters, 5, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalMaxPooling1D()(x)      # keeps the strongest dip response anywhere
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(x_in, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=["accuracy"])
    return m


def build_two_view_cnn(n_local=200):
    """Global view (400 pts, whole orbit) + local view (zoomed on the dip), two conv branches joined."""
    from tensorflow import keras
    from tensorflow.keras import layers

    def branch(n, filters):
        i = keras.Input(shape=(n, 1))
        x = i
        for f in filters:
            x = layers.Conv1D(f, 5, padding="same", activation="relu")(x)
            x = layers.MaxPooling1D(2)(x)
        return i, layers.Flatten()(x)

    gi, g = branch(N_POINTS, (16, 32, 64))
    li, l = branch(n_local, (16, 32, 64))
    x = layers.Concatenate()([g, l])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model([gi, li], out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m


def build_gru():
    from tensorflow import keras
    from tensorflow.keras import layers
    x_in = keras.Input(shape=(N_POINTS, 1))
    x = layers.AveragePooling1D(4)(x_in)           # 400 -> 100 steps so the GRU trains fast
    x = layers.GRU(48)(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(x_in, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=["accuracy"])
    return m


def fit_keras(model, Xtr, ytr, Xva, yva, cw):
    import tensorflow as tf
    model.fit(
        Xtr, ytr, validation_data=(Xva, yva), epochs=60, batch_size=32, class_weight=cw,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                     restore_best_weights=True)],
    )
    return model


def scores(y, prob):
    from sklearn.metrics import accuracy_score, roc_auc_score
    return {
        "accuracy": float(accuracy_score(y, (prob >= 0.5).astype(int))),
        "roc_auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else None,
    }


def main():
    import tensorflow as tf
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    d = np.load(FOLDED_DATASET_PATH, allow_pickle=True)
    X, y, split, names = d["X"], d["y"], d["split"], d["koi_name"]
    P = prepare(X)
    tr, va, te = split == "train", split == "val", split == "test"
    Xtr, ytr, Xva, yva, Xte, yte = P[tr], y[tr], P[va], y[va], P[te], y[te]

    n_pos, n_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    cw = {0: len(ytr) / (2 * n_neg), 1: len(ytr) / (2 * n_pos)}

    probs = {}

    cnn = tf.keras.models.load_model(MODELS_DIR / "folded_best.keras")
    probs["ExoFinder CNN"] = cnn.predict(Xte[..., None], verbose=0).ravel()

    deeper = fit_keras(build_deeper_cnn(), Xtr[..., None], ytr, Xva[..., None], yva, cw)
    probs["Deeper CNN"] = deeper.predict(Xte[..., None], verbose=0).ravel()

    gru = fit_keras(build_gru(), Xtr[..., None], ytr, Xva[..., None], yva, cw)
    probs["GRU"] = gru.predict(Xte[..., None], verbose=0).ravel()

    if "X_local" in d.files:
        PL = prepare(d["X_local"])
        two = fit_keras(build_two_view_cnn(PL.shape[1]), [Xtr[..., None], PL[tr][..., None]], ytr,
                        [Xva[..., None], PL[va][..., None]], yva, cw)
        probs["Two-view CNN"] = two.predict([Xte[..., None], PL[te][..., None]], verbose=0).ravel()

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=RANDOM_SEED, n_jobs=-1).fit(Xtr, ytr)
    probs["Random Forest"] = rf.predict_proba(Xte)[:, 1]

    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.05).fit(Xtr, ytr)
    probs["Logistic Regression"] = lr.predict_proba(Xte)[:, 1]

    koi = load_koi_table(KOI_RAW)
    classes = np.array([star_class(koi.loc[n, "koi_steff"]) if n in koi.index else "Unknown"
                        for n in names[te]])

    out = {"n_test": int(te.sum()), "min_group": MIN_GROUP, "models": {}, "star_types": {}}
    for t in ["M", "K", "G", "F", "A", "B/O"]:
        n = int((classes == t).sum())
        if n:
            out["star_types"][t] = n

    for name, prob in probs.items():
        entry = {"info": MODEL_INFO[name], **scores(yte, prob), "by_star_type": {}}
        for t, n in out["star_types"].items():
            m = classes == t
            entry["by_star_type"][t] = {
                "n": n,
                "accuracy": float(((prob[m] >= 0.5).astype(int) == yte[m]).mean()),
            }
        out["models"][name] = entry
        print(f"{name:20s} acc={entry['accuracy']:.2f} auc={entry['roc_auc']:.2f}  "
              + "  ".join(f"{t}:{v['accuracy']:.2f}" for t, v in entry["by_star_type"].items()))

    out_dir = RESULTS_DIR / "model_zoo"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
