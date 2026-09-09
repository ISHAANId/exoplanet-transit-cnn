"""
Step 7: train and evaluate the CNN on one of the two datasets built by
04_build_dataset.py.

Run Experiment A (no phase folding):
    python scripts/05_train_experiment.py --variant raw

Run Experiment B (phase folded):
    python scripts/05_train_experiment.py --variant folded

Each run:
  - loads the pre-split train/val/test arrays (split was fixed, by star, in
    04_build_dataset.py -- both experiments reuse the SAME split so the same
    stars are held out for testing in both, keeping the A/B comparison fair)
  - trains the CNN from model.py with early stopping on validation loss
  - evaluates once, on the untouched test set
  - saves the trained model, training curves, confusion matrix, and a
    metrics.json with accuracy/precision/recall/F1/ROC-AUC

It deliberately does NOT hand-wave any numbers: if you haven't run
03_download_lightcurves.py and 04_build_dataset.py with enough stars yet,
this script has nothing to train on and will tell you so instead of printing
placeholder results.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    RAW_DATASET_PATH, FOLDED_DATASET_PATH, RESULTS_DIR, MODELS_DIR, RANDOM_SEED, N_POINTS,
)
from model import build_cnn  # noqa: E402


def load_split(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    X, y, split = d["X"], d["y"], d["split"]
    out = {}
    for s in ["train", "val", "test"]:
        m = split == s
        out[s] = (X[m][..., np.newaxis], y[m])  # add channel dim -> (N, n_points, 1)
    return out


def main():
    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["raw", "folded"], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    dataset_path = RAW_DATASET_PATH if args.variant == "raw" else FOLDED_DATASET_PATH
    if not dataset_path.exists():
        print(f"{dataset_path} does not exist yet. Run 04_build_dataset.py first.")
        return

    out_dir = RESULTS_DIR / f"experiment_{'A_raw' if args.variant == 'raw' else 'B_folded'}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_split(dataset_path)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data["train"], data["val"], data["test"]
    print(f"[{args.variant}] train={len(y_train)} val={len(y_val)} test={len(y_test)}")

    if len(y_train) < 6 or len(np.unique(y_train)) < 2:
        print("Not enough training data / only one class present to train at all. "
              "Run 03_download_lightcurves.py on more stars first.")
        return
    if len(y_train) < 50:
        print(f"NOTE: training on only {len(y_train)} examples (smoke test). Metrics below "
              f"are a proof-of-concept that the pipeline works end-to-end, not a statistically "
              f"reliable measurement -- see PROJECT_LOG.md.")

    # Class imbalance is common (more false positives than confirmed planets,
    # or vice versa depending on the sample) -- weight the loss so the CNN
    # isn't rewarded for just predicting the majority class every time.
    n_pos, n_neg = (y_train == 1).sum(), (y_train == 0).sum()
    class_weight = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}

    model = build_cnn(N_POINTS)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                          restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(MODELS_DIR / f"{args.variant}_best.keras"),
                                            monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # ---- Final, single evaluation on the held-out test stars ----
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "variant": args.variant,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else None,
    }

    print("\n=== TEST SET RESULTS (never seen during training) ===")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, y_pred, target_names=["False Alarm", "Planet"]))

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1], ["False Alarm", "Planet"])
    ax.set_yticks([0, 1], ["False Alarm", "Planet"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix ({args.variant})")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)

    pd.DataFrame(history.history).to_csv(out_dir / "training_history.csv", index=False)

    print(f"\nSaved model, metrics, and plots -> {out_dir}")


if __name__ == "__main__":
    main()
