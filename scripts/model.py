"""
Step 6: the 1D CNN architecture.

Shared by both experiments (raw and folded) so that any performance
difference we measure in step 8 comes from the INPUT representation, not
from two different models.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(n_points: int) -> keras.Model:
    """400-point light curve -> Conv1D/Pool blocks -> Dense -> sigmoid.

    Three conv/pool blocks (instead of the two sketched in the slide) give
    the network enough receptive field to relate a dip at one end of the
    400-point window to the flat baseline at the other end, while staying
    small enough to train on a few thousand examples without overfitting
    immediately. Dropout before the final dense layer is the main defense
    against overfitting given the dataset is small relative to typical CNNs.
    """
    inputs = keras.Input(shape=(n_points, 1), name="light_curve")

    x = layers.Conv1D(16, kernel_size=5, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(32, kernel_size=5, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=5, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="planet_probability")(x)

    model = keras.Model(inputs, outputs, name="transit_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def prepare(X):
    """Rescale light curves for the network.

    Raw values sit at ~1.0 and a planet's dip is only ~0.0001-0.01 deep, so fed
    directly the CNN sees an almost constant input and collapses to one answer.
    Subtract each curve's median and use arcsinh so shallow transits stay visible
    while deep eclipsing-binary dips (and outliers) are compressed, not dominant.
    The training script and the app must both call this so they agree.
    """
    import numpy as np
    X = np.asarray(X, dtype=np.float32)
    med = np.median(X, axis=-1, keepdims=True)
    return np.arcsinh((X - med) / 3e-4)
