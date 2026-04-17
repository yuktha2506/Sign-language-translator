from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tensorflow.keras import callbacks, layers, models


BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV = BASE_DIR / "sign_mnist_train.csv"
TEST_CSV = BASE_DIR / "sign_mnist_test.csv"
MODEL_PATH = BASE_DIR / "sign_model.h5"
LABEL_MAP_PATH = BASE_DIR / "label_map.json"

RAW_LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
DISPLAY_LABELS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
]
LABEL_TO_INDEX = {label: index for index, label in enumerate(RAW_LABELS)}


def load_data(csv_path: Path):
    data = pd.read_csv(csv_path)
    labels = data["label"].map(LABEL_TO_INDEX)
    if labels.isnull().any():
        raise ValueError("Found unsupported labels in the dataset.")

    images = data.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
    return images.astype("float32"), labels.astype("int32").to_numpy()


def build_model():
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.08, 0.08),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.35),
            layers.Dense(len(DISPLAY_LABELS), activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_label_map() -> None:
    LABEL_MAP_PATH.write_text(
        json.dumps(
            {
                "raw_labels": RAW_LABELS,
                "labels": DISPLAY_LABELS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError(
            "Training data not found. Make sure sign_mnist_train.csv and sign_mnist_test.csv exist."
        )

    x_train, y_train = load_data(TRAIN_CSV)
    x_test, y_test = load_data(TEST_CSV)

    model = build_model()
    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        )
    ]

    model.fit(
        x_train,
        y_train,
        epochs=12,
        batch_size=64,
        validation_data=(x_test, y_test),
        callbacks=training_callbacks,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Validation accuracy: {accuracy:.2%}")
    print(f"Validation loss: {loss:.4f}")

    model.save(MODEL_PATH)
    save_label_map()
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved label map to {LABEL_MAP_PATH}")


if __name__ == "__main__":
    main()
