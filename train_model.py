from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras import callbacks, layers, models, optimizers


BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV = BASE_DIR / "sign_mnist_train.csv"
TEST_CSV = BASE_DIR / "sign_mnist_test.csv"
MODEL_PATH = BASE_DIR / "sign_model.h5"
LABEL_MAP_PATH = BASE_DIR / "label_map.json"
EXTERNAL_TEST_DIR = Path(r"C:\Users\Admin\Downloads\archive (1)\asl_alphabet_test")
ZERO_AS_O_DIR = Path(r"C:\Users\Admin\Downloads\archive (2)\Sign Language for Numbers\0")

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
DISPLAY_TO_INDEX = {label: index for index, label in enumerate(DISPLAY_LABELS)}
IGNORED_EXTERNAL_LABELS = {"J", "Z", "SPACE", "NOTHING"}


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


def expand_square_box(
    left: int,
    top: int,
    right: int,
    bottom: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    width = right - left
    height = bottom - top
    side = int(max(width, height) * 1.6)
    side = max(side, 40)

    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    left = max(center_x - side // 2, 0)
    top = max(center_y - side // 2, 0)
    right = min(left + side, frame_width)
    bottom = min(top + side, frame_height)

    if right - left < side:
        left = max(right - side, 0)
    if bottom - top < side:
        top = max(bottom - side, 0)

    return left, top, right, bottom


def crop_largest_hand_like_region(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    left, top, right, bottom = expand_square_box(
        x,
        y,
        x + w,
        y + h,
        image.shape[1],
        image.shape[0],
    )
    cropped = image[top:bottom, left:right]
    return cropped if cropped.size else image


def preprocess_external_roi(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)
    resized_gray = cv2.resize(gray, (28, 28))
    return (resized_gray.astype("float32") / 255.0).reshape(28, 28, 1)


def load_external_images(external_dir: Path):
    if not external_dir.exists():
        print(f"External image folder not found at {external_dir}. Skipping domain adaptation.")
        return np.empty((0, 28, 28, 1), dtype="float32"), np.empty((0,), dtype="int32")

    images: list[np.ndarray] = []
    labels: list[int] = []
    skipped: list[str] = []

    for image_path in sorted(external_dir.glob("*_test.jpg")):
        label_name = image_path.stem.replace("_test", "").upper()
        if label_name in IGNORED_EXTERNAL_LABELS:
            skipped.append(label_name)
            continue
        if label_name not in DISPLAY_TO_INDEX:
            skipped.append(label_name)
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            skipped.append(label_name)
            continue

        roi = crop_largest_hand_like_region(image)
        processed = preprocess_external_roi(roi)
        images.append(processed)
        labels.append(DISPLAY_TO_INDEX[label_name])

    if skipped:
        print(f"Ignored external labels without matching classes: {sorted(set(skipped))}")

    if not images:
        return np.empty((0, 28, 28, 1), dtype="float32"), np.empty((0,), dtype="int32")

    return np.stack(images).astype("float32"), np.array(labels, dtype="int32")


def load_zero_as_o_images(zero_dir: Path):
    if not zero_dir.exists():
        print(f"Zero-sign folder not found at {zero_dir}. Skipping extra O-class adaptation.")
        return np.empty((0, 28, 28, 1), dtype="float32"), np.empty((0,), dtype="int32")

    images: list[np.ndarray] = []
    labels: list[int] = []
    o_index = DISPLAY_TO_INDEX["O"]

    for image_path in sorted(zero_dir.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        roi = crop_largest_hand_like_region(image)
        processed = preprocess_external_roi(roi)
        images.append(processed)
        labels.append(o_index)

    if not images:
        return np.empty((0, 28, 28, 1), dtype="float32"), np.empty((0,), dtype="int32")

    print(f"Loaded {len(images)} zero-sign images and mapped them to class O.")
    return np.stack(images).astype("float32"), np.array(labels, dtype="int32")


def augment_external_samples(x_extra: np.ndarray, y_extra: np.ndarray, copies_per_image: int = 64):
    if len(x_extra) == 0:
        return x_extra, y_extra

    rng = np.random.default_rng(42)
    augmented_images = []
    augmented_labels = []

    for image, label in zip(x_extra, y_extra):
        base = (image[:, :, 0] * 255.0).astype(np.uint8)
        for _ in range(copies_per_image):
            angle = float(rng.uniform(-18, 18))
            scale = float(rng.uniform(0.88, 1.12))
            tx = float(rng.uniform(-3, 3))
            ty = float(rng.uniform(-3, 3))
            matrix = cv2.getRotationMatrix2D((14, 14), angle, scale)
            matrix[0, 2] += tx
            matrix[1, 2] += ty
            warped = cv2.warpAffine(
                base,
                matrix,
                (28, 28),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            alpha = float(rng.uniform(0.9, 1.15))
            beta = float(rng.uniform(-12, 12))
            adjusted = cv2.convertScaleAbs(warped, alpha=alpha, beta=beta)
            if rng.random() > 0.5:
                adjusted = cv2.GaussianBlur(adjusted, (3, 3), 0)

            augmented_images.append((adjusted.astype("float32") / 255.0).reshape(28, 28, 1))
            augmented_labels.append(label)

    all_images = np.concatenate([x_extra, np.stack(augmented_images).astype("float32")], axis=0)
    all_labels = np.concatenate([y_extra, np.array(augmented_labels, dtype="int32")], axis=0)
    return all_images, all_labels


def evaluate_external_images(model, x_extra: np.ndarray, y_extra: np.ndarray) -> None:
    if len(x_extra) == 0:
        return

    predictions = model.predict(x_extra, verbose=0)
    predicted_indices = np.argmax(predictions, axis=1)
    accuracy = float(np.mean(predicted_indices == y_extra))
    print(f"External image accuracy: {accuracy:.2%} on {len(x_extra)} images")

    for truth, pred, score in zip(y_extra, predicted_indices, np.max(predictions, axis=1)):
        if truth != pred:
            print(
                f"Misclassified {DISPLAY_LABELS[truth]} as {DISPLAY_LABELS[pred]} "
                f"({score:.1%} confidence)"
            )


def main() -> None:
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError(
            "Training data not found. Make sure sign_mnist_train.csv and sign_mnist_test.csv exist."
        )

    x_train, y_train = load_data(TRAIN_CSV)
    x_test, y_test = load_data(TEST_CSV)
    x_extra, y_extra = load_external_images(EXTERNAL_TEST_DIR)
    x_zero_o, y_zero_o = load_zero_as_o_images(ZERO_AS_O_DIR)

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
        verbose=1,
    )

    base_loss, base_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Base validation accuracy: {base_accuracy:.2%}")
    print(f"Base validation loss: {base_loss:.4f}")

    domain_x_parts = []
    domain_y_parts = []
    if len(x_extra) > 0:
        domain_x_parts.append(x_extra)
        domain_y_parts.append(y_extra)
    if len(x_zero_o) > 0:
        domain_x_parts.append(x_zero_o)
        domain_y_parts.append(y_zero_o)

    if domain_x_parts:
        x_domain = np.concatenate(domain_x_parts, axis=0)
        y_domain = np.concatenate(domain_y_parts, axis=0)
        print(f"Loaded {len(x_domain)} real-image samples for domain adaptation.")
        x_extra_aug, y_extra_aug = augment_external_samples(x_domain, y_domain, copies_per_image=96)

        sample_size = min(len(x_train), max(len(x_extra_aug) // 4, len(x_extra)))
        sample_indices = np.random.default_rng(123).choice(len(x_train), size=sample_size, replace=False)
        x_mix = np.concatenate([x_train[sample_indices], x_extra_aug], axis=0)
        y_mix = np.concatenate([y_train[sample_indices], y_extra_aug], axis=0)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            x_mix,
            y_mix,
            epochs=6,
            batch_size=32,
            validation_data=(x_test, y_test),
            callbacks=[
                callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=2,
                    restore_best_weights=True,
                )
            ],
            verbose=1,
        )

        evaluate_external_images(model, x_extra, y_extra)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final validation accuracy: {accuracy:.2%}")
    print(f"Final validation loss: {loss:.4f}")

    model.save(MODEL_PATH)
    save_label_map()
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved label map to {LABEL_MAP_PATH}")


if __name__ == "__main__":
    main()
