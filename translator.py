from __future__ import annotations

import json
import logging
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sign_model.h5"
LABEL_MAP_PATH = BASE_DIR / "label_map.json"
OUTPUT_PATH = BASE_DIR / "translated_text.txt"
LOG_PATH = BASE_DIR / "translator.log"

DEFAULT_LABELS = [
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

MIN_CONFIDENCE = 0.75
HISTORY_SIZE = 12
AUTO_APPEND_FRAMES = 8
RESET_FRAMES = 4
HAND_PADDING = 0.30
ROI_SIZE = 28
PREVIEW_SIZE = 180
MIN_CONTOUR_AREA_RATIO = 0.02


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("sign_translator")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_labels() -> list[str]:
    if LABEL_MAP_PATH.exists():
        saved = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
        labels = saved.get("labels")
        if isinstance(labels, list) and labels:
            return [str(item) for item in labels]
    return DEFAULT_LABELS


def validate_model_labels(model, labels: list[str]) -> None:
    output_shape = getattr(model, "output_shape", None)
    if not output_shape or output_shape[-1] is None:
        raise ValueError("Model output shape is not available for label validation.")

    class_count = int(output_shape[-1])
    if len(labels) != class_count:
        raise ValueError(
            f"Label count mismatch: model outputs {class_count} classes but "
            f"label_map provides {len(labels)} labels."
        )


def expand_square_box(
    left: int,
    top: int,
    right: int,
    bottom: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    box_width = right - left
    box_height = bottom - top
    side = int(max(box_width, box_height) * (1.0 + HAND_PADDING * 2))
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


def detect_hand_region(frame: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    min_area = frame.shape[0] * frame.shape[1] * MIN_CONTOUR_AREA_RATIO
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if not valid_contours:
        return None, None, thresh

    largest_contour = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    left, top, right, bottom = expand_square_box(
        x,
        y,
        x + w,
        y + h,
        frame.shape[1],
        frame.shape[0],
    )
    roi = frame[top:bottom, left:right]
    return roi, (left, top, right, bottom), thresh


def preprocess_hand_roi(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if roi.size == 0:
        raise ValueError("Detected hand ROI is empty.")

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
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

    # FIX: feed `thresh` (binary mask) into the model instead of raw `gray`.
    # This matches the preprocessing used during external-image domain adaptation
    # in train_model.py and ensures consistent train/inference input distribution.
    resized_thresh = cv2.resize(thresh, (ROI_SIZE, ROI_SIZE))
    normalized = resized_thresh.astype("float32") / 255.0
    model_input = normalized.reshape(1, ROI_SIZE, ROI_SIZE, 1)

    # Preview: left panel = equalized gray, right panel = binary mask
    preview_left = cv2.resize(gray, (PREVIEW_SIZE, PREVIEW_SIZE), interpolation=cv2.INTER_NEAREST)
    preview_right = cv2.resize(thresh, (PREVIEW_SIZE, PREVIEW_SIZE), interpolation=cv2.INTER_NEAREST)
    preview = np.hstack([preview_left, preview_right])
    return model_input, preview


def draw_help(
    frame: np.ndarray,
    sentence: str,
    prediction: str,
    confidence: float,
    status: str,
    preview: np.ndarray | None,
) -> None:
    help_lines = [
        "Hold one hand up clearly. The detector crops the hand automatically.",
        "SPACE=add space  B=backspace  C=clear  S=save  Q=quit",
    ]

    cv2.putText(frame, f"Prediction: {prediction}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 220, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 0), 2)
    cv2.putText(frame, f"Status: {status}", (20, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 200, 255), 2)
    cv2.putText(frame, f"Text: {sentence or '(empty)'}", (20, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    y = 430
    for line in help_lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2)
        y += 30

    if preview is not None:
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        px, py = 20, 185
        h, w = preview_bgr.shape[:2]
        frame[py : py + h, px : px + w] = preview_bgr
        cv2.rectangle(frame, (px, py), (px + w, py + h), (255, 255, 0), 2)
        cv2.putText(frame, "Crop / Mask", (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)


def main() -> None:
    logger = configure_logging()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")

    labels = load_labels()
    model = load_model(MODEL_PATH)
    validate_model_labels(model, labels)

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Could not access the webcam. Check your camera permissions.")

    history: deque[str] = deque(maxlen=HISTORY_SIZE)
    sentence = ""
    stable_prediction = "-"
    stable_confidence = 0.0
    preview = None
    last_saved_message = ""
    status = "Show one hand to the camera"
    committed_current_hold = False
    no_hand_frames = RESET_FRAMES
    last_error_message = ""

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                logger.warning("Webcam frame read failed.")
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (960, 540))

            try:
                roi, box, detection_mask = detect_hand_region(frame)
                if roi is None or box is None:
                    history.clear()
                    stable_prediction = "-"
                    stable_confidence = 0.0
                    no_hand_frames += 1
                    preview = cv2.resize(detection_mask, (PREVIEW_SIZE * 2, PREVIEW_SIZE), interpolation=cv2.INTER_NEAREST)
                    if no_hand_frames >= RESET_FRAMES:
                        committed_current_hold = False
                    status = "No hand detected. Raise one hand clearly."
                else:
                    left, top, right, bottom = box
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
                    processed_input, preview = preprocess_hand_roi(roi)

                    prediction = model.predict(processed_input, verbose=0)[0]
                    confidence = float(np.max(prediction))
                    class_index = int(np.argmax(prediction))
                    letter = labels[class_index]

                    if confidence >= MIN_CONFIDENCE:
                        history.append(letter)
                        stable_prediction, stable_count = Counter(history).most_common(1)[0]
                        stable_confidence = confidence
                        no_hand_frames = 0

                        if stable_count >= AUTO_APPEND_FRAMES and not committed_current_hold:
                            sentence += stable_prediction
                            committed_current_hold = True
                            status = f"Added '{stable_prediction}'. Lower your hand for the next letter."
                            last_saved_message = ""
                        elif committed_current_hold:
                            status = "Letter captured. Lower your hand, then show the next sign."
                        else:
                            status = "Hand found. Hold the sign steady."
                    else:
                        history.clear()
                        stable_prediction = "-"
                        stable_confidence = confidence
                        no_hand_frames += 1
                        status = "Prediction uncertain. Keep the hand centered and steady."
                        if no_hand_frames >= RESET_FRAMES:
                            committed_current_hold = False

            except (cv2.error, ValueError, RuntimeError) as exc:
                history.clear()
                stable_prediction = "-"
                stable_confidence = 0.0
                no_hand_frames += 1
                preview = None
                status = "Frame processing failed. See translator.log."
                if str(exc) != last_error_message:
                    logger.exception("Frame processing error")
                    last_error_message = str(exc)

            draw_help(
                frame=frame,
                sentence=sentence,
                prediction=stable_prediction,
                confidence=stable_confidence,
                status=status,
                preview=preview,
            )

            if last_saved_message:
                cv2.putText(frame, last_saved_message, (20, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.imshow("Sign Language To Text Translator", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord(" "):
                if sentence and not sentence.endswith(" "):
                    sentence += " "
                    committed_current_hold = True
                    last_saved_message = ""
            elif key == ord("b"):
                sentence = sentence[:-1]
                last_saved_message = ""
            elif key == ord("c"):
                sentence = ""
                last_saved_message = ""
            elif key == ord("s"):
                OUTPUT_PATH.write_text(sentence.strip(), encoding="utf-8")
                last_saved_message = f"Saved to {OUTPUT_PATH.name}"

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
