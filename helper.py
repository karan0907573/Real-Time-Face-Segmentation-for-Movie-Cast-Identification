"""Helper module for face segmentation prediction.

Contains all functions needed by the Streamlit app:
- Custom loss/metric definitions (needed to load the .keras model)
- Image preprocessing
- IoU computation
- Inference utilities
"""

import time
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image


# ── Custom metrics & losses (must match training definitions) ────────────────

def dice_coefficient(
    y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0
) -> tf.Tensor:
    """Compute Dice Coefficient: 2*|A∩B| / (|A|+|B|+smooth)."""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Combined Binary Cross-Entropy + Dice Loss (must match training)."""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    dice = 1.0 - dice_coefficient(y_true, y_pred)
    return bce + dice


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Dice Loss: 1 - dice_coefficient."""
    return 1.0 - dice_coefficient(y_true, y_pred)


# ── Model loading ────────────────────────────────────────────────────────────

CUSTOM_OBJECTS = {
    "dice_coefficient": dice_coefficient,
    "bce_dice_loss": bce_dice_loss,
    "dice_loss": dice_loss,
}


def load_keras_model(model_path: str) -> tf.keras.Model:
    """Load a .keras model with custom loss/metric objects registered."""
    return tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)


# ── Preprocessing ────────────────────────────────────────────────────────────

TARGET_SIZE = (256, 256)


def preprocess_image(
    image: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE
) -> np.ndarray:
    """Resize a single image to target_size and normalize to [0, 1].

    Args:
        image: uint8 or float array of shape (H, W, 3).
        target_size: (height, width) tuple.

    Returns:
        Float32 array of shape (target_h, target_w, 3) in [0, 1].
    """
    pil_img = Image.fromarray(
        np.uint8(image) if image.dtype != np.uint8 else image
    ).convert("RGB")
    target_h, target_w = target_size
    pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
    result = np.array(pil_img, dtype=np.float32)
    if result.max() > 1.0:
        result /= 255.0
    return result


def preprocess_images(
    images: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE
) -> np.ndarray:
    """Resize a batch of images and normalize to [0, 1].

    Args:
        images: Array of shape (N, H, W, 3).
        target_size: (height, width) tuple.

    Returns:
        Float32 array of shape (N, target_h, target_w, 3) in [0, 1].
    """
    return np.stack(
        [preprocess_image(img, target_size) for img in images], axis=0
    )


# ── Evaluation metrics ───────────────────────────────────────────────────────

def compute_iou(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-7
) -> float:
    """Compute Intersection over Union between two binary masks."""
    y_true_f = y_true.flatten().astype(np.float32)
    y_pred_f = y_pred.flatten().astype(np.float32)
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return float(intersection / (union + epsilon))


def measure_inference_time(
    model: tf.keras.Model, image: np.ndarray
) -> float:
    """Return inference time in milliseconds for a single image."""
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
    start = time.perf_counter()
    model.predict(image, verbose=0)
    end = time.perf_counter()
    return (end - start) * 1000.0


def compute_metrics(
    model: tf.keras.Model,
    val_images: np.ndarray,
    val_masks: np.ndarray,
) -> Dict[str, float]:
    """Compute Dice, IoU, F1, and avg inference time on a validation set."""
    dice_scores, iou_scores, f1_scores, times = [], [], [], []

    for i in range(len(val_images)):
        image = val_images[i]
        mask_true = val_masks[i].astype(np.float32)

        inf_time = measure_inference_time(model, image)
        times.append(inf_time)

        pred = model.predict(np.expand_dims(image, 0), verbose=0)[0]
        pred_binary = (pred >= 0.5).astype(np.float32)

        dice_val = float(
            dice_coefficient(tf.constant(mask_true), tf.constant(pred_binary)).numpy()
        )
        dice_scores.append(dice_val)
        iou_scores.append(compute_iou(mask_true, pred_binary))

        true_flat = mask_true.flatten()
        pred_flat = pred_binary.flatten()
        tp = np.sum(true_flat * pred_flat)
        fp = np.sum(pred_flat) - tp
        fn = np.sum(true_flat) - tp
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-7)
        f1_scores.append(float(f1))

    return {
        "dice_coefficient": float(np.mean(dice_scores)),
        "iou": float(np.mean(iou_scores)),
        "f1_score": float(np.mean(f1_scores)),
        "avg_inference_time_ms": float(np.mean(times)),
    }


# ── Prediction helpers (from facesegmentation_1000_data.py) ──────────────────

def predict_single_image(
    model: tf.keras.Model,
    image_path: str,
    target_size: Tuple[int, int] = TARGET_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run prediction on a single image file.

    Args:
        model: Loaded Keras model.
        image_path: Path to the image file.
        target_size: (height, width) for resizing.

    Returns:
        Tuple of (resized_image, prediction_mask, binary_mask).
        - resized_image: float32 (H, W, 3) in [0, 1]
        - prediction_mask: float32 (H, W, 1) raw sigmoid output
        - binary_mask: uint8 (H, W, 1) thresholded at 0.5
    """
    raw_img = Image.open(image_path).convert("RGB")
    target_h, target_w = target_size
    img_resized = raw_img.resize((target_w, target_h), Image.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0

    # Add batch dimension, predict, remove batch dimension
    prediction = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
    binary_mask = (prediction >= 0.5).astype(np.uint8)

    return img_array, prediction, binary_mask


def predict_from_array(
    model: tf.keras.Model,
    image: np.ndarray,
    target_size: Tuple[int, int] = TARGET_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run prediction on a numpy array or PIL-uploaded image.

    Args:
        model: Loaded Keras model.
        image: uint8 array (H, W, 3) or float32 already in [0, 1].
        target_size: (height, width) for resizing.

    Returns:
        Tuple of (preprocessed_image, prediction_mask, binary_mask).
    """
    preprocessed = preprocess_image(image, target_size)
    prediction = model.predict(np.expand_dims(preprocessed, axis=0), verbose=0)[0]
    binary_mask = (prediction >= 0.5).astype(np.uint8)

    return preprocessed, prediction, binary_mask
