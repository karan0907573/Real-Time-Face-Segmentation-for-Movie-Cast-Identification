"""Face Segmentation Cast ID - Streamlit Web Application.

Loads a trained .keras U-Net model and provides:
- Image upload and inference pipeline
- Detection log with CSV export
- Performance dashboard
"""

import io
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.ndimage import label

from helper import (
    compute_iou,
    dice_coefficient,
    dice_loss,
    load_keras_model,
    preprocess_images,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "model/face_segmentation_model_10000.keras"
TARGET_SIZE = (256, 256)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    if "detection_log" not in st.session_state:
        st.session_state.detection_log = []


# ---------------------------------------------------------------------------
# Model loading (Task 7.1)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained .keras model with custom objects."""
    try:
        model = load_keras_model(MODEL_PATH)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
        return model
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Upload handler and inference (Task 7.2)
# ---------------------------------------------------------------------------

def handle_upload(uploaded_file) -> np.ndarray:
    """Validate and preprocess an uploaded image file.

    Returns a float32 array of shape (256, 256, 3) with values in [0, 1].
    Raises ValueError for unsupported or corrupted files.
    """
    allowed = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp", "image/tiff", "image/avif"}
    if uploaded_file.type not in allowed:
        raise ValueError(
            f"Unsupported file type '{uploaded_file.type}'. "
            "Please upload a JPEG or PNG image."
        )

    try:
        pil_img = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not open image: {exc}") from exc

    img_array = np.array(pil_img, dtype=np.uint8)
    # preprocess_images expects shape (N, H, W, 3)
    preprocessed = preprocess_images(np.expand_dims(img_array, axis=0), TARGET_SIZE)
    return preprocessed[0]  # (256, 256, 3)


def run_inference(model, image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Run model inference on a single preprocessed image.

    Args:
        model: Loaded Keras model.
        image: Float32 array of shape (256, 256, 3).

    Returns:
        Tuple of (predicted_mask, processing_time_ms).
        predicted_mask has shape (256, 256, 1) with values in [0, 1].
    """
    batch = np.expand_dims(image, axis=0)  # (1, 256, 256, 3)
    start = time.perf_counter()
    pred = model.predict(batch, verbose=0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return pred[0], elapsed_ms  # (256, 256, 1), float


def compute_confidence(pred_mask: np.ndarray) -> float:
    """Mean predicted probability within face regions (values > 0.5)."""
    face_pixels = pred_mask[pred_mask > 0.5]
    if len(face_pixels) == 0:
        return 0.0
    return float(np.mean(face_pixels))


def count_face_regions(pred_mask: np.ndarray) -> int:
    """Count connected face regions in the binary predicted mask."""
    binary = (pred_mask.squeeze() >= 0.5).astype(np.uint8)
    _, num_features = label(binary)
    return int(num_features)


def create_overlay(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend original image with a semi-transparent red mask overlay.

    Args:
        original: Float32 (256, 256, 3) in [0, 1].
        mask: Float32 (256, 256, 1) binary mask.

    Returns:
        uint8 RGB array (256, 256, 3).
    """
    overlay = (original * 255).astype(np.uint8).copy()
    face_region = mask.squeeze() >= 0.5
    overlay[face_region, 0] = np.clip(
        overlay[face_region, 0].astype(np.int32) + 100, 0, 255
    ).astype(np.uint8)
    overlay[face_region, 1] = (overlay[face_region, 1] * 0.5).astype(np.uint8)
    overlay[face_region, 2] = (overlay[face_region, 2] * 0.5).astype(np.uint8)
    return overlay


def display_results(original: np.ndarray, mask: np.ndarray) -> None:
    """Display original image, predicted mask, and overlay side by side."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image((original * 255).astype(np.uint8), use_column_width=True)

    with col2:
        st.subheader("Predicted Mask")
        mask_display = (mask.squeeze() * 255).astype(np.uint8)
        st.image(mask_display, use_column_width=True, clamp=True)

    with col3:
        st.subheader("Overlay")
        overlay = create_overlay(original, mask)
        st.image(overlay, use_column_width=True)


# ---------------------------------------------------------------------------
# Detection log (Task 7.3)
# ---------------------------------------------------------------------------

def log_detection(
    filename: str,
    processing_time_ms: float,
    num_face_regions: int,
    confidence_score: float,
) -> None:
    """Append a Detection_Log entry to session state."""
    entry = {
        "filename": filename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processing_time_ms": round(processing_time_ms, 2),
        "num_face_regions": num_face_regions,
        "confidence_score": round(confidence_score, 4),
    }
    st.session_state.detection_log.append(entry)


def export_log_csv() -> bytes:
    """Export the detection log as CSV bytes."""
    df = pd.DataFrame(st.session_state.detection_log)
    return df.to_csv(index=False).encode("utf-8")


def display_detection_log() -> None:
    """Render the detection log table and CSV download button."""
    st.header("Detection Log")

    if not st.session_state.detection_log:
        st.info("No detections yet. Upload an image to get started.")
        return

    df = pd.DataFrame(st.session_state.detection_log)
    st.dataframe(df, use_container_width=True)

    csv_bytes = export_log_csv()
    st.download_button(
        label="Download Log as CSV",
        data=csv_bytes,
        file_name="detection_log.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Performance dashboard (Task 7.5)
# ---------------------------------------------------------------------------

def display_dashboard() -> None:
    """Render the performance dashboard."""
    with st.expander("Performance Dashboard", expanded=True):
        log = st.session_state.detection_log

        if not log:
            st.info("Process at least one image to see dashboard metrics.")
            return

        df = pd.DataFrame(log)

        # 11.1 – Average processing time
        avg_time = df["processing_time_ms"].mean()
        st.metric("Average Processing Time (ms)", f"{avg_time:.1f}")

        # 11.2 – Histogram of confidence scores
        st.subheader("Confidence Score Distribution")
        hist_data = (
            pd.cut(df["confidence_score"], bins=10, include_lowest=True)
            .value_counts()
            .sort_index()
        )
        hist_df = pd.DataFrame({
            "Bin": [str(interval) for interval in hist_data.index],
            "Count": hist_data.values,
        })
        st.bar_chart(hist_df.set_index("Bin"))

        # 11.3 – Dice / IoU shown when ground truth is available (stored in session)
        if "gt_metrics" in st.session_state and st.session_state.gt_metrics:
            st.subheader("Ground Truth Metrics (latest image)")
            m = st.session_state.gt_metrics
            col1, col2 = st.columns(2)
            col1.metric("Dice Coefficient", f"{m.get('dice', 0):.4f}")
            col2.metric("IoU", f"{m.get('iou', 0):.4f}")


# ---------------------------------------------------------------------------
# Main app layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Face Segmentation Cast ID",
        page_icon="🎬",
        layout="wide",
    )

    # Make expanded/lightbox images large and centered
    st.markdown("""
    <style>
    /* Streamlit image lightbox/modal */
    div[data-testid="stFullScreenFrame"] .stImage {
    #   width:100%;
      justify-content:center !important;
      align-items:center !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎬 Face Segmentation Cast ID")
    st.write("Upload a movie scene image to segment and identify face regions.")

    _init_session_state()

    # Task 7.1 – Load model
    model = load_model()
    if model is None:
        st.error(
            f"Model file `{MODEL_PATH}` not found. "
            "Please ensure the trained model is placed at the expected path."
        )
        st.stop()

    # Task 7.2 – Image upload
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "avif"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Processing image…"):
                image = handle_upload(uploaded_file)
                pred_mask, proc_time = run_inference(model, image)

            binary_mask = (pred_mask >= 0.5).astype(np.float32)
            num_regions = count_face_regions(pred_mask)
            confidence = compute_confidence(pred_mask)

            # Task 7.3 – Log detection
            log_detection(
                filename=uploaded_file.name,
                processing_time_ms=proc_time,
                num_face_regions=num_regions,
                confidence_score=confidence,
            )

            # Display results
            display_results(image, binary_mask)

            st.success(
                f"Detected {num_regions} face region(s) | "
                f"Confidence: {confidence:.2%} | "
                f"Processing time: {proc_time:.1f} ms"
            )

            # # Optional ground-truth mask upload for Dice/IoU (Req 11.3)
            # st.markdown("---")
            # st.subheader("Optional: Upload Ground Truth Mask")
            # gt_file = st.file_uploader(
            #     "Upload a ground truth binary mask (PNG) to compute Dice & IoU",
            #     type=["png"],
            #     key="gt_mask",
            # )
            # if gt_file is not None:
            #     try:
            #         gt_pil = Image.open(gt_file).convert("L")
            #         gt_arr = np.array(gt_pil.resize((256, 256)), dtype=np.float32) / 255.0
            #         gt_binary = (gt_arr >= 0.5).astype(np.float32)
            #
            #         import tensorflow as tf
            #         dice_val = float(
            #             dice_coefficient(
            #                 tf.constant(gt_binary[..., np.newaxis]),
            #                 tf.constant(binary_mask),
            #             ).numpy()
            #         )
            #         iou_val = compute_iou(gt_binary, binary_mask.squeeze())
            #         st.session_state.gt_metrics = {"dice": dice_val, "iou": iou_val}
            #         st.info(f"Dice Coefficient: {dice_val:.4f} | IoU: {iou_val:.4f}")
            #     except Exception as exc:
            #         st.warning(f"Could not compute ground truth metrics: {exc}")

        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            logger.error("Inference error: %s", exc)
            st.error(f"An error occurred during inference: {exc}")

    st.markdown("---")

    # Task 7.3 – Detection log
    display_detection_log()

    st.markdown("---")

    # Task 7.5 – Performance dashboard
    display_dashboard()


if __name__ == "__main__":
    main()
