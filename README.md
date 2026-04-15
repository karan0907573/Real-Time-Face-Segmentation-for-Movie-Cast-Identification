# Face Segmentation for Movie Cast Identification

A face segmentation system that detects and segments faces in movie scene screenshots. Built with a U-Net deep learning model (MobileNetV2 encoder + BatchNorm/Dropout decoder) and served through a Streamlit web application.

## Project Structure

```
├── app.py                          # Streamlit web application
├── helper.py                       # Prediction helpers, custom losses/metrics, model loading
├── process_data.ipynb              # Data preprocessing — Gaussian filter, ellipse masks, tf.data dataset creation
├── facesegmentation_1000_data.ipynb # Model training notebook (Google Colab)
├── facesegmentation_1000_data.py   # Python export of the training notebook
├── requirements.txt                # Python dependencies
├── model/
│   ├── face_segmentation_model_10000.keras  # Final trained model
│   └── best_face_model.weights.h5           # Best checkpoint weights
├── 10000_data/                     # Preprocessed tf.data datasets (train + val)
│   ├── train_dataset/
│   └── val_dataset/
├── kaggle face data/               # Additional Kaggle dataset (YOLO format labels)
│   ├── images/
│   ├── labels/
│   └── labels2/
├── Part 1- Train data - images.npy             # Original training data (~450 images)
├── Part 1Test Data - Prediction Image.jpeg     # Sample test image
├── tests/                          # Experimental scripts and temp files
└── README.md
```

## Setup Instructions

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # Linux / macOS
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Data

The original dataset provided ~450 annotated face images (`Part 1- Train data - images.npy`). This was insufficient for good model performance, so the dataset was expanded to ~10,000 images by incorporating a Kaggle face detection dataset (`kaggle face data/`) with bounding box labels.

### Data Processing Pipeline (`process_data.ipynb`)

1. Loads images and their bounding box annotations (both .npy and YOLO formats)
2. Converts bounding boxes to elliptical face masks with Gaussian smoothing
3. Resizes images to 256×256 and normalizes to [0, 1]
4. Splits into train/val sets
5. Saves as `tf.data.Dataset` objects for memory-efficient training

The preprocessed datasets are stored in `10000_data/train_dataset/` and `10000_data/val_dataset/`.

## Model Architecture

- **Encoder**: MobileNetV2 (ImageNet pretrained)
- **Decoder**: Custom upsampling path with skip connections, BatchNormalization, and Dropout (0.2)
- **Output**: Sigmoid activation for binary face segmentation
- **Input size**: 256×256×3
- **Loss**: Combined BCE + Dice Loss (`bce_dice_loss`)
- **Metric**: Dice Coefficient

## Training (`facesegmentation_1000_data.ipynb`)

Two-phase training strategy (run on Google Colab):

### Phase 1 — Decoder only (encoder frozen)
- Optimizer: Adam (lr=1e-3)
- Epochs: up to 30 (early stopping, patience=10)
- LR reduction on plateau (factor=0.5, patience=3)

### Phase 2 — Full fine-tuning
- Optimizer: Adam with Cosine Decay (1e-4 → 1e-6)
- Epochs: up to 50 (early stopping, patience=20)
- Best weights saved via ModelCheckpoint

### Data Augmentation
- Random horizontal flip
- Random brightness (±0.15) and contrast (0.8–1.2)
- Random rotation (±15°)

## Running the Streamlit App

```bash
streamlit run app.py
```

The app provides:

- **Image Upload** — Supports JPEG, PNG, WebP, BMP, TIFF, and AVIF formats
- **Face Segmentation** — Displays original image, predicted binary mask, and overlay
- **Detection Log** — Table of all results (filename, timestamp, processing time, face count, confidence) with CSV export
- **Performance Dashboard** — Average processing time and confidence score distribution

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web app — upload, inference, visualization, logging |
| `helper.py` | Custom losses/metrics, model loading, preprocessing, prediction utilities |
| `process_data.ipynb` | Data preprocessing — mask generation, resizing, tf.data creation |
| `facesegmentation_1000_data.ipynb` | Model training notebook (Colab) |
| `model/face_segmentation_model_10000.keras` | Final trained model |

## Evaluation Targets

| Metric | Target |
|--------|--------|
| Dice Coefficient | > 0.92 |
| IoU | > 0.88 |
| F1-Score | > 0.90 |
| Inference Time | < 100ms per image |

## Dependencies

Key libraries (see `requirements.txt` for pinned versions):

- TensorFlow 2.16.1
- NumPy 1.26.4
- Streamlit 1.38.0
- scikit-learn 1.5.2
- Pillow 10.4.0
- SciPy 1.14.1
- Pandas 2.2.3
- Matplotlib 3.9.2
