# Architecture

Documents the technical stack, defines boundaries between layers, and lists the "invariants" (rules) that the codebase must never violate.

## Technical Stack
- **Language:** Python 3
- **Deep Learning Frameworks:** PyTorch, Torchvision (and TensorFlow if needed)
- **Computer Vision:** OpenCV (`opencv-python`), YOLOv8 (`ultralytics`), Pillow
- **Data Science:** NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **Environment:** Conda / Venv, Jupyter Notebooks

## Layer Boundaries
- **`src/detection/`**: Strictly handles object detection, segmentation (YOLOv8-seg), and geometry math (Oriented Bounding Boxes). Should not know about breeds.
- **`src/pipeline/`**: The orchestrator. Takes raw images, calls detection to get crops, and passes crops to the classifier.
- **`src/models/`**: Contains the neural network architectures for breed classification.
- **`src/data/`**: Handles dataset loading, data augmentation, and preprocessing.
- **`src/training/`**: Scripts and loops specifically for training the classification models.

## Invariants (Strict Rules)
1. **Decoupled Stages:** The breed classifier must never depend directly on YOLO. It should only expect a pre-cropped standard image (e.g., 224x224 RGB array) as input.
2. **Deterministic Cropping:** The cropping mechanism (`src/pipeline/crop.py`) must always produce consistent outputs for the same bounding box to avoid data leakage or training inconsistencies.
3. **No Hardcoded Paths:** All file paths must be resolved relatively or via `config/config.yaml` / `.env` variables.
