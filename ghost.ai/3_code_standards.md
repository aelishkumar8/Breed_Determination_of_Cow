# Code Standards

Enforces consistency across the build. Ensures the AI maintains a uniform coding style from the first commit to the last.

## Naming Conventions
- **Files & Directories:** `snake_case` (e.g., `dataset.py`, `breed_classifier/`)
- **Variables & Functions:** `snake_case` (e.g., `crop_image`, `bounding_box`)
- **Classes:** `PascalCase` (e.g., `CowDetector`, `BreedClassifier`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `IMAGE_SIZE`, `BATCH_SIZE`)

## Formatting & Linting
- Use **Black** for code formatting (max line length usually 88 or 100).
- Use **Flake8** for linting.
- Imports should be grouped: standard library, third-party packages, local project imports.

## Technology Specific Conventions
### Python
- **Type Hinting:** Mandatory for all new functions and methods (e.g., `def detect(image: np.ndarray) -> List[dict]:`).
- **Docstrings:** Use Google-style or NumPy-style docstrings for all classes and complex functions.

### Machine Learning
- **Reproducibility:** Always provide a way to set random seeds (for PyTorch, NumPy, Python `random`).
- **Device Agnosticism:** Code should automatically detect and use GPU/MPS if available, but fallback gracefully to CPU (`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`).
