# 🐄 Breed Detection of Cow

A machine learning and computer vision project that helps identify and classify different breeds of cows from images.

## Overview

This project aims to develop an automated two-stage pipeline that can accurately determine the breed of a cow from images. It uses YOLOv8-seg for detecting and cleanly cropping the cow using Oriented Bounding Boxes (OBBs), and a separate deep learning model to classify the cropped image into its respective breed.

## Features

- Image-based cow breed classification
- Instance segmentation and cropping using YOLOv8-seg
- Support for multiple common cow breeds
- High accuracy in breed determination by minimizing background noise

## Project Structure

```
Breed detection of Cow/
├── data/
│   ├── raw/              # Original, unprocessed datasets
│   └── processed/        # Cleaned and preprocessed data
├── models/
│   ├── weights/          # Saved model weights / checkpoints
│   └── architectures/    # Model architecture definitions
├── notebooks/            # Jupyter notebooks for exploration & experiments
├── src/
│   ├── __init__.py
│   ├── data/             # Data loading & preprocessing utilities
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models/           # Model definitions
│   │   ├── __init__.py
│   │   └── classifier.py
│   ├── training/         # Training loop & utilities
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── evaluation/       # Evaluation metrics & scripts
│   │   ├── __init__.py
│   │   └── evaluate.py
│   └── utils/            # General utility functions
│       ├── __init__.py
│       └── helpers.py
├── config/
│   └── config.yaml       # Project configuration
├── tests/                # Unit tests
│   └── __init__.py
├── outputs/              # Training outputs, plots, results
├── ghost.ai/             # AI Workflow and documentation files
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
```

## Getting Started

### 1. Create & activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your dataset

Place your cow breed images in `data/raw/`, organized by breed:

```
data/raw/
├── breed_1/
│   ├── img_001.jpg
│   └── ...
├── breed_2/
│   ├── img_001.jpg
│   └── ...
└── ...
```

### 4. Train the model

```bash
python -m src.training.trainer
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors who have helped in developing this project
- Special thanks to the open-source community for their valuable resources and tools

## Contact

For any questions or suggestions, please open an issue in the repository.
