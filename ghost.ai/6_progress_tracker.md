# Progress Tracker

The most critical file that updates throughout the build. It holds the current phase, completed tasks, and ongoing work. Because agents often lack memory between sessions, this file allows the AI to restore full context with a single prompt at the start of a new session.

## Current Phase
**Phase 2: Breed Classification & Integration**
*(Transitioning from Phase 1: Robust Object Detection & Cropping)*

## Completed Tasks
- [x] Initial project scaffolding (`src/`, `data/`, `models/`, `notebooks/`)
- [x] Set up Python environment and `requirements.txt`
- [x] Integrate YOLOv8 for base object detection
- [x] Transition to instance segmentation (YOLOv8-seg) to compute Oriented Bounding Boxes (OBBs)
- [x] Implement robust cropping pipeline to extract cows with minimal background noise
- [x] Create `ghost.ai` system directories and files

## Ongoing Work
- [ ] Finalize testing of the OBB cropping edge cases (diagonal orientations)
- [ ] Set up dataset loaders (`src/data/dataset.py`) for the breed classification model using the cropped images

## Upcoming Work
- [ ] Select and define the baseline breed classification model architecture (`src/models/classifier.py`)
- [ ] Implement the training loop for the breed classifier (`src/training/trainer.py`)
- [ ] End-to-end integration test (Raw Image -> YOLOv8 Crop -> Classifier -> Result)
