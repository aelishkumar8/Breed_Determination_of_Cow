# Project Overview

Defines the product, its target audience, core user flows, and, crucially, what is out of scope. This keeps the AI from attempting to build unnecessary features.

## Product Definition
An automated two-stage computer vision pipeline designed to detect cows in various poses within images, cleanly crop them, and classify their specific breeds.

## Target Audience
- Dairy and beef farmers
- Agricultural researchers
- Livestock management software integrations

## Core User Flows
1. **Input:** User provides an image containing one or more cows.
2. **Detection & Segmentation:** The system uses a YOLOv8-seg model to identify the cow and compute an Oriented Bounding Box (OBB) to capture the cow accurately, even at diagonal angles.
3. **Cropping:** The image is cropped along the OBB to minimize background noise (grass, fences, other animals).
4. **Classification:** The cleanly cropped cow image is passed to a breed classification model.
5. **Output:** The system returns the breed prediction along with confidence scores and the original image annotated with the bounding box.

## Out of Scope
*Crucially, what is out of scope to keep the AI from attempting to build unnecessary features.*
- Individual cow identification (e.g., recognizing "Bessie" from her spots).
- Health diagnostics or disease detection.
- Weight or age estimation.
- Real-time video stream processing optimization (current focus is on robust image processing).
