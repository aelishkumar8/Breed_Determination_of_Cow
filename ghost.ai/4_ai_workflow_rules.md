# AI Workflow Rules

Outlines the agent's discipline. It instructs the AI on how to scope work, how to handle architectural decisions, and the rule to work on one feature unit at a time to prevent system drifts.

## Scope of Work
- When asked to "train the model" or "build the pipeline," the AI must break this down into smaller, verifiable steps (e.g., 1. Setup DataLoaders, 2. Define Model, 3. Write Training Loop).
- Do not attempt to write an entire complex ML pipeline in a single response.

## Architectural Decisions
- If a new dependency (e.g., a new pip package) is needed, propose it and explain why before modifying `requirements.txt` or `setup.py`.
- If modifying the boundary between detection and classification, request explicit user approval.

## Feature Implementation
- **CRITICAL RULE - ONE UNIT AT A TIME:** Work on exactly ONE feature unit at a time to prevent system drift. 
  - *Example:* If working on the YOLOv8 OBB cropping logic, DO NOT simultaneously modify the breed classifier's neural network layers.
- **Verification:** After implementing a function (e.g., a new image augmentation), write a quick script or notebook block to visualize/test it before moving on.
