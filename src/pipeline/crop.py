"""
Crop Pipeline for Cow Breed Detection.

HOW IT WORKS:
─────────────
1. Detector gives us segmentation masks and oriented bounding boxes (OBB).
2. This module uses the OBB to CROP a tight, rotation-corrected region
   containing ONLY the cow — minimal background noise.
3. If no mask/OBB is available, falls back to axis-aligned bounding box crop.
4. These clean crops are what we'll feed to the breed classifier.

WHY ORIENTED CROPS?
───────────────────
- A cow at 45° produces a large axis-aligned box with ~50% background.
- The OBB (from segmentation mask) rotates to tightly hug the cow.
- We rotate the image to make the cow upright, then crop the tight rectangle.
- Result: cleaner input for the classifier = better breed predictions.

PIPELINE FLOW:
──────────────
    Full Image ──▶ [YOLO-Seg Detector] ──▶ Masks + OBBs
                                                 │
                                                 ▼
                                          [Crop Module] ──▶ Oriented Cropped Cow Images
                                                                 │
                                                                 ▼
                                                       (saved to data/processed/)
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.detection.detector import CowDetector


class CropPipeline:
    """
    Detects cows in images and crops them out using oriented bounding boxes.

    Uses segmentation masks to compute tight, rotation-corrected crops
    that minimize background noise for better breed classification.
    """

    def __init__(
        self,
        detector: Optional[CowDetector] = None,
        padding_percent: float = 0.05,
        min_crop_size: int = 50,
    ):
        """
        Args:
            detector: CowDetector instance. Creates default if None.
            padding_percent: Extra padding around the bounding box (0.05 = 5%).
                             Prevents clipping parts of the cow.
            min_crop_size: Minimum width/height of a crop to keep (pixels).
                           Filters out tiny/false detections.
        """
        self.detector = detector or CowDetector()
        self.padding_percent = padding_percent
        self.min_crop_size = min_crop_size

    def _add_padding(
        self, bbox: list[int], img_h: int, img_w: int
    ) -> list[int]:
        """
        Expand the bounding box by padding_percent, clamped to image bounds.

        Example: A bbox of [100, 100, 300, 300] with 5% padding on a
        1000x1000 image becomes [90, 90, 310, 310].
        """
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1

        pad_x = int(box_w * self.padding_percent)
        pad_y = int(box_h * self.padding_percent)

        # Clamp to image boundaries
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_w, x2 + pad_x)
        y2 = min(img_h, y2 + pad_y)

        return [x1, y1, x2, y2]



    def crop_single_image(self, image_path: str) -> list[dict]:
        """
        Detect and crop cows from a single image.

        Uses the standard axis-aligned bounding box to ensure the cow 
        remains in its natural orientation (not tilted sideways), which is 
        crucial for accurate breed classification.

        Args:
            image_path: Path to the input image.

        Returns:
            List of dicts, each containing:
                - "crop": numpy array (BGR) of the cropped cow
                - "bbox": [x1, y1, x2, y2] (axis-aligned bounding box with padding)
                - "confidence": detection confidence
                - "source": original image path
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        img_h, img_w = image.shape[:2]

        # Step 1: Detect cows (with masks + OBBs from detector)
        detections = self.detector.detect(image)

        # Step 2: Crop each detection
        crops = []
        for det in detections:
            # We explicitly use the axis-aligned box (not OBB) to keep the cow upright
            padded_bbox = self._add_padding(det["bbox"], img_h, img_w)
            x1, y1, x2, y2 = padded_bbox
            crop = image[y1:y2, x1:x2].copy()

            # Skip tiny crops (likely false positives)
            if crop.shape[0] < self.min_crop_size or crop.shape[1] < self.min_crop_size:
                continue

            crops.append({
                "crop": crop,
                "bbox": padded_bbox,
                "confidence": det["confidence"],
                "source": image_path,
            })

        return crops

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        save_crops: bool = True,
    ) -> dict:
        """
        Process an entire directory of images: detect + crop all cows.

        Args:
            input_dir: Directory containing input images.
            output_dir: Directory to save cropped cow images.
            save_crops: If True, save crops as image files.

        Returns:
            Summary dict with stats about the processing.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [
            f for f in input_path.rglob("*")
            if f.suffix.lower() in valid_extensions
        ]

        total_images = len(image_files)
        total_cows = 0
        skipped = 0
        results = []

        print(f"Processing {total_images} images from {input_dir}...")
        print(f"Crops will be saved to {output_dir}\n")

        for i, img_file in enumerate(image_files, 1):
            try:
                crops = self.crop_single_image(str(img_file))

                if len(crops) == 0:
                    skipped += 1
                    print(f"  [{i}/{total_images}] {img_file.name} — no cow detected")
                    continue

                for j, crop_data in enumerate(crops):
                    total_cows += 1
                    # Preserve subdirectory structure (e.g., breed folders)
                    relative = img_file.relative_to(input_path)
                    crop_name = f"{relative.stem}_cow{j}{relative.suffix}"
                    crop_subdir = output_path / relative.parent
                    crop_subdir.mkdir(parents=True, exist_ok=True)
                    crop_path = crop_subdir / crop_name

                    if save_crops:
                        cv2.imwrite(str(crop_path), crop_data["crop"])

                    results.append({
                        "source": str(img_file),
                        "crop_path": str(crop_path),
                        "bbox": crop_data["bbox"],
                        "confidence": crop_data["confidence"],
                    })

                print(
                    f"  [{i}/{total_images}] {img_file.name} "
                    f"— {len(crops)} cow(s) cropped"
                )

            except Exception as e:
                skipped += 1
                print(f"  [{i}/{total_images}] {img_file.name} — ERROR: {e}")

        summary = {
            "total_images": total_images,
            "total_cows_cropped": total_cows,
            "images_skipped": skipped,
            "output_dir": output_dir,
            "results": results,
        }

        print(f"\n{'='*50}")
        print(f"DONE!")
        print(f"  Images processed: {total_images}")
        print(f"  Cows cropped:     {total_cows}")
        print(f"  Skipped:          {skipped}")
        print(f"  Saved to:         {output_dir}")
        print(f"{'='*50}")

        return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  Single image:  python -m src.pipeline.crop <image_path>\n"
            "  Full directory: python -m src.pipeline.crop <input_dir> [output_dir]"
        )
        sys.exit(1)

    pipeline = CropPipeline()
    input_path = sys.argv[1]

    if Path(input_path).is_dir():
        # --- Directory mode ---
        out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
        pipeline.process_directory(input_path, out_dir)
    else:
        # --- Single image mode ---
        crops = pipeline.crop_single_image(input_path)
        print(f"Found {len(crops)} cow(s) in {input_path}")
        for i, c in enumerate(crops):
            out = f"outputs/crop_{i}.jpg"
            Path("outputs").mkdir(exist_ok=True)
            cv2.imwrite(out, c["crop"])
            print(f"  Crop #{i}: saved to {out}, confidence={c['confidence']}")

