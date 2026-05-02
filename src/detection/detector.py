"""
Cow Detector using YOLOv8-Seg (Instance Segmentation).

HOW IT WORKS:
─────────────
1. We load YOLOv8m-seg — pretrained on COCO dataset (80 classes).
2. In COCO, "cow" is class index 19.
3. We run the model on an image and FILTER results to keep only cow detections.
4. Each detection gives us:
   - A bounding box [x1, y1, x2, y2]
   - A confidence score
   - A segmentation mask (pixel-level cow outline)
5. From the mask, we compute an Oriented Bounding Box (OBB) that tightly
   fits the cow regardless of its angle/pose.

WHY SEGMENTATION + OBB?
───────────────────────
- Axis-aligned boxes waste space on diagonal cows (up to 50% background).
- The segmentation mask lets us compute a rotated rectangle that hugs the cow.
- Tighter crops = less noise for the breed classifier.

USAGE:
──────
    from src.detection.detector import CowDetector

    detector = CowDetector()
    detections = detector.detect("path/to/image.jpg")
    # detections = [
    #   {"bbox": [...], "confidence": 0.92, "mask": np.ndarray, "obb": ((cx,cy),(w,h),angle)},
    #   ...
    # ]
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


# COCO class index for "cow"
COW_CLASS_ID = 19


class CowDetector:
    """
    Detects cows in images using a YOLOv8 model.

    The model is pretrained on COCO which already includes the "cow" class.
    We simply filter predictions to only return cow bounding boxes.
    """

    def __init__(
        self,
        model_path: str = "yolov8m-seg.pt",
        confidence_threshold: float = 0.4,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to YOLOv8-seg weights. Defaults to yolov8m-seg.pt
                        (medium segmentation model — provides masks + bboxes).
                        Options: yolov8n/s/m/l/x-seg.pt (small → large).
            confidence_threshold: Minimum confidence to accept a detection.
            device: "cuda", "cpu", or None (auto-select).
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device

    def detect(self, image_source) -> list[dict]:
        """
        Detect cows in an image with segmentation masks.

        Args:
            image_source: File path (str/Path) or numpy array (BGR).

        Returns:
            List of detections, each dict containing:
                - "bbox": [x1, y1, x2, y2] (axis-aligned bounding box)
                - "confidence": float (0-1)
                - "mask": numpy array (H×W bool) — segmentation mask, or None
                - "obb": ((cx, cy), (w, h), angle) — oriented bounding box from mask, or None
        """
        # Run YOLOv8-seg inference
        results = self.model(
            image_source,
            conf=self.confidence_threshold,
            classes=[COW_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            masks = result.masks  # Segmentation masks (may be None if non-seg model)

            if boxes is None or len(boxes) == 0:
                continue

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                det = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(conf, 4),
                    "mask": None,
                    "obb": None,
                }

                # Extract segmentation mask and compute oriented bounding box
                if masks is not None and idx < len(masks):
                    # Get binary mask at original image resolution
                    mask = masks[idx].data[0].cpu().numpy()  # (H, W) float
                    # Resize mask to original image dimensions if needed
                    if isinstance(image_source, np.ndarray):
                        img_h, img_w = image_source.shape[:2]
                    else:
                        img_h = int(result.orig_shape[0])
                        img_w = int(result.orig_shape[1])

                    if mask.shape != (img_h, img_w):
                        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

                    binary_mask = (mask > 0.5).astype(np.uint8)
                    det["mask"] = binary_mask

                    # Compute oriented bounding box from mask contours
                    contours, _ = cv2.findContours(
                        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        # Use the largest contour
                        largest = max(contours, key=cv2.contourArea)
                        # minAreaRect returns ((cx, cy), (width, height), angle)
                        obb = cv2.minAreaRect(largest)
                        det["obb"] = obb

                detections.append(det)

        return detections

    def detect_and_visualize(
        self, image_path: str, output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Detect cows and draw oriented bounding boxes + mask overlays on the image.

        Args:
            image_path: Path to input image.
            output_path: If provided, save the annotated image here.

        Returns:
            Annotated image as numpy array (BGR).
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        detections = self.detect(image)

        for det in detections:
            conf = det["confidence"]

            # Draw mask overlay (semi-transparent green)
            if det["mask"] is not None:
                mask_overlay = image.copy()
                mask_overlay[det["mask"] == 1] = [0, 200, 0]  # green
                cv2.addWeighted(mask_overlay, 0.3, image, 0.7, 0, image)

            # Draw oriented bounding box if available
            if det["obb"] is not None:
                obb_points = cv2.boxPoints(det["obb"])
                obb_points = np.int32(obb_points)
                cv2.drawContours(image, [obb_points], 0, (255, 165, 0), 3)  # type: ignore[call-overload]
            else:
                # Fallback: draw axis-aligned box
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            x1, y1 = det["bbox"][:2]
            label = f"cow {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (255, 165, 0),
                -1,
            )
            cv2.putText(
                image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
            )

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, image)
            print(f"Saved annotated image to {output_path}")

        return image


    @staticmethod
    def _convert_to_jpg(file_path: Path, output_dir: Path, relative: Path) -> Path:
        """
        Convert an unsupported image format to JPEG.

        JPEG is chosen for the best space/time trade-off:
        - Fastest encode/decode among lossy formats.
        - Smallest file size for photographic content (cow images).
        - Universally supported by OpenCV and YOLO.

        Args:
            file_path:  Absolute path to the source image.
            output_dir: Root output directory.
            relative:   Relative path (preserves subfolder structure).

        Returns:
            Path to the converted JPEG file (saved inside output_dir/_converted/).
        """
        converted_dir = output_dir / "_converted"
        jpg_relative = relative.with_suffix(".jpg")
        jpg_path = converted_dir / jpg_relative
        jpg_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(file_path).convert("RGB")
        img.save(jpg_path, format="JPEG", quality=95)
        return jpg_path

    def batch_detect_folder(
        self,
        input_dir: str,
        output_dir: str = "outputs/detected",
    ) -> dict:
        """
        Recursively detect cows in all images inside a folder (including subfolders).

        Works with ANY folder structure — subfolders can have any name:
            input_dir/
            ├── folder_A/
            │   ├── img_001.jpg
            │   └── img_002.heic   ← auto-converted to JPEG
            ├── some_random_folder/
            │   └── photo.avif     ← auto-converted to JPEG
            └── lone_cow.png

        Unsupported formats (e.g., .heic, .avif, .gif, .tga, .svg, etc.)
        are automatically converted to JPEG before detection.
        Output mirrors the same subfolder structure under output_dir/.

        Args:
            input_dir:  Root folder containing images (and subfolders).
            output_dir: Where to save annotated images. Defaults to "outputs/detected".

        Returns:
            Dict mapping each image path (str) to its list of detections.
        """
        # Formats that OpenCV / YOLO can read directly
        NATIVE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        # Files to always skip (not images)
        SKIP_EXTENSIONS = {".txt", ".md", ".json", ".yaml", ".yml", ".csv",
                          ".py", ".log", ".xml", ".html", ".gitkeep"}

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {input_dir}")

        # Recursively collect all files (except known non-image types)
        all_files = sorted(
            f for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() not in SKIP_EXTENSIONS
        )

        if not all_files:
            print(f"No files found in {input_dir}")
            return {}

        print(f"Scanning {len(all_files)} file(s) in '{input_dir}'...")
        all_results: dict[str, list[dict]] = {}
        converted_count = 0
        skipped_count = 0

        for file_path in all_files:
            relative = file_path.relative_to(input_path)
            ext = file_path.suffix.lower()

            # Determine the image path to use for detection
            if ext in NATIVE_EXTENSIONS:
                detect_path = file_path
            else:
                # Try to convert unsupported formats to JPEG via PIL
                try:
                    detect_path = self._convert_to_jpg(file_path, output_path, relative)
                    converted_count += 1
                    print(f"  ⟳ Converted {relative} → JPEG")
                except Exception:
                    # Not a valid image file — skip silently
                    skipped_count += 1
                    print(f"  ⊘ Skipped {relative} (not a valid image)")
                    continue

            # Mirror subfolder structure for annotated output
            save_path = output_path / relative.with_suffix(".jpg")

            detections = self.detect(str(detect_path))
            all_results[str(file_path)] = detections

            cow_count = len(detections)
            status = f"✓ {cow_count} cow(s)" if cow_count > 0 else "✗ no cows"
            print(f"  {relative} → {status}")

            # Save annotated image
            self.detect_and_visualize(str(detect_path), str(save_path))

        # Summary
        total_images = len(all_results)
        total_cows = sum(len(d) for d in all_results.values())
        print(f"\n{'='*50}")
        print(f"Done! Processed {total_images} images, detected {total_cows} cow(s) total.")
        if converted_count:
            print(f"Converted {converted_count} image(s) to JPEG.")
        if skipped_count:
            print(f"Skipped {skipped_count} non-image file(s).")
        print(f"Annotated images saved to: {output_path}")

        return all_results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image:  python -m src.detection.detector <image_path> [output_path]")
        print("  Folder:        python -m src.detection.detector <folder_path> [output_folder]")
        sys.exit(1)

    input_path = sys.argv[1]
    detector = CowDetector()

    if Path(input_path).is_dir():
        # --- Folder mode: loop through all images in subfolders ---
        out_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs/detected"
        detector.batch_detect_folder(input_path, out_dir)
    else:
        # --- Single image mode ---
        out_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/detected.jpg"
        detections = detector.detect(input_path)

        print(f"\nFound {len(detections)} cow(s):")
        for i, det in enumerate(detections, 1):
            print(f"  #{i}: bbox={det['bbox']}, confidence={det['confidence']}")

        detector.detect_and_visualize(input_path, out_path)
