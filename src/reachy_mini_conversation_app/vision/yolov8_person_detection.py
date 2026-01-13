"""YOLOv8 Person Detection - Head Tracker for Reachy Mini.

OPTION 1: Uses YOLOv8 to detect full persons and estimates head position
as the top-center of the person's bounding box.

PROS:
- Works at longer distances (0.5m - 10m)
- More robust to occlusion (partial visibility OK)
- Handles multiple people well
- No external model downloads needed (auto-downloads from Ultralytics)

CONS:
- Less precise than face detection
- Estimates head position (not direct detection)

USAGE:
    Just replace your current yolo_head_tracker.py with this file.
    No other code changes needed!
"""

from __future__ import annotations
import logging
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from supervision import Detections
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "To use YOLO head tracker, please install: pip install ultralytics supervision",
    ) from e

logger = logging.getLogger(__name__)


class HeadTracker:
    """YOLOv8-based head tracker using person detection.
    
    Detects full persons using YOLOv8 trained on COCO dataset and 
    estimates head position as the top-center of the person bounding box.
    """

    def __init__(
        self,
        model_name: str = "yolov8n-person.pt",
        confidence_threshold: float = 0.3,
        device: str = "cpu",
    ) -> None:
        """Initialize YOLOv8 person detector.

        Args:
            model_name: YOLOv8 model variant to use:
                - yolov8n.pt (nano)   - fastest, 6MB, ~45 FPS on CPU
                - yolov8s.pt (small)  - balanced, 22MB, ~30 FPS on CPU
                - yolov8m.pt (medium) - accurate, 52MB, ~15 FPS on CPU
                - yolov8l.pt (large)  - high accuracy, 87MB, ~8 FPS on CPU
                - yolov8x.pt (xlarge) - best, 136MB, ~5 FPS on CPU
            confidence_threshold: Minimum confidence for detection (0.0-1.0)
                Lower = more detections but more false positives
                Higher = fewer but more confident detections
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.confidence_threshold = confidence_threshold
        self.device = device

        try:
            # Load YOLOv8 model
            # Will auto-download from Ultralytics on first run
            logger.info(f"Loading YOLOv8 model: {model_name}")
            self.model = YOLO(model_name)
            self.model.to(device)
            logger.info(f"✓ YOLOv8 model '{model_name}' loaded on {device}")
        except Exception as e:
            logger.error(f"✗ Failed to load YOLOv8 model: {e}")
            logger.error("Make sure to: pip install ultralytics")
            raise

    def _select_best_person(self, detections: Detections) -> int | None:
        """Select the best person to track.

        Selection strategy:
        - Filters by confidence threshold
        - Prioritizes larger persons (closer to camera)
        - Uses weighted score: 70% confidence + 30% size

        Args:
            detections: Supervision detections object containing all persons

        Returns:
            Index of best person to track, or None if no valid detections
        """
        if detections.xyxy.shape[0] == 0:
            return None

        # Check if confidence scores are available
        if detections.confidence is None:
            logger.warning("No confidence scores available")
            return None

        # Filter by confidence threshold
        valid_mask = detections.confidence >= self.confidence_threshold
        if not np.any(valid_mask):
            logger.debug(f"No persons above confidence threshold {self.confidence_threshold}")
            return None

        valid_indices = np.where(valid_mask)[0]

        # Calculate bounding box areas (larger = closer/more important)
        boxes = detections.xyxy[valid_indices]
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights

        # Combine confidence and area for final score
        # 70% weight on confidence, 30% on size
        confidences = detections.confidence[valid_indices]
        normalized_areas = areas / np.max(areas)  # Normalize to [0, 1]
        scores = confidences * 0.7 + normalized_areas * 0.3

        # Return index of highest scoring person
        best_idx = valid_indices[np.argmax(scores)]
        
        logger.debug(
            f"Selected person {best_idx}: "
            f"confidence={confidences[np.argmax(scores)]:.2f}, "
            f"area={areas[np.argmax(scores)]:.0f}px²"
        )
        
        return int(best_idx)

    def _bbox_to_mp_coords(
        self, 
        bbox: NDArray[np.float32], 
        w: int, 
        h: int,
        use_head_position: bool = True
    ) -> NDArray[np.float32]:
        """Convert bounding box to normalized coordinates.

        Converts pixel coordinates to MediaPipe-style [-1, 1] range where:
        - (-1, -1) = top-left corner
        - (0, 0)   = center
        - (1, 1)   = bottom-right corner

        Args:
            bbox: Bounding box [x1, y1, x2, y2] in pixels
            w: Image width in pixels
            h: Image height in pixels
            use_head_position: If True, use top-center (head estimate)
                             If False, use center of box

        Returns:
            Normalized position [x, y] in [-1, 1] range
        """
        if use_head_position:
            # Estimate head position as top-center of person bounding box
            # Assumption: YOLOv8 person detection includes head-to-toe,
            # so top of box ≈ head location
            center_x = (bbox[0] + bbox[2]) / 2.0  # Horizontal center
            center_y = bbox[1]  # Top of bounding box
        else:
            # Use center of bounding box
            center_x = (bbox[0] + bbox[2]) / 2.0
            center_y = (bbox[1] + bbox[3]) / 2.0

        # Normalize from [0, w] and [0, h] to [-1, 1]
        norm_x = (center_x / w) * 2.0 - 1.0
        norm_y = (center_y / h) * 2.0 - 1.0

        return np.array([norm_x, norm_y], dtype=np.float32)

    def get_head_position(
        self, 
        img: NDArray[np.uint8]
    ) -> Tuple[NDArray[np.float32] | None, float | None]:
        """Get head position from person detection.

        Main detection method. Detects persons in the image using YOLOv8
        and estimates head position.

        Detection flow:
        1. Run YOLOv8 inference on image
        2. Filter to only 'person' class (COCO class 0)
        3. Select best person (largest + highest confidence)
        4. Estimate head as top-center of person box
        5. Normalize to [-1, 1] coordinates

        Args:
            img: Input image in BGR format (OpenCV/cv2 format)
                Shape should be (height, width, 3)

        Returns:
            Tuple of (head_position, roll_angle) where:
                - head_position: [x, y] in [-1, 1] normalized coordinates
                                Returns None if no person detected
                - roll_angle: Head roll angle in radians (always 0.0 for bbox)
                            Returns None if no person detected
        
        Example:
            >>> tracker = HeadTracker()
            >>> head_pos, roll = tracker.get_head_position(image)
            >>> if head_pos is not None:
            >>>     print(f"Head at x={head_pos[0]:.2f}, y={head_pos[1]:.2f}")
        """
        h, w = img.shape[:2]

        try:
            # Run YOLOv8 inference
            # classes=[0] means only detect 'person' (COCO class 0)
            # COCO dataset has 80 classes, we only need persons
            results = self.model(
                img,
                classes=[0],  # Only detect persons (class 0 in COCO)
                verbose=False,  # Suppress console output
                conf=self.confidence_threshold,  # Minimum confidence
            )
            
            # Convert Ultralytics format to Supervision format
            # Supervision provides easier-to-use detection handling
            detections = Detections.from_ultralytics(results[0])

            # Select best person to track
            person_idx = self._select_best_person(detections)
            
            if person_idx is None:
                logger.debug("No person detected above confidence threshold")
                return None, None

            # Get bounding box coordinates
            bbox = detections.xyxy[person_idx]

            # Log detection confidence for debugging
            if detections.confidence is not None:
                confidence = detections.confidence[person_idx]
                logger.debug(f"✓ Person detected with confidence: {confidence:.2f}")

            # Convert to normalized head position
            head_position = self._bbox_to_mp_coords(
                bbox, w, h, use_head_position=True
            )

            # Roll angle estimation
            # Note: Cannot estimate precise roll angle from bounding box alone
            # Would need pose estimation or face landmarks for accurate roll
            roll = 0.0

            return head_position, roll

        except Exception as e:
            logger.error(f"✗ Error in head position detection: {e}")
            return None, None

    def get_all_detections(
        self, 
        img: NDArray[np.uint8]
    ) -> list[Tuple[NDArray[np.float32], float]]:
        """Get all person detections in the image.

        Useful for multi-person tracking scenarios.

        Args:
            img: Input image in BGR format

        Returns:
            List of (head_position, confidence) tuples for each detected person
            Empty list if no persons detected
        
        Example:
            >>> tracker = HeadTracker()
            >>> all_people = tracker.get_all_detections(image)
            >>> print(f"Detected {len(all_people)} people")
            >>> for head_pos, conf in all_people:
            >>>     print(f"Person at {head_pos} with confidence {conf:.2f}")
        """
        h, w = img.shape[:2]
        detections_list = []

        try:
            # Run YOLOv8 inference
            results = self.model(
                img,
                classes=[0],  # Only persons
                verbose=False,
                conf=self.confidence_threshold,
            )
            
            detections = Detections.from_ultralytics(results[0])

            if detections.confidence is None:
                return detections_list

            # Process all valid detections
            for i in range(len(detections.xyxy)):
                if detections.confidence[i] >= self.confidence_threshold:
                    bbox = detections.xyxy[i]
                    head_pos = self._bbox_to_mp_coords(bbox, w, h, True)
                    confidence = float(detections.confidence[i])
                    detections_list.append((head_pos, confidence))

            logger.debug(f"✓ Found {len(detections_list)} person(s)")
            return detections_list

        except Exception as e:
            logger.error(f"✗ Error getting all detections: {e}")
            return detections_list

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold dynamically.
        
        Args:
            threshold: New confidence threshold (0.0 - 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self.confidence_threshold = threshold
        logger.info(f"Updated confidence threshold to {threshold}")


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of YOLOv8 Person Detection head tracker.
    """
    import cv2
    
    # Initialize tracker
    print("Initializing YOLOv8 Person Detection tracker...")
    tracker = HeadTracker(
        model_name="yolov8n-person.pt",  # Use nano model (fastest)
        confidence_threshold=0.3,
        device="cpu"  # Change to "cuda" if you have GPU
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("\nTracking persons... Press 'q' to quit")
    print("Stand in front of the camera!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get head position
        head_pos, roll = tracker.get_head_position(frame)
        
        if head_pos is not None:
            # Convert normalized coords back to pixels for visualization
            h, w = frame.shape[:2]
            pixel_x = int((head_pos[0] + 1) * w / 2)
            pixel_y = int((head_pos[1] + 1) * h / 2)
            
            # Draw marker at estimated head position
            cv2.circle(frame, (pixel_x, pixel_y), 10, (0, 255, 0), -1)
            cv2.putText(
                frame, 
                f"Head: ({head_pos[0]:.2f}, {head_pos[1]:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            print(f"Head position: x={head_pos[0]:+.2f}, y={head_pos[1]:+.2f}")
        else:
            cv2.putText(
                frame,
                "No person detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        # Display
        cv2.imshow('YOLOv8 Person Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()