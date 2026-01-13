"""YOLOv8-Face Detection - Head Tracker for Reachy Mini.

OPTION 2: Uses YOLOv8 fine-tuned specifically for face detection.
Detects faces directly (not full persons) for more precise tracking.

PROS:
- More precise face location (actual face, not estimated from person box)
- Better for close-range interactions (0.3m - 3m)
- Optimized specifically for faces

CONS:
- Requires downloading separate face detection model
- Worse at long distances
- More sensitive to face angle/orientation
- Struggles with occlusion (partial face visibility)

SETUP:
    1. Download YOLOv8-Face model:
       wget https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt
    
    2. Place model in your project directory or specify path in constructor
    
    3. Replace your current yolo_head_tracker.py with this file

ALTERNATIVE MODELS:
    - YOLOv8n-face.pt (6MB, ~45 FPS)  - Recommended
    - YOLOv8s-face.pt (22MB, ~30 FPS) - More accurate
    - Or use any YOLOv8-face model from HuggingFace
"""

from __future__ import annotations
import logging
from typing import Tuple
from pathlib import Path

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
    """YOLOv8-Face based head tracker.
    
    Uses YOLOv8 model fine-tuned specifically for face detection.
    Provides more precise face tracking than person detection.
    """

    def __init__(
        self,
        model_path: str = "yolov8n-face.pt",
        confidence_threshold: float = 0.3,
        device: str = "cpu",
    ) -> None:
        """Initialize YOLOv8-Face detector.

        Args:
            model_path: Path to YOLOv8-Face model file
                Download from: https://github.com/akanametov/yolo-face
                Options:
                - yolov8n-face.pt (nano)   - 6MB, ~45 FPS, recommended
                - yolov8s-face.pt (small)  - 22MB, ~30 FPS, more accurate
                - yolov8m-face.pt (medium) - 52MB, ~15 FPS, high accuracy
            confidence_threshold: Minimum confidence for face detection (0.0-1.0)
                Recommended: 0.3-0.5 for faces
                Lower = more faces detected (including false positives)
                Higher = only high-confidence faces
            device: Device to run inference on ('cpu' or 'cuda')
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            ImportError: If required packages not installed
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model_path = model_path

        # Check if model file exists
        if not Path(model_path).exists():
            logger.error(f"✗ Model file not found: {model_path}")
            logger.error("Download from: https://github.com/akanametov/yolo-face")
            logger.error("Command: wget https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            # Load YOLOv8-Face model
            logger.info(f"Loading YOLOv8-Face model from: {model_path}")
            self.model = YOLO(model_path)
            self.model.to(device)
            logger.info(f"✓ YOLOv8-Face model loaded on {device}")
        except Exception as e:
            logger.error(f"✗ Failed to load YOLOv8-Face model: {e}")
            raise

    def _select_best_face(self, detections: Detections) -> int | None:
        """Select the best face to track.

        Selection strategy:
        - Filters by confidence threshold
        - Prioritizes larger faces (closer to camera)
        - Uses weighted score: 70% confidence + 30% size
        - Assumes largest face = person closest to robot

        Args:
            detections: Supervision detections object containing all faces

        Returns:
            Index of best face to track, or None if no valid detections
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
            logger.debug(f"No faces above confidence threshold {self.confidence_threshold}")
            return None

        valid_indices = np.where(valid_mask)[0]

        # Calculate face bounding box areas
        boxes = detections.xyxy[valid_indices]
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights

        # Combine confidence and area for final score
        # Larger faces (closer people) get priority
        confidences = detections.confidence[valid_indices]
        normalized_areas = areas / np.max(areas)  # Normalize to [0, 1]
        scores = confidences * 0.7 + normalized_areas * 0.3

        # Return index of highest scoring face
        best_idx = valid_indices[np.argmax(scores)]
        
        logger.debug(
            f"Selected face {best_idx}: "
            f"confidence={confidences[np.argmax(scores)]:.2f}, "
            f"area={areas[np.argmax(scores)]:.0f}px²"
        )
        
        return int(best_idx)

    def _bbox_to_mp_coords(
        self, 
        bbox: NDArray[np.float32], 
        w: int, 
        h: int
    ) -> NDArray[np.float32]:
        """Convert face bounding box to normalized coordinates.

        For face detection, we use the CENTER of the face bounding box
        (unlike person detection which uses top-center).

        Converts pixel coordinates to MediaPipe-style [-1, 1] range where:
        - (-1, -1) = top-left corner
        - (0, 0)   = center
        - (1, 1)   = bottom-right corner

        Args:
            bbox: Face bounding box [x1, y1, x2, y2] in pixels
            w: Image width in pixels
            h: Image height in pixels

        Returns:
            Normalized face center [x, y] in [-1, 1] range
        """
        # Face center (middle of face bounding box)
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
        """Get head position from face detection.

        Main detection method. Detects faces in the image using YOLOv8-Face
        and returns the face center position.

        Detection flow:
        1. Run YOLOv8-Face inference on image
        2. Filter faces by confidence threshold
        3. Select best face (largest + highest confidence)
        4. Get face center from bounding box
        5. Normalize to [-1, 1] coordinates

        Args:
            img: Input image in BGR format (OpenCV/cv2 format)
                Shape should be (height, width, 3)

        Returns:
            Tuple of (head_position, roll_angle) where:
                - head_position: [x, y] in [-1, 1] normalized coordinates
                                Face center position
                                Returns None if no face detected
                - roll_angle: Head roll angle in radians
                            Always 0.0 (YOLOv8-Face doesn't give rotation)
                            Returns None if no face detected
        
        Example:
            >>> tracker = HeadTracker(model_path="yolov8n-face.pt")
            >>> head_pos, roll = tracker.get_head_position(image)
            >>> if head_pos is not None:
            >>>     print(f"Face at x={head_pos[0]:.2f}, y={head_pos[1]:.2f}")
        """
        h, w = img.shape[:2]

        try:
            # Run YOLOv8-Face inference
            # Unlike standard YOLOv8, this model only detects faces
            # No need to specify classes
            results = self.model(
                img,
                verbose=False,  # Suppress console output
                conf=self.confidence_threshold,  # Minimum confidence
            )
            
            # Convert Ultralytics format to Supervision format
            detections = Detections.from_ultralytics(results[0])

            # Select best face to track
            face_idx = self._select_best_face(detections)
            
            if face_idx is None:
                logger.debug("No face detected above confidence threshold")
                return None, None

            # Get face bounding box coordinates
            bbox = detections.xyxy[face_idx]

            # Log detection confidence for debugging
            if detections.confidence is not None:
                confidence = detections.confidence[face_idx]
                logger.debug(f"✓ Face detected with confidence: {confidence:.2f}")

            # Convert to normalized face center position
            face_center = self._bbox_to_mp_coords(bbox, w, h)

            # Roll angle estimation
            # Note: YOLOv8-Face only gives bounding boxes, not keypoints
            # Cannot estimate precise roll angle without facial landmarks
            # For roll estimation, would need models like MediaPipe Face Mesh
            roll = 0.0

            return face_center, roll

        except Exception as e:
            logger.error(f"✗ Error in face detection: {e}")
            return None, None

    def get_all_detections(
        self, 
        img: NDArray[np.uint8]
    ) -> list[Tuple[NDArray[np.float32], float]]:
        """Get all face detections in the image.

        Useful for multi-person scenarios or group tracking.

        Args:
            img: Input image in BGR format

        Returns:
            List of (face_position, confidence) tuples for each detected face
            Empty list if no faces detected
        
        Example:
            >>> tracker = HeadTracker(model_path="yolov8n-face.pt")
            >>> all_faces = tracker.get_all_detections(image)
            >>> print(f"Detected {len(all_faces)} face(s)")
            >>> for face_pos, conf in all_faces:
            >>>     print(f"Face at {face_pos} with confidence {conf:.2f}")
        """
        h, w = img.shape[:2]
        detections_list = []

        try:
            # Run YOLOv8-Face inference
            results = self.model(
                img,
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
                    face_pos = self._bbox_to_mp_coords(bbox, w, h)
                    confidence = float(detections.confidence[i])
                    detections_list.append((face_pos, confidence))

            logger.debug(f"✓ Found {len(detections_list)} face(s)")
            return detections_list

        except Exception as e:
            logger.error(f"✗ Error getting all detections: {e}")
            return detections_list

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold dynamically.
        
        Args:
            threshold: New confidence threshold (0.0 - 1.0)
                Recommended for faces: 0.3 - 0.5
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
    Example usage of YOLOv8-Face head tracker.
    
    REQUIREMENTS:
    1. Download model:
       wget https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt
    
    2. Install packages:
       pip install ultralytics supervision opencv-python
    """
    import cv2
    
    # Check if model exists
    model_file = "yolov8n-face.pt"
    if not Path(model_file).exists():
        print(f"ERROR: Model file '{model_file}' not found!")
        print("\nDownload it with:")
        print("wget https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt")
        exit(1)
    
    # Initialize tracker
    print("Initializing YOLOv8-Face tracker...")
    tracker = HeadTracker(
        model_path=model_file,
        confidence_threshold=0.4,  # Slightly higher for faces
        device="cpu"  # Change to "cuda" if you have GPU
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("\nTracking faces... Press 'q' to quit")
    print("Look at the camera!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get face position
        face_pos, roll = tracker.get_head_position(frame)
        
        if face_pos is not None:
            # Convert normalized coords back to pixels for visualization
            h, w = frame.shape[:2]
            pixel_x = int((face_pos[0] + 1) * w / 2)
            pixel_y = int((face_pos[1] + 1) * h / 2)
            
            # Draw marker at face center
            cv2.circle(frame, (pixel_x, pixel_y), 10, (0, 255, 0), -1)
            cv2.putText(
                frame, 
                f"Face: ({face_pos[0]:.2f}, {face_pos[1]:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Also show all faces
            all_faces = tracker.get_all_detections(frame)
            cv2.putText(
                frame,
                f"Faces detected: {len(all_faces)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            
            print(f"Face center: x={face_pos[0]:+.2f}, y={face_pos[1]:+.2f}")
        else:
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        # Display
        cv2.imshow('YOLOv8-Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()