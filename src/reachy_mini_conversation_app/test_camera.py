"""Test camera and face detection."""
import cv2
import logging
from reachy_mini import ReachyMini
from camera_worker import CameraWorker
from vision.yolo_head_tracker import HeadTracker

logging.basicConfig(level=logging.INFO)

print('Connecting to Reachy Mini...')
reachy = ReachyMini()

print('Loading YOLO face tracker...')
tracker = HeadTracker()

print('Starting camera worker...')
camera = CameraWorker(reachy, head_tracker=tracker)
camera.start()

print('\n==========================================================')
print('CAMERA + FACE DETECTION TEST')
print('==========================================================')
print('Green boxes = detected faces')
print('Press Q to quit, T to toggle tracking')
print('==========================================================\n')

cv2.namedWindow('Reachy Vision + Face Detection', cv2.WINDOW_NORMAL)

try:
    import time
    time.sleep(1)  # Let camera warm up
    
    while True:
        frame = camera.get_latest_frame()
        
        if frame is not None:
            # Use get_head_position - returns (eye_center, confidence)
            result = tracker.get_head_position(frame)
            
            if result is not None:
                eye_center, confidence = result
                face_detected = eye_center is not None
            else:
                eye_center = None
                confidence = 0.0
                face_detected = False
            
            # Draw face status
            status_color = (0, 255, 0) if face_detected else (0, 0, 255)
            cv2.putText(frame, f'Face detected: {"YES" if face_detected else "NO"}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            if face_detected:
                cv2.putText(frame, f'Confidence: {confidence:.2f}', 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw eye center point if detected
            if eye_center is not None:
                h, w, _ = frame.shape
                # Convert normalized coordinates (-1 to 1) to pixel coordinates
                eye_center_norm = (eye_center + 1) / 2
                eye_x = int(eye_center_norm[0] * w)
                eye_y = int(eye_center_norm[1] * h)
                
                # Draw crosshair at eye center
                cv2.drawMarker(frame, (eye_x, eye_y), (0, 255, 255), 
                              cv2.MARKER_CROSS, 30, 3)
                cv2.putText(frame, 'Eye Center', (eye_x + 20, eye_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Draw circle around detected region
                cv2.circle(frame, (eye_x, eye_y), 80, (0, 255, 0), 2)
            
            # Show face tracking status
            offsets = camera.get_face_tracking_offsets()
            tracking_on = camera.is_head_tracking_enabled
            
            tracking_color = (0, 255, 255) if tracking_on else (128, 128, 128)
            cv2.putText(frame, f'Head Tracking: {"ON" if tracking_on else "OFF"}', 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracking_color, 2)
            
            # Show tracking offsets if active
            if tracking_on and any(abs(o) > 0.01 for o in offsets):
                cv2.putText(frame, f'X: {offsets[0]:+.3f}  Y: {offsets[1]:+.3f}  Z: {offsets[2]:+.3f}', 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f'Roll: {offsets[3]:+.3f}  Pitch: {offsets[4]:+.3f}  Yaw: {offsets[5]:+.3f}', 
                           (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Reachy Vision + Face Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print('\nQuitting...')
            break
        elif key == ord('t'):
            # Toggle head tracking
            new_state = not camera.is_head_tracking_enabled
            camera.set_head_tracking_enabled(new_state)
            print(f'Head tracking: {"ON" if new_state else "OFF"}')

except KeyboardInterrupt:
    print('\nInterrupted by user')
except Exception as e:
    print(f'\nError: {e}')
    import traceback
    traceback.print_exc()
finally:
    print('\nCleaning up...')
    camera.stop()
    cv2.destroyAllWindows()
    print('Done!')