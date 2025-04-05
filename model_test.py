import cv2
import threading
import mediapipe as mp
import torch
import numpy as np
import os
from ultralytics import YOLO
import queue
import time

# load reference image and keypoints
ref_image_path = "images/person_15.jpg"
ref_image = cv2.imread(ref_image_path)
ref_h, ref_w, _ = ref_image.shape

# load reference labels keypoints
label_file = "labels/person_15_shoulder.txt"
ref_keypoints = []

with open(label_file, "r") as file:
    lines = file.readlines()
    for line in lines:
        parts = list(map(float, line.strip().split()))
        class_id = int(parts[0])
        x_center, y_center, width, height = parts[1:5]
        keypoints = []
        
        for i in range(5, len(parts), 3):  # Each keypoint has x, y, conf
            if i + 2 < len(parts):  # Ensure we have all 3 values
                kx, ky, conf = parts[i], parts[i+1], parts[i+2]
                keypoints.append((kx, ky, conf))
        
        # Calculate bounding box coordinates
        ref_x_min = (x_center - width/2) * ref_w
        ref_y_min = (y_center - height/2) * ref_h
        ref_bbox_width = width * ref_w
        ref_bbox_height = height * ref_h
        
        # Convert keypoints to relative coordinates
        keypoint_list = []
        for kx, ky, conf in keypoints:
            rel_x = (kx * ref_w - ref_x_min) / ref_bbox_width
            rel_y = (ky * ref_h - ref_y_min) / ref_bbox_height
            rel_x = max(0, min(1, rel_x))
            rel_y = max(0, min(1, rel_y))
            keypoint_list.append((rel_x, rel_y, conf))
        
        ref_keypoints.append((class_id, keypoint_list))
        
massage_type = "shoulder"

# initialize mediapipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    model = YOLO("model/new_best_seg.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    print("continue without model")
    model = None

print("Webcam initialized. Press 'q' to quit")

# Add shared data structures
hand_frame_queue = queue.Queue(maxsize=2)
hand_results_queue = queue.Queue(maxsize=2)
stop_threads = False

# configure frame processing settings
process_every_n_frames = 2
frame_count = 0

if model is not None:
    model.conf = 0.25  # Lower confidence threshold
    model.iou = 0.45   # Lower IoU threshold

model_frame_queue = queue.Queue(maxsize=2)
model_results_queue = queue.Queue(maxsize=2)
last_inference_time = 0
MIN_INFERENCE_INTERVAL = 0.05

last_valid_results = None
result_persistence_time = 0.5  # seconds
cache_duration = 0.5  # seconds
confidence_threshold = 0.25
smoothing_alpha = 0.5  # Smoothing factor for keypoints
debug_mode = True  # Set to True to enable debug mode

last_result_time = 0
results_cache = {}
last_valid_mask = None
last_valid_box = None
last_confidence = 0
smoothed_mask = None
smoothed_box = None

# Function to make smooth detection results
def smooth_detection(new_mask, new_box, new_conf, alpha):
    """Smooth detection results between frames"""
    global last_valid_mask, last_valid_box, smoothed_mask, smoothed_box
    
    if last_valid_mask is None or new_conf > 0.8:  # High confidence = less smoothing
        smoothed_mask = new_mask
        smoothed_box = new_box
    else:
        # Interpolate mask points
        try:
            smoothed_mask = alpha * np.array(last_valid_mask) + (1 - alpha) * np.array(new_mask)
            smoothed_mask = smoothed_mask.astype(np.int32)
            # Smooth bounding box
            smoothed_box = alpha * np.array(last_valid_box) + (1 - alpha) * np.array(new_box)
        except:
            smoothed_mask = new_mask
            smoothed_box = new_box
    
    last_valid_mask = smoothed_mask
    last_valid_box = smoothed_box
    return smoothed_mask, smoothed_box

def cache_results(mask, bbox_data, timestamp, confidence):
    """Enhanced caching with confidence weighting"""
    global results_cache, last_valid_results
    current_time = time.time()
    
    # If no detection (mask and bbox_data are None), clean the entire cache
    if mask is None or bbox_data is None:
        results_cache.clear()
        last_valid_results = None
        return
    
    # Clean old cache entries with decay
    results_cache = {
        k: v for k, v in results_cache.items()
        if current_time - v['timestamp'] < cache_duration * v['confidence']
    }
    
    if confidence >= confidence_threshold:
        cache_key = hash(str(bbox_data))
        # Smooth with existing cache entry if available
        if cache_key in results_cache:
            old_entry = results_cache[cache_key]
            alpha = min(confidence, 0.9)  # Higher confidence = less smoothing
            smoothed_mask, smoothed_box = smooth_detection(
                mask, bbox_data, confidence, alpha
            )
        else:
            smoothed_mask = mask
            smoothed_box = bbox_data
        
        results_cache[cache_key] = {
            'mask': smoothed_mask,
            'bbox': smoothed_box,
            'timestamp': timestamp,
            'confidence': confidence
        }
        
def get_cached_results(bbox_data):
    cache_key = hash(str(bbox_data))
    if cache_key in results_cache:
        return results_cache[cache_key]
    return None

def hand_detection_thread():
    """Thread function for hand detection"""
    global stop_threads
    
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while not stop_threads:
            try:
                frame = hand_frame_queue.get_nowait()
                if frame is None:
                    continue
                
                # Process frame for hand detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
                # Put results in queue
                hand_results_queue.put(results)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Hand detection error: {e}")

def model_inference_thread():
    """Thread function for YOLO model inference"""
    global stop_threads
    while not stop_threads:
        try:
            frame = model_frame_queue.get(timeout=0.1)
            if frame is None:
                continue
            
            # Process frame for model inference
            process_frame = cv2.resize(frame, (640, 480))
            results = model(process_frame, verbose=False)
            model_results_queue.put((results, frame.shape))
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Model inference error: {e}")
            
# Start hand detection thread
hand_thread = threading.Thread(target=hand_detection_thread)
hand_thread.start()

if model is not None:
    model_thread = threading.Thread(target=model_inference_thread)
    model_thread.daemon = True
    model_thread.start()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    current_time = time.time()
    frame_count += 1
    output_frame = frame.copy()
    
    if current_time - last_inference_time >= MIN_INFERENCE_INTERVAL:
        try:
            if not model_frame_queue.full():
                model_frame_queue.put_nowait(frame)
                last_inference_time = current_time
        except:
            pass
    
    # Process model results if available
    try:
        results, orig_shape = model_results_queue.get_nowait()
        if results and model is not None:
            current_time = time.time()
            target_masks = []
            
            # Process detections
            for r in results:
                if r.masks is None:
                    continue
                
                class_indices = r.boxes.cls.cpu().numpy().astype(int)
                class_names = [model.names[idx] for idx in class_indices]
                
                for i, (mask, class_name, conf) in enumerate(zip(r.masks.xy, class_names, r.boxes.conf)):
                    if class_name == massage_type:
                        target_masks.append((mask, conf.item(), r.boxes.xyxy[i].cpu().numpy()))
            
            # Enhanced confidence handling
            if target_masks:
                target_masks.sort(key=lambda x: x[1], reverse=True)
                mask, confidence, box = target_masks[0]
                
                if debug_mode:
                    print(f"Detection confidence: {confidence:.2f}")
                
                # Apply temporal smoothing
                smoothed_mask, smoothed_box = smooth_detection(
                    mask, box, confidence, smoothing_alpha
                )
                
                # Update cache with smoothed values
                cache_results(smoothed_mask, smoothed_box, current_time, confidence)
                last_valid_results = (smoothed_mask, smoothed_box)
                last_result_time = current_time
                
                # Use smoothed results for visualization
                mask, box = smoothed_mask, smoothed_box
            
            elif last_valid_results and (current_time - last_result_time) < result_persistence_time:
                mask, box = last_valid_results
                if debug_mode:
                    print("Using cached results")
            else:
                cached = get_cached_results(box if 'box' in locals() else None)
                if cached:
                    mask = cached['mask']
                    box = cached['bbox']
                    if debug_mode:
                        print("Using fall-back cache")
            
            if 'mask' in locals():
                mask_np = np.array(mask, dtype=np.int32)
                x_min, y_min = np.min(mask_np, axis=0)
                x_max, y_max = np.max(mask_np, axis=0)
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                
                # Scale coordinates back to original frame size
                scale_x = frame.shape[1] / orig_shape[1]
                scale_y = frame.shape[0] / orig_shape[0]
                x_min *= scale_x
                y_min *= scale_y
                bbox_width *= scale_x
                bbox_height *= scale_y
                
                # Simplified point drawing - just show all points
                for _, keypoints in ref_keypoints:
                    for kx_rel, ky_rel, conf in keypoints:
                        mapped_x = int(x_min + kx_rel * bbox_width)
                        mapped_y = int(y_min + ky_rel * bbox_height)
                        
                        # Draw point if it's inside the mask
                        if cv2.pointPolygonTest(mask_np, (mapped_x, mapped_y), False) >= 0:
                            cv2.circle(output_frame, (mapped_x, mapped_y), 5, (0, 255, 0), -1)
                            print(f"Point Position: ({mapped_x}, {mapped_y})")
    except queue.Empty:
        # Use cached results when queue is empty
        current_time = time.time()
        if last_valid_results and (current_time - last_result_time) < result_persistence_time:
            
            mask, box = last_valid_results
            # Continue with visualization using last valid results
            mask_np = np.array(mask, dtype=np.int32)
            x_min, y_min = np.min(mask_np, axis=0)
            x_max, y_max = np.max(mask_np, axis=0)
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            
            # Scale coordinates back to original frame size
            scale_x = frame.shape[1] / orig_shape[1]
            scale_y = frame.shape[0] / orig_shape[0]
            x_min *= scale_x
            y_min *= scale_y
            bbox_width *= scale_x
            bbox_height *= scale_y
            
            # Simplified point drawing - just show all points
            for _, keypoints in ref_keypoints:
                for kx_rel, ky_rel, conf in keypoints:
                    mapped_x = int(x_min + kx_rel * bbox_width)
                    mapped_y = int(y_min + ky_rel * bbox_height)
                    
                    # Draw point if it's inside the mask
                    if cv2.pointPolygonTest(mask_np, (mapped_x, mapped_y), False) >= 0:
                        cv2.circle(output_frame, (mapped_x, mapped_y), 5, (0, 255, 0), -1)
                        print(f"Point Position: ({mapped_x}, {mapped_y})")
        
        pass
    
    # Process hand detection every frame for smoothness
    try:
        hand_frame_queue.put_nowait(frame)
    except queue.Full:
        try:
            hand_frame_queue.get_nowait()  # Remove old frame
            hand_frame_queue.put_nowait(frame)
        except:
            pass
    
    # Draw hand landmarks if detected
    try:
        hand_results = hand_results_queue.get_nowait()
    except queue.Empty:
        hand_results = None

    if hand_results and hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                output_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )

    cv2.imshow('Realtime Detection', output_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # If 'q' is pressed, break the loop
    if key == ord('q'):
        stop_threads = True
        break

cap.release()
cv2.destroyAllWindows()

# Clean up threads before exit
stop_threads = True
hand_thread.join()
if model is not None:
    model_thread.join(timeout=1.0)

print("Application closed.")