import cv2
import threading
import mediapipe as mp
import torch
import numpy as np
import os
from ultralytics import YOLO
import queue
import time  # Add this import

output_dir = "output_images/"
os.makedirs(output_dir, exist_ok=True)

ref_image_path = "images/person_15.jpg"
ref_image = cv2.imread(ref_image_path)
ref_h, ref_w, _ = ref_image.shape

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

active_line = 0  # Track which line we're currently processing
active_pair_index = 0  # Track which pair within the current line is active

def get_keypoint_pairs(keypoints_list):
    """
    Create pairs for each line of keypoints
    Returns a list of pairs for each line
    """
    all_line_pairs = []
    
    for class_id, keypoints in keypoints_list:
        line_pairs = []
        num_points = len(keypoints)
        mid_point = num_points // 2
        # Create pairs between first half and second half points
        for i in range(mid_point):
            line_pairs.append((i, i + mid_point))
        all_line_pairs.append(line_pairs)
    
    return all_line_pairs

def get_sequential_keypoint_pairs(keypoints_list):
    all_line_pairs = []
    for class_id, keypoints in keypoints_list:
        line_pairs = []
        num_points = len(keypoints)
        if num_points > 1:
            mid_point = num_points // 2
            for i in range(0, mid_point - 1, 2):  # Step in groups of 2
                line_pairs.append((i, i + 1))
                if i + mid_point < num_points - 1:
                    line_pairs.append((i + mid_point, i + 1 + mid_point))
        all_line_pairs.append(line_pairs)   
    return all_line_pairs

TARGET_CLASS_NAME = "shoulder"
TARGET_WIDTH = 800  # Standard width to calculate relative circle size

GREEN_COLOR = (0, 255, 0)  # Active pair color
DEEP_GREEN_COLOR = (0, 102, 0)  # Inactive pair color

color_flipped = False  # Replace color_state with this simpler toggle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize camera first
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load YOLO model with error handling
try:
    model = YOLO("model/new_best_seg.pt")
    # Warm up the model
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy_img, verbose=False)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Continuing with camera feed only...")
    model = None

print("Webcam initialized. Press 'q' to quit, 's' to save current frame.")

# Add shared data structures
hand_frame_queue = queue.Queue(maxsize=2)
hand_results_queue = queue.Queue(maxsize=2)
stop_threads = False

# Add frame processing settings
PROCESS_EVERY_N_FRAMES = 2  # Process every Nth frame
frame_counter = 0
size_ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / TARGET_WIDTH
base_radius = int(12 * size_ratio)
base_radius = max(5, min(base_radius, 30))

# Optimize model settings
if model is not None:
    model.conf = 0.25  # Lower confidence threshold
    model.iou = 0.45   # Lower IoU threshold

# Add new queues for model inference
model_frame_queue = queue.Queue(maxsize=2)
model_results_queue = queue.Queue(maxsize=2)
last_inference_time = 0
MIN_INFERENCE_INTERVAL = 0.05  # 20 FPS max for model inference

# Add caching and smoothing mechanisms
last_valid_results = None
RESULT_PERSISTENCE_TIME = 0.5  # Increased from 0.1
CACHE_DURATION = 1.0  # Increased from 0.5
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detection
SMOOTHING_ALPHA = 0.7  # Temporal smoothing factor
DEBUG_MODE = True  # Enable debug printing

last_result_time = 0
results_cache = {}
last_valid_mask = None
last_valid_box = None
last_confidence = 0
smoothed_mask = None
smoothed_box = None

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
    global results_cache
    current_time = time.time()
    
    # Clean old cache entries with decay
    results_cache = {
        k: v for k, v in results_cache.items()
        if current_time - v['timestamp'] < CACHE_DURATION * v['confidence']
    }
    
    if confidence >= CONFIDENCE_THRESHOLD:
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

# Start model inference thread if model is loaded
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
    frame_counter += 1
    output_frame = frame.copy()
    
    # Queue frame for model inference with time control
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
                    if class_name == TARGET_CLASS_NAME:
                        target_masks.append((mask, conf.item(), r.boxes.xyxy[i].cpu().numpy()))
            
            # Enhanced confidence handling
            if target_masks:
                target_masks.sort(key=lambda x: x[1], reverse=True)
                mask, confidence, box = target_masks[0]
                
                if DEBUG_MODE:
                    print(f"Detection confidence: {confidence:.2f}")
                
                # Apply temporal smoothing
                smoothed_mask, smoothed_box = smooth_detection(
                    mask, box, confidence, SMOOTHING_ALPHA
                )
                
                # Update cache with smoothed values
                cache_results(smoothed_mask, smoothed_box, current_time, confidence)
                last_valid_results = (smoothed_mask, smoothed_box)
                last_result_time = current_time
                
                # Use smoothed results for visualization
                mask, box = smoothed_mask, smoothed_box
            
            elif last_valid_results and (current_time - last_result_time) < RESULT_PERSISTENCE_TIME:
                mask, box = last_valid_results
                if DEBUG_MODE:
                    print("Using cached results")
            else:
                cached = get_cached_results(box if 'box' in locals() else None)
                if cached:
                    mask = cached['mask']
                    box = cached['bbox']
                    if DEBUG_MODE:
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
                
                all_line_pairs = get_keypoint_pairs(ref_keypoints)
                sequential_pairs = get_sequential_keypoint_pairs(ref_keypoints)
                
                for idx, (class_id, keypoints) in enumerate(ref_keypoints):
                    # Draw sequential connections only for current line
                    if idx == active_line:  # Only draw sequential connections for active line
                        for pair in sequential_pairs[idx]:
                            k1, k2 = pair
                            kx1, ky1, _ = keypoints[k1]
                            kx2, ky2, _ = keypoints[k2]
                            
                            mapped_x1 = int(x_min + kx1 * bbox_width)
                            mapped_y1 = int(y_min + ky1 * bbox_height)
                            mapped_x2 = int(x_min + kx2 * bbox_width)
                            mapped_y2 = int(y_min + ky2 * bbox_height)
                            
                            # Draw sequential connections in yellow for active line
                            if cv2.pointPolygonTest(mask_np, (mapped_x1, mapped_y1), False) >= 0 and \
                               cv2.pointPolygonTest(mask_np, (mapped_x2, mapped_y2), False) >= 0:
                                cv2.line(output_frame, (mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (0, 102, 0), 2)
                    
                    # Only process active line points and pairs
                    if idx != active_line:
                        # Draw inactive points from other lines
                        for kidx, (kx_rel, ky_rel, conf) in enumerate(keypoints):
                            mapped_x = int(x_min + kx_rel * bbox_width)
                            mapped_y = int(y_min + ky_rel * bbox_height)
                            if cv2.pointPolygonTest(mask_np, (mapped_x, mapped_y), False) >= 0:
                                cv2.circle(output_frame, (mapped_x, mapped_y), base_radius, DEEP_GREEN_COLOR, -1)
                        continue
                    
                    # Process points for active line
                    line_pairs = all_line_pairs[idx]
                    for kidx, (kx_rel, ky_rel, conf) in enumerate(keypoints):
                        mapped_x = int(x_min + kx_rel * bbox_width)
                        mapped_y = int(y_min + ky_rel * bbox_height)
                        
                        # Check if this point is part of the active pair
                        is_active_point = False
                        if active_pair_index < len(line_pairs):
                            current_pair = line_pairs[active_pair_index]
                            is_active_point = kidx in current_pair
                        
                        color = GREEN_COLOR if is_active_point else DEEP_GREEN_COLOR
                        
                        # Draw point
                        if cv2.pointPolygonTest(mask_np, (mapped_x, mapped_y), False) >= 0:
                            cv2.circle(output_frame, (mapped_x, mapped_y), base_radius, color, -1)
                            # Print position only for active points
                            if is_active_point:
                                print(f"Active Keypoint Position: ({mapped_x}, {mapped_y})")
    except queue.Empty:
        # Use cached results when queue is empty
        current_time = time.time()
        if last_valid_results and (current_time - last_result_time) < RESULT_PERSISTENCE_TIME:
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
            
            all_line_pairs = get_keypoint_pairs(ref_keypoints)
            sequential_pairs = get_sequential_keypoint_pairs(ref_keypoints)
            
            for idx, (class_id, keypoints) in enumerate(ref_keypoints):
                # Draw sequential connections only for current line
                if idx == active_line:  # Only draw sequential connections for active line
                    for pair in sequential_pairs[idx]:
                        k1, k2 = pair
                        kx1, ky1, _ = keypoints[k1]
                        kx2, ky2, _ = keypoints[k2]
                        
                        mapped_x1 = int(x_min + kx1 * bbox_width)
                        mapped_y1 = int(y_min + ky1 * bbox_height)
                        mapped_x2 = int(x_min + kx2 * bbox_width)
                        mapped_y2 = int(y_min + ky2 * bbox_height)
                        
                        # Draw sequential connections in yellow for active line
                        if cv2.pointPolygonTest(mask_np, (mapped_x1, mapped_y1), False) >= 0 and \
                           cv2.pointPolygonTest(mask_np, (mapped_x2, mapped_y2), False) >= 0:
                            cv2.line(output_frame, (mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (0, 102, 0), 2)
                
                # Only process active line points and pairs
                if idx != active_line:
                    # Draw inactive points from other lines
                    for kidx, (kx_rel, ky_rel, conf) in enumerate(keypoints):
                        mapped_x = int(x_min + kx_rel * bbox_width)
                        mapped_y = int(y_min + ky_rel * bbox_height)
                        if cv2.pointPolygonTest(mask_np, (mapped_x, mapped_y), False) >= 0:
                            cv2.circle(output_frame, (mapped_x, mapped_y), base_radius, DEEP_GREEN_COLOR, -1)
                    continue
                
                # Process points for active line
                line_pairs = all_line_pairs[idx]
                for kidx, (kx_rel, ky_rel, conf) in enumerate(keypoints):
                    mapped_x = int(x_min + kx_rel * bbox_width)
                    mapped_y = int(y_min + ky_rel * bbox_height)
                    
                    # Check if this point is part of the active pair
                    is_active_point = False
                    if active_pair_index < len(line_pairs):
                        current_pair = line_pairs[active_pair_index]
                        is_active_point = kidx in current_pair
                    
                    color = GREEN_COLOR if is_active_point else DEEP_GREEN_COLOR
                    
                    # Draw point
                    if cv2.pointPolygonTest(mask_np, (mapped_x, mapped_y), False) >= 0:
                        cv2.circle(output_frame, (mapped_x, mapped_y), base_radius, color, -1)
                        # Print position only for active points
                        if is_active_point:
                            print(f"Active Keypoint Position: ({mapped_x}, {mapped_y})")
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
    # If 'c' is pressed, cycle to next pair
    elif key == ord('c'):
        if active_line < len(all_line_pairs):
            active_pair_index = (active_pair_index + 1) % len(all_line_pairs[active_line])
            # If we've cycled through all pairs in this line, move to next line
            if active_pair_index == 0:
                active_line = (active_line + 1) % len(all_line_pairs)
            print(f"Line {active_line + 1}, Pair {active_pair_index + 1}: {all_line_pairs[active_line][active_pair_index]}")
    # If 's' is pressed, save the current frame
    elif key == ord('s'):
        timestamp = int(cv2.getTickCount())
        save_path = os.path.join(output_dir, f"capture_{timestamp}.jpg")
        cv2.imwrite(save_path, output_frame)
        print(f"Saved current frame to {save_path}")

cap.release()
cv2.destroyAllWindows()

# Clean up threads before exit
stop_threads = True
hand_thread.join()
if model is not None:
    model_thread.join(timeout=1.0)

print("Application closed.")