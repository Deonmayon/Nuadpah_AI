import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO

model = YOLO("model/new_best_seg.pt")

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

cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or change if you have multiple cameras

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam initialized. Press 'q' to quit, 's' to save current frame.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Get frame dimensions
    frame_h, frame_w, _ = frame.shape
    
    original_frame = frame.copy()
    processing_frame = frame
    scale_factor = 1.0
    
    if frame_w > 1500 or frame_h > 1500:
        scale_factor = 1500 / max(frame_w, frame_h)
        new_w = int(frame_w * scale_factor)
        new_h = int(frame_h * scale_factor)
        processing_frame = cv2.resize(frame, (new_w, new_h))
    
    size_ratio = frame_w / TARGET_WIDTH
    base_radius = int(12 * size_ratio)
    base_radius = max(5, min(base_radius, 30))
    
    results = model(processing_frame, verbose=False)
    
    output_frame = original_frame.copy()

    target_masks = []
    for r in results:
        if r.masks is None:
            continue
            
        class_indices = r.boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[idx] for idx in class_indices]
        
        for i, (mask, class_name, conf) in enumerate(zip(r.masks.xy, class_names, r.boxes.conf)):
            if class_name == TARGET_CLASS_NAME:
                if scale_factor != 1.0:
                    scaled_mask = mask / scale_factor
                    target_masks.append((scaled_mask, conf.item(), r.boxes.xyxy[i].cpu().numpy() / scale_factor))
                else:
                    target_masks.append((mask, conf.item(), r.boxes.xyxy[i].cpu().numpy()))
    
    # Sort by confidence and take the most confident detection
    if target_masks:
        target_masks.sort(key=lambda x: x[1], reverse=True)
        mask, confidence, box = target_masks[0]
        
        mask_np = np.array(mask, dtype=np.int32)
        
        x_min, y_min = np.min(mask_np, axis=0)
        x_max, y_max = np.max(mask_np, axis=0)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
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
    
    cv2.imshow('Realtime Detection', output_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # If 'q' is pressed, break the loop
    if key == ord('q'):
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
print("Application closed.")