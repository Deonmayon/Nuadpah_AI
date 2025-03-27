class Config:
    SOCKETIO_PORT = 5000
    HOST = '0.0.0.0'
    CAMERA_INDEX = 0
    HAND_MODEL_CONFIG = {
        'max_num_hands': 2,
        'min_detection_confidence': 0.7,
        'min_tracking_confidence': 0.5
    }
    TARGET_LANDMARKS = [4, 8, 12, 16, 20]  # Landmark points for fingertips of (thumb, index, middle, ring, pinky)