import cv2
import mediapipe as mp
from typing import Dict, Any
from config import Config

class HandTracker:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(**Config.HAND_MODEL_CONFIG)
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        
    def get_landmarks(self) -> Dict[str, Any]:
        success, frame = self.cap.read()
        if not success:
            return {}
        
        # convert the frame to RGB for MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Bring the image to MediaPipe
        results = self.hands.process(image)
        
        # This is output's structure for landmarks
        landmarks_data = {
            "left_hand": {},
            "right_hand": {}
        }

        # Map result from MediaPipe
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Define hand type -> Left or Right
                hand_type = self._get_hand_type(results, hand_idx)

                # Process landmarks for the hand
                self._process_landmarks(hand_landmarks, landmarks_data[hand_type])
                
        return landmarks_data
    
    def _get_hand_type(self, results, hand_idx: int) -> str:
        # Define hand type with index
        hand_type = results.multi_handedness[hand_idx].classification[0].label

        # Swap hand type (do this if you are not flip image before) 
        return 'left_hand' if hand_type == 'Right' else 'right_hand'
    
    def _process_landmarks(self, hand_landmarks, output: Dict[str, Any]):
        # Select only config landmarks and store them in output
        for landmark_id in Config.TARGET_LANDMARKS:
            landmark = hand_landmarks.landmark[landmark_id]
            output[str(landmark_id)] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            }
    
    def release(self):
        self.cap.release()
        self.hands.close()