from flask_socketio import SocketIO, emit
from typing import Any
import time
from config import Config
from services.hand_detection_model import HandTracker

class SocketService:
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.hand_tracker = HandTracker()
        self._register_events()
        
    def _register_events(self):
        """ลงทะเบียน Socket.IO events"""
        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
            self.start_hand_tracking()
            
    def start_hand_tracking(self):
        """เริ่มต้นส่งข้อมูลมือ"""
        def tracking_loop():
            while True:
                landmarks = self.hand_tracker.get_landmarks()
                self.emit_landmarks(landmarks)
                time.sleep(0.05)  # 20 FPS
                
        self.socketio.start_background_task(tracking_loop)
    
    def emit_landmarks(self, data: Any):
        """ส่งข้อมูล landmarks ไปยัง client"""
        self.socketio.emit('hand_landmarks', data)
    
    def cleanup(self):
        """ทำความสะอาดทรัพยากร"""
        self.hand_tracker.release()