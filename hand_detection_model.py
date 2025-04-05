import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,         
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5  
)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("เปิดกล้องไม่ได้")
    exit()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("อ่านไม่ได้")
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # เข้าถึงเฉพาะ THUMB_TIP (ปลายนิ้วโป้ง)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            h, w, _ = image.shape
            cx, cy = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # วาดวงกลมเฉพาะที่ปลายนิ้วโป้ง
            cv2.circle(image, (cx, cy), 10, (0, 255, 0), -1)

    cv2.imshow('Thumb Tip Only', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("ปิด")
        break

cap.release()
cv2.destroyAllWindows()