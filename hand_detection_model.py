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
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("ปิด")
        break

cap.release()
cv2.destroyAllWindows()

