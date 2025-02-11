import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

WIDTH, HEIGHT = 640, 480

num_marks = 5
mark_positions = [(WIDTH // (num_marks+1)) * (i + 1) for i in range(num_marks)]

def detect_finger_position(hand_landmarks):
    """Xác định vị trí của đầu ngón trỏ trên trục X"""
    index_finger_tip = hand_landmarks.landmark[8] 
    finger_x = int(index_finger_tip.x * WIDTH)  
    return finger_x

cap = cv2.VideoCapture(0)

last_selected_mark = None  
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    for i, pos in enumerate(mark_positions):
        cv2.circle(frame, (pos, 50), 20, (0, 255, 0), -1) 
        cv2.putText(frame, str(i), (pos - 10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

   
    selected_mark = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_x = detect_finger_position(hand_landmarks)

            for i, pos in enumerate(mark_positions):
                if abs(finger_x - pos) < 20:
                    selected_mark = i + 1
                    break

    if selected_mark and selected_mark != last_selected_mark:
        print(f'Output_cmd: {(selected_mark * 25)-25}')
        last_selected_mark = selected_mark  

    cv2.imshow("Dieu Chinh", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
