import cv2
import mediapipe as mp
import pyautogui
import time
#YERS
# Mediapipe setups
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Webcam
cap = cv2.VideoCapture(0)

# Screen size
screen_w, screen_h = pyautogui.size()

# Cooldown timers
last_single_click_time = 0
last_double_click_time = 0
last_right_click_time = 0
click_cooldown = 2  # seconds

# Track click and hold state
is_left_mouse_down = False

def get_smile_percentage(landmarks):
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]

    mouth_width = ((right_mouth.x - left_mouth.x) ** 2 + (right_mouth.y - left_mouth.y) ** 2) ** 0.5
    mouth_height = ((bottom_lip.x - top_lip.x) ** 2 + (bottom_lip.y - top_lip.y) ** 2) ** 0.5

    ratio = mouth_height / mouth_width

    if ratio < 0.18:
        return 0
    elif ratio < 0.20:
        return 25
    elif ratio < 0.23:
        return 50
    elif ratio < 0.26:
        return 75
    else:
        return 100

def is_palm_open(hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    open_fingers = 0

    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            open_fingers += 1

    return open_fingers >= 4

def is_index_finger_up_only(hand_landmarks):
    """
    Check if right hand index finger is up and all others are down.
    """
    finger_tips = {
        "thumb": 4,
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20
    }
    finger_pips = {
        "thumb": 2,
        "index": 6,
        "middle": 10,
        "ring": 14,
        "pinky": 18
    }

    # Index finger up?
    index_up = hand_landmarks.landmark[finger_tips["index"]].y < hand_landmarks.landmark[finger_pips["index"]].y

    # Other fingers down?
    middle_down = hand_landmarks.landmark[finger_tips["middle"]].y > hand_landmarks.landmark[finger_pips["middle"]].y
    ring_down = hand_landmarks.landmark[finger_tips["ring"]].y > hand_landmarks.landmark[finger_pips["ring"]].y
    pinky_down = hand_landmarks.landmark[finger_tips["pinky"]].y > hand_landmarks.landmark[finger_pips["pinky"]].y
    thumb_down = hand_landmarks.landmark[finger_tips["thumb"]].x > hand_landmarks.landmark[finger_pips["thumb"]].x  # For thumb use x coord (approx)

    return index_up and middle_down and ring_down and pinky_down and thumb_down

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb_frame)
    hands_results = hands.process(rgb_frame)

    img_h, img_w, _ = frame.shape

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            nose_tip = landmarks[1]
            x = int((nose_tip.x - 0.5) * 4 * screen_w + screen_w / 2)
            y = int((nose_tip.y - 0.5) * 4 * screen_h + screen_h / 2)

            x = max(0, min(screen_w - 1, x))
            y = max(0, min(screen_h - 1, y))

            pyautogui.moveTo(x, y)

            smile_percent = get_smile_percentage(landmarks)
            cv2.putText(frame, f"Smile: {smile_percent}%", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            current_time = time.time()

            if smile_percent == 50 and current_time - last_single_click_time > click_cooldown:
                pyautogui.click()
                last_single_click_time = current_time
                print("Single Click - Smile 50%")

            if smile_percent == 100 and current_time - last_double_click_time > click_cooldown:
                pyautogui.doubleClick()
                last_double_click_time = current_time
                print("Double Click - Smile 100%")

    # Hand detection for right click (right hand palm open) & left click hold (right index finger up only)
    if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
        for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
            if handedness.classification[0].label == 'Right':
                # Right hand palm open for right click
                if is_palm_open(hand_landmarks):
                    current_time = time.time()
                    if current_time - last_right_click_time > click_cooldown:
                        pyautogui.click(button='right')
                        last_right_click_time = current_time
                        print("Right Click - Right hand palm open detected")

                # Right hand index finger up only → click and hold left button
                if is_index_finger_up_only(hand_landmarks):
                    if not is_left_mouse_down:
                        pyautogui.mouseDown(button='left')
                        is_left_mouse_down = True
                        print("Left mouse button DOWN - index finger up")
                else:
                    if is_left_mouse_down:
                        pyautogui.mouseUp(button='left')
                        is_left_mouse_down = False
                        print("Left mouse button UP - index finger not up")

    cv2.imshow("Nose Mouse & Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
