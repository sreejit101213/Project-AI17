import cv2, time, pyautogui, os, sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Force Python to look in the script's directory for the model
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0] if sys.argv[0] else __file__))
os.chdir(SCRIPT_DIR)
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

SCROLL_SPEED = 300
CAM_WIDTH, CAM_HEIGHT = 640, 480
latest_result = None

# MediaPipe connections map (Which landmark connects to which)
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),       # Thumb
    (0,5), (5,6), (6,7), (7,8),       # Index
    (5,9), (9,10), (10,11), (11,12),  # Middle
    (9,13), (13,14), (14,15), (15,16),# Ring
    (13,17), (0,17), (17,18), (18,19), (19,20) # Pinky
]

def receive_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int): # type: ignore
    global latest_result
    latest_result = result

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    result_callback=receive_result
)
detector = vision.HandLandmarker.create_from_options(options)

def draw_custom_landmarks(frame, landmarks, w, h):
    # Draw connection lines
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    # Draw joint dots
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

def detect_gesture(hand_landmarks, handedness_label):
    fingers = []
    tips = [8, 12, 16, 20]
    for tip in tips:
        if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
            fingers.append(1)

    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    if (handedness_label == "Right" and thumb_tip.x > thumb_ip.x) or \
       (handedness_label == "Left" and thumb_tip.x < thumb_ip.x):
        fingers.append(1)
    
    return "scroll up" if sum(fingers) == 5 else "scroll down" if len(fingers) == 0 else "none"

cap = cv2.VideoCapture(1)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)
last_scroll = p_time = 0

WIN = "Gesture Control"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    detector.detect_async(mp_image, int(time.time() * 1000))
    
    gesture, handedness_label = "none", "Unknown"

    if latest_result and latest_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(latest_result.hand_landmarks):
            if idx < len(latest_result.handedness):
                handedness_label = latest_result.handedness[idx][0].category_name
            
            gesture = detect_gesture(hand_landmarks, handedness_label)
            
            # Use our custom drawing function instead of legacy tools
            draw_custom_landmarks(frame, hand_landmarks, w, h)

            if (time.time() - last_scroll) > 0.5:
                if gesture == "scroll up":
                    pyautogui.scroll(SCROLL_SPEED)
                elif gesture == "scroll down":
                    pyautogui.scroll(-SCROLL_SPEED)
                last_scroll = time.time()

    c_time = time.time()
    fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
    p_time = c_time
    
    cv2.putText(frame, f"FPS: {int(fps)} | Hand: {handedness_label} | Gesture: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imshow(WIN, frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('e') or key == 27:
        break
    try:
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1: break
    except:
        break

detector.close()
cap.release()
cv2.destroyAllWindows()