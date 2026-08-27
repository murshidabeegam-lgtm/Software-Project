import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Finger Tip IDs: [Thumb, Index, Middle, Ring, Pinky]
tip_ids = [4, 8, 12, 16, 20]

def detect_gesture(lm):
    """
    Translates hand sign landmarks into letters using finger state logic.
    lm: List of 21 normalized hand landmarks (lm[i].x, lm[i].y, lm[i].z)
    """
    fingers = []

    # 1. Thumb State (Checks relative position to joint 3 based on hand orientation)
    if lm[tip_ids[0]].x > lm[tip_ids[0] - 1].x:
        fingers.append(1)  # Thumb open/extended
    else:
        fingers.append(0)  # Thumb tucked

    # 2. 4 Fingers State (Index, Middle, Ring, Pinky)
    # Checks if finger tip Y-coordinate is higher up (smaller value) than its lower PIP joint (tip - 2)
    for id in range(1, 5):
        if lm[tip_ids[id]].y < lm[tip_ids[id] - 2].y:
            fingers.append(1)  # Extended
        else:
            fingers.append(0)  # Folded

    # 3. Gesture Rule Translator
    # fingers array structure: [Thumb, Index, Middle, Ring, Pinky]
    
    # Sign 'A': All 4 fingers folded, thumb resting alongside hand
    if fingers == [1, 0, 0, 0, 0] or fingers == [0, 0, 0, 0, 0]:
        return "A", 0.95

    # Sign 'B': 4 fingers up straight, thumb tucked in
    elif fingers == [0, 1, 1, 1, 1]:
        return "B", 0.92

    # Sign 'V' or '2': Index and Middle up, others down
    elif fingers == [0, 1, 1, 0, 0]:
        return "V", 0.98

    # Sign 'L': Thumb and Index extended outwards
    elif fingers == [1, 1, 0, 0, 0]:
        return "L", 0.96

    # Sign 'Y': Thumb and Pinky extended out
    elif fingers == [1, 0, 0, 0, 1]:
        return "Y", 0.94

    # Sign 'FIVE' / Open Palm: All fingers up
    elif fingers == [1, 1, 1, 1, 1]:
        return "5", 0.99

    # Default fallback when position doesn't match predefined rules
    return "UNKNOWN", 0.00


def process_frame(frame):
    """
    Takes an OpenCV BGR image frame, extracts landmarks, returns overlay rendering 
    along with predicted sign letter and confidence score.
    """
    # Flip frame for natural mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)

    detected_letter = "None"
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Render Skeleton (Green points for joints, Red for connecting bones)
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(78, 237, 140), thickness=3, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 99, 132), thickness=2)
            )

            # Perform sign classification translation
            detected_letter, confidence = detect_gesture(hand_landmarks.landmark)

    # Return RGB frame for Streamlit display along with detected results
    annotated_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return annotated_rgb, detected_letter, confidence
