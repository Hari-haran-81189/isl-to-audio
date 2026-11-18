
import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyttsx3
import time
from collections import deque, Counter

# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

saved_data = joblib.load("models/alphabet_model.joblib")

model = saved_data["model"]
feature_columns = saved_data["feature_columns"]

print("Alphabet AI model loaded!")

# ==========================================
# 2. SETTINGS
# ==========================================

STABLE_FRAMES = 8
MIN_STABLE_VOTES = 6
CONFIDENCE_THRESHOLD = 0.60

WORD_END_DELAY = 2.0
HAND_RELEASE_DELAY = 0.35

# ==========================================
# 3. TEXT TO SPEECH
# ==========================================



muted = False


def speak_text(text):

    if not muted and text:

        print("Speaking:", text)

        speaker = pyttsx3.init(driverName="sapi5")

        speaker.setProperty("rate", 145)

        speaker.say(text)
        speaker.runAndWait()

        speaker.stop()

        del speaker



# ==========================================
# 4. MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ==========================================
# 5. NORMALIZE LANDMARKS
# ==========================================

def normalize_hand(landmarks):

    points = []
    wrist = landmarks.landmark[0]

    for point in landmarks.landmark:

        x = point.x - wrist.x
        y = point.y - wrist.y
        z = point.z - wrist.z

        points.append([x, y, z])

    scale = max(
        (x*x + y*y + z*z) ** 0.5
        for x, y, z in points
    )

    if scale == 0:
        scale = 1

    features = []

    for x, y, z in points:

        features.extend([
            x / scale,
            y / scale,
            z / scale
        ])

    return features


# ==========================================
# 6. EXTRACT FEATURES
# ==========================================

def extract_features(results):

    left_features = [0.0] * 63
    right_features = [0.0] * 63

    left_present = 0
    right_present = 0

    if results.multi_hand_landmarks:

        for landmark, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            label = handedness.classification[0].label

            features = normalize_hand(landmark)

            if label == "Left":

                left_features = features
                left_present = 1

            elif label == "Right":

                right_features = features
                right_present = 1

    if not (left_present or right_present):
        return None

    return (
        [left_present, right_present]
        + left_features
        + right_features
    )


# ==========================================
# 7. CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera could not open.")

    hands.close()
    speaker.stop()

    raise SystemExit


# ==========================================
# 8. WORD BUILDER VARIABLES
# ==========================================

prediction_history = deque(maxlen=STABLE_FRAMES)

word = ""
last_completed_word = ""

current_letter = ""
confidence = 0.0

last_accepted_letter = ""
awaiting_new_sign = True

hand_missing_since = None

last_accepted_time = 0


# ==========================================
# 9. LIVE RECOGNITION
# ==========================================

try:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        # Draw hand landmarks

        if results.multi_hand_landmarks:

            for landmark in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    landmark,
                    mp_hands.HAND_CONNECTIONS
                )

        features = extract_features(results)

        # ==================================
        # 10. HAND DETECTED
        # ==================================

        if features is not None:

            hand_missing_since = None

            input_data = np.array(
                features,
                dtype=np.float32
            ).reshape(1, -1)

            if input_data.shape[1] != len(feature_columns):
                raise ValueError(
                    "Feature count does not match trained model!"
                )

            probabilities = model.predict_proba(input_data)[0]

            best_index = np.argmax(probabilities)

            prediction = model.classes_[best_index]

            confidence = float(probabilities[best_index])

            current_letter = str(prediction)

            if confidence >= CONFIDENCE_THRESHOLD:

                prediction_history.append(current_letter)

            else:

                prediction_history.clear()

            # ==================================
            # 11. CHECK STABLE LETTER
            # ==================================

            stable_letter = ""

            if len(prediction_history) == STABLE_FRAMES:

                counts = Counter(prediction_history)

                candidate, votes = counts.most_common(1)[0]

                if votes >= MIN_STABLE_VOTES:

                    stable_letter = candidate

            # ==================================
            # 12. ACCEPT LETTER ONLY ONCE
            # ==================================

            if stable_letter:

                if (
                    awaiting_new_sign
                    or stable_letter != last_accepted_letter
                ):

                    word += stable_letter

                    last_accepted_letter = stable_letter

                    awaiting_new_sign = False

                    last_accepted_time = time.monotonic()

                    print("Letter added:", stable_letter)
                    print("Current word:", word)

        # ==================================
        # 13. NO HAND DETECTED
        # ==================================

        else:

            current_letter = ""
            confidence = 0.0

            prediction_history.clear()

            if hand_missing_since is None:

                hand_missing_since = time.monotonic()

            missing_duration = (
                time.monotonic() - hand_missing_since
            )

            # Allow the same letter to be signed again
            # after briefly removing the hand.

            if missing_duration >= HAND_RELEASE_DELAY:

                awaiting_new_sign = True
                last_accepted_letter = ""

            # ==================================
            # 14. WORD COMPLETION
            # ==================================

            if (
                word
                and missing_duration >= WORD_END_DELAY
            ):

                completed_word = word

                last_completed_word = completed_word

                print("\nWord completed:", completed_word)

                speak_text(completed_word)

                word = ""

                last_accepted_letter = ""
                awaiting_new_sign = True

        # ==================================
        # 15. DISPLAY INFORMATION
        # ==================================

        cv2.putText(
            frame,
            f"LETTER: {current_letter or '-'}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"CONFIDENCE: {confidence:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"WORD: {word or '-'}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"LAST WORD: {last_completed_word or '-'}",
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"AUDIO: {'MUTED' if muted else 'ON'}",
            (20, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "C: Clear | B: Backspace | M: Mute | Q: Quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "ISL Translator - Version 2",
            frame
        )

        # ==================================
        # 16. KEYBOARD CONTROLS
        # ==================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):

            word = ""

            prediction_history.clear()

            print("Current word cleared.")

        elif key == ord("b"):

            word = word[:-1]

            print("Updated word:", word)

        elif key == ord("m"):

            muted = not muted


            

            print("Audio muted." if muted else "Audio enabled.")

        elif key == ord("q"):

            break


finally:

    cap.release()

    hands.close()


    cv2.destroyAllWindows()

    print("ISL Translator stopped.")
