import math
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

hand_length_reference = None
hand_length_alpha = 0.1
click_threshold = 50


def calibrate_click_threshold():
    print("Calibrating... Open and close your hand a few times.")
    hand_lengths = []
    start_time = time.time()
    while time.time() - start_time < 5:
        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            thumb = hand.landmark[4]
            index = hand.landmark[8]
            h, w, c = img.shape
            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)
            length = math.hypot(x2 - x1, y2 - y1)
            hand_lengths.append(length)

    if hand_lengths:
        min_length = min(hand_lengths)
        max_length = max(hand_lengths)
        global click_threshold
        click_threshold = (min_length + max_length) / 2
        print(f"Calibration complete. Click threshold set to {click_threshold:.2f}")

calibrate_click_threshold()

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

                thumb = hand.landmark[4]
                index = hand.landmark[8]
                h, w, c = img.shape
                x1, y1 = int(thumb.x * w), int(thumb.y * h)
                x2, y2 = int(index.x * w), int(index.y * h)
                length = math.hypot(x2 - x1, y2 - y1)

                if hand_length_reference is None:
                    hand_length_reference = length
                else:
                    hand_length_reference = (1 - hand_length_alpha) * hand_length_reference + hand_length_alpha * length

                move_x = np.interp(index.x, [0, 1], [0, screen_width])
                move_y = np.interp(index.y, [0, 1], [0, screen_height])

                if length < click_threshold:
                    pyautogui.click()

                cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (0, 0, 255), cv2.FILLED)

                pyautogui.moveTo(move_x, move_y)

                cv2.putText(img, f'Hand Gesture: {length:.2f}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('Hand Gesture Mouse', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("Script terminated by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
