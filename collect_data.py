#Data collection using the live video feed -> Press 'o' to record Open Fist , 't' to record Thumbs Up , 'f' to record Fist data in a .csv file

import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# File setup
csv_file = "hand_data.csv"
fieldnames = [f"{axis}{i}" for i in range(21) for axis in ('x', 'y', 'z')] + ["label"]

# Create file with headers if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# Start webcam
cap = cv2.VideoCapture(2)

print("Press keys: 'f' for Fist, 'o' for Open, 't' for Thumbs Up, 'q' to Quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Show frame
            cv2.imshow("Hand Tracker", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

            # Label and Save Data
            label_map = {'f': 'fist', 'o': 'open', 't': 'thumbs_up'}
            if chr(key) in label_map:
                label = label_map[chr(key)]
                row = landmarks + [label]
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                print(f"Saved: {label}")

    else:
        cv2.imshow("Hand Tracker", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
