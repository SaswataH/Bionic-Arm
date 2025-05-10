import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load model and label encoder
model = joblib.load('hand_gesture_svm_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            features = []
            for landmark in hand_landmarks.landmark:
                features.append(landmark.x)
                features.append(landmark.y)
                features.append(landmark.z)

            if len(features) == 63:
                prediction = model.predict([features])
                gesture = label_encoder.inverse_transform(prediction)[0]
                
                cv2.putText(frame, gesture, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0,255,0), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Hand Gesture Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
