# Testing live video capture using OBS Droid Cam and Open CV 

import cv2

cap = cv2.VideoCapture(2)  # replace with your index

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't read frame.")
        break
    cv2.imshow("OBS Cam Feed", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
