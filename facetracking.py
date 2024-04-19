import cv2
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np
import time

cap = cv2.VideoCapture(0)
ws, hs = 1280, 720
cap.set(3, ws)
cap.set(4, hs)

if not cap.isOpened():
    print("Camera couldn't Access!!!")
    exit()

detector = FaceDetector()
prev_face_x, prev_face_y = None, None
prev_time = time.time()

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img, draw=False)

    if bboxs:
        # Get the head position (top-center of the bounding box)
        head_x = bboxs[0]["bbox"][0] + bboxs[0]["bbox"][2] // 2
        head_y = bboxs[0]["bbox"][1]

        # Draw the red circle at the head position
        cv2.circle(img, (head_x, head_y), 80, (0, 0, 255), 2)
        cv2.circle(img, (head_x, head_y), 15, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, "TARGET LOCKED", (850, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Calculate face motion trajectory and speed
        if prev_face_x is not None and prev_face_y is not None:
            # Calculate motion trajectory
            trajectory = np.sqrt((head_x - prev_face_x) ** 2 + (head_y - prev_face_y) ** 2)

            # Calculate time difference
            current_time = time.time()
            time_diff = current_time - prev_time

            # Calculate speed
            speed = trajectory / time_diff

            # Print motion trajectory and speed
            print("Motion Trajectory:", trajectory)
            print("Speed:", speed, "pixels per second")

            # Draw motion trajectory line
            cv2.line(img, (prev_face_x, prev_face_y), (head_x, head_y), (0, 255, 0), 2)

            # Draw arrow to indicate direction of motion
            cv2.arrowedLine(img, (prev_face_x, prev_face_y), (head_x, head_y), (255, 0, 0), 2)

            # Display speed value
            cv2.putText(img, f"Speed: {speed:.2f} pixels per second", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)

            # Update previous face position and time
            prev_face_x, prev_face_y = head_x, head_y
            prev_time = current_time
        else:
            # Initialize previous face position and time
            prev_face_x, prev_face_y = head_x, head_y
            prev_time = time.time()
    else:
        cv2.putText(img, "NO TARGET", (880, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.circle(img, (640, 360), 80, (0, 0, 255), 2)
        cv2.circle(img, (640, 360), 15, (0, 0, 255), cv2.FILLED)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
