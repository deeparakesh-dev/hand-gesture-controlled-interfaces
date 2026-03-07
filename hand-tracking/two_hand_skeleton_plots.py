import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not opening")
    exit()

print("Camera opened successfully. Press ESC to stop and show plots.")

# Lists to store motion signals
left_x, left_y = [], []
right_x, right_y = [], []

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:

        for i, handLms in enumerate(results.multi_hand_landmarks):

            # Draw skeleton
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Compute centroid of hand
            xs = [lm.x * w for lm in handLms.landmark]
            ys = [lm.y * h for lm in handLms.landmark]

            cx = int(np.mean(xs))
            cy = int(np.mean(ys))

            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

            hand_label = results.multi_handedness[i].classification[0].label

            if hand_label == "Left":
                left_x.append(cx)
                left_y.append(cy)
            else:
                right_x.append(cx)
                right_y.append(cy)

    cv2.imshow("Hand Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# -------- PLOT SIGNALS --------

plt.figure(figsize=(12,6))

plt.subplot(2,2,1)
plt.plot(left_x)
plt.title("Left Hand Horizontal Motion x(t)")
plt.xlabel("Time (frames)")
plt.ylabel("X Position (pixels)")
plt.grid()

plt.subplot(2,2,2)
plt.plot(left_y)
plt.title("Left Hand Vertical Motion y(t)")
plt.xlabel("Time (frames)")
plt.ylabel("Y Position (pixels)")
plt.grid()

plt.subplot(2,2,3)
plt.plot(right_x)
plt.title("Right Hand Horizontal Motion x(t)")
plt.xlabel("Time (frames)")
plt.ylabel("X Position (pixels)")
plt.grid()

plt.subplot(2,2,4)
plt.plot(right_y)
plt.title("Right Hand Vertical Motion y(t)")
plt.xlabel("Time (frames)")
plt.ylabel("Y Position (pixels)")
plt.grid()

plt.tight_layout()
plt.show()