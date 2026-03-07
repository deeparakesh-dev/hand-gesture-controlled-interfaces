import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# -------- MediaPipe Setup --------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # only one hand
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_cx, prev_cy = None, None
MOVE_THRESHOLD = 30

# Lists to store motion signals
x_signal = []
y_signal = []

print("Skeleton-based hand gesture system started. Press ESC to exit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    command = ""

    if results.multi_hand_landmarks:

        handLms = results.multi_hand_landmarks[0]

        # Draw skeleton
        mp_draw.draw_landmarks(
            frame, handLms, mp_hands.HAND_CONNECTIONS
        )

        # Compute hand centroid
        xs = [lm.x * w for lm in handLms.landmark]
        ys = [lm.y * h for lm in handLms.landmark]

        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

        # Show centroid
        cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

        # Save motion signals
        x_signal.append(cx)
        y_signal.append(cy)

        if prev_cx is not None and prev_cy is not None:

            dx = cx - prev_cx
            dy = cy - prev_cy

            if abs(dx) > abs(dy):
                if dx > MOVE_THRESHOLD:
                    command = "RIGHT"
                elif dx < -MOVE_THRESHOLD:
                    command = "LEFT"
            else:
                if dy > MOVE_THRESHOLD:
                    command = "DOWN"
                elif dy < -MOVE_THRESHOLD:
                    command = "UP"

        prev_cx, prev_cy = cx, cy

    else:
        prev_cx, prev_cy = None, None

    # Display command
    cv2.putText(
        frame,
        f"Command: {command}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 255, 0),
        3
    )

    cv2.imshow("Skeleton Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# -------- Plot motion signals --------
if len(x_signal) > 10:

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(x_signal)
    plt.title("Horizontal Motion Signal x(t)")
    plt.xlabel("Time (frames)")
    plt.ylabel("X Position (pixels)")
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(y_signal)
    plt.title("Vertical Motion Signal y(t)")
    plt.xlabel("Time (frames)")
    plt.ylabel("Y Position (pixels)")
    plt.grid()

    plt.tight_layout()
    plt.show()

else:
    print("Not enough data captured to plot.")