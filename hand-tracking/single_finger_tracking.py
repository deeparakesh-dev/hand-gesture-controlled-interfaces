import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,   # keep 1 for signal clarity
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

x_signal = []
y_signal = []

if not cap.isOpened():
    print("ERROR: Camera not opening")
    exit()

print("Tracking started... Press ESC to stop.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        # Index finger tip = landmark 8
        lm = hand.landmark[8]
        cx, cy = int(lm.x * w), int(lm.y * h)

        x_signal.append(cx)
        y_signal.append(cy)

        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

    cv2.imshow("Motion Signal Capture", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

# --------- SIGNAL PLOTTING ---------

plt.figure(figsize=(10, 4))
plt.plot(x_signal, label="x(t)")
plt.plot(y_signal, label="y(t)")
plt.title("Hand Motion Signals")
plt.xlabel("Time (frames)")
plt.ylabel("Position (pixels)")
plt.legend()
plt.grid()
plt.show()