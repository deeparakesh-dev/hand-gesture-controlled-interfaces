import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
def is_fist(hand):
    # Index fingertip (8) and palm center (0)
    tip = hand.landmark[8]
    palm = hand.landmark[0]

    dist = ((tip.x - palm.x)**2 + (tip.y - palm.y)**2) ** 0.5
    return dist < 0.08   # threshold for fist
def recognize_letter(stroke_x, stroke_y):
    if len(stroke_x) < 30:
        return "?"

    width = max(stroke_x) - min(stroke_x)
    height = max(stroke_y) - min(stroke_y)

    # Start–end distance
    start_end = ((stroke_x[0] - stroke_x[-1])**2 +
                 (stroke_y[0] - stroke_y[-1])**2) ** 0.5

    # Total path length
    path_len = 0
    for i in range(1, len(stroke_x)):
        path_len += ((stroke_x[i] - stroke_x[i-1])**2 +
                     (stroke_y[i] - stroke_y[i-1])**2) ** 0.5

    # --- O detection ---
    if start_end < 40 and path_len > 3 * max(width, height):
        return "O"

    # --- I detection ---
    if height > 3 * width:
        return "I"

    # --- horizontal line ---
    if width > 3 * height:
        return "-"

    # --- L detection ---
    if width > 50 and height > 50:
        return "L"

    return "?"
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
MIN_MOVE_DISTANCE = 5
# ========== MediaPipe Setup ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

prev_x, prev_y = None, None
x_buffer, y_buffer = [], []
SMOOTHING_WINDOW = 15

stroke_x, stroke_y = [], []

board = None

print("Air writing started. Press ESC to exit.")

# ========== MAIN LOOP ==========
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if board is None:
        board = np.ones_like(frame) * 40   # dark gray board

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        if is_fist(hand):
            # PEN UP
            prev_x, prev_y = None, None
            x_buffer.clear()
            y_buffer.clear()
            drawing = False
            continue
        else:
            drawing = True
        lm = hand.landmark[8]  # index finger tip

        raw_x = int(lm.x * w)
        raw_y = int(lm.y * h)

        # -------- Smoothing --------
        x_buffer.append(raw_x)
        y_buffer.append(raw_y)

        if len(x_buffer) > SMOOTHING_WINDOW:
            x_buffer.pop(0)
            y_buffer.pop(0)

        x = int(sum(x_buffer) / len(x_buffer))
        y = int(sum(y_buffer) / len(y_buffer))

        # -------- VISUAL FEEDBACK --------
        cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)   # 🔴 RED DOT
        cv2.circle(board, (x, y), 3, (255, 255, 255), -1)  # ⚪ WHITE DOT

        if prev_x is not None:
            dist = ((x - prev_x)**2 + (y - prev_y)**2) ** 0.5
            if dist > MIN_MOVE_DISTANCE:
                cv2.line(board, (prev_x, prev_y), (x, y), (255, 255, 255), 10)

        prev_x, prev_y = x, y

        stroke_x.append(x)
        stroke_y.append(y)

    else:
        prev_x, prev_y = None, None
        x_buffer.clear()
        y_buffer.clear()

    cv2.putText(
        board,
        "Air Writing Whiteboard",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    board_resized = cv2.resize(board, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    combined = np.hstack((frame_resized, board_resized))
    cv2.putText(
    board,
    f"Detected: {letter if 'letter' in globals() else ''}",
    (20, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
    )
    cv2.imshow("Whiteboard Only", board)
    cv2.imshow("Webcam (Left) | Whiteboard (Right)", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ========== SIGNAL PLOTS ==========
if len(stroke_x) > 10:
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(stroke_x)
    plt.title("Horizontal Motion Signal x(t)")
    plt.xlabel("Time (frames)")
    plt.ylabel("Pixels")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(stroke_y, color="orange")
    plt.title("Vertical Motion Signal y(t)")
    plt.xlabel("Time (frames)")
    plt.ylabel("Pixels")
    plt.grid()

    plt.tight_layout()
    plt.show()
else:
    print("Not enough data captured to plot.")

letter = recognize_letter(stroke_x, stroke_y)
print("Recognized character:", letter)

with open("air_written_text.txt", "a") as f:
    f.write(letter)

print("Saved to air_written_text.txt")