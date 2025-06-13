import cv2
import mediapipe as mp

# Initialize Mediapipe Hand module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define button coordinates and color
buttons = [
    {"pos": (100, 100), "size": (150, 100), "color": (0, 0, 255)},   # Red
    {"pos": (300, 100), "size": (150, 100), "color": (0, 0, 255)},
    {"pos": (500, 100), "size": (150, 100), "color": (0, 0, 255)},
]

def is_inside(x, y, btn):
    bx, by = btn["pos"]
    bw, bh = btn["size"]
    return bx <= x <= bx + bw and by <= y <= by + bh

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1) as hands:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Index finger tip = landmark 8
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)

                # Draw a circle on fingertip
                cv2.circle(frame, (x, y), 10, (255, 255, 255), -1)

                # Check for button click
                for btn in buttons:
                    if is_inside(x, y, btn):
                        btn["color"] = (0, 255, 0)  # Turn green
                    else:
                        btn["color"] = (0, 0, 255)  # Back to red

        # Draw buttons
        for i, btn in enumerate(buttons):
            x, y = btn["pos"]
            w_, h_ = btn["size"]
            cv2.rectangle(frame, (x, y), (x + w_, y + h_), btn["color"], -1)
            cv2.putText(frame, f"Button {i+1}", (x + 10, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("Virtual Buttons", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
