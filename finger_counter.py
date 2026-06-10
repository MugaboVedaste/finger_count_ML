import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

tips = [4, 8, 12, 16, 20]

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        fingers = 0

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:

                # Thumb
                if hand.landmark[tips[0]].x < hand.landmark[tips[0]-1].x:
                    fingers += 1

                # Other fingers
                for tip in tips[1:]:
                    if hand.landmark[tip].y < hand.landmark[tip-2].y:
                        fingers += 1

                mp_draw.draw_landmarks(
                    img,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.putText(
            img,
            f'Fingers: {fingers}',
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Finger Counter", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()