# import libraries
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    """
    Calculate the angle between three points
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def change_mode(mode_letter: str):
    global counter, mode, angle
    counter = 0
    angle = 0
    mode = mode_letter
    print(f"debug: mode - {mode_letter}")


cap = cv2.VideoCapture("./output.mp4")  # video interface

# set vars
mode = "n"
counter = 0
stage = "None"


## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # detect mode and monitor
            if mode == "n":
                pass
            elif mode == "b":  # do bicep curl detection
                # Get coordinates
                shoulder = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                ]
                elbow = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                ]
                wrist = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                ]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Visualize angle
                cv2.putText(
                    image,
                    str(angle),
                    tuple(np.multiply(elbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Curl counter logic
                if angle > 150:
                    stage = "down"
                if angle < 90 and stage == "down":
                    stage = "up"
                    counter += 1
                    print(f"debug: bicep counter = {counter}")  # debug
            elif mode == "p":  # do pushup detection
                # Get coordinates
                shoulder = [
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                ]
                elbow = [
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                ]
                wrist = [
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                ]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Visualize angle
                cv2.putText(
                    image,
                    f"Angle: {angle:.2f}",
                    tuple(np.multiply(elbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # Push-up counter logic
                if angle > 160:
                    position = "up"
                if angle < 90 and position == "up":
                    stage = "down"
                if angle > 160 and stage == "down":
                    stage = "up"
                    counter += 1
                    position = None
                    print(f"debug: pushup counter = {counter}")  # debug

        except Exception as E:
            print(E)

        # display info
        cv2.putText(
            image,
            f"Reps: {counter}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # common stats
        # Display stage
        cv2.putText(
            image,
            f"Stage: {stage}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Render detections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        cv2.imshow("Mediapipe Feed", image)

        # controls
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        elif cv2.waitKey(10) & 0xFF == ord("p"):  # set mode - pushup
            change_mode("p")
        elif cv2.waitKey(10) & 0xFF == ord("b"):  # set mode - bicep curl
            change_mode("b")

    cap.release()
    cv2.destroyAllWindows()
