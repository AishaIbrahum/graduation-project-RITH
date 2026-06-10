from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize Mediapipe Pose module and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

def flip_image(frame):
    # Flip the frame horizontally
    return cv2.flip(frame, 1)

def gen_frames():
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Flip the frame
            frame = flip_image(frame)

            # Convert the image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Detect landmarks
            results = pose.process(image)

            # Convert the image back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render landmarks on the image
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(60, 179, 113), thickness=4, circle_radius=4),
                mp_drawing.DrawingSpec(color=(60, 179, 113), thickness=4, circle_radius=4)
            )

            # Extract landmark coordinates
            try:
                landmarks = results.pose_landmarks.landmark

                # Calculate the arm raise angle
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                # Check if the arm raise angle is within the correct range
                if angle >= CORRECT_ARM_RAISE_ANGLE - INCORRECT_ARM_RAISE_ANGLE_THRESHOLD and angle <= CORRECT_ARM_RAISE_ANGLE + INCORRECT_ARM_RAISE_ANGLE_THRESHOLD:
                    feedback = "Correct"
                else:
                    feedback = "Incorrect"

                # Render feedback
                cv2.putText(image, f"Feedback: {feedback}",
                            (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (70, 130, 180), 2, cv2.LINE_AA)

                # Render arm raise angle
                cv2.putText(image, f"Arm Raise Angle: {angle} degrees",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (70, 130, 180), 2, cv2.LINE_AA)

            except:
                pass

            # Convert the frame to a format suitable for streaming
            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

CORRECT_ARM_RAISE_ANGLE = 90.0
INCORRECT_ARM_RAISE_ANGLE_THRESHOLD = 30.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
