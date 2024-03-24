from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture('Pushups.mp4')
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.counter = 0
        self.direction = 0

    def __del__(self):
        self.cap.release()

    def calculate_angle(self, a, b, c):
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(np.degrees(radians))
        return angle

    def is_arm_straight(self, shoulder, elbow, wrist):
        angle = self.calculate_angle(shoulder, elbow, wrist)
        return angle < 160

    def is_pushup_position(self, left_shoulder, left_elbow, left_wrist, right_shoulder, right_elbow, right_wrist):
        return self.is_arm_straight(left_shoulder, left_elbow, left_wrist) and self.is_arm_straight(right_shoulder, right_elbow, right_wrist)

    def get_frame(self):
        ret, frame = self.cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)

            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            if self.is_pushup_position(left_shoulder, left_elbow, left_wrist, right_shoulder, right_elbow, right_wrist):
                if self.direction == 0:
                    self.counter += 1
                    self.direction = 1
            else:
                self.direction = 0

            cv2.putText(image, f'Total Push-ups: {int(self.counter)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(image, (10, 50), (190 - int(left_angle), 60), (0, 255, 0), -1)

            # Display counter on the video feed
            cv2.putText(image, f'Repetitions: {int(self.counter)}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print("Error:", e)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

video_stream = VideoCamera()

@app.route('/')
def index():
    return render_template('ind.html', counter=video_stream.counter)

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
