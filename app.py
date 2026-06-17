import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import tensorflow as tf
import time

st.set_page_config(layout="wide")
st.title("🧍 Core-Correct: Real-Time Pose Estimation System")

col1, col2 = st.columns([1, 3])

with col1:
    run = st.checkbox("Start Camera")
    model_type = st.selectbox("Select Model", ["MediaPipe", "MoveNet"])
    show_angles = st.checkbox("Show Joint Angles")
    smoothing = st.slider("Temporal Smoothing", 0.0, 1.0, 0.5)
    detection_conf = st.slider("Detection Confidence", 0.3, 1.0, 0.5)

frame_window = col2.image([])

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=detection_conf,
    min_tracking_confidence=0.5
)

@st.cache_resource
def load_movenet():
    model = tf.saved_model.load(
        "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    )
    return model

movenet = None
if model_type == "MoveNet":
    movenet = load_movenet()

prev_landmarks = None

def smooth_landmarks(curr, prev, alpha):
    if prev is None:
        return curr
    return alpha * prev + (1 - alpha) * curr

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def run_movenet(frame):
    img = cv2.resize(frame, (192, 192))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = tf.cast(img, dtype=tf.int32)

    outputs = movenet.signatures['serving_default'](img)
    keypoints = outputs['output_0'].numpy()

    return keypoints

def draw_movenet(frame, keypoints):
    h, w, _ = frame.shape
    points = keypoints[0][0]

    coords = []
    for kp in points:
        y, x, conf = kp
        coords.append([x, y])

        if conf > 0.3:
            cv2.circle(frame, (int(x*w), int(y*h)), 5, (0,255,0), -1)

    return np.array(coords)

cap = cv2.VideoCapture(0)
prev_time = 0

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Camera not accessible")
        break

    frame = cv2.flip(frame, 1)

    landmarks = None

    if model_type == "MediaPipe":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            h, w, _ = frame.shape
            landmarks = []

            for lm in result.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y])

            landmarks = np.array(landmarks)

            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

    elif model_type == "MoveNet":
        keypoints = run_movenet(frame)
        landmarks = draw_movenet(frame, keypoints)

    if landmarks is not None:
        landmarks = smooth_landmarks(
            landmarks,
            prev_landmarks,
            smoothing
        )
        prev_landmarks = landmarks

        if show_angles and len(landmarks) > 16:
            try:
                shoulder = landmarks[6]
                elbow = landmarks[8]
                wrist = landmarks[10]

                angle = calculate_angle(shoulder, elbow, wrist)

                h, w, _ = frame.shape
                x, y = int(elbow[0]*w), int(elbow[1]*h)

                cv2.putText(frame, f"{int(angle)}°",
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,255,0), 2)

                if angle < 40:
                    cv2.putText(frame, "Arm too bent!",
                                (30,50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0,0,255), 3)
            except:
                pass

    curr_time = time.time()
    fps = 1/(curr_time-prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255,255,0), 2)

    frame_window.image(frame, channels="BGR")

cap.release()