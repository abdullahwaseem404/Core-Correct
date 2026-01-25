import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

st.set_page_config(layout="wide")
st.title("Live Pose Detection")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

run = st.checkbox("Start Camera")

cap = cv2.VideoCapture(0)
frame_window = st.image([])

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Camera not accessible")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    frame_window.image(frame, channels="BGR")

cap.release()
