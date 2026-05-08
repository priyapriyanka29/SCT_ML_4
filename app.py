import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import mediapipe as mp
from tensorflow.keras.models import load_model

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="🖐",
    layout="wide"
)

# --------------------------------
# CUSTOM CSS
# --------------------------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
}

section[data-testid="stSidebar"] {
    background-color: #161A23;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------
# TITLE
# --------------------------------
st.markdown("""
<h1 style='color:white; font-size:48px;'>
🖐 Hand Gesture Recognition System
</h1>

<p style='color:gray; font-size:18px;'>
Real-time gesture recognition using OpenCV, MediaPipe & TensorFlow
</p>
""", unsafe_allow_html=True)

# --------------------------------
# LOAD MODEL
# --------------------------------
model = load_model("gesture_model.h5")

# --------------------------------
# GESTURE LABELS
# --------------------------------
gesture_names = [
    "Palm",
    "L",
    "Fist",
    "Fist Moved",
    "Thumb",
    "Index",
    "OK",
    "Palm Moved",
    "C",
    "Down"
]

# --------------------------------
# SIDEBAR
# --------------------------------
st.sidebar.title("⚙ Control Panel")

run = st.sidebar.checkbox("Start Camera")

st.sidebar.success("System Ready")

# --------------------------------
# SESSION STATE
# --------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------
# MEDIAPIPE
# --------------------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# --------------------------------
# LAYOUT
# --------------------------------
col1, col2 = st.columns([3, 1])

FRAME_WINDOW = col1.image([])

# --------------------------------
# RIGHT PANEL
# --------------------------------
with col2:

    st.subheader("📊 Detection Info")

    total = len(st.session_state.history)

    last_gesture = "None"

    if total > 0:
        last_gesture = st.session_state.history[-1]["Gesture"]

    st.metric("Total Detections", total)

    st.metric("Last Gesture", last_gesture)

    st.markdown("---")

    st.subheader("📝 History")

    if total > 0:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df.tail(10),
            use_container_width=True
        )

# --------------------------------
# CAMERA
# --------------------------------
camera = cv2.VideoCapture(0)

while run:

    success, frame = camera.read()

    if not success:
        st.error("Camera not working")
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands
    results = hands.process(rgb)

    detected_gesture = "No Hand"

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, c = frame.shape

            x_min, y_min = w, h
            x_max, y_max = 0, 0

            # Bounding box
            for lm in hand_landmarks.landmark:

                x = int(lm.x * w)
                y = int(lm.y * h)

                x_min = min(x, x_min)
                y_min = min(y, y_min)

                x_max = max(x, x_max)
                y_max = max(y, y_max)

            # Padding
            padding = 20

            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)

            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Crop hand
            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size != 0:

                # Resize
                hand_img = cv2.resize(hand_img, (64, 64))

                # Normalize
                hand_img = hand_img / 255.0

                # Expand dimensions
                hand_img = np.expand_dims(hand_img, axis=0)

                # Prediction
                prediction = model.predict(
                    hand_img,
                    verbose=0
                )

                detected_gesture = gesture_names[
                    np.argmax(prediction)
                ]

                # Put text
                cv2.putText(
                    frame,
                    f"Gesture: {detected_gesture}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                # Save history
                st.session_state.history.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Gesture": detected_gesture
                })

    # Show frame
    FRAME_WINDOW.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

camera.release()

# --------------------------------
# FOOTER
# --------------------------------
st.markdown("---")

st.caption(
    "Developed using Streamlit, OpenCV, MediaPipe & TensorFlow"
)