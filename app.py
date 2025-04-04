import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import base64

#  Set page config at the VERY TOP
st.set_page_config(page_title="Fake Video Detection", layout="wide")

# Load the trained model
MODEL_PATH = "my_xception_model.keras"
model = load_model(MODEL_PATH)

# Function to preprocess a frame before prediction
def preprocess_frame(frame):
    frame = cv2.resize(frame, (model.input_shape[1], model.input_shape[2]))  # Resize
    frame = frame / 255.0  # Normalize pixel values
    frame = frame.reshape((1,) + frame.shape)  # Reshape for model input
    return frame

# Function to set a background image with error handling
def set_background(image_path):
    if not os.path.exists(image_path):
        st.error(f"🚨 Background image not found at {image_path}")
        return
    
    with open(image_path, "rb") as f:
        bin_str = base64.b64encode(f.read()).decode()
    
    page_bg_img = f"""
    <style>
    .stApp {{
    background-image: url("data:image/jpeg;base64,{bin_str}");
    background-size: cover;
    background-position: center;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set a cool background image (change the path accordingly)
BACKGROUND_IMAGE_PATH = "D:/Major Project/Deepfake Detection/background/4.jpg"  # Update this path!
set_background(BACKGROUND_IMAGE_PATH)

# Streamlit UI Setup
st.markdown(
    "<h1 style='text-align: center; color: Black;'>🔍 Fake Video Detection</h1>",
    unsafe_allow_html=True,
)

# Sidebar for uploading videos
st.sidebar.header("📁 Upload a Video File")
uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

# If a video is uploaded, process it
if uploaded_file is not None:
    st.sidebar.success("✅ Video uploaded successfully!")
    temp_video_path = "uploaded_video.mp4"

    # Save uploaded file to disk
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    # Video processing
    cap = cv2.VideoCapture(temp_video_path)
    frame_holder = st.empty()  # Placeholder for video frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.sidebar.error("End of video or error reading file.")
            break

        # Preprocess frame and get prediction
        preprocessed_frame = preprocess_frame(frame)
        prediction = model.predict(preprocessed_frame)[0][0]
        confidence = f"{prediction * 100:.2f}%"

        # Display result on frame
        label = "Fake Video" if prediction > 0.3 else "Real Video"
        color = (0, 0, 255) if prediction > 0.3 else (0, 255, 0)
        cv2.putText(frame, f"{label} ({confidence})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Convert frame to RGB format for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_holder.image(frame, channels="RGB")

    cap.release()

# Footer
st.sidebar.markdown("🛠 Developed with **Streamlit & OpenCV**")
