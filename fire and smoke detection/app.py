import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO

# Load YOLOv8 model
model = YOLO('best.pt')

def run_inference(image_bytes):
    np_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    results = model(img)
    result_image = results[0].plot()
    return result_image

st.set_page_config(page_title="Fire & Smoke Detection", page_icon="🔥", layout="wide")
st.title("🔥 Fire and Smoke Detection System 🔥")
st.markdown("### Upload an image to detect fire and smoke")

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("This model detects fire and smoke in images using YOLOv8.")
    st.markdown("---")
    st.write("**Developed for real-time fire hazard detection**")
    st.write("📌 Ensure good image quality for better results.")

image_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    image_bytes = image_file.read()
    st.markdown("### Uploaded Image")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_bytes, caption="Original Image", use_container_width=False, width=300)

    with st.spinner("Detecting..."):
        result_image = run_inference(image_bytes)
    
    with col2:
        st.markdown("### Detection Result") 
        st.image(result_image, caption="Detected Fire and Smoke", use_container_width=False, width=300)
    
    result_bytes = cv2.imencode(".jpg", result_image)[1].tobytes()
    btn = st.download_button(
        label="💾 Download Result Image",
        data=result_bytes,
        file_name="detection_result.jpg",
        mime="image/jpeg"
    )
