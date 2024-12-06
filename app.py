import torch
import cv2
import numpy as np
import tempfile
import os
import shutil
import time
import streamlit as st
from datetime import datetime
from PIL import Image

# Performance Optimization: Use torch.hub.load with device selection
def load_yolo_model(device='auto'):
    """Load YOLOv5 model with device optimization."""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device=device)
        return model, device
    except Exception as e:
        st.error(f"Model loading failed: {e}. Falling back to CPU.")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device='cpu')
        return model, 'cpu'

# Global model initialization
model, DEVICE = load_yolo_model()

def detect_objects_in_image(image):
    """Detect objects in the image using YOLOv5 with performance optimizations."""
    try:
        # Ensure image is converted to correct format
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Performance: Add inference mode and disable gradient computation
        with torch.no_grad():
            results = model(image)
        return results
    except Exception as e:
        st.error(f"Image detection error: {e}")
        return None

def get_compatible_codec():
    """Try different codecs to find a compatible one with error handling."""
    codecs = [
        ('mp4v', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi'),
        ('WMV1', '.wmv')
    ]
    temp_filename = tempfile.mktemp(suffix='.avi')
    test_size = (640, 480)
    for codec, ext in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(temp_filename, fourcc, 20, test_size)
            if out.isOpened():
                out.release()
                os.remove(temp_filename)
                return codec, ext
        except Exception as e:
            print(f"Codec {codec} not compatible: {e}")
    return 'XVID', '.avi'

def detect_objects_in_video(uploaded_file, progress_bar, status_text, frame_placeholder, conf_threshold=0.25, skip_frames=1):
    """Optimized video object detection with improved performance."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        shutil.copyfileobj(uploaded_file, temp_file)
        temp_video_path = temp_file.name

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file at {temp_video_path}.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    codec, ext = get_compatible_codec()
    output_path = tempfile.mktemp(suffix=ext)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), fps // max(1, skip_frames), (width, height))

    # Performance: Pre-configure model settings
    model.conf = conf_threshold
    model.iou = 0.45  # Intersection over Union threshold
    
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue
            
            # Performance: Use no_grad context
            with torch.no_grad():
                results = model(frame)
                annotated_frame = np.array(results.render()[0])
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(annotated_frame, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), 
                                 caption=f'Processing frame {frame_count}/{total_frames}', 
                                 use_column_width=True)
            out.write(annotated_frame)
            
            progress = min(float(frame_count) / total_frames, 1.0)
            progress_bar.progress(progress)
            
            elapsed_time = time.time() - start_time
            fps_rate = processed_count / elapsed_time if elapsed_time > 0 else 0
            status_text.markdown(f'<span style="color: yellow;">Processing frame {frame_count}/{total_frames} ({fps_rate:.1f} fps)</span>', unsafe_allow_html=True)
            processed_count += 1
            
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        os.remove(temp_video_path)
    
    return output_path

def detect_objects_in_live_camera(conf_threshold=0.25):
    """Optimized live camera detection with performance improvements."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # Set lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        st.error("Error accessing the camera.")
        return

    st_frame = st.empty()
    model.conf = conf_threshold
    model.iou = 0.45
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read from camera.")
                break
            
            # Performance: Use no_grad context
            with torch.no_grad():
                results = model(frame)
                annotated_frame = np.array(results.render()[0])
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(annotated_frame, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Live Camera Feed", use_column_width=True)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    st.set_page_config(
        page_title="VisionHack 2024 - Object Detection",
        page_icon="🎥",
        layout="centered",  # Reduced screen size
        initial_sidebar_state="collapsed"  # Sidebar starts collapsed
    )

    # Caching the CSS for faster loading
    st.markdown("""
        <style>
            body {
                margin: 0 auto;
                max-width: 900px; /* Limit the content width */
            }
            .title {
                color: #FFD700;
                text-align: center;
                font-size: 2rem; /* Adjusted font size */
                font-weight: bold;
                margin-bottom: 20px;
            }
            .subtitle {
                color: #FF4500;
                text-align: center;
                font-size: 1rem;
                margin-bottom: 30px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='title'>VisionHack 2024: Object Detection </div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Team Members: <span style='color: lightblue;'>Pratham M (2347138)</span> and <span style='color: lightgreen;'>Johanan Joshua (2347119)</span></div>", unsafe_allow_html=True)

    st.sidebar.title("Control Panel")
    st.sidebar.markdown(f"🚀 Current Device: {DEVICE.upper()}")
    st.sidebar.markdown("---")
    
    file_type = st.sidebar.selectbox("Select input type:", ("📷 Image", "🎥 Video", "📹 Live Camera"))
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
    playback_speed = st.sidebar.slider("Playback Speed (Video)", 0.5, 2.0, 1.0, step=0.1)

    if file_type == "📷 Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)
            if st.button("🔍 Detect Objects in Image"):
                with st.spinner("Analyzing Image..."):
                    results = detect_objects_in_image(Image.open(uploaded_file))
                    if results:
                        st.image(np.array(results.render()[0]), caption="Detected Objects", use_column_width=True)

    elif file_type == "🎥 Video":
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
        if uploaded_file:
            with st.spinner("Analyzing Video..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_placeholder = st.empty()
                output_video_path = detect_objects_in_video(uploaded_file, progress_bar, status_text, frame_placeholder, conf_threshold)
                if output_video_path:
                    st.video(output_video_path)

    elif file_type == "📹 Live Camera":
        if st.button("Start Live Feed Detection"):
            with st.spinner(f"Starting Live Detection on {DEVICE.upper()}..."):
                detect_objects_in_live_camera(conf_threshold)

    st.markdown(f"<div class='team'>Team FLOW | VisionHack 2024 | Running on {DEVICE.upper()}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
