import torch
import cv2
import numpy as np
import tempfile
import os
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects_in_image(image):
    results = model(image)
    return results

def get_compatible_codec():
    """Try different codecs to find a compatible one"""
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
        except:
            continue
            
    return 'XVID', '.avi'  # Default fallback

def detect_objects_in_video(uploaded_file, progress_bar, status_text, frame_placeholder, conf_threshold=0.25, skip_frames=2):
    model = load_model()

    # Create a temporary file for the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        shutil.copyfileobj(uploaded_file, temp_file)  # Save the uploaded file to the temporary file
        temp_video_path = temp_file.name

    # Verify the video file path
    st.write(f"Temporary video file path: {temp_video_path}")  # Debugging

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = tempfile.mktemp(suffix='.mp4')
    out = cv2.VideoWriter(output_path, codec, fps // skip_frames, (width, height))

    frame_count = 0
    processed_count = 0
    start_time = time.time()
    model.conf = conf_threshold
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue
            
            results = model(frame)
            annotated_frame = np.array(results.render()[0])
            
            # Add timestamp to frame
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
            status_text.text(f'Processing frame {frame_count}/{total_frames} ({fps_rate:.1f} fps)')
            processed_count += 1
            
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        os.remove(temp_video_path)  # Clean up temporary video file
    
    return output_path