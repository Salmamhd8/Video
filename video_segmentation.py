import os

os.environ["MPLCONFIGDIR"] = "/tmp"
os.environ["YOLO_CONFIG_DIR"] = "/tmp"
import urllib.request
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
from PIL import Image

# Configuration de la page
st.set_page_config(layout="wide")
st.title("🎥 Advanced Video Segmentation")


@st.cache_resource(ttl=24 * 3600)
@st.cache_resource(ttl=24 * 3600)
def load_model():
    try:
        cache_dir = "/tmp/yolo_cache"  # <- RÉPERTOIRE AUTORISÉ
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "yolov8n-seg.pt")

        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1_000_000:
            with st.spinner("Downloading YOLOv8 model (20MB)..."):
                urllib.request.urlretrieve(
                    "https://ultralytics.com/assets/yolov8n-seg.pt",
                    model_path
                )
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None


def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir="/tmp") as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name
    except Exception as e:
        st.error(f"File save error: {str(e)}")
        return None


def process_frame(frame, model, class_id, bg_option, bg_value=None, bg_img=None):
    results = model(frame, classes=[class_id], conf=0.5)
    if len(results[0]) > 0:
        mask = results[0].masks[0].data[0].cpu().numpy() * 255
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        if bg_option == "Color":
            background = np.full_like(frame, bg_value)
        elif bg_option == "Blur":
            background = cv2.blur(frame, (50, 50))
        elif bg_option == "Custom" and bg_img is not None:
            background = cv2.resize(bg_img, (frame.shape[1], frame.shape[0]))
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            frame[:, :, 3] = mask.astype(np.uint8)
            background = np.zeros_like(frame)

        if bg_option != "Transparent":
            result = cv2.bitwise_and(frame, frame, mask=mask.astype(np.uint8)) + \
                     cv2.bitwise_and(background, background, mask=255 - mask.astype(np.uint8))
        else:
            result = frame
        return result
    return frame


# UI Components
uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        selected_class = st.selectbox("Object to segment", ["person", "car", "dog", "cat"])
    with col2:
        bg_option = st.radio("Background", ["Color", "Blur", "Transparent", "Custom"])

    bg_value = None
    bg_img = None
    if bg_option == "Color":
        bg_value = st.color_picker("Background color", "#00FF00")
        bg_value = np.array([int(bg_value.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)
    elif bg_option == "Custom":
        bg_img = st.file_uploader("Background image", type=["jpg", "jpeg", "png"])
        if bg_img:
            bg_img = np.array(Image.open(bg_img).convert('RGB'))

    if st.button("Process Video"):
        input_path = save_uploaded_file(uploaded_file)
        if input_path:
            output_path = "/tmp/output.mp4"
            try:
                cap = cv2.VideoCapture(input_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                progress_bar = st.progress(0)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processed = process_frame(
                        frame, model,
                        class_id=["person", "car", "dog", "cat"].index(selected_class),
                        bg_option=bg_option,
                        bg_value=bg_value,
                        bg_img=bg_img
                    )
                    out.write(processed)
                    progress_bar.progress(int(cap.get(cv2.CAP_PROP_POS_FRAMES) / frame_count))

            finally:
                cap.release()
                out.release()
                if os.path.exists(input_path):
                    os.unlink(input_path)

            st.success("Processing complete!")
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    st.download_button(
                        "Download Processed Video",
                        f,
                        "processed_video.mp4",
                        "video/mp4"
                    )
                os.unlink(output_path)


