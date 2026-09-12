
import time
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# =========================================================
# CẤU HÌNH APP
# =========================================================
st.set_page_config(
    page_title="AI Nhắc Tư Thế Ngồi",
    page_icon="🪑",
    layout="wide",
)

MODEL_PATH = "model/posture_model.keras"

# PHẢI đúng thứ tự class lúc train trên Colab
CLASS_NAMES = [
    "leaning_backward",
    "leaning_left",
    "leaning_right",
    "upright",
]

CORRECT_CLASS = "upright"

IMAGE_SIZE = (224, 224)

# Nếu lúc train dùng image / 255.0 -> "0_1"
# Nếu lúc train dùng MobileNetV2 preprocess_input -> "mobilenet"
PREPROCESS_MODE = "mobilenet"

MIN_CONFIDENCE = 0.65
BAD_POSTURE_SECONDS = 10.0
ALERT_COOLDOWN_SECONDS = 15.0
SMOOTHING_FRAMES = 8

DISPLAY_NAMES = {
    "leaning_backward": "Ngả về sau",
    "leaning_left": "Nghiêng trái",
    "leaning_right": "Nghiêng phải",
    "upright": "Tư thế đúng",
    "unknown": "Chưa xác định",
}

MESSAGES = {
    "leaning_backward": "Bạn đang ngả người ra phía sau. Hãy điều chỉnh tư thế ngồi.",
    "leaning_left": "Bạn đang nghiêng người sang trái. Hãy ngồi cân bằng lại.",
    "leaning_right": "Bạn đang nghiêng người sang phải. Hãy ngồi cân bằng lại.",
    "upright": "Tư thế ngồi đúng.",
    "unknown": "Chưa xác định rõ tư thế.",
}

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    model_ok = True
except Exception as e:
    model = None
    model_ok = False
    model_error = str(e)

# =========================================================
# XỬ LÝ ẢNH / INFERENCE
# =========================================================
def preprocess_frame(frame_bgr):
    img = cv2.resize(frame_bgr, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    if PREPROCESS_MODE == "mobilenet":
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    else:
        img = img / 255.0

    return np.expand_dims(img, axis=0)

def predict_posture(frame_bgr):
    x = preprocess_frame(frame_bgr)
    pred = model.predict(x, verbose=0)[0]

    class_id = int(np.argmax(pred))
    confidence = float(pred[class_id])

    if class_id >= len(CLASS_NAMES):
        label = "unknown"
    else:
        label = CLASS_NAMES[class_id]

    if confidence < MIN_CONFIDENCE:
        label = "unknown"

    probabilities = {
        CLASS_NAMES[i]: float(pred[i])
        for i in range(min(len(pred), len(CLASS_NAMES)))
    }

    return label, confidence, probabilities

def majority_vote(history):
    valid = [x for x in history if x != "unknown"]
    if not valid:
        return "unknown"
    return max(set(valid), key=valid.count)

def result_box(label, confidence):
    if label == CORRECT_CLASS:
        st.success(
            f"✅ **{DISPLAY_NAMES[label]}** — độ tin cậy: {confidence * 100:.1f}%"
        )
    elif label == "unknown":
        st.warning(
            f"⚠️ **Chưa xác định rõ tư thế** — độ tin cậy: {confidence * 100:.1f}%"
        )
    else:
        st.error(
            f"❌ **{DISPLAY_NAMES.get(label, label)}** — độ tin cậy: {confidence * 100:.1f}%\n\n"
            f"{MESSAGES.get(label, 'Hãy điều chỉnh lại tư thế ngồi.')}"
        )

# =========================================================
# VIDEO PROCESSOR CHO CAMERA
# =========================================================
class PostureVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.history = deque(maxlen=SMOOTHING_FRAMES)
        self.bad_start_time = None
        self.last_alert_time = 0.0

        self.current_label = "unknown"
        self.current_confidence = 0.0
        self.bad_seconds = 0.0
        self.alert_message = ""
        self.should_alert = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if model is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        raw_label, confidence, _ = predict_posture(img)
        self.history.append(raw_label)
        stable_label = majority_vote(list(self.history))

        now = time.time()

        if stable_label in ("unknown", CORRECT_CLASS):
            self.bad_start_time = None
            self.bad_seconds = 0.0
            self.should_alert = False
        else:
            if self.bad_start_time is None:
                self.bad_start_time = now

            self.bad_seconds = now - self.bad_start_time

            if (
                self.bad_seconds >= BAD_POSTURE_SECONDS
                and now - self.last_alert_time >= ALERT_COOLDOWN_SECONDS
            ):
                self.alert_message = MESSAGES.get(
                    stable_label,
                    "Bạn đang ngồi sai tư thế. Hãy điều chỉnh lại tư thế ngồi."
                )
                self.should_alert = True
                self.last_alert_time = now

        self.current_label = stable_label
        self.current_confidence = confidence

        if stable_label == CORRECT_CLASS:
            color = (0, 200, 0)
        elif stable_label == "unknown":
            color = (0, 200, 255)
        else:
            color = (0, 0, 255)

        cv2.putText(
            img,
            f"Posture: {DISPLAY_NAMES.get(stable_label, stable_label)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        cv2.putText(
            img,
            f"Confidence: {confidence * 100:.1f}%",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        if stable_label not in (CORRECT_CLASS, "unknown"):
            remaining = max(0.0, BAD_POSTURE_SECONDS - self.bad_seconds)
            cv2.putText(
                img,
                f"Alert after: {remaining:.1f}s",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =========================================================
# HEADER
# =========================================================
st.title("🪑 AI Nhắc Tư Thế Ngồi")
st.caption(
    "Hai tính năng chính: nhận diện realtime bằng camera và kiểm tra tư thế từ ảnh tải lên."
)

if not model_ok:
    st.error(
        "Không load được model. Hãy copy file `posture_model.keras` vào "
        "`model/posture_model.keras`."
    )
    st.code(model_error)
    st.stop()

# =========================================================
# 2 TÍNH NĂNG CHÍNH
# =========================================================
tab_camera, tab_upload = st.tabs([
    "📷 Camera realtime",
    "🖼️ Upload ảnh",
])

# ---------------------------------------------------------
# TAB 1: CAMERA REALTIME
# ---------------------------------------------------------
with tab_camera:
    st.subheader("1. Phát hiện tư thế bằng camera")
    st.write(
        "AI phân tích liên tục hình ảnh từ camera. "
        "Nếu phát hiện tư thế sai liên tục **10 giây**, hệ thống sẽ phát lời nhắc."
    )

    left, right = st.columns([2, 1])

    with left:
        ctx = webrtc_streamer(
            key="posture-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=PostureVideoProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            async_processing=True,
        )

    with right:
        st.markdown("### Trạng thái")
        status_box = st.empty()
        confidence_box = st.empty()
        time_box = st.empty()
        message_box = st.empty()
        speech_box = st.empty()

        st.markdown("---")
        st.write(f"**Ngưỡng tin cậy:** {MIN_CONFIDENCE * 100:.0f}%")
        st.write(f"**Cảnh báo sau:** {BAD_POSTURE_SECONDS:.0f} giây")
        st.write(f"**Làm mượt:** {SMOOTHING_FRAMES} frame")

    if ctx.state.playing:
        while ctx.state.playing:
            processor = ctx.video_processor

            if processor:
                label = processor.current_label
                confidence = processor.current_confidence
                bad_seconds = processor.bad_seconds

                status_box.metric(
                    "Tư thế",
                    DISPLAY_NAMES.get(label, label)
                )

                confidence_box.metric(
                    "Độ tin cậy",
                    f"{confidence * 100:.1f}%"
                )

                time_box.metric(
                    "Thời gian sai liên tục",
                    f"{bad_seconds:.1f}s"
                )

                if label == CORRECT_CLASS:
                    message_box.success("✅ Tư thế hiện tại đang đúng.")

                elif label == "unknown":
                    message_box.info("Chưa xác định rõ tư thế.")

                else:
                    remain = max(0.0, BAD_POSTURE_SECONDS - bad_seconds)

                    if bad_seconds < BAD_POSTURE_SECONDS:
                        message_box.warning(
                            f"Phát hiện tư thế sai. "
                            f"Sẽ cảnh báo sau {remain:.1f} giây nếu không chỉnh lại."
                        )

                    if processor.should_alert:
                        message = processor.alert_message
                        message_box.error("🔊 " + message)

                        safe_message = message.replace("\\", "\\\\").replace("'", "\\'")
                        speech_box.components.v1.html(
                            f"""
                            <script>
                                const msg = new SpeechSynthesisUtterance('{safe_message}');
                                msg.lang = 'vi-VN';
                                msg.rate = 1.0;
                                window.speechSynthesis.cancel();
                                window.speechSynthesis.speak(msg);
                            </script>
                            """,
                            height=0,
                        )

                        processor.should_alert = False

            time.sleep(0.5)

# ---------------------------------------------------------
# TAB 2: UPLOAD ẢNH
# ---------------------------------------------------------
with tab_upload:
    st.subheader("2. Kiểm tra tư thế từ ảnh")
    st.write(
        "Tải lên một ảnh có người đang ngồi. "
        "AI sẽ phân loại tư thế và hiển thị độ tin cậy."
    )

    uploaded_file = st.file_uploader(
        "Chọn ảnh",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col_img, col_result = st.columns([1.2, 1])

        with col_img:
            st.image(
                image,
                caption="Ảnh đã tải lên",
                use_container_width=True
            )

        # PIL RGB -> numpy RGB -> OpenCV BGR
        image_np = np.array(image)
        frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        label, confidence, probabilities = predict_posture(frame_bgr)

        with col_result:
            st.markdown("### Kết quả AI")
            result_box(label, confidence)

            st.markdown("#### Xác suất theo từng lớp")

            sorted_probs = sorted(
                probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for class_name, prob in sorted_probs:
                display_name = DISPLAY_NAMES.get(class_name, class_name)
                st.write(f"**{display_name}**")
                st.progress(min(max(prob, 0.0), 1.0))
                st.caption(f"{prob * 100:.1f}%")

            if label != "unknown":
                st.info(
                    "Đây là kết quả dự đoán của mô hình AI, "
                    "không phải chẩn đoán y tế."
                )

st.markdown("---")
st.caption(
    "TMA2-AI Team — Demo AI Sitting Posture Classification"
)
