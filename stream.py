import streamlit as st
import cv2 as cv
import numpy as np
import requests
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="ASL Live Detection", layout="wide")

st.title("ASL Live Recognition (API Based)")

API_URL = "http://127.0.0.1:8000/predict"

def call_api(frame_bgr):
    _, img_encoded = cv.imencode(".jpg", frame_bgr)
    response = requests.post(
        API_URL,
        files={"file": img_encoded.tobytes()},
        timeout=5
    )
    return response.json()

class ASLVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_letter = ""
        self.last_conf = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        try:
            result = call_api(img)
            detections = result.get("detections", [])

            for det in detections:
                x1 = det["bbox"]["x1"]
                y1 = det["bbox"]["y1"]
                x2 = det["bbox"]["x2"]
                y2 = det["bbox"]["y2"]

                letter = det["letter"]
                conf = det["confidence"]

                self.last_letter = letter
                self.last_conf = conf

                color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)

                cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv.putText(
                    img,
                    f"{letter} ({conf:.2f})",
                    (x1, y1 - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

        except Exception:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_ctx = webrtc_streamer(
    key="asl-live",
    video_processor_factory=ASLVideoProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")

st.subheader("Live Prediction")

if webrtc_ctx.video_processor:
    st.metric(
        "Detected Letter",
        webrtc_ctx.video_processor.last_letter,
        f"{webrtc_ctx.video_processor.last_conf*100:.1f}% confidence",
    )
else:
    st.info("Start the camera to begin detection.")
