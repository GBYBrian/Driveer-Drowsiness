import os
import av
import threading
import streamlit as st
import streamlit_nested_layout
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer # 在streamlit应用中实现webrtc功能，支持试试音视流

from audio_handling import AudioFrameHandler
from drowsy_detection import VideoFrameHandler
# from ads import css_string


# Define the audio file to use.
eye_alarm_file_path = os.path.join("audio", "wake_up.wav")
mouth_alarm_file_path = os.path.join("audio", "are_you_tried.wav")
# Streamlit Components
st.set_page_config(
    page_title="Drowsiness Detection ",
    page_icon="https://learnopencv.com/wp-content/uploads/2017/12/favicon.png",
    layout="wide",  # centered, wide
    initial_sidebar_state="expanded",
)


col1, col2 = st.columns(spec=[6, 2], gap="medium") # 第一列占6个单位，第二列占2个单位，总宽度为8

with col1:
    st.title("Drowsiness Detection!!!🥱😪😴")
    with st.container():
        c1, c2 = st.columns(spec=[1, 1])
        with c1:
            # The amount of time (in seconds) to wait before sounding the alarm.警报前的等待时间
            # 在第一列c1中，添加一个滑块，用于设置等待时间0-5，默认1，步长0，25
            WAIT_TIME = st.slider("Seconds to wait before sounding alarm:", 0.0, 5.0, 1.0, 0.25)
            
        with c2:
            # Lowest valid value of Eye Aspect Ratio. Ideal values [0.15, 0.2].EAR阈值
            EAR_THRESH = st.slider("Eye Aspect Ratio threshold:", 0.0, 0.4, 0.18, 0.01)
            MOU_THRSH = st.slider("Mouth Aspect Ratio threshold",0.0, 0.4, 0.2, 0.01)

thresholds = {
    "EAR_THRESH": EAR_THRESH, # 判断用户是否处于困倦状态的阈值
    "WAIT_TIME": WAIT_TIME, # 检测到困倦后，出发警报前的等待时间
    "MOU_THRESH": MOU_THRSH # 嘴巴张开的程度判定为打哈欠的阈值
}

# For streamlit-webrtc
video_handler = VideoFrameHandler()
audio_handler = AudioFrameHandler()

lock = threading.Lock()  # For thread-safe access & to prevent race-condition.
shared_state = {"play_eye_alarm": False,
                "play_mouth_alarm": False}

# av模块：av是一个py库，用于处理音频和视频数据，提供了对多种媒体格式的支持，并允许用户读取，写入和处理音视频流
# av.VideoFrame: av.VideoFrame是av模块中的一个类，表示视频帧

# 该函数接受一个av.VideoFrame类型的参数frame,这是一个表示视频帧的对象
# video_frame_callback是一个回调函数，通常用于处理从视频流中读取的每一帧的数据

def video_frame_callback(frame: av.VideoFrame):
    frame = frame.to_ndarray(format="bgr24")  # Decode and convert frame to RGB

    frame, play_eye_alarm, play_mouth_alarm = video_handler.process(frame, thresholds)  # Process frame
    with lock:
        shared_state["play_eye_alarm"] = play_eye_alarm  # Update shared state
        shared_state["play_mouth_alarm"] = play_mouth_alarm

    return av.VideoFrame.from_ndarray(frame, format="bgr24")  # Encode and return BGR frame


def audio_frame_callback(frame: av.AudioFrame):
    with lock:  # access the current “play_alarm” state
        play_eye_alarm = shared_state["play_eye_alarm"]
        play_mouth_alarm = shared_state["play_mouth_alarm"]
    # if play_eye_alarm == True:
    #     audio_handler.update_audio_file_path(eye_alarm_file_path)
    #     new_frame: av.AudioFrame = audio_handler.process(frame, play_sound=play_eye_alarm) # 通过play_sound即play_alarm状态来执行报警器
    # else:
    #     new_frame: av.AudioFrame = audio_handler.process(frame, False)
    if play_eye_alarm == True:
        audio_handler.update_audio_file_path(eye_alarm_file_path)
    new_frame: av.AudioFrame = audio_handler.process(frame, play_sound=play_eye_alarm) # 通过play_sound即play_alarm状态来执行报警器

    if play_mouth_alarm == True:
        audio_handler.update_audio_file_path(mouth_alarm_file_path)
    new_frame: av.AudioFrame = audio_handler.process(frame, play_sound=play_mouth_alarm) # 通过play_sound即play_alarm状态来执行报警器
    return new_frame


# https://github.com/whitphx/streamlit-webrtc/blob/main/streamlit_webrtc/config.py
with col1:
    ctx = webrtc_streamer(
        key="drowsiness-detection",
        # key = "local-stream",
        video_frame_callback=video_frame_callback,
        audio_frame_callback=audio_frame_callback,
        # rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  # Add this to config for cloud deployment.
        media_stream_constraints={"video": {"height": {"ideal": 480}}, "audio": True},
        # media_stream_constraints={"video": True, "audio": True},
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=False),
    )
