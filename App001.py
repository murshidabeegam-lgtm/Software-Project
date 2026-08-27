import streamlit as st
import pyttsx3
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import cv2
import numpy as np

# 1. Import your refactored translation core
from translation_core import process_frame

# Page Configuration
st.set_page_config(page_title="viva la vida", layout="centered")

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

def speak_text(text):
    if text.strip():
        engine.say(text)
        engine.runAndWait()

# Dark Theme UI Styling
st.markdown("""
    <style>
    .stApp { background-color: #0E1317; color: #FFFFFF; }
    .app-header { text-align: center; margin-bottom: 15px; }
    .app-title { color: #4EED8C; font-size: 28px; font-weight: bold; }
    .info-card { background-color: #161D23; border-radius: 16px; padding: 15px; border: 1px solid #232E37; margin-top: 15px;}
    .sentence-box { background-color: #161D23; border-radius: 16px; padding: 15px; border: 1px solid #232E37; margin-top: 15px; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
    <div class="app-header">
        <div class="app-title">viva la vida 🍉</div>
        <div style="color: #A0AAB0; font-size: 14px;">Sign Language Translator</div>
    </div>
""", unsafe_allow_html=True)

# Persistent Session State Variables
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'last_letter' not in st.session_state:
    st.session_state.last_letter = "None"

# Camera Feed Input
img_file_buffer = st.camera_input("Live Camera Feed", label_visibility="collapsed")

detected_letter = "None"
confidence_display = "0%"

if img_file_buffer is not None:
    # Read frame buffer as OpenCV image
    bytes_data = img_file_buffer.getvalue()
    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # 2. Run backend sign translation pipeline
    annotated_frame, letter, conf = process_frame(cv_img)
    
    # Display the processed camera stream with MediaPipe overlays
    st.image(annotated_frame, use_container_width=True)

    if letter != "UNKNOWN" and letter != "None":
        detected_letter = letter
        confidence_display = f"{int(conf * 100)}%"

        # Append letter to current sentence when gesture changes
        if letter != st.session_state.last_letter:
            st.session_state.sentence += letter
            st.session_state.last_letter = letter

# Status Cards Display
st.markdown(f"""
    <div class="info-card">
        <div style="display: flex; justify-content: space-between;">
            <div>
                <div style="font-size: 12px; color: #A0AAB0;">Detected Sign</div>
                <div style="font-size: 22px; font-weight: bold;">Letter: <span style="color:#4EED8C;">{detected_letter}</span></div>
            </div>
            <div>
                <div style="font-size: 12px; color: #A0AAB0; text-align: right;">Confidence</div>
                <div style="font-size: 22px; font-weight: bold; color: #4EED8C;">{confidence_display}</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div class="sentence-box">
        <div style="color: #A0AAB0; font-size: 13px; margin-bottom: 8px;">Your Sentence</div>
        <div style="font-size: 20px; font-weight: 500;">{st.session_state.sentence} |</div>
    </div>
""", unsafe_allow_html=True)

# Control Buttons Layout
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔊\n\nSpeak"):
        speak_text(st.session_state.sentence)

with col2:
    if st.button("🧹\n\nClear Text"):
        st.session_state.sentence = ""
        st.session_state.last_letter = "None"
        st.rerun()

with col3:
    if st.button("⏸️\n\nPause Camera"):
        st.info("Camera paused.")