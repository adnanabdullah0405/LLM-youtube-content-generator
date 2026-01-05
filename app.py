import os
import time
import math
import subprocess
import tempfile
import re
from datetime import datetime
from typing import TypedDict, Annotated

import pandas as pd
import streamlit as st
import imageio_ffmpeg as ffmpeg
from google.cloud import speech
import google.generativeai as genai
from pydub import AudioSegment
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

# ======================================================
# Load Env + Configure
# ======================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

os.environ["FFMPEG_BINARY"] = ffmpeg.get_ffmpeg_exe()

# ======================================================
# LangGraph State Definition
# ======================================================
class AgentState(TypedDict):
    video_path: str
    chunks_video: list
    extracted_audio: str
    audio_chunks: list
    full_urdu: str
    english_text: str
    description: str

# ======================================================
# Node: Set Google Credentials from Streamlit Secrets
# ======================================================
def set_google_credentials(_: dict) -> dict:
    try:
        service_account_json = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            temp_file.write(service_account_json.encode("utf-8"))
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file.name
        return {}
    except Exception as e:
        st.error(f"Error setting credentials: {e}")
        return {}

# ======================================================
# Node: Upload Video File
# ======================================================
def upload_video(_: dict) -> dict:
    uploaded = st.file_uploader("Upload Video", type=["mp4","mov","avi","mkv"])
    if not uploaded:
        raise ValueError("No video uploaded")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        return {"video_path": tmp.name}

# ======================================================
# Node: Compress Video if large
# ======================================================
def compress_video(state: AgentState) -> dict:
    video_path = state["video_path"]
    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    
    if size_mb > 200:
        compressed_path = "compressed.mp4"
        cmd = [os.environ["FFMPEG_BINARY"], "-i", video_path, "-vcodec", "libx264", "-crf", "28", compressed_path]
        subprocess.run(cmd, check=True)
        return {"video_path": compressed_path}
    
    return {"video_path": video_path}

# ======================================================
# Node: Split Video into chunks based on size
# ======================================================
def split_video(state: AgentState) -> dict:
    video_path = state["video_path"]
    duration_cmd = [os.environ["FFMPEG_BINARY"], "-i", video_path, "-f", "null", "-"]
    result = subprocess.run(duration_cmd, capture_output=True, text=True)
    
    try:
        duration = float(result.stderr.split("Duration: ")[1].split(",")[0].split(":")[2])
    except Exception:
        duration = 0.0
    
    output_files = []
    if duration > 0:
        chunk_secs = 60 * 3  # 3 min per chunk
        total_chunks = math.ceil(duration / chunk_secs)
        for i in range(total_chunks):
            out_file = f"video_chunk_{i}.mp4"
            cmd = [
                os.environ["FFMPEG_BINARY"], "-i", video_path,
                "-ss", str(i * chunk_secs),
                "-t", str(chunk_secs),
                out_file
            ]
            subprocess.run(cmd, check=True)
            output_files.append(out_file)
    
    return {"chunks_video": output_files}

# ======================================================
# Node: Extract Audio from Chunks
# ======================================================
def extract_audio(state: AgentState) -> dict:
    all_chunks = state.get("chunks_video", [])
    audio_files = []
    
    for vid in all_chunks:
        base = vid.rsplit(".",1)[0]
        wav_path = f"{base}.wav"
        cmd = [os.environ["FFMPEG_BINARY"], "-y", "-i", vid, "-ac", "1", "-ar", "16000", wav_path]
        subprocess.run(cmd, check=True)
        audio_files.append(wav_path)
    
    return {"audio_chunks": audio_files}

# ======================================================
# Node: Transcribe Audio (Google STT)
# ======================================================
def transcribe_audio(state: AgentState) -> dict:
    client = speech.SpeechClient()
    all_text = ""
    for audio in state["audio_chunks"]:
        with open(audio,"rb") as f:
            content = f.read()
        audio_req = speech.RecognitionAudio(content=content)
        config_req = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ur-PK",
            alternative_language_codes=["en-US"]
        )
        resp = client.recognize(config=config_req, audio=audio_req)
        for r in resp.results:
            all_text += r.alternatives[0].transcript + " "
    return {"full_urdu": all_text.strip()}

# ======================================================
# Node: Translate using Gemini
# ======================================================
def translate_text(state: AgentState) -> dict:
    urdu = state.get("full_urdu","")
    prompt = f"Translate the following Urdu text to English:\n\n{urdu}"
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return {"english_text": response.text}

# ======================================================
# Node: Generate YouTube Description
# ======================================================
def generate_description(state: AgentState) -> dict:
    text = state["english_text"]
    prompt = f"Create a YouTube description from this English text:\n{text}"
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return {"description": response.text}

# ======================================================
# Node: Clean & Format Description
# ======================================================
def clean_output(state: AgentState) -> dict:
    desc = state["description"]
    
    # remove blank lines
    lines = desc.split("\n")
    filtered = [l for l in lines if l.strip()]
    desc_clean = "\n".join(filtered)
    
    # limit length
    if len(desc_clean) > 2000:
        desc_clean = desc_clean[:2000]
    
    return {"description": desc_clean}

# ======================================================
# Build LangGraph Workflow
# ======================================================
workflow = StateGraph(AgentState)

workflow.add_node("set_google_creds", set_google_credentials)
workflow.add_node("upload_video", upload_video)
workflow.add_node("compress_video", compress_video)
workflow.add_node("split_video", split_video)
workflow.add_node("extract_audio", extract_audio)
workflow.add_node("transcribe_audio", transcribe_audio)
workflow.add_node("translate_text", translate_text)
workflow.add_node("generate_desc", generate_description)
workflow.add_node("clean_output", clean_output)

workflow.set_entry_point("set_google_creds")

workflow.add_edge("set_google_creds", "upload_video")
workflow.add_edge("upload_video", "compress_video")
workflow.add_edge("compress_video", "split_video")
workflow.add_edge("split_video", "extract_audio")
workflow.add_edge("extract_audio", "transcribe_audio")
workflow.add_edge("transcribe_audio", "translate_text")
workflow.add_edge("translate_text", "generate_desc")
workflow.add_edge("generate_desc", "clean_output")
workflow.add_edge("clean_output", END)

app = workflow.compile()

# ======================================================
# Run Agent from Streamlit
# ======================================================
if st.button("Generate Description"):
    st.info("Processing...")
    result = app.invoke({})
    if "description" in result:
        st.text_area("Final YouTube Description", result["description"], height=300)
