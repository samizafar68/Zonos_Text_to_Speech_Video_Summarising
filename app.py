import streamlit as st
import whisper
import requests
import os
import torch
import torchaudio
from pytube import YouTube
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from dotenv import load_dotenv

# Load environment variables (for local development; use Streamlit secrets for deployment)
load_dotenv()

# Set page config (must be the first Streamlit command)
st.set_page_config(page_title="Video to Speech Summarizer with Zonos", page_icon="🎥", layout="wide")

# Custom CSS for UI styling
st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
            color: #d4af37;
        }
        .stButton>button {
            background-color: #8b5e3c;
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
        .stTextInput input, .stSelectbox select {
            background-color: #3e2723;
            color: white;
            border-radius: 8px;
        }
        .stAudio audio {
            width: 100%;
        }
        .css-1d391kg p {
            color: #d4af37;
        }
    </style>
""", unsafe_allow_html=True)

# Configure API keys (use Streamlit secrets for deployment)
GEMINI_API_KEY = st.secrets.get("google", {}).get("gemini_api_key", os.getenv("GEMINI_API_KEY"))

# Load Whisper model (cached for performance)
@st.cache_resource
def load_whisper():
    try:
        model = whisper.load_model("small", device="cpu")  # Use CPU for Streamlit Cloud
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        return None

whisper_model = load_whisper()

# Load Zonos model (cached for performance)
@st.cache_resource
def load_zonos():
    try:
        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cpu")  # Use CPU for Streamlit Cloud
        return model
    except Exception as e:
        st.error(f"Error loading Zonos model: {str(e)}")
        return None

zonos_model = load_zonos()

# Language mapping for user selection
language_map = {
    "en-us": "English (US)",
    "en-gb": "English (UK)",
    "es-es": "Spanish (Spain)",
    "es-mx": "Spanish (Mexico)",
    "fr-fr": "French (France)",
    "de-de": "German (Germany)",
    "it-it": "Italian (Italy)",
    "pt-br": "Portuguese (Brazil)",
    "ru-ru": "Russian (Russia)",
    "zh-cn": "Chinese (Mandarin)",
    "ja-jp": "Japanese (Japan)",
    "ko-kr": "Korean (Korea)",
    "hi-in": "Hindi (India)"
}

# Function: Download audio from video link and return audio tensor
def download_audio_from_video(video_url):
    try:
        yt = YouTube(video_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_path = audio_stream.download(output_path="temp", filename="audio.mp3")
        wav, sampling_rate = torchaudio.load(audio_path)
        return audio_path, wav, sampling_rate
    except Exception as e:
        st.error(f"Error downloading audio: {str(e)}")
        return None, None, None

# Function: Speech to Text (Whisper)
def convert_speech_to_text(audio_path):
    if whisper_model is None:
        st.error("Whisper model not loaded.")
        return None
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Speech-to-Text Error: {str(e)}")
        return None

# Function: Summarize Text (Gemini API)
def summarize_text(text):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": f"Provide a concise summary of the following text, focusing on the most important points:\n\n{text}"}]}]
        }

        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            response_json = response.json()
            summarized_text = response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return summarized_text if summarized_text else "Summary not found"
        else:
            st.error(f"Gemini API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Summarization Error: {str(e)}")
        return None

# Function: Text to Speech (Zonos with speaker embedding)
def text_to_speech(text, language, speaker_wav, sampling_rate):
    if zonos_model is None:
        st.error("Zonos model not loaded.")
        return None
    try:
        speaker = zonos_model.make_speaker_embedding(speaker_wav, sampling_rate)
        cond_dict = make_cond_dict(text=text, speaker=speaker, language=language)
        conditioning = zonos_model.prepare_conditioning(cond_dict)
        codes = zonos_model.generate(conditioning)
        wavs = zonos_model.autoencoder.decode(codes).cpu()
        
        audio_path = "output.wav"
        torchaudio.save(audio_path, wavs[0], zonos_model.autoencoder.sampling_rate)
        return audio_path
    except Exception as e:
        st.error(f"Text-to-Speech Error: {str(e)}")
        return None

# Streamlit UI Layout
st.sidebar.markdown("<h1 style='text-align: center; color: #d4af37'>🎥 Video to Speech Summarizer</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center;'>Upload a video link to get a summarized speech output using Zonos TTS</p>", unsafe_allow_html=True)

st.subheader("🎥 Enter Video Link")
video_url = st.text_input("Enter a YouTube video URL")
language = st.selectbox("Select Output Language", list(language_map.keys()), format_func=lambda x: language_map[x])

if st.button("Process Video") and video_url:
    with st.spinner("Processing video..."):
        # Step 1: Download audio from video
        audio_path, wav, sampling_rate = download_audio_from_video(video_url)
        if audio_path and wav is not None:
            # Step 2: Convert audio to text
            text = convert_speech_to_text(audio_path)
            if text:
                st.write(f"**Extracted Text:** {text}")
                
                # Step 3: Summarize text using Gemini API
                summarized_text = summarize_text(text)
                if summarized_text:
                    st.write(f"**Summarized Text:** {summarized_text}")
                    
                    # Step 4: Convert summarized text to speech using Zonos with speaker embedding
                    audio_path = text_to_speech(summarized_text, language, wav, sampling_rate)
                    if audio_path and os.path.exists(audio_path):
                        st.subheader("🎧 Summarized Speech")
                        st.audio(audio_path, format="audio/wav")
                        os.remove(audio_path)  # Clean up temporary file
            os.remove(audio_path)  # Clean up temporary file

# Footer
st.markdown("""
    <hr>
    <p style="text-align: center; color: #d4af37;">© 2025 Video to Speech Summarizer | Powered by Zonos TTS</p>
""", unsafe_allow_html=True)
