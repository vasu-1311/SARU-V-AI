import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import numpy as np
import asyncio
import edge_tts
from groq import Groq
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
import tempfile
import soundfile as sf

# ================= ENV =================
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    st.error("Missing GROQ_API_KEY")
    st.stop()

client = Groq(api_key=API_KEY)

# ================= PAGE =================
st.set_page_config(page_title="Sariva AI", page_icon="✨", layout="wide")

# ================= SESSION =================
defaults = {
    "username": "",
    "resume_text": "",
    "chat_history": [],
    "interview_history": [],
    "question_count": 0,
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ================= NAME GATE =================
if not st.session_state.username:
    st.title("Sariva AI")
    name = st.text_input("Enter your name")
    if st.button("Start"):
        if name.strip():
            st.session_state.username = name.strip()
            st.rerun()
    st.stop()

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("### 🧠 Modes")
    st.write(f"👤 {st.session_state.username}")

    mode = st.radio(
        "",
        ["🤖 Assistant", "🎯 Entry Interview"],
        label_visibility="collapsed"
    )

    if st.button("Reset All"):
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.session_state.username = st.session_state.username
        st.rerun()

# ================= HEADER =================
st.title("Sariva AI")

# ================= RESUME =================
uploaded = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded:
    reader = PdfReader(uploaded)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    st.session_state.resume_text = text
    st.success("Resume Loaded")

# ======================================================
# ================= ASSISTANT MODE =====================
# ======================================================

if mode == "🤖 Assistant":

    if not st.session_state.chat_history:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"Hi {st.session_state.username}, how can I help you today?"
        })

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask anything..."):

        st.session_state.chat_history.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": f"""
You are Sariva AI assistant.

Be:
- Logical
- Structured
- Clear
- Professional
- Resume-aware if needed

Resume:
{st.session_state.resume_text[:4000]}
"""
                }
            ] + st.session_state.chat_history
        )

        reply = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

# ======================================================
# ================= INTERVIEW MODE =====================
# ======================================================

elif mode == "🎯 Entry Interview":

    if not st.session_state.resume_text:
        st.info("Upload resume to begin interview.")
        st.stop()

    async def speak(text):
        communicate = edge_tts.Communicate(text, voice="en-IN-NeerjaNeural")
        file = "response.mp3"
        await communicate.save(file)
        return file

    st.markdown("### 🎯 Entry-Level Interview")

    # Ask first question
    if st.session_state.question_count == 0:

        prompt = f"""
You are a professional entry-level interviewer.

Ask ONE question only.
Do NOT answer it.
Stay within resume context.

Resume:
{st.session_state.resume_text[:4000]}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        question = response.choices[0].message.content

        st.session_state.interview_history.append({"role": "assistant", "content": question})
        st.session_state.question_count += 1

    # Show question
    st.markdown("### 👔 Interviewer")
    st.write(st.session_state.interview_history[-1]["content"])
    st.audio(asyncio.run(speak(st.session_state.interview_history[-1]["content"])), autoplay=True)

    # Audio capture
    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.frames = []

        def recv(self, frame: av.AudioFrame):
            self.frames.append(frame.to_ndarray())
            return frame

        def clear(self):
            self.frames = []

    ctx = webrtc_streamer(
        key="interview",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    st.info("Click START above, speak, then click Record Answer.")

    if ctx.state.playing and st.button("Record Answer"):

        frames = ctx.audio_processor.frames

        if not frames:
            st.warning("No audio captured.")
            st.stop()

        audio_data = np.concatenate(frames, axis=1)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, audio_data.T, 48000)

        ctx.audio_processor.clear()

        with open(tmp.name, "rb") as f:
            transcript = client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3"
            )

        answer = transcript.text.strip()

        st.markdown("### 🧑 Candidate")
        st.write(answer)

        st.session_state.interview_history.append({"role": "user", "content": answer})

        # Next question
        follow_prompt = f"""
Continue interview.

Ask ONE next question only.
Do NOT answer it.
Stay entry-level.
Stay within resume.

Conversation:
{st.session_state.interview_history}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            messages=[{"role": "user", "content": follow_prompt}]
        )

        next_question = response.choices[0].message.content

        st.session_state.interview_history.append({"role": "assistant", "content": next_question})
        st.session_state.question_count += 1

        st.markdown("### 👔 Next Question")
        st.write(next_question)
        st.audio(asyncio.run(speak(next_question)), autoplay=True)

    if st.button("End Interview"):

        final_prompt = f"""
Evaluate entry-level candidate.

Transcript:
{st.session_state.interview_history}

Provide:
- Technical score /10
- Communication score /10
- Confidence score /10
- Strengths
- Areas to improve
"""

        final = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            messages=[{"role": "user", "content": final_prompt}]
        )

        st.markdown("## 📊 Final Evaluation")
        st.write(final.choices[0].message.content)