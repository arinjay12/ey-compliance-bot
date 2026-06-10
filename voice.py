"""
voice.py
--------
Level 5 (input): local speech-to-text with faster-whisper.

A spoken question (WAV bytes from Streamlit's microphone widget) is transcribed
on the CPU, fully offline — nothing leaves the machine, consistent with the rest
of the bot. faster-whisper decodes/resamples the audio internally (via PyAV), so
no system ffmpeg is required.

The model (~140 MB for base.en) is downloaded once on first use, then cached.
"""
import io
import os
import re
import tempfile

_model = None
WHISPER_SIZE = "base.en"   # English-only; good speed/accuracy balance on CPU


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe(wav_bytes: bytes) -> str:
    """Transcribe recorded audio (WAV bytes) to text. Returns '' on failure or
    if no speech was detected."""
    if not wav_bytes:
        return ""
    try:
        segments, _info = get_model().transcribe(
            io.BytesIO(wav_bytes),
            language="en",
            beam_size=1,        # fastest decoding
            vad_filter=True,    # skip silence at the ends
        )
        return " ".join(seg.text for seg in segments).strip()
    except Exception:
        return ""


# ── Text-to-speech (Level 5 output) ─────────────────────────────────────────
# Offline TTS via pyttsx3 (Windows' built-in SAPI5 voice). No internet, no data
# leaves the machine — consistent with the rest of the bot. We render to a WAV
# file and return the bytes so the browser can play it with st.audio.

_SOURCE_TAG_RE = re.compile(r"\[\s*Source:.*?\]", re.S | re.I)


def _clean_for_speech(text: str) -> str:
    """Strip citation tags and markdown so the spoken answer sounds natural."""
    text = _SOURCE_TAG_RE.sub("", text)         # drop "[Source: ... | Page 2]"
    text = re.sub(r"[*_`#>]+", " ", text)        # markdown symbols
    text = re.sub(r"\s+", " ", text).strip()
    return text


def synthesize(text: str) -> bytes | None:
    """Convert answer text to speech (WAV bytes) with the offline SAPI5 voice.
    Returns None on failure. A fresh engine per call avoids pyttsx3's known
    run-loop reuse issues."""
    text = _clean_for_speech(text)
    if not text:
        return None
    path = None
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)          # words per minute (default ~200)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        engine.save_to_file(text, path)
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:
            pass
        with open(path, "rb") as f:
            data = f.read()
        return data or None
    except Exception:
        return None
    finally:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception:
                pass
