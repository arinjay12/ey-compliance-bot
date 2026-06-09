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
