from __future__ import annotations

import os
import tempfile
import wave
from io import BytesIO

import numpy as np
import sounddevice as sd


def transcribe_from_microphone(
    duration_s: float = 5.0,
    sample_rate: int = 16000,
    model_id: str = "scribe_v2",
    language_code: str = "eng",
) -> str:
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is required for ElevenLabs speech-to-text.")

    print(f"[speech->text] Recording for {duration_s:.1f}s...")
    audio = sd.rec(int(duration_s * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    wav_bytes = _wav_bytes(audio, sample_rate=sample_rate)

    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=api_key)
    transcription = client.speech_to_text.convert(
        file=BytesIO(wav_bytes),
        model_id=model_id,
        tag_audio_events=False,
        language_code=language_code,
        diarize=False,
    )
    text = getattr(transcription, "text", None)
    if text is None and isinstance(transcription, dict):
        text = transcription.get("text")
    return str(text or "").strip()


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    with tempfile.SpooledTemporaryFile() as handle:
        with wave.open(handle, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio.tobytes())
        handle.seek(0)
        return handle.read()
