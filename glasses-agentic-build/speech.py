from __future__ import annotations

import os
import subprocess
import threading
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable


@dataclass
class SpeechController:
    enabled: bool = True
    provider: str = "system"
    elevenlabs_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"
    elevenlabs_model_id: str = "eleven_flash_v2_5"
    output_format: str = "mp3_44100_128"

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self.provider = self.provider.lower().strip()
        self._elevenlabs_client = None
        self._current_process: subprocess.Popen | None = None
        self._last_text = ""
        self._last_started_at = 0.0

    def start(self) -> None:
        if not self.enabled:
            print("[speech] Speech output disabled; terminal output only.")
            return
        if self.provider == "elevenlabs":
            self._init_elevenlabs()
            print("[speech] ElevenLabs speech output enabled.")
        else:
            print("[speech] macOS system speech output enabled.")

    def speak_urgent(self, text: str) -> None:
        self._say(text, urgent=True)

    def speak_normal(self, text: str) -> None:
        self._say(text, urgent=False)

    def stop(self) -> None:
        with self._lock:
            self._stop_current_locked()

    def _say(self, text: str, urgent: bool = False) -> None:
        if not self.enabled or not text:
            return
        with self._lock:
            try:
                if self._should_skip_locked(text, urgent=urgent):
                    return
                if urgent:
                    self._stop_current_locked()
                if self.provider == "elevenlabs":
                    process = self._say_elevenlabs(text)
                else:
                    process = self._say_system(text, urgent=urgent)
                self._current_process = process
                self._last_text = text
                self._last_started_at = time.time()
            except Exception as exc:
                print(f"[speech] Speech failed: {exc}")

    def _say_system(self, text: str, urgent: bool = False) -> subprocess.Popen:
        voice_args = ["say", text]
        if urgent:
            voice_args = ["say", "-r", "210", text]
        return subprocess.Popen(voice_args)

    def _init_elevenlabs(self) -> None:
        api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is required for ElevenLabs speech.")
        from elevenlabs.client import ElevenLabs

        self._elevenlabs_client = ElevenLabs(api_key=api_key)

    def _say_elevenlabs(self, text: str) -> subprocess.Popen:
        if self._elevenlabs_client is None:
            self._init_elevenlabs()
        audio = self._elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=self.elevenlabs_voice_id,
            model_id=self.elevenlabs_model_id,
            output_format=self.output_format,
        )
        audio_bytes = _audio_to_bytes(audio)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
            handle.write(audio_bytes)
            path = handle.name
        return subprocess.Popen(["afplay", path])

    def _should_skip_locked(self, text: str, urgent: bool) -> bool:
        if self._current_process is None:
            return False
        if self._current_process.poll() is not None:
            self._current_process = None
            return False
        if urgent:
            return text == self._last_text and time.time() - self._last_started_at < 1.0
        return True

    def _stop_current_locked(self) -> None:
        if self._current_process is None:
            return
        if self._current_process.poll() is None:
            self._current_process.terminate()
            try:
                self._current_process.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                self._current_process.kill()
        self._current_process = None


def _audio_to_bytes(audio: bytes | Iterable[bytes]) -> bytes:
    if isinstance(audio, bytes):
        return audio
    return b"".join(chunk for chunk in audio if chunk)
