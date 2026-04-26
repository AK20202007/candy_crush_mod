"""
ElevenLabs Speech Integration for Assistive Navigation

Provides high-quality text-to-speech (TTS) and speech-to-text (STT)
using ElevenLabs API for natural, human-like voice interaction.

Features:
- High-quality TTS with natural voices
- Real-time speech-to-text for voice commands
- Multiple voice options
- Streaming audio playback
- Fallback to system TTS if ElevenLabs unavailable

Requirements:
- pip install elevenlabs
- ELEVENLABS_API_KEY environment variable
"""

from __future__ import annotations

import io
import os
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Optional, Callable

import requests

# Try to import elevenlabs
try:
    from elevenlabs import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    print("[elevenlabs] ElevenLabs package not installed. Using fallback TTS.")

try:
    import sounddevice as sd
    import soundfile as sf
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False
    print("[elevenlabs] Sounddevice/soundfile not available. Using fallback audio.")

from user_interface import AlertType, UserInterface


class ElevenLabsSpeechController:
    """
    ElevenLabs-based speech controller with TTS and STT capabilities.
    
    Provides natural, high-quality speech synthesis and accurate
    speech recognition for assistive navigation.
    """
    
    # Default voice IDs (using standard ElevenLabs preset voices)
    DEFAULT_VOICES = {
        "default": "21m00Tcm4TlvDq8ikWAM",  # Rachel - Natural female
        "calm": "AZnzlk1XvdvUeBnXmlld",     # Emily - Calm female  
        "professional": "TxGEqnHWrfWFTfGW9XjX", # Josh - Professional male
        "warm": "XB0fDUnXU5powFXDhCwa",      # Bella - Warm female
        "male": "pNInz6obpgDQGcFmaJgB",     # Adam - Male voice
        "custom": "21m00Tcm4TlvDq8ikWAM",   # Default fallback
    }
    
    def __init__(
        self,
        user_interface: Optional[UserInterface] = None,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: str = "eleven_flash_v2_5",
        fallback_to_system: bool = True
    ):
        self.ui = user_interface or UserInterface()
        self.fallback_to_system = fallback_to_system
        
        # API configuration
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id or self.DEFAULT_VOICES["default"]
        self.model_id = model_id
        
        # ElevenLabs client
        self.client = None
        if self.api_key:
            try:
                self.client = ElevenLabs(api_key=self.api_key)
            except Exception as e:
                print(f"[elevenlabs] Error creating client: {e}")
        
        # Speech queues
        self._critical_queue: queue.Queue[str] = queue.Queue()
        self._urgent_queue: queue.Queue[str] = queue.Queue()
        self._normal_queue: queue.Queue[str] = queue.Queue()
        self._info_queue: queue.Queue[str] = queue.Queue()
        
        # Control
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="ElevenLabsSpeechWorker", daemon=True)
        
        # Statistics
        self._messages_spoken = 0
        self._messages_suppressed = 0
        self.is_playing = False
        self._api_calls = 0
        self._api_errors = 0
        
        # Audio playback
        self._current_stream = None
        self._audio_buffer = io.BytesIO()
        
    def _init_client(self) -> bool:
        """Initialize ElevenLabs client."""
        if not ELEVENLABS_AVAILABLE:
            return False
        
        if not self.api_key:
            print("[elevenlabs] Warning: No API key found. Set ELEVENLABS_API_KEY environment variable.")
            return False
        
        try:
            self.client = ElevenLabs(api_key=self.api_key)
            print("[elevenlabs] Client initialized successfully")
            return True
        except Exception as e:
            print(f"[elevenlabs] Error initializing client: {e}")
            return False
    
    def start(self) -> None:
        """Start the speech worker thread."""
        self._thread.start()
        print("[elevenlabs] Speech controller started")
        
        if not self.api_key and self.fallback_to_system:
            print("[elevenlabs] Using fallback system TTS")
    
    def stop(self) -> None:
        """Stop the speech worker."""
        self._stop.set()
        for q in [self._critical_queue, self._urgent_queue, self._normal_queue, self._info_queue]:
            try:
                q.put_nowait("")
            except queue.Full:
                pass
        print("[elevenlabs] Speech controller stopped")
    
    def pause(self) -> None:
        """Pause speech output."""
        self._pause.set()
        if self._current_stream:
            try:
                sd.stop()
            except:
                pass
        print("[elevenlabs] Paused")
    
    def resume(self) -> None:
        """Resume speech output."""
        self._pause.clear()
        print("[elevenlabs] Resumed")
    
    def speak(
        self,
        message: str,
        alert_type: AlertType = AlertType.INFO,
        priority: int = 50,
        formatted: bool = False,
        immediate: bool = False
    ) -> bool:
        """Queue a message for speaking."""
        if self._stop.is_set():
            return False
        
        if not formatted:
            message = self.ui.format_message_for_user(message)
        
        # Check if we should speak
        force = alert_type == AlertType.CRITICAL or priority >= 90 or immediate
        if not self.ui.should_speak(message, alert_type, priority, force):
            self._messages_suppressed += 1
            return False
        
        # For immediate speech, clear lower priority queues
        if immediate or alert_type == AlertType.CRITICAL or priority >= 90:
            self._drain_lower_priority_queues()
            self._critical_queue.put(message)
        elif alert_type == AlertType.WARNING or priority >= 70:
            self._urgent_queue.put(message)
        elif alert_type == AlertType.GUIDANCE:
            self._normal_queue.put(message)
        else:
            self._info_queue.put(message)
        
        return True
    
    def speak_critical(self, message: str, immediate: bool = True) -> bool:
        """Speak a critical safety alert."""
        return self.speak(message, AlertType.CRITICAL, priority=100, immediate=immediate)
    
    def speak_warning(self, message: str, immediate: bool = False) -> bool:
        """Speak a warning alert."""
        return self.speak(message, AlertType.WARNING, priority=80, immediate=immediate)
    
    def speak_guidance(self, message: str, immediate: bool = False) -> bool:
        """Speak navigation guidance."""
        return self.speak(message, AlertType.GUIDANCE, priority=60, immediate=immediate)
    
    def speak_info(self, message: str, immediate: bool = False) -> bool:
        """Speak general information."""
        return self.speak(message, AlertType.INFO, priority=30, immediate=immediate)
    
    def _drain_lower_priority_queues(self) -> None:
        """Clear normal and info queues when critical alert arrives."""
        for q in [self._normal_queue, self._info_queue]:
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
    
    def _get_next_message(self) -> Optional[tuple[str, AlertType]]:
        """Get next message from queues in priority order."""
        try:
            msg = self._critical_queue.get_nowait()
            if msg:
                return (msg, AlertType.CRITICAL)
        except queue.Empty:
            pass
        
        try:
            msg = self._urgent_queue.get_nowait()
            if msg:
                return (msg, AlertType.WARNING)
        except queue.Empty:
            pass
        
        try:
            msg = self._normal_queue.get_nowait()
            if msg:
                return (msg, AlertType.GUIDANCE)
        except queue.Empty:
            pass
        
        try:
            msg = self._info_queue.get_nowait()
            if msg:
                return (msg, AlertType.INFO)
        except queue.Empty:
            pass
        
        return None
    
    def _speak_with_elevenlabs(self, message: str) -> bool:
        """Synthesize speech using ElevenLabs API."""
        if not self.client:
            return False
        
        try:
            self._api_calls += 1
            
            # Generate audio using ElevenLabs
            audio_stream = self.client.text_to_speech.convert(
                text=message,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format="mp3_44100_128",
            )
            
            # Collect audio data
            audio_data = b""
            for chunk in audio_stream:
                audio_data += chunk
            
            # Play audio
            self._play_audio(audio_data)
            
            return True
            
        except Exception as e:
            self._api_errors += 1
            print(f"[elevenlabs] TTS error: {e}")
            return False
    
    def _play_audio(self, audio_data: bytes) -> None:
        """Play audio data."""
        if not SOUND_AVAILABLE:
            print("[elevenlabs] Sound playback not available")
            return
        
        try:
            # Stop any existing playback
            sd.stop()
            
            # Load audio from bytes
            audio_buffer = io.BytesIO(audio_data)
            data, samplerate = sf.read(audio_buffer)
            
            # Use default device to avoid conflicts
            sd.default.device = None
            
            # Play with shorter timeout to avoid hanging
            sd.play(data, samplerate)
            
            # Wait for playback with timeout
            start_time = time.time()
            while sd.get_stream().active:
                if time.time() - start_time > 10:  # 10 second timeout
                    sd.stop()
                    break
                time.sleep(0.1)
            
        except Exception as e:
            # Suppress common macOS audio errors
            if "Unknown Error" not in str(e) and "err='-50'" not in str(e):
                print(f"[elevenlabs] Audio playback error: {e}")
            else:
                # Common macOS audio error - suppress for cleaner output
                pass
    
    def _speak_with_fallback(self, message: str) -> bool:
        """Fallback to system TTS (pyttsx3)."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()
            return True
        except Exception as e:
            print(f"[elevenlabs] Fallback TTS error: {e}")
            return False
    
    def _worker(self) -> None:
        """Main speech worker thread."""
        print("[elevenlabs] Speech worker ready")
        
        while not self._stop.is_set():
            if self._pause.is_set():
                time.sleep(0.1)
                continue
            
            result = self._get_next_message()
            if result is None:
                continue
            
            message, alert_type = result
            
            if not message.strip():
                continue
            
            self.is_playing = True
            try:
                # Try ElevenLabs first, then fallback
                success = False
                if self.api_key:
                    success = self._speak_with_elevenlabs(message)
                
                if not success and self.fallback_to_system:
                    success = self._speak_with_fallback(message)
            finally:
                self.is_playing = False
            
            if success:
                self._messages_spoken += 1
                # Messages are now sequenced naturally without purging the queue
        
    def recognize_speech(
        self,
        timeout: float = 10.0,
        language: str = "en",
        on_partial: Optional[Callable[[str], None]] = None
    ) -> Optional[str]:
        """
        Recognize speech from microphone using ElevenLabs STT.
        
        Args:
            timeout: Maximum time to listen (seconds)
            language: Language code (e.g., "en", "es", "fr")
            on_partial: Callback for partial results
            
        Returns:
            Recognized text or None if failed
        """
        if not self.client:
            print("[elevenlabs] STT not available - no client")
            return None
        
        if not SOUND_AVAILABLE:
            print("[elevenlabs] STT not available - sound recording not available")
            return None
        
        try:
            print("[elevenlabs] Listening... (speak now)")
            
            # Record audio
            import sounddevice as sd
            import numpy as np
            
            samplerate = 16000
            duration = timeout
            
            # Record
            recording = sd.rec(
                int(samplerate * duration),
                samplerate=samplerate,
                channels=1,
                dtype=np.int16
            )
            sd.wait()
            
            # Convert to bytes
            audio_bytes = recording.tobytes()
            
            # Save to temporary file
            temp_file = "/tmp/elevenlabs_stt.wav"
            with wave.open(temp_file, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(samplerate)
                wav_file.writeframes(audio_bytes)
            
            # Send to ElevenLabs STT
            with open(temp_file, 'rb') as f:
                response = requests.post(
                    "https://api.elevenlabs.io/v1/speech-to-text",
                    headers={
                        "xi-api-key": self.api_key,
                    },
                    data={
                        "model_id": "scribe_v1",
                    },
                    files={
                        "file": f,
                    },
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "").strip()
                if text:
                    print(f"[elevenlabs] Recognized: '{text}'")
                    return text
                else:
                    print("[elevenlabs] No speech recognized")
                    return None
            else:
                print(f"[elevenlabs] STT API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[elevenlabs] STT error: {e}")
            return None
    
    def is_idle(self) -> bool:
        """Return True if no audio is playing and all queues are empty."""
        has_items = any(q.qsize() > 0 for q in [self._critical_queue, self._urgent_queue, self._normal_queue, self._info_queue])
        return not has_items and not getattr(self, 'is_playing', False)

    def get_statistics(self) -> dict:
        """Get speech controller statistics."""
        return {
            "messages_spoken": self._messages_spoken,
            "messages_suppressed": self._messages_suppressed,
            "api_calls": self._api_calls,
            "api_errors": self._api_errors,
            "elevenlabs_available": self.client is not None,
            "using_fallback": (self.client is None or self.api_key is None) and self.fallback_to_system,
            "queues": {
                "critical": self._critical_queue.qsize(),
                "urgent": self._urgent_queue.qsize(),
                "normal": self._normal_queue.qsize(),
                "info": self._info_queue.qsize(),
            }
        }
    
    def clear_queues(self) -> None:
        """Clear all speech queues."""
        for q in [self._critical_queue, self._urgent_queue, self._normal_queue, self._info_queue]:
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
        print("[elevenlabs] Queues cleared")


def create_elevenlabs_controller(
    api_key: Optional[str] = None,
    voice: str = "default",
    **kwargs
) -> ElevenLabsSpeechController:
    """
    Factory function to create ElevenLabs speech controller.
    
    Args:
        api_key: ElevenLabs API key (or set ELEVENLABS_API_KEY env var)
        voice: Voice preset ("default", "calm", "professional", "warm")
        **kwargs: Additional arguments for ElevenLabsSpeechController
        
    Returns:
        Configured ElevenLabsSpeechController
    """
    voice_id = ElevenLabsSpeechController.DEFAULT_VOICES.get(voice, voice)
    
    # Remove voice_id from kwargs if present to avoid conflict
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'voice_id'}
    
    return ElevenLabsSpeechController(
        api_key=api_key,
        voice_id=voice_id,
        **kwargs_clean
    )
