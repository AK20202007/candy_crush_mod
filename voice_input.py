"""
Voice destination capture.

The default backend uses Apple's Speech framework through PyObjC, which is
already available in the local macOS virtual environment. This keeps destination
entry off the terminal while avoiding an extra cloud STT dependency.
"""

from __future__ import annotations

import platform
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


class VoiceInputError(RuntimeError):
    pass


@dataclass
class DestinationCaptureConfig:
    timeout_s: float = 8.0
    locale: str = "en_US"
    prefer_on_device: bool = True
    attempts: int = 2
    codeword: str = "navigate"
    stop_word: str = "stop"
    use_codeword: bool = True


class DestinationListener:
    def capture(self, config: DestinationCaptureConfig) -> str:
        raise NotImplementedError


class TranscriptSessionListener:
    def capture_transcript(self, config: DestinationCaptureConfig) -> str:
        raise NotImplementedError


@dataclass
class CodewordSessionState:
    codeword: str = "navigate"
    stop_word: str = "stop"
    active: bool = False
    captured_text: str = ""
    heard_codeword: bool = False
    heard_stop: bool = False

    def update(self, transcript: str) -> None:
        text = _normalize_spaces(transcript)
        if not text:
            return

        if not self.active:
            after_codeword = _text_after_phrase(text, self.codeword)
            if after_codeword is None:
                return
            self.active = True
            self.heard_codeword = True
            text = after_codeword

        before_stop, stopped = _split_before_phrase(text, self.stop_word)
        self.captured_text = before_stop.strip()
        if stopped:
            self.heard_stop = True

    def result(self) -> str:
        return self.captured_text.strip(" ,.;:")


class MacOSSpeechDestinationListener(DestinationListener):
    def capture(self, config: DestinationCaptureConfig) -> str:
        if platform.system() != "Darwin":
            raise VoiceInputError("macOS Speech backend is only available on macOS.")

        try:
            import AVFoundation
            import Foundation
            import Speech
        except Exception as exc:
            raise VoiceInputError(f"macOS Speech framework is unavailable: {exc}") from exc

        status = self._request_authorization(Speech)
        authorized = getattr(Speech, "SFSpeechRecognizerAuthorizationStatusAuthorized", 3)
        if status != authorized:
            raise VoiceInputError(f"Speech recognition permission was not authorized; status={status}.")

        recognizer = Speech.SFSpeechRecognizer.alloc().initWithLocale_(
            Foundation.NSLocale.localeWithLocaleIdentifier_(config.locale)
        )
        if recognizer is None or not recognizer.isAvailable():
            raise VoiceInputError("Speech recognizer is not available.")

        request = Speech.SFSpeechAudioBufferRecognitionRequest.alloc().init()
        request.setShouldReportPartialResults_(True)

        supports_on_device = False
        if hasattr(recognizer, "supportsOnDeviceRecognition"):
            try:
                supports_on_device = bool(recognizer.supportsOnDeviceRecognition())
            except Exception:
                supports_on_device = False

        if config.prefer_on_device and supports_on_device and hasattr(request, "setRequiresOnDeviceRecognition_"):
            try:
                request.setRequiresOnDeviceRecognition_(True)
            except Exception:
                pass

        audio_engine = AVFoundation.AVAudioEngine.alloc().init()
        input_node = audio_engine.inputNode()
        if input_node is None:
            raise VoiceInputError("No microphone input node is available.")

        result_holder = {"text": "", "error": None}
        done = threading.Event()

        def result_handler(result, error) -> None:
            if result is not None:
                transcription = result.bestTranscription()
                if transcription is not None:
                    result_holder["text"] = str(transcription.formattedString()).strip()
                if result.isFinal():
                    done.set()
            if error is not None:
                result_holder["error"] = error
                done.set()

        task = recognizer.recognitionTaskWithRequest_resultHandler_(request, result_handler)
        fmt = input_node.outputFormatForBus_(0)

        def tap_handler(buffer, when) -> None:
            request.appendAudioPCMBuffer_(buffer)

        input_node.installTapOnBus_bufferSize_format_block_(0, 1024, fmt, tap_handler)
        audio_engine.prepare()

        started = False
        try:
            started = _start_audio_engine(audio_engine)
            if not started:
                raise VoiceInputError("Microphone audio engine did not start.")

            deadline = time.time() + max(1.0, config.timeout_s)
            last_text = ""
            last_change = time.time()
            while time.time() < deadline and not done.is_set():
                current_text = str(result_holder.get("text") or "").strip()
                if current_text and current_text != last_text:
                    last_text = current_text
                    last_change = time.time()
                if current_text and time.time() - last_change >= 1.0:
                    break
                time.sleep(0.05)
        finally:
            if started:
                audio_engine.stop()
            try:
                input_node.removeTapOnBus_(0)
            except Exception:
                pass
            try:
                request.endAudio()
            except Exception:
                pass
            try:
                task.cancel()
            except Exception:
                pass

        text = str(result_holder.get("text") or "").strip()
        if not text:
            error = result_holder.get("error")
            if error is not None:
                raise VoiceInputError(f"Speech recognition failed: {error}")
            raise VoiceInputError("No destination was heard.")
        return text

    @staticmethod
    def _request_authorization(Speech) -> int:
        holder = {"status": None}
        done = threading.Event()

        def callback(status) -> None:
            holder["status"] = status
            done.set()

        Speech.SFSpeechRecognizer.requestAuthorization_(callback)
        if not done.wait(timeout=10.0):
            raise VoiceInputError("Timed out waiting for speech recognition permission.")
        return int(holder["status"])


class MacOSCodewordSessionListener(TranscriptSessionListener):
    def capture_transcript(self, config: DestinationCaptureConfig) -> str:
        if platform.system() != "Darwin":
            raise VoiceInputError("macOS Speech backend is only available on macOS.")

        try:
            import AVFoundation
            import Foundation
            import Speech
        except Exception as exc:
            raise VoiceInputError(f"macOS Speech framework is unavailable: {exc}") from exc

        status = MacOSSpeechDestinationListener._request_authorization(Speech)
        authorized = getattr(Speech, "SFSpeechRecognizerAuthorizationStatusAuthorized", 3)
        if status != authorized:
            raise VoiceInputError(f"Speech recognition permission was not authorized; status={status}.")

        recognizer = Speech.SFSpeechRecognizer.alloc().initWithLocale_(
            Foundation.NSLocale.localeWithLocaleIdentifier_(config.locale)
        )
        if recognizer is None or not recognizer.isAvailable():
            raise VoiceInputError("Speech recognizer is not available.")

        request = Speech.SFSpeechAudioBufferRecognitionRequest.alloc().init()
        request.setShouldReportPartialResults_(True)

        if config.prefer_on_device and hasattr(recognizer, "supportsOnDeviceRecognition"):
            try:
                if recognizer.supportsOnDeviceRecognition() and hasattr(request, "setRequiresOnDeviceRecognition_"):
                    request.setRequiresOnDeviceRecognition_(True)
            except Exception:
                pass

        audio_engine = AVFoundation.AVAudioEngine.alloc().init()
        input_node = audio_engine.inputNode()
        if input_node is None:
            raise VoiceInputError("No microphone input node is available.")

        session = CodewordSessionState(codeword=config.codeword, stop_word=config.stop_word)
        result_holder = {"error": None}
        done = threading.Event()

        def result_handler(result, error) -> None:
            if result is not None:
                transcription = result.bestTranscription()
                if transcription is not None:
                    session.update(str(transcription.formattedString()))
                if result.isFinal() or session.heard_stop:
                    done.set()
            if error is not None:
                result_holder["error"] = error
                done.set()

        task = recognizer.recognitionTaskWithRequest_resultHandler_(request, result_handler)
        fmt = input_node.outputFormatForBus_(0)

        def tap_handler(buffer, when) -> None:
            request.appendAudioPCMBuffer_(buffer)

        input_node.installTapOnBus_bufferSize_format_block_(0, 1024, fmt, tap_handler)
        audio_engine.prepare()

        started = False
        try:
            started = _start_audio_engine(audio_engine)
            if not started:
                raise VoiceInputError("Microphone audio engine did not start.")
            deadline = time.time() + max(1.0, config.timeout_s)
            while time.time() < deadline and not done.is_set():
                if session.active and session.result() and not session.heard_stop:
                    # Keep recording until the stop word or timeout.
                    pass
                time.sleep(0.05)
        finally:
            if started:
                audio_engine.stop()
            try:
                input_node.removeTapOnBus_(0)
            except Exception:
                pass
            try:
                request.endAudio()
            except Exception:
                pass
            try:
                task.cancel()
            except Exception:
                pass

        text = session.result()
        if text:
            return text
        error = result_holder.get("error")
        if error is not None:
            raise VoiceInputError(f"Speech recognition failed: {error}")
        if not session.heard_codeword:
            raise VoiceInputError(f"Codeword {config.codeword!r} was not heard.")
        raise VoiceInputError("No destination was heard after the codeword.")


def _start_audio_engine(audio_engine) -> bool:
    try:
        result = audio_engine.startAndReturnError_(None)
    except TypeError:
        result = audio_engine.startAndReturnError_()
    except Exception as exc:
        raise VoiceInputError(f"Could not start microphone audio engine: {exc}") from exc

    if isinstance(result, tuple):
        return bool(result[0])
    if result is None:
        return True
    return bool(result)


def capture_destination_by_voice(
    speak: Callable[[str], None],
    config: Optional[DestinationCaptureConfig] = None,
    listener: Optional[DestinationListener] = None,
) -> str:
    cfg = config or DestinationCaptureConfig()
    backend = listener or MacOSSpeechDestinationListener()
    last_error: Optional[Exception] = None

    for attempt in range(1, max(1, cfg.attempts) + 1):
        speak("Where would you like to go?")
        time.sleep(0.3)
        try:
            destination = backend.capture(cfg).strip()
        except Exception as exc:
            last_error = exc
            speak("I did not catch that.")
            continue

        if destination:
            speak(f"Navigating to {destination}.")
            return destination

    raise VoiceInputError(f"Could not capture a destination by voice. Last error: {last_error}")


def capture_destination_with_codeword(
    speak: Callable[[str], None],
    config: Optional[DestinationCaptureConfig] = None,
    listener: Optional[TranscriptSessionListener] = None,
) -> str:
    cfg = config or DestinationCaptureConfig()
    backend = listener or MacOSCodewordSessionListener()
    last_error: Optional[Exception] = None

    for attempt in range(1, max(1, cfg.attempts) + 1):
        speak(f"Say {cfg.codeword}, then your destination, then {cfg.stop_word}.")
        time.sleep(0.3)
        try:
            destination = backend.capture_transcript(cfg).strip()
        except Exception as exc:
            last_error = exc
            speak("I did not catch that.")
            continue

        if destination:
            speak(f"Navigating to {destination}.")
            return destination
        speak("I did not catch that.")

    raise VoiceInputError(f"Could not capture a destination after the codeword. Last error: {last_error}")


def parse_codeword_session(transcript: str, codeword: str = "navigate", stop_word: str = "stop") -> str:
    session = CodewordSessionState(codeword=codeword, stop_word=stop_word)
    session.update(transcript)
    return session.result()


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _text_after_phrase(text: str, phrase: str) -> Optional[str]:
    phrase = _normalize_spaces(phrase).lower()
    match = re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text.lower())
    if not match:
        return None
    return text[match.end() :].strip()


def _split_before_phrase(text: str, phrase: str) -> tuple[str, bool]:
    phrase = _normalize_spaces(phrase).lower()
    match = re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text.lower())
    if not match:
        return text, False
    return text[: match.start()].strip(), True
