from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_input import (  # noqa: E402
    DestinationCaptureConfig,
    DestinationListener,
    TranscriptSessionListener,
    VoiceInputError,
    capture_destination_by_voice,
    capture_destination_with_codeword,
    parse_codeword_session,
)


class FakeListener(DestinationListener):
    def __init__(self, results) -> None:
        self.results = list(results)

    def capture(self, config: DestinationCaptureConfig) -> str:
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return str(result)


class FakeTranscriptListener(TranscriptSessionListener):
    def __init__(self, results) -> None:
        self.results = list(results)

    def capture_transcript(self, config: DestinationCaptureConfig) -> str:
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return parse_codeword_session(str(result), codeword=config.codeword, stop_word=config.stop_word)


def test_voice_destination_capture_returns_spoken_destination() -> None:
    spoken = []

    destination = capture_destination_by_voice(
        speak=spoken.append,
        config=DestinationCaptureConfig(timeout_s=0.1, attempts=1),
        listener=FakeListener(["Central Library"]),
    )

    assert destination == "Central Library"
    assert spoken[0] == "Where would you like to go?"
    assert spoken[-1] == "Navigating to Central Library."


def test_voice_destination_capture_retries_after_failed_attempt() -> None:
    spoken = []

    destination = capture_destination_by_voice(
        speak=spoken.append,
        config=DestinationCaptureConfig(timeout_s=0.1, attempts=2),
        listener=FakeListener([VoiceInputError("too quiet"), "Room 204"]),
    )

    assert destination == "Room 204"
    assert "I did not catch that." in spoken


def test_voice_destination_capture_raises_after_all_attempts_fail() -> None:
    spoken = []

    try:
        capture_destination_by_voice(
            speak=spoken.append,
            config=DestinationCaptureConfig(timeout_s=0.1, attempts=1),
            listener=FakeListener([VoiceInputError("no permission")]),
        )
    except VoiceInputError as exc:
        assert "Could not capture" in str(exc)
    else:
        raise AssertionError("expected VoiceInputError")


def test_codeword_session_extracts_text_between_codeword_and_stop_word() -> None:
    assert parse_codeword_session("hello navigate Rieber Hall UCLA stop thanks") == "Rieber Hall UCLA"
    assert parse_codeword_session("navigate De Neve Plaza") == "De Neve Plaza"
    assert parse_codeword_session("please go to Rieber Hall stop") == ""


def test_codeword_destination_capture_returns_destination() -> None:
    spoken = []

    destination = capture_destination_with_codeword(
        speak=spoken.append,
        config=DestinationCaptureConfig(timeout_s=0.1, attempts=1, codeword="navigate", stop_word="stop"),
        listener=FakeTranscriptListener(["navigate Rieber Hall UCLA stop"]),
    )

    assert destination == "Rieber Hall UCLA"
    assert spoken[0] == "Say navigate, then your destination, then stop."
    assert spoken[-1] == "Navigating to Rieber Hall UCLA."


def test_codeword_destination_capture_retries_when_codeword_missing() -> None:
    spoken = []

    destination = capture_destination_with_codeword(
        speak=spoken.append,
        config=DestinationCaptureConfig(timeout_s=0.1, attempts=2, codeword="navigate", stop_word="stop"),
        listener=FakeTranscriptListener(["Rieber Hall stop", "navigate Powell Library stop"]),
    )

    assert destination == "Powell Library"
    assert "I did not catch that." in spoken


def main() -> None:
    test_voice_destination_capture_returns_spoken_destination()
    test_voice_destination_capture_retries_after_failed_attempt()
    test_voice_destination_capture_raises_after_all_attempts_fail()
    test_codeword_session_extracts_text_between_codeword_and_stop_word()
    test_codeword_destination_capture_returns_destination()
    test_codeword_destination_capture_retries_when_codeword_missing()
    print("voice input tests passed")


if __name__ == "__main__":
    main()
