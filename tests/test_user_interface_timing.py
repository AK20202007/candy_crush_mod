from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from user_interface import AlertType, UserInterface, UserPreferences  # noqa: E402


def test_high_priority_warning_breaks_global_speech_gap() -> None:
    ui = UserInterface(UserPreferences())

    assert ui.should_speak("Route loaded.", AlertType.GUIDANCE, 60) is True
    assert ui.should_speak("Door handle detected right.", AlertType.WARNING, 92) is True


def test_high_priority_warning_still_throttles_exact_repeats() -> None:
    ui = UserInterface(UserPreferences())

    assert ui.should_speak("Door handle detected right.", AlertType.WARNING, 92) is True
    assert ui.should_speak("Door handle detected right.", AlertType.WARNING, 92) is False


def test_high_priority_warning_repeats_after_warning_interval() -> None:
    ui = UserInterface(UserPreferences())

    assert ui.should_speak("Door handle detected right.", AlertType.WARNING, 92) is True
    ui._message_history["Door handle detected right."]["last_time_ms"] = int(time.time() * 1000) - 4000

    assert ui.should_speak("Door handle detected right.", AlertType.WARNING, 92) is True


def main() -> None:
    test_high_priority_warning_breaks_global_speech_gap()
    test_high_priority_warning_still_throttles_exact_repeats()
    test_high_priority_warning_repeats_after_warning_interval()
    print("user interface timing tests passed")


if __name__ == "__main__":
    main()
