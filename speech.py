"""
Text-to-speech wrapper using pyttsx3.

Runs a background thread so vision and navigation do not block on speech.
Urgent safety messages are preferred over routine navigation prompts.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import pyttsx3


class SpeechController:
    """
    Thread-safe TTS with two priority levels.

    - Urgent: obstacle / person warnings (processed before queued navigation).
    - Normal: mock navigation directions (queued FIFO).
    """

    def __init__(self) -> None:
        self._normal: queue.Queue[str] = queue.Queue()
        # PriorityQueue for errors/urgent messages. Lower number = higher priority.
        self._urgent: queue.PriorityQueue[tuple[int, float, str]] = queue.PriorityQueue()
        self._urgent_lock = threading.Lock()

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="SpeechWorker", daemon=True)

    def start(self) -> None:
        """Start the background speech worker."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker (best-effort)."""
        self._stop.set()
        self._normal.put_nowait("")  # unblock queue.get if needed

    def speak_urgent(self, message: str, priority: int = 10) -> None:
        """Queue a high-priority safety message. Lower priority number wins."""
        with self._urgent_lock:
            # Use timestamp to maintain FIFO ordering for items with the same priority
            self._urgent.put((priority, time.time(), message))

    def speak_emergency(self, message: str) -> None:
        """Queue a critical emergency message that repeats many times (Highest Priority 0)."""
        for _ in range(10):
            self.speak_urgent(message, priority=0)

    def speak_normal(self, message: str) -> None:
        """Queue a navigation-style message."""
        if self._stop.is_set():
            return
        self._normal.put(message)

    def _drain_normal_if_urgent(self) -> None:
        """Drop pending navigation lines when a warning must take priority."""
        with self._urgent_lock:
            has_urgent = not self._urgent.empty()
        if not has_urgent:
            return
        try:
            while True:
                self._normal.get_nowait()
        except queue.Empty:
            pass

    def _pop_urgent(self) -> Optional[str]:
        with self._urgent_lock:
            if self._urgent.empty():
                return None
            _, _, msg = self._urgent.get_nowait()
            return msg

    def _worker(self) -> None:
        engine = pyttsx3.init()
        try:
            rate = engine.getProperty("rate")
            engine.setProperty("rate", int(rate * 1.1))
        except Exception:
            pass

        while not self._stop.is_set():
            self._drain_normal_if_urgent()
            msg = self._pop_urgent()
            if msg is None:
                try:
                    msg = self._normal.get(timeout=0.4)
                except queue.Empty:
                    continue
            if msg == "":
                continue

            time.sleep(0.1)  # Brief pause before speaking
            engine.say(msg + " ")
            engine.runAndWait()
            time.sleep(0.2)  # Wait for hardware to finish audio buffer
