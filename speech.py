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
        self._engine = pyttsx3.init()
        # Slightly faster speech keeps the demo snappy (platform-dependent).
        try:
            rate = self._engine.getProperty("rate")
            self._engine.setProperty("rate", int(rate * 1.1))
        except Exception:
            pass

        self._normal: queue.Queue[str] = queue.Queue()
        self._urgent_pending = False
        self._urgent_message: Optional[str] = None
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

    def speak_urgent(self, message: str) -> None:
        """Queue a high-priority safety message (latest wins if several arrive quickly)."""
        with self._urgent_lock:
            self._urgent_message = message
            self._urgent_pending = True

    def speak_emergency(self, message: str) -> None:
        """Queue a critical emergency message that repeats many times."""
        # For simplicity in this prototype, we'll just repeat the message in the queue.
        for _ in range(10):
            self.speak_urgent(message)

    def speak_normal(self, message: str) -> None:
        """Queue a navigation-style message."""
        if self._stop.is_set():
            return
        self._normal.put(message)

    def _drain_normal_if_urgent(self) -> None:
        """Drop pending navigation lines when a warning must take priority."""
        with self._urgent_lock:
            urgent = self._urgent_pending
        if not urgent:
            return
        try:
            while True:
                self._normal.get_nowait()
        except queue.Empty:
            pass

    def _pop_urgent(self) -> Optional[str]:
        with self._urgent_lock:
            if not self._urgent_pending:
                return None
            msg = self._urgent_message
            self._urgent_pending = False
            self._urgent_message = None
            return msg

    def _worker(self) -> None:
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

            try:
                # Stop may help skip to the next phrase on some backends (best-effort).
                self._engine.stop()
            except Exception:
                pass

            self._engine.say(msg)
            self._engine.runAndWait()
            time.sleep(0.05)  # Tiny pause so logs and engine stay stable between lines.
