"""
Intelligent Speech Controller for Assistive Navigation

Integrates user preferences, smart timing, and context-aware feedback
to provide a production-ready audio interface for blind/low-vision users.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from typing import Optional

import pyttsx3

from user_interface import AlertType, UserInterface, UserPreferences


class IntelligentSpeechController:
    """
    Production-ready speech controller with intelligent timing and prioritization.
    
    Features:
    - Priority-based speech queue (urgent warnings interrupt navigation)
    - Smart timing to prevent message spam
    - User preference adaptation
    - Context-aware message filtering
    - Clean, actionable audio feedback
    """
    
    def __init__(self, user_interface: Optional[UserInterface] = None):
        self.ui = user_interface or UserInterface()
        
        # Speech queues by priority
        self._critical_queue: queue.Queue[str] = queue.Queue()      # Immediate danger
        self._urgent_queue: queue.Queue[str] = queue.Queue()      # Warnings
        self._normal_queue: queue.Queue[str] = queue.Queue()      # Navigation
        self._info_queue: queue.Queue[str] = queue.Queue()        # General info
        
        # Control
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="SpeechWorker", daemon=True)
        self._current_message: Optional[str] = None
        self._speaking_start_time: Optional[float] = None
        self._queue_lock = threading.Lock()
        
        # Statistics
        self._messages_spoken = 0
        self._messages_suppressed = 0
        self._last_message_type: Optional[AlertType] = None
        
    def start(self) -> None:
        """Start the speech worker thread."""
        self._thread.start()
        print("[speech] Controller started")
    
    def stop(self) -> None:
        """Stop the speech worker gracefully."""
        self._stop.set()
        # Unblock queues
        for q in [self._critical_queue, self._urgent_queue, self._normal_queue, self._info_queue]:
            try:
                q.put_nowait("")
            except queue.Full:
                pass
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=8.0)
        print("[speech] Controller stopped")
    
    def pause(self) -> None:
        """Pause speech output temporarily."""
        self._pause.set()
        print("[speech] Paused")
    
    def resume(self) -> None:
        """Resume speech output."""
        self._pause.clear()
        print("[speech] Resumed")
    
    def speak(
        self, 
        message: str, 
        alert_type: AlertType = AlertType.INFO,
        priority: int = 50,
        formatted: bool = False,
        force: bool = False,
    ) -> bool:
        """
        Queue a message for speaking with intelligent filtering.
        
        Args:
            message: The message to speak
            alert_type: Type of alert for timing rules
            priority: Priority score (0-100, higher = more urgent)
            formatted: If True, message is already user-formatted
            
        Returns:
            True if message was queued, False if suppressed
        """
        if self._stop.is_set():
            return False
        
        # Apply user formatting if needed
        if not formatted:
            message = self.ui.format_message_for_user(message)
        
        # Check if we should speak this message
        should_force = force or alert_type == AlertType.CRITICAL or priority >= 90
        if not self.ui.should_speak(message, alert_type, priority, should_force):
            self._messages_suppressed += 1
            return False
        
        with self._queue_lock:
            # Queue based on priority. Non-critical queues keep only the latest
            # item so delayed speech never catches up with stale frame messages.
            if alert_type == AlertType.CRITICAL or priority >= 90:
                self._drain_all_queues()
                self._critical_queue.put(message)
            elif alert_type == AlertType.WARNING or priority >= 70:
                self._replace_queue(self._urgent_queue, message)
            elif alert_type == AlertType.GUIDANCE:
                self._replace_queue(self._normal_queue, message)
            else:
                self._replace_queue(self._info_queue, message)
        
        return True
    
    def speak_critical(self, message: str) -> bool:
        """Speak a critical safety alert (interrupts everything)."""
        return self.speak(message, AlertType.CRITICAL, priority=100)
    
    def speak_warning(self, message: str, force: bool = False) -> bool:
        """Speak a warning alert."""
        return self.speak(message, AlertType.WARNING, priority=80, force=force)
    
    def speak_guidance(self, message: str, force: bool = False) -> bool:
        """Speak navigation guidance."""
        return self.speak(message, AlertType.GUIDANCE, priority=60, force=force)
    
    def speak_info(self, message: str, force: bool = False) -> bool:
        """Speak general information."""
        return self.speak(message, AlertType.INFO, priority=30, force=force)
    
    def speak_confirmation(self, message: str) -> bool:
        """Speak a confirmation/acknowledgment."""
        return self.speak(message, AlertType.CONFIRMATION, priority=40)
    
    def _drain_lower_priority_queues(self) -> None:
        """Clear normal and info queues when critical alert arrives."""
        for q in [self._normal_queue, self._info_queue]:
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

    def _drain_all_queues(self) -> None:
        """Clear every pending queue before a critical message."""
        for q in [self._critical_queue, self._urgent_queue, self._normal_queue, self._info_queue]:
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

    @staticmethod
    def _replace_queue(q: queue.Queue[str], message: str) -> None:
        """Keep only the newest message for a priority lane."""
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        q.put(message)
    
    def _get_next_message(self) -> Optional[tuple[str, AlertType]]:
        """Get next message from queues in priority order."""
        # Check critical first
        try:
            msg = self._critical_queue.get_nowait()
            if msg:
                return (msg, AlertType.CRITICAL)
        except queue.Empty:
            pass
        
        # Check urgent
        try:
            msg = self._urgent_queue.get_nowait()
            if msg:
                return (msg, AlertType.WARNING)
        except queue.Empty:
            pass
        
        # Check normal with timeout
        try:
            msg = self._normal_queue.get(timeout=0.5)
            if msg:
                return (msg, AlertType.GUIDANCE)
        except queue.Empty:
            pass
        
        # Check info with timeout
        try:
            msg = self._info_queue.get(timeout=0.3)
            if msg:
                return (msg, AlertType.INFO)
        except queue.Empty:
            pass
        
        return None
    
    def _worker(self) -> None:
        """Main speech worker thread."""
        engine = None
        if sys.platform != "darwin":
            try:
                engine = pyttsx3.init()
                rate = engine.getProperty("rate")
                engine.setProperty("rate", int(rate * self.ui.preferences.speech_rate))
                engine.setProperty("volume", 0.9)
            except Exception as e:
                print(f"[speech] Error initializing TTS: {e}")
                return
        
        while not self._stop.is_set():
            # Check if paused
            if self._pause.is_set():
                time.sleep(0.1)
                continue
            
            # Get next message
            result = self._get_next_message()
            if result is None:
                continue
            
            message, alert_type = result
            
            # Skip empty messages
            if not message.strip():
                continue
            
            try:
                # Speak the message
                self._current_message = message
                self._speaking_start_time = time.time()
                
                time.sleep(0.05)
                self._speak_message(engine, message)
                time.sleep(0.2)  # Wait for hardware to finish audio buffer
                
                self._messages_spoken += 1
                self._last_message_type = alert_type
                
                # Brief pause between messages
                time.sleep(0.1)
                
            except Exception as e:
                print(f"[speech] Error speaking message: {e}")
            finally:
                self._current_message = None
                self._speaking_start_time = None

    def _speak_message(self, engine, message: str) -> None:
        if sys.platform == "darwin":
            rate = int(190 * self.ui.preferences.speech_rate)
            subprocess.run(["say", "-r", str(rate), message], check=False)
            return
        engine.say(message + " ")
        engine.runAndWait()
    
    def get_statistics(self) -> dict:
        """Get speech controller statistics."""
        return {
            "messages_spoken": self._messages_spoken,
            "messages_suppressed": self._messages_suppressed,
            "last_message_type": self._last_message_type.name if self._last_message_type else None,
            "currently_speaking": self._current_message is not None,
            "queues": {
                "critical": self._critical_queue.qsize(),
                "urgent": self._urgent_queue.qsize(),
                "normal": self._normal_queue.qsize(),
                "info": self._info_queue.qsize(),
            }
        }

    def is_idle(self) -> bool:
        """Return True if no audio is playing and no queued speech remains."""
        has_items = any(q.qsize() > 0 for q in [self._critical_queue, self._urgent_queue, self._normal_queue, self._info_queue])
        return self._current_message is None and not has_items
    
    def clear_queues(self) -> None:
        """Clear all speech queues."""
        for q in [self._critical_queue, self._urgent_queue, self._normal_queue, self._info_queue]:
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
