"""
User Interface Layer for Assistive Navigation

Provides intelligent speech timing, user-friendly feedback, and context-aware
interaction management for blind/low-vision users.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set


class UserMode(Enum):
    """Operating modes based on user activity and intent."""
    EXPLORATION = auto()      # General walking, no specific destination
    NAVIGATION = auto()       # Following route to destination
    SEARCHING = auto()        # Looking for specific object/location
    STATIONARY = auto()       # Standing still, not moving
    ORIENTATION = auto()      # Getting bearings, understanding environment


class AlertType(Enum):
    """Types of alerts with different urgency and repeat policies."""
    CRITICAL = auto()         # Immediate danger (person, vehicle)
    WARNING = auto()          # Caution needed (obstacle, curb)
    GUIDANCE = auto()         # Navigation instructions
    INFO = auto()             # General information
    CONFIRMATION = auto()     # Acknowledgment feedback


@dataclass
class SpeechTiming:
    """Timing configuration for different alert types."""
    initial_delay_ms: int = 0           # Delay before first alert
    min_repeat_interval_ms: int = 3000  # Minimum time between repeats
    max_repeats: int = 3                 # Maximum times to repeat same message
    cooldown_after_max_ms: int = 10000   # Cooldown after max repeats reached


@dataclass
class UserPreferences:
    """User-configurable preferences for the interface."""
    verbosity: str = "normal"           # minimal, normal, verbose
    speech_rate: float = 1.0            # 0.5 to 1.5
    warning_sensitivity: str = "normal" # low, normal, high
    haptic_enabled: bool = True
    audio_cues_enabled: bool = True
    auto_repeat_warnings: bool = True
    navigation_prompt_interval_s: float = 8.0
    
    # Mode-specific settings
    indoor_mode: bool = False
    outdoor_mode: bool = False
    crossing_mode: bool = False


@dataclass
class SystemState:
    """Current state of the navigation system."""
    mode: UserMode = UserMode.EXPLORATION
    is_moving: bool = False
    speed_mps: float = 0.0
    approaching_destination: bool = False
    in_crosswalk: bool = False
    on_sidewalk: bool = True
    indoor_environment: bool = False
    last_guidance_time_ms: int = 0
    last_warning_time_ms: int = 0
    current_destination: Optional[str] = None
    next_turn_distance_m: Optional[float] = None
    next_turn_direction: Optional[str] = None
    active_detections: List[str] = field(default_factory=list)
    recent_warnings: List[str] = field(default_factory=list)


class UserInterface:
    """
    Intelligent user interface for assistive navigation.
    
    Manages:
    - Smart speech timing to prevent spam
    - Context-aware alert prioritization
    - User preference adaptation
    - Clear, actionable feedback
    """
    
    # Timing configurations for different alert types
    DEFAULT_TIMING = {
        AlertType.CRITICAL: SpeechTiming(
            initial_delay_ms=0,
            min_repeat_interval_ms=1500,
            max_repeats=5,
            cooldown_after_max_ms=5000
        ),
        AlertType.WARNING: SpeechTiming(
            initial_delay_ms=500,
            min_repeat_interval_ms=3000,
            max_repeats=3,
            cooldown_after_max_ms=8000
        ),
        AlertType.GUIDANCE: SpeechTiming(
            initial_delay_ms=0,
            min_repeat_interval_ms=8000,
            max_repeats=2,
            cooldown_after_max_ms=15000
        ),
        AlertType.INFO: SpeechTiming(
            initial_delay_ms=0,
            min_repeat_interval_ms=15000,
            max_repeats=1,
            cooldown_after_max_ms=30000
        ),
        AlertType.CONFIRMATION: SpeechTiming(
            initial_delay_ms=0,
            min_repeat_interval_ms=2000,
            max_repeats=1,
            cooldown_after_max_ms=5000
        ),
    }
    
    def __init__(self, preferences: Optional[UserPreferences] = None):
        self.preferences = preferences or UserPreferences()
        self.state = SystemState()
        self.timing = self._apply_preferences_to_timing()
        
        # Message tracking for anti-spam
        self._message_history: Dict[str, Dict] = {}  # message -> {count, last_time_ms, alert_type}
        self._recent_alerts: List[str] = []  # Last N alerts for context
        self._last_speech_time_ms: int = 0
        
        # Performance tracking
        self._start_time_ms: int = int(time.time() * 1000)
        self._alert_counts: Dict[AlertType, int] = {t: 0 for t in AlertType}
        
    def _apply_preferences_to_timing(self) -> Dict[AlertType, SpeechTiming]:
        """Adjust timing based on user preferences."""
        timing = {}
        sensitivity_mult = {
            "low": 1.5,
            "normal": 1.0,
            "high": 0.7
        }.get(self.preferences.warning_sensitivity, 1.0)
        
        for alert_type, base_timing in self.DEFAULT_TIMING.items():
            timing[alert_type] = SpeechTiming(
                initial_delay_ms=base_timing.initial_delay_ms,
                min_repeat_interval_ms=int(base_timing.min_repeat_interval_ms * sensitivity_mult),
                max_repeats=base_timing.max_repeats,
                cooldown_after_max_ms=base_timing.cooldown_after_max_ms
            )
        
        return timing
    
    def update_state(self, **kwargs) -> None:
        """Update system state with new information."""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
    
    def should_speak(
        self, 
        message: str, 
        alert_type: AlertType,
        priority: int,
        force: bool = False
    ) -> bool:
        """
        Determine if a message should be spoken based on timing rules.
        
        Args:
            message: The message to potentially speak
            alert_type: Type of alert for timing rules
            priority: Priority score (higher = more urgent)
            force: Override timing rules for critical situations
            
        Returns:
            True if message should be spoken, False otherwise
        """
        now_ms = int(time.time() * 1000)
        
        # Always speak critical alerts
        if force or alert_type == AlertType.CRITICAL:
            self._record_speech(message, alert_type, now_ms)
            return True
        
        # Check verbosity settings
        if self.preferences.verbosity == "minimal" and alert_type in [AlertType.INFO, AlertType.GUIDANCE]:
            return False
        
        # Get timing rules for this alert type
        timing = self.timing.get(alert_type, self.DEFAULT_TIMING[AlertType.INFO])
        
        # Check if we've spoken this exact message recently
        msg_record = self._message_history.get(message)
        if msg_record:
            time_since_last = now_ms - msg_record["last_time_ms"]
            repeat_count = msg_record["count"]
            
            # If max repeats reached, apply cooldown
            if repeat_count >= timing.max_repeats:
                if time_since_last < timing.cooldown_after_max_ms:
                    return False
                else:
                    # Reset counter after cooldown
                    msg_record["count"] = 0
            
            # Check minimum repeat interval
            if time_since_last < timing.min_repeat_interval_ms:
                return False
        
        # Check global speech rate limiting
        time_since_last_speech = now_ms - self._last_speech_time_ms
        min_gap = 500 if alert_type == AlertType.CRITICAL else 1000
        if time_since_last_speech < min_gap:
            return False
        
        self._record_speech(message, alert_type, now_ms)
        return True
    
    def _record_speech(self, message: str, alert_type: AlertType, timestamp_ms: int) -> None:
        """Record that a message was spoken."""
        if message not in self._message_history:
            self._message_history[message] = {
                "count": 0,
                "last_time_ms": timestamp_ms,
                "alert_type": alert_type
            }
        
        self._message_history[message]["count"] += 1
        self._message_history[message]["last_time_ms"] = timestamp_ms
        self._last_speech_time_ms = timestamp_ms
        self._alert_counts[alert_type] += 1
        
        # Keep recent alerts list manageable
        self._recent_alerts.append(message)
        if len(self._recent_alerts) > 20:
            self._recent_alerts.pop(0)
    
    def format_message_for_user(self, base_message: str, context: Optional[Dict] = None) -> str:
        """
        Format a technical message into user-friendly language.
        
        Args:
            base_message: Original technical message
            context: Additional context for formatting
            
        Returns:
            User-friendly formatted message
        """
        # Simplify technical terms
        replacements = {
            "at 12 o'clock": "ahead",
            "at 1 o'clock": "ahead and to your right",
            "at 2 o'clock": "to your right",
            "at 3 o'clock": "to your right",
            "at 9 o'clock": "to your left",
            "at 10 o'clock": "to your left",
            "at 11 o'clock": "ahead and to your left",
            "within arm's reach": "very close",
            "less than 3 feet away": "close by",
            "in the near field": "nearby",
            "in the far field": "far ahead",
        }
        
        message = base_message
        for old, new in replacements.items():
            message = message.replace(old, new)
        
        # Add context-appropriate prefixes
        if self.state.approaching_destination and "destination" in message.lower():
            message = f"You're almost there. {message}"
        
        if self.state.in_crosswalk and "walk" not in message.lower():
            message = f"In crosswalk. {message}"
        
        return message
    
    def get_status_summary(self) -> str:
        """Get a user-friendly status summary."""
        parts = []
        
        # Mode
        mode_str = self.state.mode.name.lower().replace("_", " ")
        parts.append(f"Mode: {mode_str}")
        
        # Destination
        if self.state.current_destination:
            parts.append(f"Going to: {self.state.current_destination}")
        
        # Movement
        if self.state.is_moving:
            parts.append("Moving" if self.state.speed_mps > 0.5 else "Slow movement detected")
        else:
            parts.append("Stopped")
        
        # Environment
        env = []
        if self.state.indoor_environment:
            env.append("indoor")
        if self.state.on_sidewalk:
            env.append("on sidewalk")
        elif self.state.in_crosswalk:
            env.append("in crosswalk")
        if env:
            parts.append(f"Environment: {', '.join(env)}")
        
        return " | ".join(parts)
    
    def get_alert_statistics(self) -> Dict[str, int]:
        """Get statistics on alerts spoken."""
        return {
            "total_alerts": sum(self._alert_counts.values()),
            "critical": self._alert_counts[AlertType.CRITICAL],
            "warnings": self._alert_counts[AlertType.WARNING],
            "guidance": self._alert_counts[AlertType.GUIDANCE],
            "info": self._alert_counts[AlertType.INFO],
            "runtime_minutes": (int(time.time() * 1000) - self._start_time_ms) // 60000,
        }
    
    def clear_message_history(self) -> None:
        """Clear message history (useful when changing modes or destinations)."""
        self._message_history.clear()
        self._recent_alerts.clear()
