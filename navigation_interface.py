"""
Navigation Interface - Integration layer for user-friendly assistive navigation

Connects the agentic routing system with intelligent UI and speech controller
to provide a production-ready experience for blind/low-vision users.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from agentic_layer import AgentAction, AgentDecision, AgenticNavigationRouter
from agentic_layer.models import FrameContext
from elevenlabs_speech import ElevenLabsSpeechController, create_elevenlabs_controller
from speech_controller import AlertType, IntelligentSpeechController
from user_interface import SystemState, UserInterface, UserMode, UserPreferences


class NavigationInterface:
    """
    Main interface integrating vision, routing, and intelligent user feedback.
    
    This class manages:
    - Context-aware decision making
    - Intelligent speech timing
    - User-friendly status updates
    - Mode transitions and adaptation
    """
    
    def __init__(
        self,
        router: AgenticNavigationRouter,
        preferences: Optional[UserPreferences] = None
    ):
        self.router = router
        self.ui = UserInterface(preferences)
        self.speech = ElevenLabsSpeechController(user_interface=self.ui)
        
        # State tracking
        self._last_decision: Optional[AgentDecision] = None
        self._last_decision_time_ms: int = 0
        self._consecutive_same_decisions: int = 0
        self._last_spoken_message: Optional[str] = None
        
        # Context history
        self._recent_detections: List[str] = []
        self._recent_warnings: List[str] = []
        self._frame_count: int = 0
        
        # Guidance state
        self._last_guidance_time_ms: int = 0
        self._guidance_repeat_count: int = 0
        
    def start(self) -> None:
        """Initialize and start all components."""
        self.speech.start()
        self.speak_info("Navigation system ready. Say 'navigate' followed by your destination to begin.")
        print("[interface] Navigation interface started")
    
    def stop(self) -> None:
        """Stop all components gracefully."""
        self.speech.stop()
        print("[interface] Navigation interface stopped")
    
    def process_decision(self, decision: AgentDecision, ctx: FrameContext) -> bool:
        """
        Process an agentic decision and provide appropriate user feedback.
        
        Args:
            decision: The agentic routing decision
            ctx: Frame context with current environment state
            
        Returns:
            True if feedback was provided, False otherwise
        """
        self._frame_count += 1
        now_ms = ctx.timestamp_ms or int(time.time() * 1000)
        
        # Update UI state from context
        self._update_state_from_context(ctx)
        
        # Determine if we should speak this decision
        if not self._should_provide_feedback(decision, now_ms):
            return False
        
        # Classify and handle the decision
        alert_type, priority = self._classify_decision(decision)
        
        # Format message for user
        user_message = self._format_decision_message(decision)
        
        # Provide feedback
        success = self.speech.speak(
            user_message,
            alert_type=alert_type,
            priority=priority,
            formatted=True
        )
        
        if success:
            self._last_decision = decision
            self._last_decision_time_ms = now_ms
            self._last_spoken_message = user_message
            
            # Track consecutive same decisions
            if self._last_spoken_message == user_message:
                self._consecutive_same_decisions += 1
            else:
                self._consecutive_same_decisions = 0
        
        return success
    
    def _update_state_from_context(self, ctx: FrameContext) -> None:
        """Update UI state based on frame context."""
        # Update detections
        detections = [d.label for d in ctx.detections] if ctx.detections else []
        self._recent_detections = detections
        self.ui.update_state(active_detections=detections)
        
        # Update warnings
        warnings = [w.message for w in ctx.warnings] if ctx.warnings else []
        self._recent_warnings = warnings
        self.ui.update_state(recent_warnings=warnings)
        
        # Update movement state
        if ctx.motion:
            is_moving = ctx.motion.is_moving if hasattr(ctx.motion, 'is_moving') else False
            speed = ctx.motion.speed_mps if hasattr(ctx.motion, 'speed_mps') else 0.0
            self.ui.update_state(is_moving=is_moving, speed_mps=speed)
        
        # Update route state
        if ctx.route:
            destination = ctx.route.destination if hasattr(ctx.route, 'destination') else None
            next_instruction = ctx.route.next_instruction if hasattr(ctx.route, 'next_instruction') else None
            approaching = ctx.route.approaching_destination if hasattr(ctx.route, 'approaching_destination') else False
            
            self.ui.update_state(
                current_destination=destination,
                approaching_destination=approaching
            )
        
        # Update scene state
        if ctx.scene:
            indoor = ctx.scene.location_type in ['hallway', 'room', 'indoor'] if hasattr(ctx.scene, 'location_type') else False
            on_sidewalk = ctx.scene.location_type == 'sidewalk' if hasattr(ctx.scene, 'location_type') else True
            
            self.ui.update_state(
                indoor_environment=indoor,
                on_sidewalk=on_sidewalk
            )
    
    def _should_provide_feedback(self, decision: AgentDecision, now_ms: int) -> bool:
        """Determine if we should provide feedback for this decision."""
        # Always provide feedback for non-silent actions
        if decision.action == AgentAction.SILENT:
            return False
        
        # Don't repeat the exact same message too frequently
        if decision.message == self._last_spoken_message:
            time_since_last = now_ms - self._last_decision_time_ms
            
            # Different timing based on action type
            if decision.action == AgentAction.WARN:
                min_interval = 2000  # 2 seconds for warnings
            elif decision.action == AgentAction.GUIDE:
                min_interval = 5000  # 5 seconds for guidance
            else:
                min_interval = 3000  # 3 seconds default
            
            if time_since_last < min_interval:
                return False
        
        return True
    
    def _classify_decision(self, decision: AgentDecision) -> Tuple[AlertType, int]:
        """Classify a decision into alert type and priority."""
        action = decision.action
        priority = decision.priority
        
        if action == AgentAction.WARN:
            if priority >= 95:
                return (AlertType.CRITICAL, priority)
            else:
                return (AlertType.WARNING, priority)
        
        elif action == AgentAction.GUIDE:
            return (AlertType.GUIDANCE, priority)
        
        elif action == AgentAction.ORIENT:
            return (AlertType.INFO, priority)
        
        elif action == AgentAction.ANSWER:
            return (AlertType.INFO, priority)
        
        elif action == AgentAction.ASK:
            return (AlertType.GUIDANCE, priority)
        
        elif action == AgentAction.ESCALATE:
            return (AlertType.CRITICAL, 100)
        
        else:
            return (AlertType.INFO, priority)
    
    def _format_decision_message(self, decision: AgentDecision) -> str:
        """Format decision message for user consumption."""
        if not decision.message:
            return ""
        
        message = decision.message
        
        # Add context-appropriate prefixes based on situation
        if self.ui.state.approaching_destination and decision.action == AgentAction.GUIDE:
            if "approaching" not in message.lower() and "near" not in message.lower():
                message = f"You're getting close. {message}"
        
        if self.ui.state.in_crosswalk and decision.action == AgentAction.WARN:
            if "crosswalk" not in message.lower():
                message = f"In crosswalk. {message}"
        
        # Add haptic indication for critical alerts
        if decision.haptic and decision.haptic.value != "none":
            # Message already includes haptic info in most cases
            pass
        
        return message
    
    def set_mode(self, mode: UserMode) -> None:
        """Change operating mode and announce to user."""
        old_mode = self.ui.state.mode
        self.ui.update_state(mode=mode)
        self.ui.clear_message_history()
        
        mode_names = {
            UserMode.EXPLORATION: "exploration mode",
            UserMode.NAVIGATION: "navigation mode",
            UserMode.SEARCHING: "search mode",
            UserMode.STATIONARY: "stationary mode",
            UserMode.ORIENTATION: "orientation mode"
        }
        
        if old_mode != mode:
            self.speak_info(f"Switched to {mode_names.get(mode, 'unknown mode')}")
    
    def set_destination(self, destination: str) -> None:
        """Set destination and switch to navigation mode (no speech — verification already confirmed)."""
        self.ui.update_state(
            current_destination=destination,
            approaching_destination=False,
            mode=UserMode.NAVIGATION
        )
        self.ui.clear_message_history()
    
    def clear_destination(self) -> None:
        """Clear destination and return to exploration mode."""
        self.ui.update_state(current_destination=None, approaching_destination=False)
        self.set_mode(UserMode.EXPLORATION)
    
    def speak_critical(self, message: str) -> bool:
        """Speak a critical safety alert."""
        return self.speech.speak_critical(message)
    
    def speak_warning(self, message: str) -> bool:
        """Speak a warning alert."""
        return self.speech.speak_warning(message)
    
    def speak_guidance(self, message: str) -> bool:
        """Speak navigation guidance."""
        return self.speech.speak_guidance(message)
    
    def speak_info(self, message: str) -> bool:
        """Speak general information."""
        return self.speech.speak_info(message)
    
    def get_status(self) -> Dict:
        """Get current system status."""
        return {
            "ui_state": self.ui.get_status_summary(),
            "speech_stats": self.speech.get_statistics(),
            "alert_stats": self.ui.get_alert_statistics(),
            "frame_count": self._frame_count,
            "recent_detections": self._recent_detections[:5],
            "recent_warnings": self._recent_warnings[:3],
        }
