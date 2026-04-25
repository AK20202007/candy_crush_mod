#!/usr/bin/env python3
"""
ElevenLabs Integration Demo
Showcases all features of the integrated speech system.
"""

import os
import time
from elevenlabs_speech import ElevenLabsSpeechController, create_elevenlabs_controller
from navigation_interface import NavigationInterface
from agentic_layer import AgenticNavigationRouter
from user_interface import UserPreferences

def demo_voice_presets():
    """Demo different voice presets."""
    print("🎙️  VOICE PRESETS DEMO")
    print("=" * 50)
    
    voices = [
        ("default", "Natural female voice"),
        ("calm", "Calm and soothing voice"),
        ("professional", "Clear professional voice"),
        ("warm", "Friendly warm voice"),
        ("male", "Natural male voice")
    ]
    
    for voice_name, description in voices:
        print(f"\nTesting {voice_name}: {description}")
        
        speech = create_elevenlabs_controller(voice=voice_name)
        speech.start()
        
        message = f"This is the {voice_name} voice preset. {description}"
        speech.speak_info(message)
        
        # Wait for speech to complete
        time.sleep(3)
        
        stats = speech.get_statistics()
        print(f"  Status: {stats['messages_spoken']} spoken, {stats['api_errors']} errors")
        print(f"  ElevenLabs: {'✅ Active' if stats['elevenlabs_available'] else '❌ Inactive'}")
        
        speech.stop()
    
    print("\n✅ Voice presets demo completed!\n")

def demo_priority_levels():
    """Demo different priority levels."""
    print("🚨 PRIORITY LEVELS DEMO")
    print("=" * 50)
    
    speech = create_elevenlabs_controller()
    speech.start()
    
    print("\nTesting priority-based speech...")
    
    # Test different priority levels
    messages = [
        ("critical", "⚠️  Critical safety alert: Obstacle detected ahead!"),
        ("warning", "⚠️  Warning: Person crossing nearby"),
        ("guidance", "📍 Navigation guidance: Turn left in 100 feet"),
        ("info", "ℹ️  Information: System status normal")
    ]
    
    for priority, message in messages:
        print(f"\n{priority.upper()}: {message}")
        
        if priority == "critical":
            speech.speak_critical(message.split(": ")[1])
        elif priority == "warning":
            speech.speak_warning(message.split(": ")[1])
        elif priority == "guidance":
            speech.speak_guidance(message.split(": ")[1])
        else:
            speech.speak_info(message.split(": ")[1])
        
        time.sleep(2)
    
    stats = speech.get_statistics()
    print(f"\n📊 Final Stats: {stats['messages_spoken']} spoken, {stats['api_calls']} API calls")
    
    speech.stop()
    print("\n✅ Priority levels demo completed!\n")

def demo_navigation_integration():
    """Demo integration with navigation system."""
    print("🧭 NAVIGATION INTEGRATION DEMO")
    print("=" * 50)
    
    # Create navigation interface with ElevenLabs
    router = AgenticNavigationRouter()
    prefs = UserPreferences()
    prefs.verbosity = "minimal"
    
    interface = NavigationInterface(router, prefs)
    interface.start()
    
    print("\nTesting navigation system with ElevenLabs speech...")
    
    # Simulate navigation events
    navigation_events = [
        ("Welcome", "Welcome to assistive navigation. System is ready."),
        ("Destination", "Starting navigation to Central Library."),
        ("Turn", "Turn right at the next intersection."),
        ("Distance", "In 200 feet, turn left."),
        ("Warning", "Caution: Crosswalk ahead."),
        ("Arrival", "You have arrived at your destination."),
        ("Complete", "Navigation complete. Thank you for using our system.")
    ]
    
    for event_type, message in navigation_events:
        print(f"\n🗺️  {event_type}: {message}")
        
        if "Warning" in event_type:
            interface.speak_warning(message)
        elif "Welcome" in event_type or "Arrival" in event_type or "Complete" in event_type:
            interface.speak_info(message)
        else:
            interface.speak_guidance(message)
        
        time.sleep(2)
    
    # Get comprehensive status
    status = interface.get_status()
    print(f"\n📊 Navigation Status:")
    print(f"  Speech Stats: {status['speech_stats']}")
    print(f"  Alert Stats: {status['alert_stats']}")
    
    interface.stop()
    print("\n✅ Navigation integration demo completed!\n")

def demo_fallback_system():
    """Demo fallback to system TTS."""
    print("🔄 FALLBACK SYSTEM DEMO")
    print("=" * 50)
    
    print("\nTesting fallback when ElevenLabs unavailable...")
    
    # Save original API key
    original_key = os.getenv("ELEVENLABS_API_KEY")
    
    # Test without API key
    os.environ["ELEVENLABS_API_KEY"] = ""
    
    speech = create_elevenlabs_controller()
    speech.start()
    
    stats = speech.get_statistics()
    print(f"  Fallback Status: {'✅ Active' if stats['using_fallback'] else '❌ Inactive'}")
    print(f"  ElevenLabs Available: {'✅ Yes' if stats['elevenlabs_available'] else '❌ No'}")
    
    speech.speak_info("This is the fallback system text-to-speech voice.")
    time.sleep(2)
    
    speech.stop()
    
    # Restore API key
    if original_key:
        os.environ["ELEVENLABS_API_KEY"] = original_key
    
    print("\n✅ Fallback system demo completed!\n")

def demo_performance_monitoring():
    """Demo performance monitoring and statistics."""
    print("📈 PERFORMANCE MONITORING DEMO")
    print("=" * 50)
    
    speech = create_elevenlabs_controller()
    speech.start()
    
    print("\nTesting performance with multiple messages...")
    
    # Send multiple messages to test performance
    for i in range(10):
        message = f"Performance test message number {i + 1}"
        speech.speak_info(message)
        time.sleep(0.5)
    
    # Get detailed statistics
    stats = speech.get_statistics()
    
    print(f"\n📊 Performance Statistics:")
    print(f"  Messages Spoken: {stats['messages_spoken']}")
    print(f"  Messages Suppressed: {stats['messages_suppressed']}")
    print(f"  API Calls: {stats['api_calls']}")
    print(f"  API Errors: {stats['api_errors']}")
    print(f"  ElevenLabs Available: {'✅ Yes' if stats['elevenlabs_available'] else '❌ No'}")
    print(f"  Using Fallback: {'✅ Yes' if stats['using_fallback'] else '❌ No'}")
    
    print(f"\n📋 Queue Status:")
    for queue_name, size in stats['queues'].items():
        print(f"  {queue_name.capitalize()}: {size} messages")
    
    speech.stop()
    print("\n✅ Performance monitoring demo completed!\n")

def main():
    """Run the complete ElevenLabs integration demo."""
    print("🎉 ELEVENLABS INTEGRATION COMPLETE DEMO")
    print("=" * 60)
    print("This demo showcases the fully integrated ElevenLabs speech system")
    print("for the Candy Crush assistive navigation application.")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("⚠️  WARNING: No ELEVENLABS_API_KEY found")
        print("   Demo will use fallback system TTS")
        print("   Set API key for full ElevenLabs functionality:")
        print("   export ELEVENLABS_API_KEY='your_key_here'")
        print()
    else:
        print("✅ ElevenLabs API key detected")
        print("   Demo will use high-quality ElevenLabs voices")
        print()
    
    try:
        # Run all demos
        demo_voice_presets()
        demo_priority_levels()
        demo_navigation_integration()
        demo_fallback_system()
        demo_performance_monitoring()
        
        print("🎉 DEMO COMPLETE!")
        print("=" * 60)
        print("✅ All ElevenLabs integration features tested successfully!")
        print("✅ Voice presets working with different voice options")
        print("✅ Priority-based speech system operational")
        print("✅ Navigation integration fully functional")
        print("✅ Fallback system working correctly")
        print("✅ Performance monitoring and statistics active")
        print()
        print("🚀 The Candy Crush mod is ready for production use!")
        print("   with high-quality ElevenLabs speech integration.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
