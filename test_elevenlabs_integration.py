#!/usr/bin/env python3
"""
Test script for ElevenLabs speech integration.
"""

import os
import sys
from elevenlabs_speech import ElevenLabsSpeechController, create_elevenlabs_controller
from voice_input import ElevenLabsDestinationListener, DestinationCaptureConfig

def test_tts():
    """Test text-to-speech functionality."""
    print("Testing ElevenLabs TTS...")
    
    try:
        # Create controller
        speech = create_elevenlabs_controller()
        speech.start()
        
        # Test basic speech
        print("Speaking test message...")
        speech.speak_info("ElevenLabs text-to-speech is working correctly.")
        
        # Wait for speech to complete
        import time
        time.sleep(3)
        
        # Get statistics
        stats = speech.get_statistics()
        print(f"TTS Statistics: {stats}")
        
        speech.stop()
        print("TTS test completed successfully!")
        return True
        
    except Exception as e:
        print(f"TTS test failed: {e}")
        return False

def test_stt():
    """Test speech-to-text functionality."""
    print("Testing ElevenLabs STT...")
    
    try:
        # Create listener
        listener = ElevenLabsDestinationListener()
        
        print("Listening for speech (5 seconds)...")
        config = DestinationCaptureConfig(timeout_s=5.0)
        
        # This will record and transcribe
        result = listener.capture(config)
        
        print(f"Transcription result: '{result}'")
        print("STT test completed successfully!")
        return True
        
    except Exception as e:
        print(f"STT test failed: {e}")
        return False

def main():
    """Run integration tests."""
    print("=" * 60)
    print("ELEVENLABS INTEGRATION TEST")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("WARNING: No ELEVENLABS_API_KEY environment variable found.")
        print("Tests may fail without a valid API key.")
        print("Set it with: export ELEVENLABS_API_KEY='your_key_here'")
        print()
    
    # Test TTS
    tts_success = test_tts()
    print()
    
    # Test STT (optional, requires microphone)
    if "--skip-stt" not in sys.argv:
        stt_success = test_stt()
    else:
        print("Skipping STT test (use --skip-stt to skip microphone test)")
        stt_success = True
    
    print()
    print("=" * 60)
    if tts_success and stt_success:
        print("✅ ALL TESTS PASSED - ElevenLabs integration is working!")
    else:
        print("❌ SOME TESTS FAILED - Check the error messages above")
    print("=" * 60)

if __name__ == "__main__":
    main()
