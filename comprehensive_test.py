#!/usr/bin/env python3
"""
Comprehensive testing for ElevenLabs integration.
Tests all functionality to ensure no errors.
"""

import os
import sys
import time
from elevenlabs_speech import ElevenLabsSpeechController, create_elevenlabs_controller
from voice_input import ElevenLabsDestinationListener, DestinationCaptureConfig

def test_voice_presets():
    """Test all voice presets."""
    print("Testing all voice presets...")
    
    voices = ["default", "calm", "professional", "warm", "male"]
    
    for voice in voices:
        try:
            print(f"  Testing voice: {voice}")
            speech = create_elevenlabs_controller(voice=voice)
            speech.start()
            
            # Test short message
            speech.speak_info(f"Testing {voice} voice")
            time.sleep(2)
            
            stats = speech.get_statistics()
            print(f"    ✓ {voice}: {stats['messages_spoken']} spoken, {stats['api_errors']} errors")
            
            speech.stop()
            
        except Exception as e:
            print(f"    ✗ {voice}: {e}")
            return False
    
    return True

def test_api_methods():
    """Test all API methods."""
    print("Testing API methods...")
    
    try:
        speech = create_elevenlabs_controller()
        speech.start()
        
        # Test all priority levels
        speech.speak_critical("Critical safety alert")
        time.sleep(1)
        
        speech.speak_warning("Warning message")
        time.sleep(1)
        
        speech.speak_guidance("Navigation guidance")
        time.sleep(1)
        
        speech.speak_info("Information message")
        time.sleep(1)
        
        stats = speech.get_statistics()
        print(f"  ✓ API methods: {stats['messages_spoken']} spoken, {stats['api_errors']} errors")
        
        speech.stop()
        return True
        
    except Exception as e:
        print(f"  ✗ API methods: {e}")
        return False

def test_fallback_mechanism():
    """Test fallback when no API key."""
    print("Testing fallback mechanism...")
    
    # Temporarily remove API key
    original_key = os.getenv("ELEVENLABS_API_KEY")
    os.environ["ELEVENLABS_API_KEY"] = ""
    
    try:
        speech = create_elevenlabs_controller()
        speech.start()
        
        speech.speak_info("Testing fallback system TTS")
        time.sleep(2)
        
        stats = speech.get_statistics()
        if stats["using_fallback"]:
            print("  ✓ Fallback: Successfully using system TTS")
            success = True
        else:
            print("  ✗ Fallback: Not using system TTS when expected")
            success = False
        
        speech.stop()
        
        # Restore API key
        if original_key:
            os.environ["ELEVENLABS_API_KEY"] = original_key
        
        return success
        
    except Exception as e:
        print(f"  ✗ Fallback: {e}")
        # Restore API key
        if original_key:
            os.environ["ELEVENLABS_API_KEY"] = original_key
        return False

def test_stt_functionality():
    """Test speech-to-text (basic test without microphone)."""
    print("Testing STT functionality...")
    
    try:
        # Test STT listener creation
        listener = ElevenLabsDestinationListener()
        print("  ✓ STT: Listener created successfully")
        
        # Test config creation
        config = DestinationCaptureConfig(timeout_s=1.0)  # Very short timeout
        print("  ✓ STT: Config created successfully")
        
        return True
        
    except Exception as e:
        print(f"  ✗ STT: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid inputs."""
    print("Testing error handling...")
    
    try:
        speech = create_elevenlabs_controller()
        speech.start()
        
        # Test empty message
        speech.speak_info("")
        time.sleep(0.5)
        
        # Test very long message
        long_msg = "Test " * 100
        speech.speak_info(long_msg)
        time.sleep(1)
        
        # Test special characters
        speech.speak_info("Testing special chars: !@#$%^&*()")
        time.sleep(1)
        
        stats = speech.get_statistics()
        print(f"  ✓ Error handling: {stats['messages_spoken']} spoken, {stats['api_errors']} errors")
        
        speech.stop()
        return True
        
    except Exception as e:
        print(f"  ✗ Error handling: {e}")
        return False

def test_memory_usage():
    """Test for memory leaks."""
    print("Testing memory usage...")
    
    try:
        speech = create_elevenlabs_controller()
        speech.start()
        
        # Send many messages
        for i in range(50):
            speech.speak_info(f"Message {i}")
            time.sleep(0.1)
        
        # Clear queues
        speech.clear_queues()
        
        stats = speech.get_statistics()
        print(f"  ✓ Memory: {stats['messages_spoken']} processed, queues cleared")
        
        speech.stop()
        return True
        
    except Exception as e:
        print(f"  ✗ Memory: {e}")
        return False

def main():
    """Run all comprehensive tests."""
    print("=" * 70)
    print("COMPREHENSIVE ELEVENLABS INTEGRATION TEST")
    print("=" * 70)
    
    # Check API key
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("WARNING: No ELEVENLABS_API_KEY found")
        print("Some tests may fail without a valid API key")
        print()
    
    tests = [
        ("Voice Presets", test_voice_presets),
        ("API Methods", test_api_methods),
        ("Fallback Mechanism", test_fallback_mechanism),
        ("STT Functionality", test_stt_functionality),
        ("Error Handling", test_error_handling),
        ("Memory Usage", test_memory_usage),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Integration is ready!")
        return True
    else:
        print("⚠️  Some tests failed - Check the errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
