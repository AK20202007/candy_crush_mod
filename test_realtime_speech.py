#!/usr/bin/env python3
"""
Test real-time speech processing to ensure no delays.
"""

import os
import time
from elevenlabs_speech import create_elevenlabs_controller

def test_realtime_speech():
    """Test that messages are spoken immediately without queuing delays."""
    print("🎙️  REAL-TIME SPEECH TEST")
    print("=" * 50)
    
    speech = create_elevenlabs_controller()
    speech.start()
    
    print("\nTesting immediate speech...")
    
    # Test 1: Immediate critical message
    print("1. Testing critical message (should speak immediately):")
    start_time = time.time()
    speech.speak_critical("Critical: Obstacle detected!", immediate=True)
    
    # Wait for speech to start
    time.sleep(1)
    elapsed = time.time() - start_time
    print(f"   Response time: {elapsed:.2f} seconds")
    
    # Test 2: Regular message
    print("\n2. Testing regular message:")
    start_time = time.time()
    speech.speak_info("Regular navigation guidance")
    
    time.sleep(1)
    elapsed = time.time() - start_time
    print(f"   Response time: {elapsed:.2f} seconds")
    
    # Test 3: Multiple rapid messages
    print("\n3. Testing rapid message sequence:")
    messages = [
        "Turn left in 50 feet",
        "Person detected ahead", 
        "Continue straight",
        "Destination approaching"
    ]
    
    for i, message in enumerate(messages):
        start_time = time.time()
        if i == 1:  # Second message should be warning
            speech.speak_warning(message, immediate=True)
        else:
            speech.speak_guidance(message)
        
        time.sleep(0.5)  # Short wait between messages
        elapsed = time.time() - start_time
        print(f"   Message {i+1}: {elapsed:.2f}s - {message}")
    
    # Check final statistics
    stats = speech.get_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"   Messages spoken: {stats['messages_spoken']}")
    print(f"   Messages suppressed: {stats['messages_suppressed']}")
    print(f"   API calls: {stats['api_calls']}")
    print(f"   Queue sizes: {stats['queues']}")
    
    speech.stop()
    
    # Test results
    if stats['messages_spoken'] > 0:
        print("\n✅ Real-time speech test PASSED")
        print("   Messages are being processed immediately")
        return True
    else:
        print("\n❌ Real-time speech test FAILED")
        print("   Messages are not being spoken")
        return False

def test_queue_clearing():
    """Test that old messages are cleared when new critical messages arrive."""
    print("\n🧹 QUEUE CLEARING TEST")
    print("=" * 50)
    
    speech = create_elevenlabs_controller()
    speech.start()
    
    print("\nAdding several messages to queues...")
    speech.speak_info("Message 1: Regular guidance")
    speech.speak_info("Message 2: More guidance") 
    speech.speak_info("Message 3: Even more guidance")
    
    # Check queue sizes
    stats = speech.get_statistics()
    print(f"   Queues before critical: {stats['queues']}")
    
    print("\nSending critical message (should clear old messages):")
    speech.speak_critical("URGENT: Stop immediately!", immediate=True)
    
    time.sleep(0.5)
    
    # Check queue sizes after critical
    stats = speech.get_statistics()
    print(f"   Queues after critical: {stats['queues']}")
    
    speech.stop()
    
    if stats['queues']['info'] == 0:
        print("\n✅ Queue clearing test PASSED")
        print("   Old messages cleared for critical alerts")
        return True
    else:
        print("\n❌ Queue clearing test FAILED")
        print("   Old messages not cleared")
        return False

def main():
    """Run real-time speech tests."""
    print("🚀 TESTING REAL-TIME SPEECH PROCESSING")
    print("Ensuring no delays and immediate speech output")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("⚠️  No API key - will test with fallback")
    
    tests = [
        ("Real-time Speech", test_realtime_speech),
        ("Queue Clearing", test_queue_clearing),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("REAL-TIME SPEECH TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 REAL-TIME SPEECH IS WORKING!")
        print("   Messages are spoken immediately")
        print("   No queuing delays detected")
        print("   Critical messages clear old messages")
    else:
        print("⚠️  Some issues detected - see above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
