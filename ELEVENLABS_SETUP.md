# ElevenLabs Speech Integration Setup

This guide helps you set up the ElevenLabs speech integration for high-quality text-to-speech (TTS) and speech-to-text (STT) in the assistive navigation system.

## 🎯 What You Get

### Text-to-Speech (TTS)
- **Natural, human-like voices** - No robotic speech
- **Multiple voice options** - Choose the best voice for your needs
- **Fast response** - Low latency for real-time navigation
- **High quality** - Professional-grade speech synthesis

### Speech-to-Text (STT)
- **Accurate voice commands** - Speak your destination naturally
- **Hands-free operation** - No typing required
- **Fast recognition** - Real-time transcription

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Microphone** for voice input (optional)
3. **Speakers/headphones** for audio output
4. **ElevenLabs account** - Free tier available

## 🚀 Setup Instructions

### Step 1: Get ElevenLabs API Key

1. Go to [https://elevenlabs.io](https://elevenlabs.io)
2. Sign up for a free account (or log in)
3. Navigate to your Profile → API Keys
4. Copy your API key

### Step 2: Install Dependencies

```bash
cd /Users/krishayg/CascadeProjects/candy-crush-mod/candy_crush_mod
pip install -r requirements.txt
```

This installs:
- `elevenlabs` - ElevenLabs Python SDK
- `sounddevice` - Audio playback/recording
- `soundfile` - Audio file handling

### Step 3: Set API Key

**Option A: Environment Variable (Recommended)**
```bash
# macOS/Linux
export ELEVENLABS_API_KEY="your_api_key_here"

# Windows (Command Prompt)
set ELEVENLABS_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:ELEVENLABS_API_KEY="your_api_key_here"
```

**Option B: Command Line Argument**
```bash
python main_elevenlabs.py --api-key "your_api_key_here"
```

**Option C: Configuration File**
Create `~/.assistive_nav/preferences.json`:
```json
{
  "elevenlabs_api_key": "your_api_key_here"
}
```

### Step 4: Run the System

```bash
python main_elevenlabs.py
```

## 🎙️ Voice Options

### Voice Presets

Choose a voice that matches your preference:

```bash
# Default - Natural, balanced tone
python main_elevenlabs.py --voice default

# Calm - Relaxed, soothing voice
python main_elevenlabs.py --voice calm

# Professional - Clear, authoritative tone
python main_elevenlabs.py --voice professional

# Warm - Friendly, approachable voice
python main_elevenlabs.py --voice warm
```

### Custom Voice ID

Use a specific ElevenLabs voice:

```bash
python main_elevenlabs.py --voice-id "XB0fDUnXU5powFXDhCwa"
```

Find more voices at: https://elevenlabs.io/voice-library

## 🗣️ Using Voice Input

### Speak Your Destination

```bash
# Enable voice input
python main_elevenlabs.py --use-voice
```

The system will:
1. Say: "Listening for destination. Please speak clearly."
2. Listen for 10 seconds
3. Transcribe your speech
4. Confirm: "Starting navigation to [destination]"

### Voice Input Tips

- **Speak clearly** - Natural pace, not too fast
- **Quiet environment** - Reduces background noise
- **Short phrases** - "Library" vs "I want to go to the library please"
- **Wait for prompt** - Don't speak until you hear "Listening..."

## 🔧 Troubleshooting

### "No ELEVENLABS_API_KEY found"

**Solution:** Set the API key
```bash
export ELEVENLABS_API_KEY="your_key"
python main_elevenlabs.py
```

### "ElevenLabs package not installed"

**Solution:** Install dependencies
```bash
pip install elevenlabs sounddevice soundfile
```

### "Sounddevice/soundfile not available"

**Solution:** Install audio libraries
```bash
# macOS
brew install portaudio

# Ubuntu/Debian
sudo apt-get install libportaudio2 libsndfile1

# Then reinstall
pip install sounddevice soundfile --force-reinstall
```

### No Audio Output

**Check speakers/headphones**
```bash
# Test audio
python -c "import sounddevice as sd; print(sd.query_devices())"
```

**Set default output device**
```python
import sounddevice as sd
sd.default.device = 1  # Try different numbers
```

### Microphone Not Working

**Check microphone access**
```bash
# macOS - grant permission in System Preferences → Security & Privacy
# Linux - check alsamixer
alsamixer
```

**Test recording**
```bash
python -c "
import sounddevice as sd
import numpy as np
print('Recording 3 seconds...')
audio = sd.rec(int(3 * 16000), 16000, 1, dtype=np.int16)
sd.wait()
print('Done!')
"
```

### High Latency (Slow Response)

**Use faster model**
```python
# In elevenlabs_speech.py, change model_id:
self.model_id = "eleven_turbo_v2_5"  # Faster but slightly lower quality
```

**Check internet connection** - ElevenLabs requires API calls

### API Rate Limits

**Free tier limits:**
- 10,000 characters/month for TTS
- Limited STT minutes

**Upgrade:** https://elevenlabs.io/pricing

### Fallback to System TTS

If ElevenLabs fails, the system automatically uses pyttsx3 (system TTS).

To force fallback:
```bash
# Don't set API key
unset ELEVENLABS_API_KEY
python main_elevenlabs.py
```

## 💡 Usage Examples

### Basic Usage
```bash
python main_elevenlabs.py
```

### With Specific Voice
```bash
python main_elevenlabs.py --voice calm
```

### Voice Input + Calm Voice
```bash
python main_elevenlabs.py --use-voice --voice calm
```

### Direct Destination
```bash
python main_elevenlabs.py -d "Library"
```

### Text Input Only (No Voice)
```bash
python main_elevenlabs.py --typed-destination
```

### Custom API Key
```bash
python main_elevenlabs.py --api-key "your_key"
```

## 🎛️ Advanced Configuration

### Environment Variables

```bash
# Required
export ELEVENLABS_API_KEY="your_key"

# Optional
export ELEVENLABS_VOICE="calm"
export ELEVENLABS_MODEL="eleven_flash_v2_5"
```

### Voice Settings

Edit `elevenlabs_speech.py`:

```python
# Voice settings for TTS
voice_settings = VoiceSettings(
    stability=0.5,      # 0.0-1.0 (lower = more emotive)
    similarity_boost=0.75,  # 0.0-1.0 (clarity vs naturalness)
    style=0.5,          # Speaking style (if supported)
    use_speaker_boost=True
)
```

### STT Language

```python
# In elevenlabs_speech.py, recognize_speech method
# Change language code
result = self.speech.recognize_speech(
    timeout=10.0,
    language="es"  # Spanish
)
```

## 📊 Monitoring Usage

The system tracks API usage:

```
[stats] Runtime: 60.5 seconds
[stats] Messages spoken: 25
[stats] ElevenLabs API calls: 25
[stats] Using ElevenLabs: True
```

## 🆘 Getting Help

### ElevenLabs Support
- Documentation: https://docs.elevenlabs.io
- API Reference: https://api.elevenlabs.io/docs
- Discord: https://discord.gg/elevenlabs

### System Issues
1. Check this guide first
2. Verify API key is set: `echo $ELEVENLABS_API_KEY`
3. Test API connectivity: `curl -H "xi-api-key: YOUR_KEY" https://api.elevenlabs.io/v1/voices`
4. Check logs for specific errors

## ✅ Quick Checklist

Before running:
- [ ] API key obtained from elevenlabs.io
- [ ] API key set in environment or config
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Microphone working (if using voice input)
- [ ] Speakers/headphones connected
- [ ] Internet connection active

## 🎉 Ready to Use!

```bash
export ELEVENLABS_API_KEY="your_key"
python main_elevenlabs.py
```

Enjoy natural, high-quality speech for your assistive navigation! 🗣️
