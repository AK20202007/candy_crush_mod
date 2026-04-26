# Assistive Navigation System 🧭

**A smart navigation assistant for blind and low-vision users.**

This system uses your device's camera and AI to detect obstacles, provide safety warnings, and guide you to your destination - all through clear, spoken feedback.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the System

```bash
python main.py
```

### Phone / laptop thin client (stream frames to this machine)

Runs YOLO + router here; a browser on your phone sends JPEGs and speaks responses locally.

```bash
python3 -m pip install -r requirements.txt
python3 stream_server.py
```

Open `http://localhost:8765` (or `http://<this-computer-LAN-IP>:8765`). Mobile cameras usually need **HTTPS** on non-localhost; use e.g. `ngrok http 8765` and open the https URL on the phone.

**Expo Go app (same API):** `cd mobile-stream && npm install && npm run start` (LAN Metro) or **`npm run start:tunnel`** if the QR host is unreachable. **iPhone + Expo Go:** plain `http://192.168…` to `stream_server` often **times out**. On the Mac, with `stream_server.py` already listening on **8765**, run **`brew install cloudflared`** once, then **`./scripts/tunnel_stream_server.sh`** (or `cloudflared tunnel --url http://127.0.0.1:8765`). Paste the printed **`https://….trycloudflare.com`** origin into the app (no path) and tap **Ping server**. (ngrok works too if you prefer it.)

### 3. Enter Your Destination

When prompted, type where you want to go:
```
Enter destination: Library
```

That's it! The system will start guiding you.

---

## 🎯 What It Does

### Smart Safety Warnings
- **"Stop. Person ahead, very close."** - Alerts when someone is in your path
- **"Curb ahead."** - Warns about sidewalk edges
- **"Vehicle approaching."** - Detects cars, bikes, buses

### Navigation Guidance
- **"Walk straight."** - Clear directional guidance
- **"Turn left in 20 feet."** - Turn-by-turn directions
- **"You are approaching your destination."** - Arrival notification

### Smart Features
- ✅ **No repetitive warnings** - Says it once, not constantly
- ✅ **Prioritizes safety** - Critical alerts interrupt navigation
- ✅ **Context aware** - Adapts to walking, stopped, indoor, outdoor
- ✅ **Customizable** - Adjust speech speed, verbosity, sensitivity

---

## 📱 Usage Examples

### Basic Navigation
```bash
# Interactive mode - type your destination
python main.py

# Direct navigation - specify destination
python main.py -d "Student Center"

# Text input (no voice)
python main.py --typed-destination
```

### User Experience Levels

**New User** (more guidance and warnings):
```bash
python main.py --verbosity verbose --sensitivity high
```

**Experienced User** (balanced):
```bash
python main.py  # default settings
```

**Advanced User** (minimal interruptions):
```bash
python main.py --verbosity minimal --sensitivity low
```

### Adjust Speech Speed
```bash
# Slower speech
python main.py --speech-rate 0.8

# Faster speech
python main.py --speech-rate 1.2
```

---

## 🎮 How to Use

### Starting Navigation
1. Run `python main.py`
2. Type your destination (e.g., "Library", "Cafe", "Royce Hall")
3. Press Enter
4. Camera opens and navigation begins

### During Navigation
- **Walk normally** - system provides guidance automatically
- **Listen for warnings** - safety alerts take priority
- **Press 'q'** in camera window to stop navigation
- **Press Ctrl+C** in terminal to stop/exit

### Stopping
- **Press 'q'** - Stop camera, stay in system (enter new destination)
- **Press Ctrl+C** - Stop current navigation
- **Press Ctrl+C again** - Exit completely
- **Type 'quit'** - Exit at any prompt

---

## ⚙️ Configuration

### User Preferences File

Create `~/.assistive_nav/preferences.json`:

```json
{
  "verbosity": "normal",
  "speech_rate": 1.0,
  "warning_sensitivity": "normal",
  "navigation_prompt_interval_s": 8.0,
  "haptic_enabled": true
}
```

**Options:**
- `verbosity`: `minimal`, `normal`, `verbose`
- `speech_rate`: `0.5` to `1.5` (0.8 = slower, 1.2 = faster)
- `warning_sensitivity`: `low`, `normal`, `high`
- `navigation_prompt_interval_s`: `5` to `15` seconds

---

## 🔧 Command Line Options

```bash
python main.py [OPTIONS]

Options:
  -d, --destination TEXT        Direct destination (skip prompt)
  -t, --typed-destination       Use text input (not voice)
  -v, --verbosity [minimal|normal|verbose]
                              Feedback level (default: normal)
  -s, --sensitivity [low|normal|high]
                              Warning frequency (default: normal)
  -r, --speech-rate FLOAT       Speech speed 0.5-1.5 (default: 1.0)
  -c, --camera INT             Camera index (default: 0)
  --help                       Show all options
```

---

## 📋 Requirements

- Python 3.8+
- Webcam (built-in or external)
- Microphone (optional, for voice input)
- Dependencies in `requirements.txt`:
  - OpenCV (camera)
  - Ultralytics/YOLO (object detection)
  - pyttsx3 (text-to-speech)
  - Other ML libraries

---

## ⚠️ Important Safety Notes

**This system is an ASSISTIVE TOOL, not a replacement for:**
- White cane
- Guide dog
- Human assistance
- Your own judgment

**Always:**
- Use standard mobility techniques
- Cross streets with extreme caution
- Listen to environmental sounds
- Be aware of system limitations

**Limitations:**
- Cannot detect all obstacles
- May have false positives/negatives
- Works best in good lighting
- Requires clear camera view

---

## 📚 Documentation

- `USER_INTERFACE_GUIDE.md` - Complete user guide
- `QUICK_START.md` - Quick reference
- `IMPROVEMENTS_SUMMARY.md` - Technical details

---

## 🤝 How It Works

1. **Camera captures video** - 6-10 frames per second
2. **AI detects objects** - People, cars, obstacles, signs
3. **System decides what to say** - Priority: Safety > Navigation > Info
4. **Speaks clearly** - No technical jargon, simple directions
5. **Adapts to context** - Different behavior when walking vs. stopped

---

## 🆘 Troubleshooting

### Too Many Warnings
```bash
python main.py --sensitivity low --verbosity minimal
```

### Missing Important Alerts
```bash
python main.py --sensitivity high
```

### Camera Not Working
```bash
# Find available cameras
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"

# Use specific camera
python main.py --camera 1
```

### Speech Too Fast/Slow
```bash
python main.py --speech-rate 0.8  # slower
python main.py --speech-rate 1.2  # faster
```

---

## 📦 Project Structure

```
.
├── main.py                    # Main application (start here)
├── user_interface.py          # Smart speech timing
├── speech_controller.py       # Priority-based speech
├── navigation_interface.py    # Context-aware guidance
├── vision.py                  # Camera & object detection
├── agentic_layer/             # AI decision-making
├── config/                    # Profiles & settings
├── docs/                      # Additional documentation
├── default_preferences.json   # Default settings
└── requirements.txt           # Python dependencies
```

---

## 💡 Tips for Best Experience

1. **Start with normal settings** - adjust based on comfort
2. **Test in familiar areas first** - get used to the system
3. **Use high sensitivity in new places** - more safety warnings
4. **Use minimal verbosity when comfortable** - fewer interruptions
5. **Keep camera lens clean** - better detection accuracy
6. **Good lighting helps** - system works best in daylight

---

## 🎓 Learning the System

### Week 1: Getting Started
- Use `verbose` mode to hear everything
- Walk familiar routes
- Learn the different warning sounds

### Week 2: Comfort Building  
- Switch to `normal` mode
- Try new destinations
- Adjust speech rate if needed

### Week 3+: Expert Usage
- Use `minimal` mode for routine trips
- High sensitivity only in busy areas
- Customize preferences file

---

## 🔬 Technical Details

- **YOLOv8** for object detection (person, car, chair, etc.)
- **Agentic routing** for intelligent decision-making
- **Smart timing** prevents message spam
- **Context awareness** adapts to user state
- **Configurable profiles** for different scenarios

See `IMPROVEMENTS_SUMMARY.md` for technical architecture.

---

## 📞 Getting Help

1. Check this README first
2. Read `QUICK_START.md` for quick reference
3. See `USER_INTERFACE_GUIDE.md` for complete guide
4. Run `python main.py --help` for command options

---

## ⭐ Key Features Summary

✅ **Smart speech** - No annoying repetition  
✅ **Safety first** - Critical alerts interrupt everything  
✅ **Context aware** - Adapts to your situation  
✅ **User friendly** - Clear, simple language  
✅ **Customizable** - Adjust to your needs  
✅ **Reliable** - Handles errors gracefully  

---

## 🎉 Ready to Start?

```bash
python main.py
```

**Welcome to smarter navigation!** 🚀

---

*Version 2.0 - Production Ready*
