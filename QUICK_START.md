# Quick Start Guide - Assistive Navigation System

## Run the Improved Interface

### Basic Usage

```bash
# Navigate to the project directory
cd /Users/krishayg/CascadeProjects/candy-crush-mod/candy_crush_mod

# Run the improved interface
python main_improved.py

# Or with specific options
python main_improved.py --destination "Library" --typed-destination
```

## Available Commands

### Setting Destination

**Interactive Mode** (with voice input):
```bash
python main_improved.py
```
System will ask: "Where would you like to go?" and listen for your voice input.

**Text Input Mode**:
```bash
python main_improved.py --typed-destination
```
System will prompt: "Enter destination:"

**Direct Destination**:
```bash
python main_improved.py -d "Library"
```

### User Experience Levels

**Beginner** (maximum guidance):
```bash
python main_improved.py --verbosity verbose --sensitivity high
```

**Experienced** (balanced):
```bash
python main_improved.py --verbosity normal --sensitivity normal
```

**Expert** (minimal interruptions):
```bash
python main_improved.py --verbosity minimal --sensitivity low
```

### Adjusting Speech

**Faster speech**:
```bash
python main_improved.py --speech-rate 1.2
```

**Slower speech**:
```bash
python main_improved.py --speech-rate 0.8
```

## Configuration

### Create Preferences File

Create file: `~/.assistive_nav/preferences.json`

```json
{
  "verbosity": "normal",
  "speech_rate": 1.0,
  "warning_sensitivity": "normal",
  "navigation_prompt_interval_s": 8.0,
  "haptic_enabled": true,
  "auto_repeat_warnings": true
}
```

Settings are automatically saved when you exit the program.

## System Commands

While running:
- **Ctrl+C** - Stop navigation and return to destination prompt
- **Ctrl+C twice** - Exit program completely
- Say **"stop"** - Pause navigation (if voice commands enabled)

## What to Expect

### Smart Speech Behavior

**Critical Alerts** (Immediate danger):
- Speak immediately, interrupt everything else
- Example: "Stop. Person ahead, very close."
- Repeat every 1.5 seconds if still present (max 5 times)

**Warnings** (Caution needed):
- Speak after 0.5 second delay
- Example: "Curb ahead."
- Repeat every 3 seconds (max 3 times)

**Navigation Guidance**:
- Speak when safe to do so
- Example: "Walk straight."
- Repeat every 8 seconds (max 2 times)

**General Info**:
- Speak sparingly
- Example: "Approaching destination."
- Repeat every 15 seconds (max 1 time)

### Message Examples

**Before (Old System)**:
```
[debug] action=warn priority=100 haptic=stop
[debug] candidates: warn (100), guide (60), silent (0)
Stop. Person at 12 o'clock, within arm's reach.
```

**After (New System)**:
```
Stop. Person ahead, very close.
```

## Troubleshooting

### Too Many Warnings
```bash
python main_improved.py --sensitivity low --verbosity minimal
```

### Missing Important Alerts
```bash
python main_improved.py --sensitivity high
```

### Camera Not Working
```bash
# List available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# Use specific camera
python main_improved.py --camera 1
```

### Speech Too Fast/Slow
```bash
python main_improved.py --speech-rate 0.8  # slower
python main_improved.py --speech-rate 1.2  # faster
```

## Testing the Interface

### Quick Test
```bash
# Test imports
python -c "from user_interface import UserInterface; print('OK')"

# Test with typed input
python main_improved.py --typed-destination -d "Test Location"

# Exit after testing: Press Ctrl+C, then Ctrl+C again
```

### Full System Test
1. Start with: `python main_improved.py --typed-destination`
2. Enter destination: `Test Location`
3. System should start camera and begin detection
4. Move around to trigger different alerts
5. Press Ctrl+C to stop, then Ctrl+C again to exit
6. Check statistics printed at end

## Getting Help

### Command Help
```bash
python main_improved.py --help
```

### Full Documentation
- `USER_INTERFACE_GUIDE.md` - Complete user guide
- `IMPROVEMENTS_SUMMARY.md` - Technical improvements overview

### Status During Operation
The system prints status updates showing:
- Current mode (exploration/navigation/stationary)
- Active detections
- Recent warnings
- Alert statistics

## Tips for Best Experience

1. **Start with normal settings** and adjust based on comfort
2. **Use high sensitivity in unfamiliar areas** for more safety warnings
3. **Use minimal verbosity when familiar** with the route
4. **Test voice input first** with `--typed-destination` if unsure
5. **Check camera angle** - should capture walking direction view
6. **Set realistic expectations** - system assists but doesn't replace cane/guide dog

## Safety Notes

⚠️ **Important**: This system is an **assistive tool**, not a replacement for:
- White cane
- Guide dog
- Human assistance
- Personal judgment

Always:
- Use standard mobility techniques
- Cross streets with caution
- Listen to environmental sounds
- Be aware of system limitations

## Next Steps

1. Test basic functionality with `--typed-destination`
2. Adjust settings to personal preference
3. Practice in safe, familiar environment
4. Gradually use in new environments
5. Provide feedback for further improvements

---

**Ready to start? Run:**
```bash
python main_improved.py --typed-destination
```
