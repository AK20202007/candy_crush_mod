# Assistive Navigation System - User Interface Guide

## Overview

This production-ready assistive navigation system provides intelligent, context-aware guidance for blind and low-vision users. The interface has been redesigned to eliminate repetitive speech, provide clear actionable feedback, and adapt to different user needs.

## Key Improvements

### 1. Smart Speech Timing
- **No more spam**: Messages are intelligently filtered to prevent repetitive warnings
- **Adaptive intervals**: Different timing for critical alerts (1.5s) vs guidance (8s)
- **Context awareness**: Adjusts frequency based on user activity and environment
- **Repeat limits**: Maximum 3-5 repeats before automatic cooldown

### 2. Context-Aware Decision Making
The system automatically detects and adapts to:
- **User mode**: Exploration, Navigation, Searching, Stationary, Orientation
- **Movement state**: Walking, stopped, speed detection
- **Environment**: Indoor/outdoor, sidewalk vs crosswalk
- **Approaching destination**: Special guidance when near target

### 3. User-Friendly Messages
- **Clear directions**: "ahead" instead of "at 12 o'clock"
- **Distance descriptions**: "very close", "close by", "nearby"
- **Context prefixes**: "In crosswalk", "You're getting close"
- **No debug clutter**: Clean output without technical details

### 4. User Preferences
Configurable settings include:
- **Verbosity**: Minimal, Normal, Verbose
- **Warning sensitivity**: Low, Normal, High
- **Speech rate**: 0.5x to 1.5x speed
- **Navigation interval**: Time between guidance prompts

## How It Works

### When to Say What

#### Critical Alerts (Speak Immediately)
- Person in path within arm's reach
- Vehicle approaching
- Immediate obstacle (stairs, drop-off)
- **Timing**: No delay, repeat every 1.5s if still present

#### Warnings (Speak with Context)
- Obstacles further away
- Curb or sidewalk edge
- Potential hazards
- **Timing**: 0.5s delay, repeat every 3s, max 3 repeats

#### Navigation Guidance (Speak When Needed)
- Turn instructions
- Distance updates
- Route confirmations
- **Timing**: No delay, repeat every 8s, max 2 repeats

#### General Info (Speak Sparingly)
- Mode changes
- Status updates
- Environment descriptions
- **Timing**: 15s+ intervals, max 1 repeat

### Decision Prioritization

1. **Critical safety alerts** always take priority (interrupt everything)
2. **Warnings** come next, but won't interrupt critical alerts
3. **Navigation guidance** waits for safety messages to complete
4. **Info messages** only when nothing else is pending

### Smart Filtering Examples

**Scenario 1: Person Detection**
```
Frame 1: "Stop. Person ahead, very close." ✓ (speaks)
Frame 2: "Stop. Person ahead, very close." ✗ (suppressed, <1.5s)
Frame 3: "Stop. Person ahead, very close." ✓ (speaks, 1.5s+ passed)
Frame 6: "Stop. Person ahead, very close." ✓ (speaks)
Frame 10: "Stop. Person ahead, very close." ✗ (max repeats reached, cooldown)
```

**Scenario 2: Mixed Alerts**
```
Pending: "Walk straight" (guidance)
New alert: "Stop. Person ahead" (critical)
Result: Clears guidance, speaks critical alert immediately
```

**Scenario 3: Approaching Destination**
```
Regular: "Walk straight"
Near destination: "You're getting close. Walk straight"
Very near: "You're almost there. Walk straight"
```

## Usage

### Basic Commands

```bash
# Interactive mode (default)
python main_improved.py

# Direct navigation
python main_improved.py --destination "Library"

# Text input instead of voice
python main_improved.py --typed-destination

# Minimal feedback (fewer announcements)
python main_improved.py --verbosity minimal

# More frequent warnings
python main_improved.py --sensitivity high

# Faster speech
python main_improved.py --speech-rate 1.2
```

### User Profiles

**Beginner User**
```bash
python main_improved.py --verbosity verbose --sensitivity high
```
- More detailed guidance
- Frequent warnings
- Shorter intervals between prompts

**Experienced User**
```bash
python main_improved.py --verbosity normal --sensitivity normal
```
- Balanced guidance
- Standard warning frequency
- Moderate intervals

**Expert User**
```bash
python main_improved.py --verbosity minimal --sensitivity low
```
- Essential info only
- Only critical warnings
- Longer intervals between prompts

### Configuration File

Create `~/.assistive_nav/preferences.json`:

```json
{
  "verbosity": "normal",
  "speech_rate": 1.0,
  "warning_sensitivity": "normal",
  "haptic_enabled": true,
  "navigation_prompt_interval_s": 8.0
}
```

Settings are automatically saved on exit and loaded on startup.

## Interface Modes

### Exploration Mode (Default)
- General walking without specific destination
- Detects and warns about obstacles
- Provides orientation cues
- Switches to navigation when destination set

### Navigation Mode
- Active route following
- Turn-by-turn guidance
- Distance updates
- Destination approach warnings

### Searching Mode
- Looking for specific objects (door, chair, etc.)
- Object-finding guidance
- Directional hints

### Stationary Mode
- User is stopped/not moving
- Reduces guidance frequency
- Maintains safety warnings
- Periodic orientation updates

### Orientation Mode
- Getting bearings
- Environmental description
- Spatial awareness cues

## Message Examples

### Before (Old System)
```
[vision] ~4.4 FPS over last 3.2s
[nav->agent] Walk straight
[agent->speech] action=warn priority=100 haptic=stop: Stop. Person ahead, at 12 o'clock, within arm's reach.
[debug] All candidates:
  1. warn (priority=100) by ['safety']: Stop: person at 12 o'clock, within arm's reach.
  2. guide (priority=60) by ['wayfinding']: Walk straight.
  3. silent (priority=0) by []: 
[agent->speech] action=warn priority=100 haptic=stop: Stop. Person ahead, at 12 o'clock, within arm's reach.
```

### After (New System)
```
Stop. Person ahead, very close.
Walk straight.
```

## Technical Architecture

### Components

1. **UserInterface** (`user_interface.py`)
   - State management
   - Smart timing rules
   - Message formatting
   - Preference handling

2. **IntelligentSpeechController** (`speech_controller.py`)
   - Priority queues (critical, urgent, normal, info)
   - Anti-spam filtering
   - User preference adaptation
   - Clean TTS interface

3. **NavigationInterface** (`navigation_interface.py`)
   - Decision processing
   - Context awareness
   - Mode management
   - Status tracking

4. **NavigationApp** (`main_improved.py`)
   - Main application logic
   - User input handling
   - Configuration management
   - Graceful shutdown

### Data Flow

```
Vision System → Agentic Router → Navigation Interface → Speech Controller → User
                    ↓
            User Interface (state, timing, preferences)
```

## Tips for Users

1. **Start with normal verbosity** - Adjust based on comfort
2. **Use high sensitivity in unfamiliar areas** - More warnings for safety
3. **Use minimal verbosity when familiar** - Less interruption
4. **Set appropriate speech rate** - Not too fast to understand
5. **Clear destination when done** - Returns to exploration mode

## Troubleshooting

### Too Many Warnings
- Reduce sensitivity: `--sensitivity low`
- Reduce verbosity: `--verbosity minimal`
- Increase intervals: Modify `navigation_prompt_interval_s`

### Missing Important Alerts
- Increase sensitivity: `--sensitivity high`
- Increase verbosity: `--verbosity verbose`
- Check camera alignment and lighting

### Speech Too Fast/Slow
- Adjust rate: `--speech-rate 0.8` (slower) or `--speech-rate 1.2` (faster)

### System Not Responding
- Check camera permissions
- Verify microphone access (for voice input)
- Restart with `--typed-destination` to bypass voice

## Future Enhancements

- Gesture/voice acknowledgment system
- Machine learning for personalized timing
- Multi-language support
- Integration with smart glasses/haptic devices
- Cloud-based route optimization
- User behavior learning

## Support

For issues or questions:
1. Check the configuration file
2. Review this guide
3. Test with different settings
4. Report bugs with specific scenarios

---

**Version**: 2.0 (Production Ready)
**Last Updated**: 2024
