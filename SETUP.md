# Gesture Stimulus App - Setup & Migration Guide

## Overview

Gesture recording application (v4) for multi-device simultaneous EXG data collection.
- Supports up to 4 ESP32 devices via USB
- EXG-only (ECG + EMG, no IMU/PPG)
- 10 gesture types, 5 reps each, 6 sub-actions per gesture

## Project Structure

```
gesture-stimulus-app/
├── stimulus_v4.py        # Main application
├── gesture_images/       # Gesture reference images (PNG) and demo videos (MP4)
│   ├── gesture_0.png ~ gesture_10.png
│   ├── g1.mp4 ~ g10.mp4
│   └── SS.jpeg
├── requirements.txt      # Python dependencies
├── .gitignore
└── SETUP.md              # This file
```

## Prerequisites

- **Python**: 3.12+ (tested on 3.13.9)
- **OS**: Windows / Linux / macOS (cross-platform audio support)
- **Hardware**: ESP32 devices with EXG firmware (optional - has simulation mode)

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Curtchen/gesture_stimulus_app.git
cd gesture-stimulus-app

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Linux only (requires root/sudo, one-time setup)
sudo apt install python3-tk
sudo usermod -aG dialout $USER
# Re-login after this step for serial port access to take effect

# 5. Run
python stimulus_v4.py
```

## Dependencies

| Package         | Version  | Purpose                          |
|-----------------|----------|----------------------------------|
| numpy           | >=2.0    | Numerical operations             |
| matplotlib      | >=3.10   | Real-time waveform display       |
| pillow          | >=12.0   | Gesture image rendering          |
| pyserial        | >=3.5    | ESP32 serial communication       |
| neurokit2       | >=0.2.12 | ECG signal processing            |
| opencv-python   | >=4.13   | MP4 gesture video playback       |

All except `neurokit2` are optional - the app will run in degraded mode without them.

Audio beep uses `winsound` on Windows, `aplay` on Linux, `afplay` on macOS (all built-in, no extra packages needed).

## Key Configuration

Edit the `CONFIG` dict at the top of `stimulus_v4.py`:

| Parameter              | Default          | Description                       |
|------------------------|------------------|-----------------------------------|
| `on_duration`          | 3s               | Action display time               |
| `off_duration`         | 3s               | Rest between actions              |
| `actions_per_gesture`  | 6                | Sub-actions per round             |
| `num_repetitions`      | 5                | Repetitions per gesture           |
| `baud_rate`            | 921600           | ESP32 serial baud rate            |
| `gesture_images_dir`   | `./gesture_images` | Path to gesture images          |
| `output_dir`           | `./recordings`   | Data output directory             |
| `fullscreen`           | False            | Fullscreen mode                   |
| `window_size`          | 1400x850         | Window dimensions                 |

## Output

Recordings are saved to `./recordings/` with timestamped folders per session.

## Platform Notes

### Linux
- Install tkinter: `sudo apt install python3-tk`
- Serial port permission: `sudo usermod -aG dialout $USER` (re-login required)
- Audio uses `aplay` (pre-installed on most distros)

### macOS
- Install tkinter: `brew install python-tk`
- Audio uses `afplay` (pre-installed)

### Windows
- No extra setup needed

## Migration Checklist

When moving to a new machine:

1. Clone the repo
2. Set up Python 3.12+ and virtual environment
3. `pip install -r requirements.txt`
4. **Linux only**: `sudo apt install python3-tk` and add user to `dialout` group
5. Connect ESP32 devices via USB (auto-detected)
6. Verify `gesture_images/` contains all 11 PNGs and 10 MP4s
7. Run `python stimulus_v4.py` to test

## Controls

| Key     | Action                  |
|---------|-------------------------|
| ENTER   | Start recording         |
| SPACE   | Pause                   |
| W       | Toggle waveform display |
| 1-4     | Switch subject view     |
| ESC     | Exit                    |
