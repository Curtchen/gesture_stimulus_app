"""
Gesture Recording v4 - Multi-Device Simultaneous Collection
============================================================
- Supports up to 4 ESP32 devices via USB (Hub + laptop ports)
- EXG-only (ECG + EMG, no IMU/PPG)
- Auto-scans available COM ports
- Subject switcher: view any subject's ECG/EMG waveforms
- v4 changes: 3 sub-actions per round x 2 rounds = 6 actions/gesture,
  5 reps (was 10), continue dialog after each rep

Controls:
  ENTER = Start | SPACE = Pause | W = Toggle Waveform | ESC = Exit
  1-4   = Switch subject view
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import json
import os
import random
import time
from datetime import datetime
import threading
import struct
import numpy as np
from pathlib import Path
from collections import deque

# Try to import serial
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠️ pyserial not installed. Running in SIMULATION mode.")

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ matplotlib not installed. Waveform display disabled.")

# Try to import winsound
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

# Try to import OpenCV for mp4 video playback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ opencv-python not installed. Video gesture display disabled.")

import neurokit2 as nk

# ==================== CONFIGURATION ====================

CONFIG = {
    # Timing parameters (in seconds)
    "on_duration": 3,
    "off_duration": 3,
    "actions_per_gesture": 6,
    "transition_duration": 3,
    "hold_duration": 10,
    "freeform_duration": 10,

    # Demo
    "demo_gestures": 2,

    # Experiment parameters
    "num_repetitions": 5,

    # Display settings
    "fullscreen": False,
    "window_size": "1400x850",
    "background_color": "#1a1a2e",
    "text_color": "#ffffff",
    "accent_color": "#4ecca3",
    "warning_color": "#ff6b6b",
    "off_color": "#ffd700",
    "steady_color": "#7b68ee",
    "freeform_color": "#00bfff",
    "transition_color": "#e0a0ff",

    # Waveform display
    "waveform_window": 5,
    "sidebar_width": 420,
    "show_waveform": True,

    # Paths
    "gesture_images_dir": "./gesture_images",
    "output_dir": "./recordings",

    # Serial/DAQ settings
    "baud_rate": 921600,
    "enable_daq": True,
}

# Gesture definitions
GESTURES = {
    1: "Hand Close",
    2: "Hand Open",
    3: "Wrist Flexion",
    4: "Wrist Extension",
    5: "Pointing Index",
    6: "V Sign",
    7: "Flexion of Little Finger",
    8: "Tripod Grasp",
    9: "Flexion of Thumb",
    10: "Flexion of Middle Finger",
}

# Frame definitions (EXG only)
FRAME_HEADER = 0xAA
EXG_TYPE = 0x55
EXG_FRAME_SIZE = 14
EXG_FS = 2000

# Subject colors for UI differentiation
SUBJECT_COLORS = ["#ff6b6b", "#4ecdc4", "#ffd93d", "#a29bfe"]


# ==================== PORT SCANNER ====================

def scan_serial_ports():
    """Scan for available serial ports that look like ESP32 devices."""
    if not SERIAL_AVAILABLE:
        return []

    ports = serial.tools.list_ports.comports()
    esp_ports = []

    for p in ports:
        desc = (p.description or "").lower()
        # ESP32-S3 shows up as USB JTAG/serial debug unit or CP210x or CH340
        if any(kw in desc for kw in [
            "usb jtag", "serial", "cp210", "ch340", "ch910",
            "usb-serial", "uart", "esp"
        ]):
            esp_ports.append({
                "port": p.device,
                "description": p.description,
                "hwid": p.hwid,
            })

    # Sort by port name for consistency
    esp_ports.sort(key=lambda x: x["port"])
    return esp_ports


# ==================== DATA COLLECTOR (EXG ONLY) ====================

class DataCollector:
    """Handles real-time EXG acquisition for a single device."""

    def __init__(self, port: str, subject_id: str, baud_rate: int = 921600,
                 window_sec: int = 5, simulate: bool = False):
        self.port = port
        self.subject_id = subject_id
        self.baud_rate = baud_rate
        self.serial = None
        self.window_sec = window_sec
        self.simulate = simulate

        # Data storage
        self.exg_data = []
        self.exg_timestamps = []
        self.exg_seq = []

        # Repetition markers
        self.rep_start_exg = 0

        # Display buffers (4 channels)
        exg_buf_size = window_sec * EXG_FS
        self.exg_display = [deque([2048] * exg_buf_size, maxlen=exg_buf_size) for _ in range(4)]

        # Counters
        self.exg_count = 0

        # State
        self.running = False
        self.session_start_time = 0
        self.collect_thread = None

    def start(self):
        """Start data collection."""
        self.session_start_time = time.time()

        if self.simulate or not SERIAL_AVAILABLE:
            print(f"⚠️ [{self.subject_id}] Simulation mode on port {self.port}")
            self.running = True
            self.collect_thread = threading.Thread(target=self._simulate_data, daemon=True)
            self.collect_thread.start()
            return True

        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=0.1
            )
            self.serial.reset_input_buffer()
            self.running = True

            self.collect_thread = threading.Thread(target=self._collect_loop, daemon=True)
            self.collect_thread.start()

            print(f"✅ [{self.subject_id}] Serial port {self.port} opened")
            return True

        except Exception as e:
            print(f"❌ [{self.subject_id}] Failed to open {self.port}: {e}")
            return False

    def stop(self):
        """Stop data collection."""
        self.running = False
        if self.collect_thread:
            self.collect_thread.join(timeout=2.0)
        if self.serial and self.serial.is_open:
            self.serial.close()

    def get_current_indices(self):
        return {"exg_idx": self.exg_count}

    def mark_repetition_start(self):
        self.rep_start_exg = len(self.exg_data)

    def get_repetition_data(self):
        exg_data = self.exg_data[self.rep_start_exg:]
        exg_ts = self.exg_timestamps[self.rep_start_exg:]
        exg_seq = self.exg_seq[self.rep_start_exg:]
        exg_arr = np.array(exg_data, dtype=np.uint16) if exg_data else np.zeros((0, 4), dtype=np.uint16)
        return {
            'exg': exg_arr,
            'exg_timestamps': np.array(exg_ts, dtype=np.float64),
            'exg_seq': np.array(exg_seq, dtype=np.uint32),
        }

    def snapshot_and_clear(self):
        """Convert accumulated data to compact numpy arrays and free the Python lists.

        Returns a dict of numpy arrays. Call this periodically (e.g. after each gesture)
        to prevent unbounded memory growth from Python list overhead.
        """
        n = len(self.exg_data)
        if n == 0:
            return {
                'exg': np.zeros((0, 4), dtype=np.uint16),
                'exg_timestamps': np.zeros(0, dtype=np.float64),
                'exg_seq': np.zeros(0, dtype=np.uint32),
            }

        exg_arr = np.array(self.exg_data, dtype=np.uint16)
        ts_arr = np.array(self.exg_timestamps, dtype=np.float64)
        seq_arr = np.array(self.exg_seq, dtype=np.uint32)

        # Clear Python lists to free memory (display deque is unaffected)
        self.exg_data.clear()
        self.exg_timestamps.clear()
        self.exg_seq.clear()
        self.rep_start_exg = 0

        return {'exg': exg_arr, 'exg_timestamps': ts_arr, 'exg_seq': seq_arr}

    def _collect_loop(self):
        """Data collection loop - EXG frames only."""
        buffer = b''

        while self.running:
            try:
                if self.serial.in_waiting:
                    buffer += self.serial.read(self.serial.in_waiting)

                while len(buffer) >= 2:
                    if buffer[0] != FRAME_HEADER:
                        buffer = buffer[1:]
                        continue

                    frame_type = buffer[1]

                    if frame_type == EXG_TYPE:
                        if len(buffer) < EXG_FRAME_SIZE:
                            break

                        frame_data = buffer[2:EXG_FRAME_SIZE]
                        seq, ecg1, ecg2, emg1, emg2 = struct.unpack('<IHHHH', frame_data)
                        current_time = time.time() - self.session_start_time

                        self.exg_data.append([ecg1, ecg2, emg1, emg2])
                        self.exg_timestamps.append(current_time)
                        self.exg_seq.append(seq)

                        self.exg_display[0].append(ecg1)
                        self.exg_display[1].append(ecg2)
                        self.exg_display[2].append(emg1)
                        self.exg_display[3].append(emg2)

                        self.exg_count += 1
                        buffer = buffer[EXG_FRAME_SIZE:]
                    else:
                        # Skip unknown frame types
                        buffer = buffer[1:]

                time.sleep(0.001)

            except Exception:
                time.sleep(0.01)

    def _simulate_data(self):
        """Simulate EXG data for testing."""
        last_exg = time.time()
        # Random phase offset so different subjects look different
        phase_offset = random.uniform(0, 2 * np.pi)
        hr_sim = random.uniform(1.0, 1.5)  # different simulated HR per subject

        while self.running:
            current_time = time.time()
            rel_time = current_time - self.session_start_time

            if current_time - last_exg >= 1 / EXG_FS:
                ecg1 = int(2048 + 500 * np.sin(2 * np.pi * hr_sim * rel_time + phase_offset)
                           + np.random.randn() * 30)
                ecg2 = int(2048 + 400 * np.sin(2 * np.pi * hr_sim * rel_time + phase_offset + 0.1)
                           + np.random.randn() * 30)
                emg1 = int(2048 + np.random.randn() * 150)
                emg2 = int(2048 + np.random.randn() * 150)

                self.exg_data.append([ecg1, ecg2, emg1, emg2])
                self.exg_timestamps.append(rel_time)
                self.exg_seq.append(self.exg_count)  # simulated sequential seq

                self.exg_display[0].append(ecg1)
                self.exg_display[1].append(ecg2)
                self.exg_display[2].append(emg1)
                self.exg_display[3].append(emg2)

                self.exg_count += 1
                last_exg = current_time

            time.sleep(0.0001)


# ==================== EVENT LOGGER ====================

class EventLogger:
    """Records experiment events with per-repetition saving. EXG only."""

    def __init__(self, subject_id, subject_dir, collector=None):
        self.subject_id = subject_id
        self.subject_dir = subject_dir
        self.collector = collector
        self.session_start_unix = 0

        self.all_events = []
        self.rep_events = []
        self.rep_start_time = 0
        self.metadata = {}

        self.data_dir = os.path.join(subject_dir, "repetitions")
        self.events_dir = os.path.join(subject_dir, "events")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.events_dir, exist_ok=True)

    def start_session(self):
        self.session_start_unix = time.time()
        self.metadata = {
            "subject_id": self.subject_id,
            "session_start": datetime.now().isoformat(),
            "session_start_unix": self.session_start_unix,
            "config": {
                "on_duration": CONFIG["on_duration"],
                "off_duration": CONFIG["off_duration"],
                "actions_per_gesture": CONFIG["actions_per_gesture"],
                "transition_duration": CONFIG["transition_duration"],
                "hold_duration": CONFIG["hold_duration"],
                "freeform_duration": CONFIG["freeform_duration"],
                "num_repetitions": CONFIG["num_repetitions"],
            },
            "sampling_rates": {"exg": EXG_FS},
            "gestures": GESTURES,
        }
        self.log_event("SESSION_START", {})

    def start_repetition(self, rep_num):
        self.rep_events = []
        self.rep_start_time = time.time()

    def log_event(self, event_type, data):
        current_time = time.time()
        elapsed_ms = int((current_time - self.session_start_unix) * 1000) if self.session_start_unix else 0
        rep_elapsed_ms = int((current_time - self.rep_start_time) * 1000) if self.rep_start_time else 0

        event = {
            "timestamp": datetime.now().isoformat(),
            "unix_time": current_time,
            "elapsed_ms": elapsed_ms,
            "rep_elapsed_ms": rep_elapsed_ms,
            "event_type": event_type,
            "data": data,
        }

        if self.collector:
            event["sample_indices"] = self.collector.get_current_indices()

        self.all_events.append(event)
        self.rep_events.append(event)

        idx_str = ""
        if "sample_indices" in event:
            idx = event["sample_indices"]
            idx_str = f" [E:{idx['exg_idx']}]"
        print(f"[{self.subject_id}][{elapsed_ms:>7}ms]{idx_str} {event_type}")

    def save_repetition(self, rep_num, gesture_order):
        events_file = os.path.join(self.events_dir, f"rep_{rep_num:02d}_events.json")
        output = {
            "metadata": {
                "subject_id": self.subject_id,
                "repetition": rep_num,
                "gesture_order": gesture_order,
                "config": self.metadata.get("config", {}),
                "sampling_rates": self.metadata.get("sampling_rates", {}),
                "gestures": GESTURES,
            },
            "events": self.rep_events,
            "summary": {
                "total_events": len(self.rep_events),
                "total_gestures": len([e for e in self.rep_events if e["event_type"] == "GESTURE_START"]),
                "total_actions": len([e for e in self.rep_events if e["event_type"] == "ACTION_START"]),
            }
        }
        with open(events_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"✅ [{self.subject_id}] Rep {rep_num} events saved")
        return events_file

    def save_session_summary(self, completed_reps):
        summary_file = os.path.join(self.subject_dir, "session_info.json")
        output = {
            "metadata": self.metadata,
            "session_end": datetime.now().isoformat(),
            "completed_repetitions": completed_reps,
            "total_events": len(self.all_events),
            "folder_structure": {
                "repetitions": "Contains .npz data files for each repetition",
                "events": "Contains .json event logs for each repetition",
            }
        }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"✅ [{self.subject_id}] Session summary saved")


# ==================== MAIN APPLICATION ====================

class MultiDeviceRecorderApp:
    """Multi-device gesture recording with subject-switchable waveform view."""

    def __init__(self, config):
        self.config = config
        self.root = tk.Tk()
        self.root.title("Gesture Recording v4 - Multi-Device")

        # Multi-device state
        self.devices = {}          # {subject_id: DataCollector}
        self.loggers = {}          # {subject_id: EventLogger}
        self.subject_ids = []      # ordered list of subject IDs
        self.subject_ports = {}    # {subject_id: port_name}
        self.subject_dirs = {}     # {subject_id: directory}
        self.active_subject = None # currently viewed subject

        # Experiment state
        self.current_repetition = 0
        self.current_gesture_idx = 0
        self.current_action = 0
        self.gesture_order = []
        self.is_running = False
        self.is_paused = False
        self.experiment_started = False
        self.sidebar_visible = config["show_waveform"]

        # Waveform
        self.fig = None
        self.canvas = None
        self.lines = {}
        self.axes = []

        # HR computation (App-level, only for active subject)
        self.ecg_hr_chest = 0
        self.ecg_hr_wrist = 0
        self.hr_thread = None
        self.hr_running = False

        # Video animation state
        self._video_after_id = None
        self._video_frame_idx = 0
        self._video_current_id = None

        # Free form small video animation state
        self._ff_video_after_id = None
        self._ff_video_frame_idx = 0
        self._ff_video_current_id = None

        # Setup
        self._setup_window()
        self._setup_layout()
        self._load_gesture_images()

    def _setup_window(self):
        if self.config["fullscreen"]:
            self.root.attributes("-fullscreen", True)
        else:
            self.root.geometry(self.config["window_size"])
        self.root.configure(bg=self.config["background_color"])
        self.root.bind_all("<Escape>", lambda e: self._on_escape())
        self.root.bind_all("<space>", lambda e: self._toggle_pause())
        self.root.bind_all("<Return>", lambda e: self._start_experiment())
        self.root.bind_all("<w>", lambda e: self._toggle_sidebar())
        # Number keys to switch subject view
        for i in range(1, 5):
            self.root.bind_all(str(i), lambda e, idx=i: self._switch_subject_view(idx))
        self.root.protocol("WM_DELETE_WINDOW", self._on_escape)
        self.root.focus_force()

    def _setup_layout(self):
        bg = self.config["background_color"]
        fg = self.config["text_color"]
        accent = self.config["accent_color"]

        self.main_container = tk.Frame(self.root, bg=bg)
        self.main_container.pack(expand=True, fill="both")

        # ========== MAIN CONTENT (CENTER) ==========
        self.main_frame = tk.Frame(self.main_container, bg=bg)
        self.main_frame.pack(side="left", expand=True, fill="both")

        # Top bar
        self.top_frame = tk.Frame(self.main_frame, bg=bg)
        self.top_frame.pack(fill="x", padx=20, pady=10)

        self.progress_label = tk.Label(
            self.top_frame, text="Press ENTER to setup",
            font=("Arial", 16, "bold"), bg=bg, fg=accent
        )
        self.progress_label.pack(side="left")

        self.status_label = tk.Label(
            self.top_frame, text="Ready",
            font=("Arial", 14), bg=bg, fg=fg
        )
        self.status_label.pack(side="right")

        # Center content
        self.center_frame = tk.Frame(self.main_frame, bg=bg)
        self.center_frame.pack(expand=True, fill="both", padx=40, pady=20)

        self.gesture_label = tk.Label(
            self.center_frame, text="", font=("Arial", 32, "bold"), bg=bg, fg=fg
        )
        self.gesture_label.pack(pady=(20, 10))

        self.action_label = tk.Label(
            self.center_frame, text="", font=("Arial", 18), bg=bg, fg="#888888"
        )
        self.action_label.pack(pady=(0, 10))

        self.image_label = tk.Label(self.center_frame, bg=bg)
        self.image_label.pack(expand=True, pady=20)

        self.timer_label = tk.Label(
            self.center_frame, text="00:00", font=("Arial", 72, "bold"), bg=bg, fg=fg
        )
        self.timer_label.pack(pady=(20, 10))

        self.phase_label = tk.Label(
            self.center_frame, text="", font=("Arial", 32), bg=bg, fg=accent
        )
        self.phase_label.pack(pady=(0, 20))

        # ========== FREE FORM OVERLAY ==========
        self.ff_frame = tk.Frame(self.center_frame, bg=bg)
        # Not packed by default - shown only during free form

        self.ff_timer_label = tk.Label(
            self.ff_frame, text="00:00", font=("Arial", 28), bg=bg, fg=fg
        )
        self.ff_timer_label.pack(pady=(5, 5))

        # Spacer to push content to vertical center
        self.ff_spacer_top = tk.Frame(self.ff_frame, bg=bg)
        self.ff_spacer_top.pack(expand=True)

        self.ff_phase_label = tk.Label(
            self.ff_frame, text="FREE FORM",
            font=("Arial", 16), bg=bg, fg="#888888"
        )
        self.ff_phase_label.pack(pady=(0, 5))

        self.ff_title_label = tk.Label(
            self.ff_frame, text="Stretch a little bit",
            font=("Arial", 52, "bold"), bg=bg, fg=self.config["freeform_color"]
        )
        self.ff_title_label.pack(pady=(0, 20))

        self.ff_next_label = tk.Label(
            self.ff_frame, text="", font=("Arial", 20), bg=bg, fg="#aaaaaa"
        )
        self.ff_next_label.pack(pady=(5, 5))

        self.ff_video_label = tk.Label(self.ff_frame, bg=bg)
        self.ff_video_label.pack(pady=(5, 0))

        # Spacer to push content to vertical center
        self.ff_spacer_bottom = tk.Frame(self.ff_frame, bg=bg)
        self.ff_spacer_bottom.pack(expand=True)

        # Bottom bar
        self.bottom_frame = tk.Frame(self.main_frame, bg=bg)
        self.bottom_frame.pack(fill="x", padx=20, pady=10)

        self.progress_bar = ttk.Progressbar(self.bottom_frame, length=500, mode="determinate")
        self.progress_bar.pack(pady=(0, 10))

        self.instruction_label = tk.Label(
            self.bottom_frame,
            text="ENTER=Start | SPACE=Pause | W=Waveform | 1-4=Switch Subject | ESC=Exit",
            font=("Arial", 10), bg=bg, fg="#666666"
        )
        self.instruction_label.pack()

        # ========== SIDEBAR (RIGHT) - WAVEFORMS ==========
        self.sidebar_frame = tk.Frame(
            self.main_container, bg="#16213e", width=self.config["sidebar_width"]
        )
        if self.sidebar_visible:
            self.sidebar_frame.pack(side="right", fill="y")
        self.sidebar_frame.pack_propagate(False)

        # Sidebar header
        sidebar_header = tk.Frame(self.sidebar_frame, bg="#16213e")
        sidebar_header.pack(fill="x", padx=5, pady=5)

        tk.Label(
            sidebar_header, text="Signal Monitor",
            font=("Arial", 12, "bold"), bg="#16213e", fg="#ffffff"
        ).pack(side="left")

        self.toggle_btn = tk.Button(
            sidebar_header, text="Hide [W]", command=self._toggle_sidebar,
            font=("Arial", 9), bg="#333333", fg="#ffffff", relief="flat"
        )
        self.toggle_btn.pack(side="right")

        # Subject selector buttons
        self.subject_btn_frame = tk.Frame(self.sidebar_frame, bg="#16213e")
        self.subject_btn_frame.pack(fill="x", padx=5, pady=(0, 5))

        self.subject_buttons = {}  # populated after setup

        # Active subject indicator
        self.active_subject_label = tk.Label(
            self.sidebar_frame, text="No devices connected",
            font=("Arial", 14, "bold"), bg="#16213e", fg="#ffffff"
        )
        self.active_subject_label.pack(fill="x", padx=5, pady=(0, 3))

        # HR display
        self.hr_frame = tk.Frame(self.sidebar_frame, bg="#16213e")
        self.hr_frame.pack(fill="x", padx=5, pady=(0, 5))

        self.hr_chest_label = tk.Label(
            self.hr_frame, text="HR Chest: -- bpm",
            font=("Arial", 14, "bold"), bg="#16213e", fg="#ff6b6b"
        )
        self.hr_chest_label.pack(anchor="w")

        self.hr_wrist_label = tk.Label(
            self.hr_frame, text="HR Wrist: -- bpm",
            font=("Arial", 14, "bold"), bg="#16213e", fg="#4ecdc4"
        )
        self.hr_wrist_label.pack(anchor="w")

        # Waveform area
        self.waveform_frame = tk.Frame(self.sidebar_frame, bg="#16213e")
        self.waveform_frame.pack(expand=True, fill="both", padx=5, pady=5)

        if MATPLOTLIB_AVAILABLE:
            self._setup_waveform_display()

    def _setup_waveform_display(self):
        window_sec = self.config["waveform_window"]
        self.fig = Figure(figsize=(4, 5), facecolor='#16213e', dpi=80)

        channel_info = [
            ('ECG Chest', '#ff6b6b', EXG_FS),
            ('ECG Wrist', '#4ecdc4', EXG_FS),
            ('EMG 1', '#95e1d3', EXG_FS),
            ('EMG 2', '#a8e6cf', EXG_FS),
        ]

        self.axes = []
        self.lines = {}

        for i, (name, color, fs) in enumerate(channel_info):
            ax = self.fig.add_subplot(4, 1, i + 1)
            ax.set_facecolor('#1a1a2e')
            ax.set_ylabel(name, fontsize=7, color='white')
            ax.tick_params(colors='white', labelsize=5)
            ax.set_xlim(0, window_sec)
            ax.set_ylim(0, 4095)
            ax.grid(True, alpha=0.2, color='white')

            for spine in ax.spines.values():
                spine.set_color('#333333')

            if i < 3:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (s)', fontsize=7, color='white')

            t = np.linspace(0, window_sec, window_sec * fs)
            init_data = [2048] * len(t)
            line, = ax.plot(t, init_data, color=color, linewidth=0.5)
            self.lines[i] = line
            self.axes.append(ax)

        self.fig.tight_layout(pad=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.waveform_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill="both")

    def _toggle_sidebar(self):
        if self.sidebar_visible:
            self.sidebar_frame.pack_forget()
            self.sidebar_visible = False
        else:
            self.sidebar_frame.pack(side="right", fill="y")
            self.sidebar_visible = True

    def _switch_subject_view(self, idx):
        """Switch waveform display to subject at index idx (1-based)."""
        if idx <= len(self.subject_ids):
            self.active_subject = self.subject_ids[idx - 1]
            color = SUBJECT_COLORS[(idx - 1) % len(SUBJECT_COLORS)]
            self.active_subject_label.config(
                text=f"Viewing: {self.active_subject} ({self.subject_ports.get(self.active_subject, '?')})",
                fg=color
            )
            # Reset HR display (will be recomputed for new subject)
            self.ecg_hr_chest = 0
            self.ecg_hr_wrist = 0

            # Update button highlight
            for sid, btn in self.subject_buttons.items():
                if sid == self.active_subject:
                    btn.config(relief="sunken", bg="#4ecca3")
                else:
                    btn.config(relief="raised", bg="#333333")

    def _load_gesture_images(self):
        self.gesture_images = {}      # {id: PhotoImage} static fallback
        self.gesture_video_frames = {} # {id: [PhotoImage, ...]} video frames
        self.gesture_video_frames_small = {} # {id: [PhotoImage, ...]} small preview frames
        self.gesture_video_fps = {}    # {id: float}
        img_dir = self.config["gesture_images_dir"]
        os.makedirs(img_dir, exist_ok=True)

        # Load steady state image
        self.ss_image = None
        for ext in ("SS.jpeg", "SS.jpg", "SS.png"):
            ss_path = os.path.join(img_dir, ext)
            if os.path.exists(ss_path):
                img = Image.open(ss_path).resize((280, 280), Image.Resampling.LANCZOS)
                self.ss_image = ImageTk.PhotoImage(img)
                print(f"📷 Steady State image loaded: {ss_path}")
                break

        for gesture_id, gesture_name in GESTURES.items():
            # Try mp4 first
            mp4_path = os.path.join(img_dir, f"g{gesture_id}.mp4")
            if CV2_AVAILABLE and os.path.exists(mp4_path):
                frames, fps = self._load_video_frames(mp4_path)
                small_frames, _ = self._load_video_frames(mp4_path, size=(140, 140))
                if frames:
                    self.gesture_video_frames[gesture_id] = frames
                    self.gesture_video_frames_small[gesture_id] = small_frames
                    self.gesture_video_fps[gesture_id] = fps
                    self.gesture_images[gesture_id] = frames[0]
                    print(f"🎬 Gesture {gesture_id}: {len(frames)} frames @ {fps:.1f}fps from {mp4_path}")
                    continue

            # Fallback to png
            img_path = os.path.join(img_dir, f"gesture_{gesture_id}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img = img.resize((280, 280), Image.Resampling.LANCZOS)
            else:
                img = self._create_placeholder_image(gesture_id, gesture_name)
            self.gesture_images[gesture_id] = ImageTk.PhotoImage(img)

    def _load_video_frames(self, path, size=(280, 280)):
        """Load all frames from an mp4 file as PhotoImage list."""
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS)
            frames.append(ImageTk.PhotoImage(img))
        cap.release()
        return frames, fps

    def _clear_image_text(self):
        """Clear any text on image_label (used after freeform)."""
        self.image_label.config(text="", font=("Arial", 1))

    def _show_ff_overlay(self, gesture_id, gesture_name):
        """Show the free form overlay, hiding normal center widgets."""
        # Hide normal widgets
        for w in (self.gesture_label, self.action_label, self.image_label,
                  self.timer_label, self.phase_label):
            w.pack_forget()
        # Show overlay
        self.ff_frame.pack(expand=True, fill="both")
        self.ff_phase_label.config(text="FREE FORM")
        self.ff_title_label.config(text="Stretch a little bit",
                                   fg=self.config["freeform_color"])
        self.ff_next_label.config(text=f"Next: {gesture_name}")
        self._start_ff_video(gesture_id)

    def _hide_ff_overlay(self):
        """Hide the free form overlay, restore normal center widgets."""
        self._stop_ff_video()
        self.ff_frame.pack_forget()
        # Clear all normal widgets before re-packing to avoid flash
        self.gesture_label.config(text="")
        self.action_label.config(text="")
        self.image_label.config(image="")
        self._clear_image_text()
        self.timer_label.config(text="")
        self.phase_label.config(text="")
        # Restore normal widgets in order
        self.gesture_label.pack(pady=(20, 10))
        self.action_label.pack(pady=(0, 10))
        self.image_label.pack(expand=True, pady=20)
        self.timer_label.pack(pady=(20, 10))
        self.phase_label.pack(pady=(0, 20))

    def _start_video(self, gesture_id, loop=True, duration_ms=None):
        """Start video playback for a gesture.

        Args:
            gesture_id: which gesture video to play
            loop: if True, loop forever; if False, play once then stop
            duration_ms: if set, stretch/compress playback to fit this duration
        """
        self._stop_video()
        self._clear_image_text()
        if gesture_id not in self.gesture_video_frames:
            if gesture_id in self.gesture_images:
                self.image_label.config(image=self.gesture_images[gesture_id])
            return
        self._video_current_id = gesture_id
        self._video_frame_idx = 0
        self._video_loop = loop
        frames = self.gesture_video_frames[gesture_id]
        if duration_ms:
            # Fit all frames into the given duration
            self._video_delay = max(1, int(duration_ms / len(frames)))
        else:
            fps = self.gesture_video_fps.get(gesture_id, 30.0)
            self._video_delay = max(1, int(1000 / fps))
        self._animate_video()

    def _animate_video(self):
        """Display next frame and schedule the one after."""
        gid = self._video_current_id
        if gid is None or gid not in self.gesture_video_frames:
            return
        frames = self.gesture_video_frames[gid]
        self.image_label.config(image=frames[self._video_frame_idx])
        self._video_frame_idx += 1
        if self._video_frame_idx >= len(frames):
            if self._video_loop:
                self._video_frame_idx = 0
            else:
                self._video_current_id = None
                return
        self._video_after_id = self.root.after(self._video_delay, self._animate_video)

    def _stop_video(self):
        """Stop any running video animation."""
        if self._video_after_id is not None:
            self.root.after_cancel(self._video_after_id)
            self._video_after_id = None
        self._video_current_id = None
        self._video_frame_idx = 0

    def _show_static_frame(self, gesture_id):
        """Stop video and show the first frame (relaxed pose)."""
        self._stop_video()
        self._clear_image_text()
        if gesture_id in self.gesture_video_frames:
            self.image_label.config(image=self.gesture_video_frames[gesture_id][0])
        elif gesture_id in self.gesture_images:
            self.image_label.config(image=self.gesture_images[gesture_id])

    def _start_ff_video(self, gesture_id):
        """Start looping small video in the free form preview label."""
        self._stop_ff_video()
        if gesture_id not in self.gesture_video_frames_small:
            return
        self._ff_video_current_id = gesture_id
        self._ff_video_frame_idx = 0
        fps = self.gesture_video_fps.get(gesture_id, 30.0)
        self._ff_video_delay = max(1, int(1000 / fps))
        self._animate_ff_video()

    def _animate_ff_video(self):
        gid = self._ff_video_current_id
        if gid is None or gid not in self.gesture_video_frames_small:
            return
        frames = self.gesture_video_frames_small[gid]
        self.ff_video_label.config(image=frames[self._ff_video_frame_idx])
        self._ff_video_frame_idx = (self._ff_video_frame_idx + 1) % len(frames)
        self._ff_video_after_id = self.root.after(self._ff_video_delay, self._animate_ff_video)

    def _stop_ff_video(self):
        if self._ff_video_after_id is not None:
            self.root.after_cancel(self._ff_video_after_id)
            self._ff_video_after_id = None
        self._ff_video_current_id = None
        self._ff_video_frame_idx = 0
        self.ff_video_label.config(image="")

    def _create_placeholder_image(self, gesture_id, gesture_name):
        img = Image.new('RGB', (280, 280), color='#2d3436')
        draw = ImageDraw.Draw(img)
        draw.rectangle([5, 5, 275, 275], outline='#4ecca3', width=2)
        try:
            font_large = ImageFont.truetype("arial.ttf", 70)
            font_small = ImageFont.truetype("arial.ttf", 16)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        text = str(gesture_id)
        bbox = draw.textbbox((0, 0), text, font=font_large)
        tw = bbox[2] - bbox[0]
        draw.text(((280 - tw) // 2, 70), text, fill='#4ecca3', font=font_large)
        bbox = draw.textbbox((0, 0), gesture_name, font=font_small)
        tw = bbox[2] - bbox[0]
        draw.text(((280 - tw) // 2, 180), gesture_name, fill='#ffffff', font=font_small)
        return img

    # ==================== SETUP DIALOG ====================

    def _show_setup_dialog(self):
        """Show setup dialog: scan ports, assign subject IDs."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Multi-Device Setup")
        dialog.geometry("550x500")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg="#1a1a2e")

        # --- Scan section ---
        tk.Label(dialog, text="Multi-Device Setup",
                 font=("Arial", 16, "bold"), bg="#1a1a2e", fg="#ffffff").pack(pady=(15, 10))

        scan_frame = tk.Frame(dialog, bg="#1a1a2e")
        scan_frame.pack(fill="x", padx=20, pady=5)

        tk.Label(scan_frame, text="Starting Subject ID:",
                 font=("Arial", 12), bg="#1a1a2e", fg="#ffffff").pack(side="left")

        start_id_entry = tk.Entry(scan_frame, font=("Arial", 14), width=8)
        start_id_entry.insert(0, "S01")
        start_id_entry.pack(side="left", padx=10)

        # Port list frame
        port_list_frame = tk.Frame(dialog, bg="#1a1a2e")
        port_list_frame.pack(fill="both", expand=True, padx=20, pady=10)

        tk.Label(port_list_frame, text="Detected Ports (check to use):",
                 font=("Arial", 11, "bold"), bg="#1a1a2e", fg="#4ecca3").pack(anchor="w")

        # Scrollable port list
        port_canvas = tk.Canvas(port_list_frame, bg="#1a1a2e", highlightthickness=0)
        port_inner = tk.Frame(port_canvas, bg="#1a1a2e")
        port_canvas.pack(fill="both", expand=True)
        port_canvas.create_window((0, 0), window=port_inner, anchor="nw")

        port_vars = {}  # {port: BooleanVar}
        port_labels = {}

        def do_scan():
            # Clear old
            for w in port_inner.winfo_children():
                w.destroy()
            port_vars.clear()

            ports = scan_serial_ports()

            if not ports:
                tk.Label(port_inner,
                         text="No serial ports found!\n(Will use simulation mode)",
                         font=("Arial", 11), bg="#1a1a2e", fg="#ff6b6b").pack(pady=10)
                # Add simulation entries
                for i in range(4):
                    sim_port = f"SIM{i + 1}"
                    var = tk.BooleanVar(value=(i < 1))  # default: 1 simulated device
                    port_vars[sim_port] = var
                    row = tk.Frame(port_inner, bg="#1a1a2e")
                    row.pack(fill="x", pady=2)
                    cb = tk.Checkbutton(row, variable=var, bg="#1a1a2e",
                                        activebackground="#1a1a2e", selectcolor="#333333")
                    cb.pack(side="left")
                    tk.Label(row, text=f"{sim_port}  (Simulation)",
                             font=("Arial", 11), bg="#1a1a2e", fg="#ffd700").pack(side="left")
            else:
                for p in ports:
                    var = tk.BooleanVar(value=True)
                    port_vars[p["port"]] = var
                    row = tk.Frame(port_inner, bg="#1a1a2e")
                    row.pack(fill="x", pady=2)
                    cb = tk.Checkbutton(row, variable=var, bg="#1a1a2e",
                                        activebackground="#1a1a2e", selectcolor="#333333")
                    cb.pack(side="left")
                    tk.Label(row, text=f"{p['port']}  -  {p['description']}",
                             font=("Arial", 11), bg="#1a1a2e", fg="#ffffff").pack(side="left")

            status_lbl.config(text=f"Found {len(ports)} port(s)")

        scan_btn = tk.Button(scan_frame, text="Scan Ports", command=do_scan,
                             font=("Arial", 11), bg="#4ecca3", fg="#000000", width=10)
        scan_btn.pack(side="right")

        status_lbl = tk.Label(dialog, text="Click 'Scan Ports' to detect devices",
                              font=("Arial", 10), bg="#1a1a2e", fg="#888888")
        status_lbl.pack(pady=5)

        # --- Result ---
        result = [None]

        def on_ok():
            selected = [port for port, var in port_vars.items() if var.get()]
            if not selected:
                messagebox.showwarning("Error", "Select at least one port!")
                return
            if len(selected) > 4:
                messagebox.showwarning("Error", "Maximum 4 devices!")
                return

            start_str = start_id_entry.get().strip()

            # Parse starting ID: e.g. "S01" -> prefix="S", num=1
            prefix = ""
            num_str = ""
            for ch in start_str:
                if ch.isdigit():
                    num_str += ch
                else:
                    if not num_str:
                        prefix += ch

            start_num = int(num_str) if num_str else 1
            num_digits = len(num_str) if num_str else 2

            assignments = []
            for i, port in enumerate(selected):
                sid = f"{prefix}{str(start_num + i).zfill(num_digits)}"
                assignments.append((sid, port))

            result[0] = assignments
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        btn_frame = tk.Frame(dialog, bg="#1a1a2e")
        btn_frame.pack(pady=15)
        tk.Button(btn_frame, text="Start", command=on_ok,
                  font=("Arial", 12, "bold"), bg="#4ecca3", fg="#000000", width=12).pack(side="left", padx=10)
        tk.Button(btn_frame, text="Cancel", command=on_cancel,
                  font=("Arial", 12), bg="#333333", fg="#ffffff", width=12).pack(side="left", padx=10)

        start_id_entry.bind("<Return>", lambda e: on_ok())
        dialog.bind("<Escape>", lambda e: on_cancel())

        # Auto scan on open
        dialog.after(200, do_scan)

        self.root.wait_window(dialog)
        return result[0]

    # ==================== DEVICE & SESSION MANAGEMENT ====================

    def _create_subject_folders(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for sid in self.subject_ids:
            d = os.path.join(self.config["output_dir"], f"{sid}_{timestamp}")
            os.makedirs(d, exist_ok=True)
            self.subject_dirs[sid] = d
            print(f"📁 [{sid}] {d}")

    def _start_all_devices(self):
        """Start data collection on all devices."""
        for sid in self.subject_ids:
            port = self.subject_ports[sid]
            simulate = port.startswith("SIM")
            collector = DataCollector(
                port=port,
                subject_id=sid,
                baud_rate=self.config["baud_rate"],
                window_sec=self.config["waveform_window"],
                simulate=simulate,
            )
            if collector.start():
                self.devices[sid] = collector
            else:
                messagebox.showwarning("DAQ", f"Failed to open {port} for {sid}")

    def _stop_all_devices(self):
        self._stop_hr_thread()
        for sid, collector in self.devices.items():
            collector.stop()

    def _create_subject_buttons(self):
        """Create subject selector buttons in sidebar."""
        for w in self.subject_btn_frame.winfo_children():
            w.destroy()
        self.subject_buttons.clear()

        for i, sid in enumerate(self.subject_ids):
            color = SUBJECT_COLORS[i % len(SUBJECT_COLORS)]
            btn = tk.Button(
                self.subject_btn_frame,
                text=f"{sid} [{i + 1}]",
                command=lambda idx=i + 1: self._switch_subject_view(idx),
                font=("Arial", 10, "bold"),
                bg="#333333", fg=color,
                relief="raised", width=8,
            )
            btn.pack(side="left", padx=2, pady=2)
            self.subject_buttons[sid] = btn

    # ==================== WAVEFORM & STATUS UPDATES ====================

    def _update_waveform(self):
        """Update waveform for the active subject."""
        if self.active_subject and self.active_subject in self.devices:
            collector = self.devices[self.active_subject]

            if MATPLOTLIB_AVAILABLE and self.sidebar_visible:
                try:
                    for i in range(4):
                        data = list(collector.exg_display[i])
                        self.lines[i].set_ydata(data)
                        if len(data) > 0:
                            dmin, dmax = min(data), max(data)
                            margin = (dmax - dmin) * 0.1 + 50
                            self.axes[i].set_ylim(dmin - margin, dmax + margin)
                    self.canvas.draw_idle()
                except Exception:
                    pass

        self.root.after(50, self._update_waveform)

    def _update_hr_display(self):
        """Update HR labels from App-level computed values."""
        self.hr_chest_label.config(
            text=f"HR Chest: {self.ecg_hr_chest} bpm" if self.ecg_hr_chest > 0 else "HR Chest: -- bpm"
        )
        self.hr_wrist_label.config(
            text=f"HR Wrist: {self.ecg_hr_wrist} bpm" if self.ecg_hr_wrist > 0 else "HR Wrist: -- bpm"
        )
        self.root.after(5000, self._update_hr_display)

    def _start_hr_thread(self):
        """Start a single background thread for HR computation."""
        self.hr_running = True
        self.hr_thread = threading.Thread(target=self._hr_compute_loop, daemon=True)
        self.hr_thread.start()

    def _stop_hr_thread(self):
        self.hr_running = False

    def _hr_compute_loop(self):
        """Single thread: compute HR every 5s for the active subject only."""
        while self.hr_running:
            sid = self.active_subject
            if sid and sid in self.devices:
                collector = self.devices[sid]
                self._compute_hr_for_collector(collector)
            time.sleep(5)

    def _compute_hr_for_collector(self, collector):
        """Compute real-time HR from a collector's display buffer using NeuroKit2.

        Uses exg_display deque (fixed-size sliding window) which is always available
        regardless of whether exg_data has been cleared for memory management.
        Stores results in self.ecg_hr_chest / self.ecg_hr_wrist.
        """
        min_samples = 3 * EXG_FS

        for ch_idx, attr in [(0, 'ecg_hr_chest'), (1, 'ecg_hr_wrist')]:
            try:
                signal = np.array(collector.exg_display[ch_idx], dtype=np.float64)

                if len(signal) < min_samples:
                    continue

                cleaned = nk.ecg_clean(signal, sampling_rate=EXG_FS)
                _, info = nk.ecg_peaks(cleaned, sampling_rate=EXG_FS)
                peaks = info.get("ECG_R_Peaks", [])

                if len(peaks) < 2:
                    continue

                rr = np.diff(peaks) / EXG_FS * 1000  # ms
                valid_rr = rr[(rr >= 300) & (rr <= 2000)]

                if len(valid_rr) > 0:
                    hr = 60000.0 / np.mean(valid_rr)
                    setattr(self, attr, int(round(hr)))
            except Exception:
                pass

    # ==================== EXPERIMENT FLOW ====================

    def _start_experiment(self):
        if self.is_running:
            return

        if not self.experiment_started:
            # Phase 1: Setup dialog
            assignments = self._show_setup_dialog()
            if not assignments:
                return

            self.subject_ids = [a[0] for a in assignments]
            for sid, port in assignments:
                self.subject_ports[sid] = port

            self._create_subject_folders()
            self._start_all_devices()
            self._create_subject_buttons()
            # Set first subject as active
            self.active_subject = self.subject_ids[0]
            self._switch_subject_view(1)

            # Start update loops
            self._update_waveform()
            self._update_hr_display()
            self._start_hr_thread()

            self.experiment_started = True
            n = len(self.subject_ids)
            self.progress_label.config(
                text=f"{n} device(s) connected - Press ENTER to begin recording"
            )
            self.status_label.config(text="Monitoring...", fg="#ffd700")
            return

        # Phase 2: Ask about demo (must not block main thread)
        def ask_demo():
            do_demo = messagebox.askyesno(
                "Demo Session",
                "Would you like to run a demo session first?\n\n"
                f"This will show {self.config['demo_gestures']} gestures "
                "to help you get familiar with the flow.\n"
                "No data will be recorded."
            )
            if do_demo:
                self.is_running = True
                threading.Thread(target=self._run_demo, daemon=True).start()
            else:
                self._begin_recording()

        self.root.after(0, ask_demo)

    def _begin_recording(self):
        """Create loggers and start the experiment."""
        for sid in self.subject_ids:
            if sid in self.devices:
                logger = EventLogger(sid, self.subject_dirs[sid], self.devices[sid])
                logger.start_session()
                self.loggers[sid] = logger

        self.is_running = True
        self.current_repetition = 0
        threading.Thread(target=self._run_experiment, daemon=True).start()

    def _run_demo(self):
        """Run a short demo session with no data recording."""
        demo_count = self.config["demo_gestures"]
        demo_gestures = random.sample(list(GESTURES.keys()), demo_count)

        def update_start():
            self.progress_label.config(text=f"DEMO SESSION - {demo_count} gestures")
            self.status_label.config(text="Demo", fg="#ffd700")

        self.root.after(0, update_start)

        for idx, gesture_id in enumerate(demo_gestures):
            self.current_gesture_idx = idx + 1
            if not self.is_running:
                break
            self._run_gesture(gesture_id, is_demo=True)

        self.is_running = False

        # After demo, ask to start recording
        start_event = threading.Event()
        self._start_after_demo = False

        def ask_start():
            self._start_after_demo = messagebox.askyesno(
                "Demo Complete",
                "Demo session complete!\n\n"
                "Start recording now?"
            )
            start_event.set()

        self.root.after(0, ask_start)
        start_event.wait()

        if self._start_after_demo:
            self._begin_recording()
        else:
            def update_ready():
                self.progress_label.config(text="Press ENTER to begin recording")
                self.status_label.config(text="Monitoring...", fg="#ffd700")
            self.root.after(0, update_ready)

    def _run_experiment(self):
        """Main experiment loop - shared stimulus, all devices record simultaneously."""
        total_reps = self.config["num_repetitions"]

        for rep in range(total_reps):
            self.current_repetition = rep + 1

            # Per-rep data buffer: {sid: [chunk1, chunk2, ...]} of numpy arrays
            self._rep_data_chunks = {sid: [] for sid in self.subject_ids}

            # Clear any leftover data in collectors from previous rep / monitoring
            for sid in self.subject_ids:
                if sid in self.devices:
                    self.devices[sid].snapshot_and_clear()

            for sid in self.subject_ids:
                if sid in self.loggers:
                    self.loggers[sid].start_repetition(self.current_repetition)
                    self.loggers[sid].log_event("REPETITION_START", {
                        "repetition": self.current_repetition,
                        "total": total_reps,
                    })

            if not self.is_running:
                break

            # Generate gesture order (shared across all subjects in same rep)
            self.gesture_order = list(GESTURES.keys())
            random.shuffle(self.gesture_order)

            for sid in self.subject_ids:
                if sid in self.loggers:
                    self.loggers[sid].log_event("GESTURE_ORDER", {
                        "repetition": self.current_repetition,
                        "order": self.gesture_order,
                    })

            # Run gestures
            for idx, gesture_id in enumerate(self.gesture_order):
                self.current_gesture_idx = idx + 1

                while self.is_paused:
                    time.sleep(0.1)
                if not self.is_running:
                    break

                self._run_gesture(gesture_id)

                # After each gesture: snapshot data to numpy & free Python lists
                for sid in self.subject_ids:
                    if sid in self.devices:
                        chunk = self.devices[sid].snapshot_and_clear()
                        self._rep_data_chunks[sid].append(chunk)

            # End repetition
            for sid in self.subject_ids:
                if sid in self.loggers:
                    self.loggers[sid].log_event("REPETITION_END", {
                        "repetition": self.current_repetition,
                    })

            # Save all subjects' data for this repetition (same file format)
            self._save_all_repetition_data(self.current_repetition)

            if not self.is_running:
                break

            # Ask whether to continue to next repetition
            if rep < total_reps - 1:
                continue_event = threading.Event()
                self._continue_result = False

                def ask_continue(r=self.current_repetition, t=total_reps):
                    self._continue_result = messagebox.askyesno(
                        "Continue?",
                        f"Repetition {r}/{t} completed and saved.\n\n"
                        f"Continue to the next repetition?"
                    )
                    continue_event.set()

                self.root.after(0, ask_continue)
                continue_event.wait()

                if not self._continue_result:
                    self.is_running = False
                    break

        self._experiment_complete()

    def _run_gesture(self, gesture_id, is_demo=False):
        """Run single gesture. If is_demo, no events are logged."""
        gesture_name = GESTURES[gesture_id]
        total_actions = self.config["actions_per_gesture"]
        on_dur = self.config["on_duration"]
        off_dur = self.config["off_duration"]
        trans_dur = self.config["transition_duration"]

        # Helper to log to all loggers (no-op during demo)
        def log_all(event_type, data):
            if is_demo:
                return
            for sid in self.subject_ids:
                if sid in self.loggers:
                    self.loggers[sid].log_event(event_type, data)

        rep_text = (f"Rep {self.current_repetition}/{self.config['num_repetitions']} | "
                    f"Gesture {self.current_gesture_idx}/{len(GESTURES)}")
        if is_demo:
            rep_text = f"DEMO | Gesture {self.current_gesture_idx}/{self.config['demo_gestures']}"

        # ===== FREE FORM =====
        log_all("FREEFORM_START", {
            "repetition": self.current_repetition,
            "gesture_idx": self.current_gesture_idx,
            "gesture_id": gesture_id,
            "gesture_name": gesture_name,
            "duration": self.config["freeform_duration"],
        })

        def update_freeform():
            self.progress_label.config(text=rep_text)
            self.status_label.config(text="Free Form", fg=self.config["freeform_color"])
            # Hide normal center widgets, show ff overlay
            self._show_ff_overlay(gesture_id, gesture_name)
            self._update_progress_bar()

        self.root.after(0, update_freeform)
        self._countdown(self.config["freeform_duration"], "freeform",
                        timer_label=self.ff_timer_label)

        log_all("FREEFORM_END", {
            "repetition": self.current_repetition,
            "gesture_idx": self.current_gesture_idx,
            "gesture_id": gesture_id,
        })

        # ===== GET READY =====
        while self.is_paused:
            time.sleep(0.1)
        if not self.is_running:
            return

        log_all("TRANSITION_START", {
            "repetition": self.current_repetition,
            "gesture_idx": self.current_gesture_idx,
            "gesture_id": gesture_id,
            "gesture_name": gesture_name,
            "phase": "get_ready",
            "duration": trans_dur,
        })

        def update_get_ready():
            self.ff_phase_label.config(text="TRANSITION")
            self.ff_title_label.config(text="Get Ready",
                                       fg=self.config["transition_color"])
            self.status_label.config(text="Transition", fg=self.config["transition_color"])
            # ff_frame should already be shown from freeform; if not, show it
            if not self.ff_frame.winfo_ismapped():
                self._show_ff_overlay(gesture_id, gesture_name)
                self.ff_title_label.config(text="Get Ready",
                                           fg=self.config["transition_color"])

        self.root.after(0, update_get_ready)
        self._countdown(trans_dur, "transition", timer_label=self.ff_timer_label)

        # Hide ff overlay, restore normal layout for ON/OFF
        self.root.after(0, self._hide_ff_overlay)
        time.sleep(0.05)  # brief pause to let UI update

        log_all("TRANSITION_END", {
            "repetition": self.current_repetition,
            "gesture_idx": self.current_gesture_idx,
            "gesture_id": gesture_id,
            "phase": "get_ready",
        })

        # ===== GESTURE START =====
        log_all("GESTURE_START", {
            "repetition": self.current_repetition,
            "gesture_idx": self.current_gesture_idx,
            "gesture_id": gesture_id,
            "gesture_name": gesture_name,
            "total_actions": total_actions,
        })

        # ===== ON / OFF CYCLES =====
        for action_num in range(1, total_actions + 1):
            self.current_action = action_num

            while self.is_paused:
                time.sleep(0.1)
            if not self.is_running:
                return

            # --- ON ---
            log_all("ACTION_ON", {
                "repetition": self.current_repetition,
                "gesture_idx": self.current_gesture_idx,
                "gesture_id": gesture_id,
                "gesture_name": gesture_name,
                "action_num": action_num,
                "total_actions": total_actions,
                "duration": on_dur,
            })

            def update_on(an=action_num, ta=total_actions, gid=gesture_id, gn=gesture_name):
                self._play_beep(1000, 200)
                self.status_label.config(text="RECORDING", fg=self.config["warning_color"])
                self.gesture_label.config(text=f"{gn}  ({an}/{ta})",
                                          font=("Arial", 32, "bold"))
                self.action_label.config(text="")
                self.phase_label.config(text="Follow the gesture", fg=self.config["warning_color"])
                self._start_video(gid, loop=False, duration_ms=int(on_dur * 1000))

            self.root.after(0, update_on)
            self._countdown(on_dur, "action")

            log_all("ACTION_ON_END", {
                "repetition": self.current_repetition,
                "gesture_idx": self.current_gesture_idx,
                "gesture_id": gesture_id,
                "action_num": action_num,
            })

            # --- OFF ---
            while self.is_paused:
                time.sleep(0.1)
            if not self.is_running:
                return

            log_all("ACTION_OFF", {
                "repetition": self.current_repetition,
                "gesture_idx": self.current_gesture_idx,
                "gesture_id": gesture_id,
                "action_num": action_num,
                "duration": off_dur,
            })

            def update_off(gid=gesture_id, gn=gesture_name):
                self._play_beep(500, 200)
                self._show_static_frame(gid)
                self.status_label.config(text="RECORDING", fg=self.config["off_color"])
                self.gesture_label.config(text=gn, font=("Arial", 32, "bold"))
                self.action_label.config(text="")
                self.phase_label.config(text="Relax", fg=self.config["off_color"])

            self.root.after(0, update_off)
            self._countdown(off_dur, "off")

            log_all("ACTION_OFF_END", {
                "repetition": self.current_repetition,
                "gesture_idx": self.current_gesture_idx,
                "gesture_id": gesture_id,
                "action_num": action_num,
            })

        log_all("GESTURE_END", {
            "repetition": self.current_repetition,
            "gesture_idx": self.current_gesture_idx,
            "gesture_id": gesture_id,
            "gesture_name": gesture_name,
        })

        # ===== TRANSITION OUT =====
        while self.is_paused:
            time.sleep(0.1)
        if not self.is_running:
            return

        log_all("TRANSITION_START", {
            "repetition": self.current_repetition,
            "gesture_idx": self.current_gesture_idx,
            "gesture_id": gesture_id,
            "phase": "post_action",
            "duration": trans_dur,
        })

        def update_transition_out():
            self._show_static_frame(gesture_id)
            self.gesture_label.config(text="Transition", font=("Arial", 32, "bold"))
            self.status_label.config(text="Transition", fg=self.config["transition_color"])
            self.action_label.config(text="Relax your hands naturally",
                                     font=("Arial", 18), fg="#888888")
            self.phase_label.config(text="", fg=self.config["transition_color"])

        self.root.after(0, update_transition_out)
        self._countdown(trans_dur, "transition")

        log_all("TRANSITION_END", {
            "repetition": self.current_repetition,
            "gesture_idx": self.current_gesture_idx,
            "gesture_id": gesture_id,
            "phase": "post_action",
        })

        # ===== HOLD =====
        while self.is_paused:
            time.sleep(0.1)
        if not self.is_running:
            return

        log_all("HOLD_START", {
            "repetition": self.current_repetition,
            "gesture_idx": self.current_gesture_idx,
            "gesture_id": gesture_id,
            "gesture_name": gesture_name,
            "duration": self.config["hold_duration"],
        })

        def update_hold():
            self._stop_video()
            if self.ss_image:
                self.image_label.config(image=self.ss_image)
            self.gesture_label.config(text="Hold", font=("Arial", 32, "bold"))
            self.action_label.config(text="")
            self.status_label.config(text="Hold", fg=self.config["steady_color"])
            self.phase_label.config(text="Try not to move", fg=self.config["steady_color"])

        self.root.after(0, update_hold)
        self._countdown(self.config["hold_duration"], "steady")

        log_all("HOLD_END", {
            "repetition": self.current_repetition,
            "gesture_idx": self.current_gesture_idx,
            "gesture_id": gesture_id,
        })

    def _countdown(self, duration, phase="action", timer_label=None):
        if timer_label is None:
            timer_label = self.timer_label
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            remaining = max(0, duration - elapsed)
            if remaining <= 0:
                break

            while self.is_paused:
                pause_start = time.time()
                while self.is_paused:
                    time.sleep(0.1)
                start_time += (time.time() - pause_start)

            if not self.is_running:
                return

            mins = int(remaining) // 60
            secs = int(remaining) % 60

            if phase == "action":
                color = self.config["warning_color"] if remaining <= 3 else self.config["accent_color"]
            elif phase == "off":
                color = self.config["off_color"]
            elif phase == "steady":
                color = self.config["steady_color"]
            elif phase == "freeform":
                color = self.config["freeform_color"]
            elif phase == "transition":
                color = self.config["transition_color"]
            else:
                color = self.config["text_color"]

            self.root.after(0, lambda c=color, t=f"{mins:02d}:{secs:02d}", lbl=timer_label:
                           lbl.config(text=t, fg=c))
            time.sleep(0.1)

    def _update_progress_bar(self):
        total = self.config["num_repetitions"] * len(GESTURES)
        current = (self.current_repetition - 1) * len(GESTURES) + self.current_gesture_idx
        self.progress_bar["value"] = (current / total) * 100

    # ==================== DATA SAVING ====================

    def _save_all_repetition_data(self, rep_num):
        """Save data for all subjects for this repetition.

        Concatenates per-gesture numpy chunks stored in self._rep_data_chunks,
        then saves with the same file names and format as before.
        """
        for sid in self.subject_ids:
            if sid in self.loggers:
                self.loggers[sid].save_repetition(rep_num, self.gesture_order)

            chunks = self._rep_data_chunks.get(sid, [])
            if not chunks:
                continue

            # Concatenate all gesture chunks into one rep-level array
            exg = np.concatenate([c['exg'] for c in chunks], axis=0) if any(c['exg'].shape[0] > 0 for c in chunks) else np.zeros((0, 4), dtype=np.uint16)
            exg_ts = np.concatenate([c['exg_timestamps'] for c in chunks])
            exg_seq = np.concatenate([c['exg_seq'] for c in chunks])

            npz_dir = os.path.join(self.subject_dirs[sid], "repetitions")
            os.makedirs(npz_dir, exist_ok=True)
            npz_file = os.path.join(npz_dir, f"rep_{rep_num:02d}_data.npz")

            np.savez(
                npz_file,
                exg=exg,
                exg_timestamps=exg_ts,
                exg_seq=exg_seq,
                fs_exg=EXG_FS,
                columns_exg=['ecg_chest', 'ecg_wrist', 'emg1', 'emg2'],
                subject_id=sid,
                repetition=rep_num,
                gesture_order=self.gesture_order,
            )

            print(f"✅ [{sid}] Rep {rep_num} data: {exg.shape}")

            # Free the chunks after saving
            self._rep_data_chunks[sid].clear()

    def _experiment_complete(self):
        for sid in self.subject_ids:
            if sid in self.loggers:
                self.loggers[sid].log_event("SESSION_END", {
                    "total_repetitions": self.current_repetition,
                    "completed": self.is_running,
                })
                self.loggers[sid].save_session_summary(self.current_repetition)

        self._stop_all_devices()
        self.is_running = False

        def update():
            self._stop_video()
            self.image_label.config(image="")
            self.progress_label.config(text="Complete!")
            self.gesture_label.config(text="Thank You!")
            self.action_label.config(text="")
            self.status_label.config(text="All Data Saved", fg=self.config["accent_color"])
            self.timer_label.config(text="✓", fg=self.config["accent_color"])
            self.phase_label.config(text="DONE", fg=self.config["accent_color"])
            self.progress_bar["value"] = 100

            dirs_str = "\n".join(f"  {sid}: {self.subject_dirs[sid]}" for sid in self.subject_ids)
            messagebox.showinfo("Done", f"Data saved:\n{dirs_str}")

        self.root.after(0, update)

    # ==================== MISC ====================

    def _play_beep(self, frequency=1000, duration=200):
        if WINSOUND_AVAILABLE:
            try:
                winsound.Beep(frequency, duration)
            except:
                pass

    def _toggle_pause(self):
        if not self.is_running:
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            for sid in self.subject_ids:
                if sid in self.loggers:
                    self.loggers[sid].log_event("PAUSED", {})
            self.status_label.config(text="PAUSED", fg="#ffcc00")
            self.phase_label.config(text="PAUSED", fg="#ffcc00")
        else:
            for sid in self.subject_ids:
                if sid in self.loggers:
                    self.loggers[sid].log_event("RESUMED", {})

    def _on_escape(self):
        if self.is_running:
            if messagebox.askyesno("Stop?", "Stop the experiment for ALL subjects?"):
                self.is_running = False
                self.is_paused = False

                # Snapshot any unsaved data from current gesture into chunks
                if hasattr(self, '_rep_data_chunks'):
                    for sid in self.subject_ids:
                        if sid in self.devices:
                            chunk = self.devices[sid].snapshot_and_clear()
                            self._rep_data_chunks.setdefault(sid, []).append(chunk)

                # Save current (partial) repetition data
                if self.current_repetition > 0:
                    self._save_all_repetition_data(self.current_repetition)

                for sid in self.subject_ids:
                    if sid in self.loggers:
                        self.loggers[sid].log_event("CANCELLED", {})
                        self.loggers[sid].save_session_summary(self.current_repetition)
                self._stop_all_devices()
        else:
            # Not running experiment - check if we're in monitoring phase
            if self.experiment_started and self.devices:
                for sid in self.subject_ids:
                    if sid in self.devices:
                        collector = self.devices[sid]
                        if collector.exg_count > 0:
                            data_dir = os.path.join(self.subject_dirs[sid], "repetitions")
                            os.makedirs(data_dir, exist_ok=True)
                            npz_file = os.path.join(data_dir, "monitoring_data.npz")

                            exg_arr = np.array(collector.exg_data, dtype=np.uint16) if collector.exg_data else np.zeros((0, 4), dtype=np.uint16)
                            ts_arr = np.array(collector.exg_timestamps, dtype=np.float64)
                            seq_arr = np.array(collector.exg_seq, dtype=np.uint32)

                            np.savez(
                                npz_file,
                                exg=exg_arr,
                                exg_timestamps=ts_arr,
                                exg_seq=seq_arr,
                                fs_exg=EXG_FS,
                                columns_exg=['ecg_chest', 'ecg_wrist', 'emg1', 'emg2'],
                                subject_id=sid,
                                repetition=0,
                            )
                            print(f"✅ [{sid}] Monitoring data saved: {exg_arr.shape} → {npz_file}")

            self._stop_all_devices()
            self.root.quit()
            self.root.destroy()

    def run(self):
        self.root.mainloop()


# ==================== MAIN ====================

def main():
    n_actions = CONFIG['actions_per_gesture']
    on = CONFIG['on_duration']
    off = CONFIG['off_duration']
    trans = CONFIG['transition_duration']

    print("=" * 60)
    print("  Gesture Recording v4 - Multi-Device")
    print("=" * 60)
    print(f"\n{len(GESTURES)} gestures × {n_actions} actions/gesture × {CONFIG['num_repetitions']} reps")
    print(f"EXG only (4 channels: ECG Chest, ECG Wrist, EMG1, EMG2)")
    print(f"Supports up to 4 simultaneous USB devices")
    print(f"\nPer-gesture flow:")
    print(f"  GetReady({trans}s) → [ON({on}s) → OFF({off}s)] × {n_actions}")
    print(f"  → Transition({trans}s) → Hold({CONFIG['hold_duration']}s) → FreeForm({CONFIG['freeform_duration']}s)")
    print(f"\nDemo session: {CONFIG['demo_gestures']} gestures (optional, no recording)")
    print(f"After each repetition, a dialog asks whether to continue.")
    print(f"\nData Structure (per subject):")
    print(f"  recordings/")
    print(f"    [subject_id]_[timestamp]/")
    print(f"      repetitions/   <- .npz files (exg only)")
    print(f"      events/        <- .json files")
    print(f"      session_info.json")
    print(f"\nControls:")
    print(f"  ENTER=Start | SPACE=Pause | W=Waveform | 1-4=Switch Subject | ESC=Exit")
    print("=" * 60)

    app = MultiDeviceRecorderApp(CONFIG)
    app.run()


if __name__ == "__main__":
    main()