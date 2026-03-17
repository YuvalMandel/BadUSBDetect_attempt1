BadUSB Detector - Program Architecture

## Setup & Installation

### Step 1 — Install htm.core from source

**`pip install htm.core` from PyPI does not work reliably. You must clone and build from source.**

#### Linux

```bash
# Install system build dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev git

# Clone and build htm.core
git clone https://github.com/htm-community/htm.core
cd htm.core
pip install -v .
cd ..
```

#### Windows

**One-time prerequisites:**

1. Install [Visual Studio 2022 Community](https://visualstudio.microsoft.com/vs/community/) (free).
   During installation select the **"Desktop development with C++"** workload
   (includes MSVC compiler and CMake).

2. Install [Git for Windows](https://git-scm.com/download/win) if not already installed.

3. Use **Python 3.8–3.10** (64-bit). htm.core does not build on Python 3.11+.
   Verify: `python --version`

**Build steps** (run in a standard Command Prompt or PowerShell — **not** Git Bash):

```bat
git clone https://github.com/htm-community/htm.core
cd htm.core
pip install -v .
cd ..
```

> If you get `cl.exe not found`, open the **"x64 Native Tools Command Prompt for VS 2022"**
> from the Start menu and re-run `pip install -v .` from inside the `htm.core` folder.

### Step 2 — Install remaining dependencies

```bash
pip install torch numpy scipy pynput
```

`tkinter` is included with Python on Windows and most Linux distributions. On Ubuntu/Debian:
```bash
sudo apt-get install -y python3-tk
```

### Step 3 — Run the detector

```bash
# From the project root (BadUSBDetect_attempt1/)
python visual_app/bad_usb_detector.py
```

The HTM model is optional — if `htm.core` is not installed the app falls back to MLP/GRU only.

---

Overview
This program detects BadUSB attacks by analyzing keyboard typing patterns using machine learning. It monitors keystroke dynamics (dwell time and flight time) in real-time and classifies typing behavior as either human or automated bot/script.

Core Architecture
1. Multi-Threaded Design
The program uses three concurrent threads for efficient real-time processing:

text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Input Listener  │───▶│   Processor     │───▶│      GUI        │
│    (Thread)     │    │    (Thread)     │    │  (Main Thread)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
       │                       │                       │
       ▼                       ▼                       ▼
 raw_events_queue     control_queue            gui_update_queue
                    (for reset commands)
Thread Responsibilities:
Input Listener Thread: Captures keyboard events (press/release) with timestamps

Processor Thread: Analyzes typing patterns, runs ML inference, detects attacks

GUI Thread: Manages user interface, displays status and alerts

2. Data Flow
text
Keyboard Events → Raw Queue → Feature Extraction → ML Models → GUI Updates
                                      ↓
                            Sliding Window (size=15)
                                      ↓
                            Dwell Time Buffer
                            Flight Time Buffer
3. Machine Learning Models
MLP Model (Statistical + Neural Network)
Input: 14 statistical features (7 from dwell + 7 from flight times)

Features include: mean, median, std, skewness, kurtosis, KS-statistics, Wasserstein distance

Architecture: 3-layer neural network (64→32→16→1)

Detection Logic: Requires 3 consecutive positive detections

Purpose: Traditional feature-based classification

GRU Model (Sequence-based)
Input: Raw sequences of (dwell, flight) pairs (window size 15 × 2 features)

Architecture: GRU layer (hidden_dim=64) + fully connected layer

Detection Logic: Single positive detection triggers alarm

Purpose: Direct sequence analysis without manual feature engineering

4. Key Data Structures
Buffers
Dwell Buffer: Stores key press durations (time between press and release)

Flight Buffer: Stores intervals between key releases

Window Size: 15 events (configurable)

Queues
raw_events_queue: Passes keyboard events to processor

gui_update_queue: Sends status updates to GUI

control_queue: Handles reset commands from GUI

5. Detection Pipeline
Event Collection

Capture KeyDown/KeyUp events with millisecond timestamps

Calculate dwell times (KeyUp - KeyDown)

Calculate flight times (KeyDown - last KeyUp)

Feature Extraction (MLP only)

Extract statistical features from windows

Compare against reference pools using KS-test and Wasserstein distance

Normalize features using pre-trained scalers

Model Inference

MLP: Process 14 features through neural network

GRU: Process sequence through GRU network

Output: Probability score (0-1)

Decision Logic

MLP: 3 consecutive probabilities > 0.8 → Alarm

GRU: Single probability > 0.8 → Alarm

Probability 0.5-0.8 → Suspicious (yellow indicator)

Probability < 0.5 → Human (green indicator)

GUI Feedback

Color-coded status light (green/yellow/red)

Real-time statistics display

Model selection radio buttons

Reset button for clearing alarm state

6. Reset Mechanism
User-initiated reset clears all buffers and active keys

Resets consecutive strike counter

Returns GUI to monitoring state

Enables fresh start after false positive detection

Dependencies
pynput: Keyboard event capture

torch: Neural network inference

numpy: Numerical operations

scipy: Statistical tests (KS-test, skewness, kurtosis)

tkinter: GUI framework

Configuration
WINDOW_SIZE: Sliding window size (default: 15)

Model paths and scaler files configurable via constants

Device auto-detection (CUDA if available, else CPU)
