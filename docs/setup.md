# Development Environment Setup

This document provides instructions for setting up the development environment
for FitFighter.

## Prerequisites

- Python 3.8+ for MediaPipe
- Unity 2022.3 LTS or newer
- Git
- A mobile device with a camera for testing

## MediaPipe Setup

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r motion-detection/requirements.txt
   ```

## Unity Setup

1. Download and install Unity Hub from [unity.com](https://unity.com/download)
2. Install Unity 2022.3 LTS through Unity Hub
3. Open the project by selecting the `/unity` directory in Unity Hub

## Development Tools Recommendations

- Visual Studio Code with Python and C# extensions
- Jupyter Notebook for prototyping MediaPipe algorithms
- Git for version control
- ADB (Android Debug Bridge) for Android testing

## Testing Setup

- For mobile testing, enable Developer Mode on your device
- Connect your device via USB or set up wireless debugging
- Ensure your device has sufficient performance capabilities (minimum
  requirements TBD during research phase)

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd fitfighter
   ```

2. Set up the Python environment as described above
3. Open the Unity project
4. Follow the proof of concept guide in `/docs/poc.md`
