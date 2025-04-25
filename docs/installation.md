# Installation Guide

This guide will help you set up FitFighter, an exercise detection library, on
your system.

## Prerequisites

Before installing FitFighter, ensure you have the following:

- Python 3.8 or higher
- pip (Python package installer)
- Webcam (for real-time detection)

## Installation Methods

### Option 1: Install from PyPI (Recommended)

The simplest way to install FitFighter is directly from PyPI:

```bash
pip install fitfighter
```

### Option 2: Install from Source

For the latest development version, you can install directly from the GitHub
repository:

```bash
git clone https://github.com/yourusername/fitfighter.git
cd fitfighter
pip install -e .
```

This installs FitFighter in development mode, allowing you to modify the code
and see the changes immediately.

## Dependencies

FitFighter depends on the following main libraries:

- **MediaPipe**: For pose estimation
- **NumPy**: For numerical operations
- **OpenCV**: For image processing and visualization

These dependencies should be automatically installed when you install
FitFighter.

## Verifying Installation

To verify that FitFighter is correctly installed, run the following command in
your Python interpreter:

```python
import fitfighter
print(fitfighter.__version__)
```

This should print the version number without any errors.

## Platform-Specific Notes

### Windows

For Windows users, you might need Microsoft Visual C++ Build Tools to install
some dependencies. You can download it from the
[Microsoft website](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

### MacOS

MacOS users might need to grant camera permissions to run real-time detection
applications:

1. Go to System Preferences > Security & Privacy > Privacy > Camera
2. Ensure your Python interpreter or application has permission to access the
   camera

### Linux

For Linux users, you may need to install additional libraries:

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx
```

## Troubleshooting

### Common Issues

1. **MediaPipe installation fails**:
   - Try installing MediaPipe separately: `pip install mediapipe`
   - Check if your Python version is compatible

2. **Camera access issues**:
   - Ensure your webcam is properly connected
   - Check system permissions for camera access

3. **OpenCV display problems**:
   - Try installing a specific version: `pip install opencv-python==4.5.5.64`

If you encounter any other issues, please report them on our
[GitHub issues page](https://github.com/yourusername/fitfighter/issues).

## Development Installation

If you want to contribute to FitFighter, install the development dependencies:

```bash
pip install -e ".[dev]"
```

This will install additional packages like pytest for testing and flake8 for
linting.
