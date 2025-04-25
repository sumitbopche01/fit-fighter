# Migration Guide

This document provides instructions for migrating from the previous codebase
structure (`motion_detection` and `fitfighter` folders) to the new consolidated
structure.

## Project Structure Changes

### Old Structure

The old codebase was split across two directories:

1. `motion_detection/` - Contains the main application code, camera utilities,
   and some detectors
2. `fitfighter/` - Contains a more modular structure with detectors, core
   components, and utilities

### New Structure

The new structure consolidates the code into a single package with a clearly
defined structure:

```
fitFighter/
├── docs/            # Documentation files
├── src/             # Source code
│   ├── core/        # Core components
│   ├── detectors/   # Exercise detectors
│   ├── utils/       # Utility functions
│   └── constants/   # Constant values
├── tests/           # Test files
└── requirements.txt # Dependencies
```

## Major Changes

1. **Package Structure**: The code now follows a standard Python package
   structure with proper separation of concerns.
2. **Imports**: All imports are now relative to the `fitFighter` package.
3. **Detector Management**: Uses the improved `ExerciseDetectorManager` from the
   original `fitfighter` codebase.
4. **Main Application**: Combines the best features from both
   `motion_detection/main.py` and detector management from the `fitfighter`
   package.

## Using the New Package

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/fitFighter.git
cd fitFighter

# Install the package in development mode
pip install -e .
```

### Running the Application

```bash
# Run directly
python -m fitFighter.src.main

# Or use the console script
fitfighter
```

## Migrating Custom Code

If you've developed custom detectors or extensions for the old codebase:

1. Create a new detector in `src/detectors/` following the pattern of existing
   detectors.
2. Make sure it inherits from `ExerciseDetector` in `src/core/base_detector.py`.
3. Update imports to match the new package structure.
4. Add your detector to the detector manager in `src/core/detector_manager.py`.

## Removed Duplications

The following duplicate files have been consolidated:

1. `motion_detection/jumping_jack_detector.py` and
   `motion_detection/exercise_detection/jumping_jack_detector.py` - Merged into
   a single detector
2. Multiple versions of exercise detectors across both directories - The more
   complete versions from `fitfighter/detectors/` are now used
3. `motion_analyzer.py` functionality - Replaced with the more robust
   `ExerciseDetectorManager`
