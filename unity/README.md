# FitFighter Unity Project

This directory contains the Unity project for FitFighter, which receives motion
data from the Python motion detection module and visualizes it in an interactive
game environment.

## Setup Instructions

### Prerequisites

1. Unity 2022.3 LTS or newer
2. TextMeshPro package
3. NativeWebSocket package

### Setting Up the Unity Project

1. **Create a new Unity project**:
   - Open Unity Hub
   - Click "New Project"
   - Select "3D" template
   - Name it "FitFighter"
   - Set the location to this `/unity` directory
   - Click "Create Project"

2. **Install required packages**:
   - In Unity, go to Window > Package Manager
   - Click the "+" button and select "Add package from git URL..."
   - Add the NativeWebSocket package:
     `https://github.com/endel/NativeWebSocket.git`
   - Install TextMeshPro package if not already included

3. **Import the scripts**:
   - Create a "Scripts" folder in your Assets directory
   - Copy the provided C# scripts into this folder:
     - `MotionDataReceiver.cs`
     - `MotionControlledObject.cs`
     - `ConnectionStatusUI.cs`

4. **Set up the scene**:
   - Create a new scene or use the default one
   - Add a cube to the scene (this will be controlled by your movements)
   - Add the `MotionControlledObject` component to the cube
   - Create an empty GameObject and add the `MotionDataReceiver` component to it
   - In the Inspector, drag the `MotionDataReceiver` into the field on the
     `MotionControlledObject`

5. **Set up the UI**:
   - Create a Canvas for UI elements
   - Add TextMeshPro elements for status text and instructions
   - Create a status panel
   - Add a reconnect button
   - Add the `ConnectionStatusUI` component to the Canvas
   - Configure the references in the Inspector

### Running the Application

1. Start the Python motion detection application with the `--unity-mode` flag:
   ```bash
   cd motion-detection
   python main.py --unity-mode
   ```

2. Run the Unity project
   - The cube should respond to your movements detected by the Python
     application
   - Punch: The cube moves forward
   - Squat: The cube moves down
   - Plank: The cube stabilizes

## Project Structure

- `Scripts/`: Contains the C# scripts for the application
  - `MotionDataReceiver.cs`: Handles WebSocket communication with the Python
    application
  - `MotionControlledObject.cs`: Controls the game object based on motion data
  - `ConnectionStatusUI.cs`: Manages the UI for connection status and
    instructions

## Customization

- Adjust the movement parameters in the `MotionControlledObject` component:
  - `Punch Force`: How far the object moves when a punch is detected
  - `Squat Height`: How far down the object moves when a squat is detected
  - `Return Speed`: How quickly the object returns to its original position
  - `Plank Stabilization Factor`: How much the object's movement is reduced when
    a plank is detected

## Troubleshooting

- If the Unity application can't connect to the Python WebSocket server:
  - Ensure the Python application is running with the `--unity-mode` flag
  - Check that the WebSocket URL in the `MotionDataReceiver` component matches
    the Python server address
  - Try clicking the reconnect button in the UI
  - Check your firewall settings if running on different machines

## Communication with Motion Detection

The Unity project will communicate with the motion detection module through:

1. Socket communication (WebSockets or UDP)
2. JSON data format for motion data
3. Real-time data processing and visualization

## Development Guidelines

- Use the Unity Input System for controls
- Follow the MVC pattern where applicable
- Implement event-based communication between components
- Optimize for mobile performance
