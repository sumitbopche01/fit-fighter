"""
Data sender module for FitFighter.

This module handles the communication between the motion detection and Unity game engine
using WebSockets for real-time data transfer.
"""

import json
import time
import asyncio
import websockets
from datetime import datetime


class DataSender:
    """Handles sending motion data to the Unity game engine."""

    def __init__(self, host="127.0.0.1", port=5678):
        """
        Initialize the data sender.

        Args:
            host: WebSocket server host (default: 127.0.0.1)
            port: WebSocket server port (default: 5678)
        """
        self.host = host
        self.port = port
        self.server = None
        self.clients = set()
        self.running = False
        self.last_sent_time = time.time()
        self.data_queue = asyncio.Queue()

    async def start_server(self):
        """Start the WebSocket server."""
        try:
            self.running = True
            self.server = await websockets.serve(
                self._handle_client, self.host, self.port
            )
            print(f"WebSocket server started at ws://{self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to start WebSocket server: {str(e)}")
            self.running = False

    async def send_data(self, landmarks, exercise_states, fps):
        """
        Send motion data to connected clients.

        Args:
            landmarks: Pose landmarks (list of x, y, z, visibility tuples)
            exercise_states: Dictionary of exercise states
            fps: Current frames per second
        """
        if not self.clients:
            return

        current_time = time.time()
        latency_ms = int((current_time - self.last_sent_time) * 1000)
        self.last_sent_time = current_time

        # Prepare landmark data
        landmark_data = []
        if landmarks:
            # Map to named landmarks based on index
            landmark_names = [
                "nose",
                "left_eye_inner",
                "left_eye",
                "left_eye_outer",
                "right_eye_inner",
                "right_eye",
                "right_eye_outer",
                "left_ear",
                "right_ear",
                "mouth_left",
                "mouth_right",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_pinky",
                "right_pinky",
                "left_index",
                "right_index",
                "left_thumb",
                "right_thumb",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
                "left_heel",
                "right_heel",
                "left_foot_index",
                "right_foot_index",
            ]

            for i, (x, y, z, confidence) in enumerate(landmarks):
                if i < len(landmark_names):
                    landmark_data.append(
                        {
                            "name": landmark_names[i],
                            "x": x,
                            "y": y,
                            "z": z,
                            "confidence": confidence,
                        }
                    )

        # Prepare message
        message = {
            "timestamp": int(datetime.now().timestamp() * 1000),
            "exercises": exercise_states,
            "landmarks": landmark_data,
            "performance": {"fps": fps, "latency_ms": latency_ms},
        }

        # Convert to JSON
        json_message = json.dumps(message)

        # Send to all clients
        await self.data_queue.put(json_message)

    async def _send_to_clients(self):
        """Send data from the queue to all connected clients."""
        while self.running:
            try:
                message = await self.data_queue.get()

                # Send to all clients
                disconnected_clients = set()
                for client in self.clients:
                    try:
                        await client.send(message)
                    except websockets.exceptions.ConnectionClosed:
                        disconnected_clients.add(client)

                # Remove disconnected clients
                self.clients -= disconnected_clients

                self.data_queue.task_done()
            except Exception as e:
                print(f"Error sending data: {str(e)}")

    async def _handle_client(self, websocket, path):
        """
        Handle WebSocket client connection.

        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        print(f"Client connected: {websocket.remote_address}")
        self.clients.add(websocket)

        try:
            # Keep connection alive
            async for message in websocket:
                # Just echo client messages for now
                await websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {websocket.remote_address}")
        finally:
            self.clients.remove(websocket)

    async def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.running = False
            self.server.close()
            await self.server.wait_closed()
            print("WebSocket server stopped")

    def __del__(self):
        """Clean up resources."""
        if self.running:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.stop())
