# Virtual Mouse Project
This project demonstrates an innovative approach to hands-free interaction with a computer, using hand gestures to perform tasks such as moving the mouse cursor, clicking, and adjusting the volume.

## Overview
This project is a Python-based implementation for controlling a computer’s mouse and volume through hand gestures, utilising computer vision techniques. The system uses OpenCV to capture video from the webcam and MediaPipe (via a custom HandTrackingModule) to detect and track hand landmarks in real-time. 

**Hand Tracking Module**: The code defines a `handDetector` class using MediaPipe to detect and track hand landmarks in real-time video feeds. The class includes methods for identifying hands in an image, finding the positions of specific hand landmarks, determining which fingers are up, and calculating the distance between two landmarks. The `main()` function captures video from the webcam, utilises the `handDetector` to process the frames, and displays the real-time video feed with hand landmarks and the calculated frame rate.

**Volume Hand Control**: The code defines a `set_volume` function that adjusts the system volume on a macOS device.

**Virtual Mouse**: The code captures video from a webcam, detects hand landmarks using a hand detection module, and interprets specific gestures to control the mouse cursor and system volume on macOS. It updates the cursor position, performs clicks, and adjusts the volume based on hand movements, with real-time visual feedback and a volume bar displayed on the screen.

## Requirements
Before running the code in this project, make sure you have installed the following packages:

* cv2
* numpy
* pyautogui
* HandTrackingModule
* time
* math
* subprocess
* mediapipe

You can install these packages using pip or conda. The code was run using Python 3.11 on a Mac.
