# 🎯 Virtual Mouse Project

This project showcases an innovative approach to hands-free interaction with a computer, utilising hand gestures to perform tasks such as:

- 🖱️ Moving the mouse cursor
- 🖱️ Clicking
- 🔊 Adjusting the volume

## 📄 Overview

This Python-based implementation enables control of a computer’s mouse and volume through hand gestures, leveraging advanced computer vision techniques.

### 🔍 Key Components

- **Hand Tracking Module**:  
  A custom `handDetector` class, built using MediaPipe, detects and tracks hand landmarks in real-time. The class provides methods for:
  - Identifying hands in an image
  - Finding the positions of specific hand landmarks
  - Determining which fingers are up
  - Calculating the distance between two landmarks
  
  The `main()` function captures video from the webcam, processes the frames with the `handDetector`, and displays the real-time video feed with annotated hand landmarks and frame rate.

- **Volume Hand Control**:  
  The `set_volume` function is responsible for adjusting the system volume on a macOS device.

- **Virtual Mouse**:  
  This component captures video from the webcam, detects hand landmarks using the hand detection module, and interprets specific gestures to:
  - Control the mouse cursor
  - Perform clicks
  - Adjust the volume

  It also provides real-time visual feedback, including a volume bar displayed on the screen.

## 🛠️ Requirements

Ensure the following packages are installed before running the code:

```bash
pip install cv2 numpy pyautogui mediapipe
