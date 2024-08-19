import cv2
import numpy as np
import pyautogui
import HandTrackingModule as htm
import time
import math
import subprocess

##########################
wCam, hCam = 640, 480
frameR = 50  # Frame Reduction
smoothening = 4
#########################
volRange = (0.0, 100.0)  # Volume range 0 to 100 to match macOS volume level

volBar = 400
volPer = 0
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
detector = htm.handDetector(maxHands=1)
wScr, hScr = pyautogui.size()  # Getting the screen size using pyautogui

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Get the current volume level from the system
    current_volume = int(subprocess.check_output(['osascript', '-e', 'output volume of (get volume settings)']).strip())
    volBar = np.interp(current_volume, [0, 100], [400, 150])  # Adjust the volume bar based on current volume
    volPer = current_volume  # Set volume percentage to the current system volume

    # 2. Get the tip of the index, middle fingers and thumb
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # index
        x2, y2 = lmList[12][1:]  # middle
        x4, y4 = lmList[4][1], lmList[4][2]  # thumb

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (75, 0, 130), 2)

        # 4. Only Index Finger: Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move Mouse
            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 10, (75, 0, 130), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 8. Both Index and middle fingers are up: Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)

            # 10. Click mouse if distance short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 5, (0, 255, 0), cv2.FILLED)
                pyautogui.click()

        # 11. Thumb and index are up: change volume
        if fingers[0] == 1 and fingers[1] == 1:
            cx, cy = (x4 + x1) // 2, (y4 + y1) // 2

            cv2.circle(img, (x4, y4), 8, (75, 0, 130), cv2.FILLED)
            cv2.circle(img, (x1, y1), 8, (75, 0, 130), cv2.FILLED)
            cv2.line(img, (x4, y4), (x1, y1), (75, 0, 130), 2)
            cv2.circle(img, (cx, cy), 8, (75, 0, 130), cv2.FILLED)

            length = math.hypot(x1 - x4, y1 - y4)

            vol = np.interp(length, [50, 300], [0, 100])
            volBar = np.interp(length, [50, 300], [400, 150])
            volPer = np.interp(length, [50, 300], [0, 100])

            # Set the system volume using AppleScript
            if length < 50:
                cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)
                new_volume = max(0, current_volume - 5)  # Decrease volume by 5, ensuring it doesn't go below 0
                subprocess.call(['osascript', '-e', f'set volume output volume {new_volume}'])

            if length > 180:
                cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)
                new_volume = min(100, current_volume + 5)  # Increase volume by 5, ensuring it doesn't go above 100
                subprocess.call(['osascript', '-e', f'set volume output volume {new_volume}'])

            # Get the updated system volume level
            current_volume = int(
                subprocess.check_output(['osascript', '-e', 'output volume of (get volume settings)']).strip())
            volPer = current_volume
            volBar = np.interp(volPer, [0, 100], [400, 150])

        # Draw Volume Bar
        cv2.rectangle(img, (12, 150), (35, 400), (0, 255, 0), 2)  # The outline of the volume bar
        cv2.rectangle(img, (12, int(volBar)), (35, 400), (0, 255, 0), cv2.FILLED)  # The filled part of the volume bar
        cv2.putText(img, f'{int(volPer)}%', (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(10)
