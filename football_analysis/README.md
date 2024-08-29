# ⚽ Football Analysis Project

This project provides a comprehensive approach to analysing football videos using computer vision, machine learning, and deep learning techniques. The focus is on tracking players and the ball, estimating camera movements, transforming perspectives, assigning teams based on jersey colours, and analysing ball possession and player speed. By utilising models such as YOLOv8, the project demonstrates how to extract and visualise key insights from soccer match videos.

## 📄 Overview

The project utilises a range of computer vision and machine learning techniques, including object detection, clustering, and perspective transformation, to analyse football match footage. The focus is on accurately tracking players, referees, and the ball, estimating camera movements, assigning team colours, and calculating player statistics, all while optimising processing efficiency with intermediate results stored in stubs for reuse.

### 🔍 Key Components

- **Input Video**:  
  This folder contains the video of a soccer match, which serves as the basis of this project.

- **Models**:  
  The output best weights model trained on the Roboflow dataset using YOLOv8. This folder contains a link to a Google Drive where the model can be downloaded.

- **Output Videos**:  
  This folder contains a shortened version of the final output video from the project.

- **Stubs**:  
  This folder contains pickle files that save the results of computationally expensive tasks, like camera movement estimation, allowing these results to be reused in future runs without reprocessing to save computation time.

- **Training**:  
  The code involves setting up and training a YOLOv8 object detection model to identify football players and other relevant objects in images from a Roboflow dataset.

- **`__init__`**:  
  This code initialises and imports the various functions, classes, and modules that have been created to be used in the `main.py` file.

- **Camera Movement Estimator**:  
  Defines a class that estimates and adjusts for camera movement in video frames by tracking specific features across frames. It stores the movement data in a pickle file and optionally displays the calculated camera movement on the video frames.

- **Main**:  
  This script brings together all elements to process the input file by tracking players and the ball, estimating camera movement, transforming views, assigning team colours, determining ball possession, calculating speed and distance, and then annotating and saving the processed video with these features.

- **Player Ball Assigner**:  
  Assigns the ball to the nearest player within a specified distance by comparing the distances between the ball and the bounding boxes of all players.

- **Speed and Ball Estimator**:  
  Defines a class that calculates players' speed and distance covered over batches of frames in a video by measuring the distance between transformed positions of players every 5 frames. It annotates these metrics onto the video frames and draws the calculated speed (in km/h) and distance (in metres) on the frames for each player by accessing their bounding boxes and foot positions.

- **Trackers**:  
  Defines a class that uses YOLO for object detection to track players, referees, and the ball in video frames, handles missing ball detections with interpolation, and visualises object positions, player team assignments, and ball possession using custom drawing functions for annotated video output.

- **Team Assigner**:  
  Defines a class that uses K-means clustering to determine player jersey colours from video frames, assigns team colours based on clustering results, and associates each player with a team by comparing their jersey colour to the clustered team colours.

- **Utils**:  
  Contains utility functions used in other areas of the project, e.g., reading and saving video frames, calculating the centre and width of bounding boxes.

- **View Transformer**:  
  Defines a class that uses a perspective transformation to map points from the original video frame to a transformed view of a football field, adjusting the positions of objects in tracking data based on their adjusted positions after correcting for camera movement.

## 🛠️ Requirements

Ensure the following packages are installed before running the code:

```bash
pip install opencv-python numpy pandas pickle-mixin ultralytics supervision roboflow

Note: The code was tested using Python 3.11 on a macOS device.