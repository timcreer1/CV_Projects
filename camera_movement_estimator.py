import pickle
import cv2
import numpy as np
import os
import sys
from utils import measure_distance, measure_xy_distance

#creating function to counteract the camera movement
class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5
        #params for optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        #extracting top of image and bottom of image for features to track
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1 #getting first 20 rows of pixels
        mask_features[:, 900:1050] = 1 #getting last 150 rows of pixels
        #dictionary for params
        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features)

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            # Looping over frame number and tracks
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    if 'position' in track_info:
                        position = track_info['position']
                        camera_movement = camera_movement_per_frame[frame_num]
                        # Subtracting x and y position of player from camera movement
                        position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                        tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
                    else:
                        print(
                            f"Warning: 'position' key not found in track_info for object {object}, frame {frame_num}, track_id {track_id}")
                        # Handle missing 'position' key as needed
                        # Example: Skip this track info or set a default position
                        continue

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read from stubs to save computation
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        #movement for x and y axis
        camera_movement = [[0, 0]] * len(frames)
        #convert image to grey image to extract featyres
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        #extracting corner features
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features) #** to expand the dict
        #looping over frames to extract features
        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            #using optical flow giving to measure distance between old features and new features
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0
            #as list in list need zip
            for i, (new, old) in enumerate(zip(new_features, old_features)):

                new_features_point = new.ravel()
                old_features_point = old.ravel()
                #finding if camera moved
                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)
            #finding if camera didn't move
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    #displaying the x and y movement
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        #looping over frames and calculating movement
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 0), 3)

            output_frames.append(frame)

        return output_frames