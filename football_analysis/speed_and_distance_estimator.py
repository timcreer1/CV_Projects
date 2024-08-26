import cv2
import sys
from utils import measure_distance, get_foot_position

#to calculate the speed of a player and distance ran
class SpeedAndDistance_Estimator():
    def __init__(self):
        # calculating the speed of the player every 5 frames
        self.frame_window = 5
        self.frame_rate = 24
    #function to add the calculations to the tracjs
    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        for object, object_tracks in tracks.items():
            # detect only the players
            if object == "ball" or object == "referees":
                continue
            number_of_frames = len(object_tracks)
            #detect the speed and distance by looping over frames batches
            for frame_num in range(0, number_of_frames, self.frame_window):
                #ensuring we don't go out of bounds
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)
                #if track is in first frame but not in last then don't do anything to that player
                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue
                    #if player is in first and last frame of the batch get positions
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue
                    #calculate the distance, time and speed
                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_meteres_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meteres_per_second * 3.6
                    #make the player into dictionary
                    if object not in total_distance:
                        total_distance[object] = {}

                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    #adding the distance to the player
                    total_distance[object][track_id] += distance_covered
                    #creating labels for the speed and distance
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        if speed is None or distance is None:
                            continue

                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40

                        position = tuple(map(int, position))
                        # Format speed and distance as whole numbers
                        cv2.putText(frame, f"{int(speed)} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{int(distance)} m", (position[0], position[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            output_frames.append(frame)

        return output_frames
