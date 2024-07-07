import cv2
import os
from modules.vehicle_detector_by_background import VehicleDetectorByBackground
from modules.vehicle_detector_by_movement import VehicleDetectorByMovement
import numpy as np

MIN_CAR_WIDTH = MIN_CAR_HEIGHT = 28
MIN_CAR_AREA = 5000
LINE_OFFSET = 10
FPS_DURATION = 2

# Check if the input bbox is overlapped more than 50% with any bbox in the previous frame 

def clear_console():
    if os.name == 'nt':  # for Windows
        os.system('cls')
    else:  # for Linux/OS X
        os.system('clear')

def show_processing_percentage(percentage):
    clear_console()
    print(f'Progress in percentage: {min(percentage, 100):.2f}%')

def start_count():
    video_path = os.getenv('VIDEO_PATH')
    output_path = os.getenv('OUTPUT_PATH')
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS) / FPS_DURATION)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    if not ret:
        return
    
    # _ is the number of channels
    height, width, _ = first_frame.shape

    processed_frame_count = 1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    combined_out = cv2.VideoWriter(os.path.join(output_path, 'combined_output.mp4'), fourcc, fps, (width*3, height))
    vehicle_detector_by_background = VehicleDetectorByBackground(first_frame, fps=fps)
    vehicle_detector_by_movement = VehicleDetectorByMovement(first_frame, fps=fps)

    while cap.isOpened():
        ret, frame = cap.read()
        processed_frame_count += 1
        if not ret:
            break
        if processed_frame_count % FPS_DURATION != 0:
            continue
        show_processing_percentage(processed_frame_count/total_frames*100)
        frame_detected_by_background = vehicle_detector_by_background.detect(frame.copy())
        frame_detected_by_movement = vehicle_detector_by_movement.detect(frame.copy())
        combined_frame = cv2.hconcat([frame, frame_detected_by_background, frame_detected_by_movement])
        combined_out.write(combined_frame)

    vehicle_detector_by_background.close()
    vehicle_detector_by_movement.close()
    clear_console()
    print(f'Total Moving Vehicles Detected by Background Filter: {vehicle_detector_by_background.total_car}')
    print(f'Total Moving Vehicles Detected by Movement Filter: {vehicle_detector_by_movement.total_car}')
    cap.release()
    combined_out.release()

if __name__ == '__main__':
    total_car = start_count()