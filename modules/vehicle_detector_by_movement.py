from modules.vehicle_detector import VehicleDetector
import cv2
from dotenv import load_dotenv
import os
from matplotlib import pyplot as plt
import numpy as np

OUTPUT_FILE_NAME = 'output_by_movement'
DETECTION_TYPE = 'Movement'
load_dotenv()

class VehicleDetectorByMovement(VehicleDetector):
    def __init__(self, frame, fps=30, fourcc=cv2.VideoWriter_fourcc(*'mp4v')):
        super().__init__(frame, fps, fourcc, OUTPUT_FILE_NAME)
        self.detection_type = DETECTION_TYPE
        # Use only the saturation channel
        s_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
        self.prev_blurred = cv2.GaussianBlur(s_channel, (21, 21), 0)
        
    def extract_background(self):
        cap = cv2.VideoCapture(self.video_path)
        backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            backSub.apply(frame)
            
        background_img = backSub.getBackgroundImage()
        cv2.imwrite(self.background_image_path, background_img)
        return background_img
    
    def get_contours(self, frame):
        s_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
        blurred = cv2.GaussianBlur(s_channel, (21, 21), 0)

        frame_diff = cv2.absdiff(self.prev_blurred, blurred)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # Use dilated then eroded to close small holes in the foreground
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        canny = cv2.Canny(eroded, 150, 200)
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.prev_blurred = blurred

        return contours