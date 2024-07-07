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
        self.prev_frame_gray = cv2.GaussianBlur(cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        contour_img = np.zeros((self.height, self.width), np.uint8)
        frame_diff = cv2.absdiff(self.prev_frame_gray, gray)
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Use dilated then eroded to close small holes in the foreground
        kernel = np.ones((5, 5),np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)

        self.prev_frame_gray = gray
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)

        cv2.line(frame, self.line_position[0], self.line_position[1], (255,127,0), 3) 
        combined_frame = cv2.hconcat([frame, cv2.merge([contour_img, contour_img, contour_img])])
        self.video_writer.write(combined_frame)
        return contours