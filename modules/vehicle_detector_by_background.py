from modules.vehicle_detector import VehicleDetector
import cv2
from dotenv import load_dotenv
import os
from matplotlib import pyplot as plt
import numpy as np

OUTPUT_FILE_NAME = 'output_by_background'
DETECTION_TYPE = 'Background'
load_dotenv()

class VehicleDetectorByBackground(VehicleDetector):
    def __init__(self, frame, fps=30, fourcc=cv2.VideoWriter_fourcc(*'mp4v')):
        super().__init__(frame, fps, fourcc, OUTPUT_FILE_NAME)
        self.detection_type = DETECTION_TYPE
        self.background_img = cv2.imread(self.background_image_path) if os.path.exists(self.background_image_path) else self.extract_background()

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
        gray_background = cv2.cvtColor(self.background_img, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        blurred_background = cv2.GaussianBlur(gray_background, (21, 21), 0)
        blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        
        foreground_mask = cv2.absdiff(blurred_background, blurred_frame)
        _, foreground_mask = cv2.threshold(foreground_mask, 25, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(foreground_mask, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        canny = cv2.Canny(eroded, 150, 200)
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours
