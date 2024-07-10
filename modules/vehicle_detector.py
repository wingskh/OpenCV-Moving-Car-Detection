from abc import ABC, abstractmethod
import os
import cv2
import numpy as np
from dotenv import load_dotenv
from helpers.bbox_helper import is_overlapped
from modules.bounding_box import BoundingBox
load_dotenv()

MIN_CAR_WIDTH = MIN_CAR_HEIGHT = 35
MIN_CAR_AREA = 2500
LINE_OFFSET = 20
FPS_DURATION = 2

class VehicleDetector(ABC):
    def __init__(self, frame, fps=30, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), output_video_name='output'):
        self.output_path = os.getenv('OUTPUT_PATH')
        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)
        self.detection_type = ''
        self.video_path = os.getenv('VIDEO_PATH')
        self.background_image_path = os.getenv('BACKGROUND_IMAGE_PATH')
        self.min_car_width = MIN_CAR_WIDTH
        self.min_car_height = MIN_CAR_HEIGHT
        self.min_car_area = MIN_CAR_AREA
        self.frame = frame
        self.height, self.width, _ = self.frame.shape
        self.line_y = int(self.height * 2 / 3)
        self.line_position = (0, self.line_y), (self.width, self.line_y)
        self.prev_detected_bbox = []
        self.video_writer = cv2.VideoWriter(
            os.path.join(self.output_path, f'{output_video_name}.mp4'), 
            fourcc, 
            fps, 
            (self.width*3, self.height)
        )
        self.total_car = 0

    @abstractmethod
    def get_contours(self, frame):
        raise NotImplementedError("Subclass must implement abstract method: get_contours")
    
    def detect(self, frame):
        ori_frame = frame.copy()
        contours = self.get_contours(frame)
        boxes = [cv2.boundingRect(contour) for contour in contours]

        center_list = []
        cur_detected_bbox = []

        cv2.line(frame, self.line_position[0], self.line_position[1], (255,127,0), 3) 
        for (x, y, w, h) in boxes:
            contour_bbox = BoundingBox(x, y, w, h)
            # Put area in the frame
            # cv2.putText(frame, "Area: "+str(contour_bbox.get_area())+f' {w} {h}', contour_bbox.get_center(), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),2)
            if contour_bbox.get_area() < MIN_CAR_AREA or w < MIN_CAR_WIDTH or h < MIN_CAR_HEIGHT:
                continue

            cv2.rectangle(frame, contour_bbox.get_left_top(), contour_bbox.get_right_bottom(), (0, 255, 0), 2)
            cv2.circle(frame, contour_bbox.get_center(), 4, (0, 0,255), -1)
            center_y = contour_bbox.get_center()[1]
            if center_y < (self.line_y + LINE_OFFSET) and center_y > (self.line_y - LINE_OFFSET):
                if not is_overlapped(self.prev_detected_bbox, contour_bbox):
                    self.total_car += 1
                    cv2.rectangle(frame, contour_bbox.get_left_top(), contour_bbox.get_right_bottom(), (0, 0, 255), 2)
                    cv2.circle(frame, contour_bbox.get_center(), 4, (0, 0, 255), -1)
                    cv2.line(frame, self.line_position[0], self.line_position[1], (0, 127, 255), 3) 
                    cv2.circle(frame, contour_bbox.get_center(), 4, (0, 255, 0), -1)
                    center_list.append(contour_bbox.get_center())
                
                cur_detected_bbox.append(contour_bbox)

        self.prev_detected_bbox = cur_detected_bbox

        text = f"Vehicle Count: {str(self.total_car)}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        start_x = (self.width - text_size[0]) // 2
        cv2.putText(frame, text, (start_x, int(self.height * 0.15)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        text = f"Detect Moving Vehicles by {self.detection_type} Filter"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        start_x = (self.width - text_size[0]) // 2
        cv2.putText(frame, text, (start_x, int(self.height * 0.075)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        contour_img = np.zeros((self.height, self.width), np.uint8)
        contour_img = cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)
        cv2.line(frame, self.line_position[0], self.line_position[1], (255,127,0), 3) 
        combined_frame = cv2.hconcat([ori_frame, frame, cv2.merge([contour_img, contour_img, contour_img])])
        self.video_writer.write(combined_frame)
        return frame

    def close(self):
        self.video_writer.release()
