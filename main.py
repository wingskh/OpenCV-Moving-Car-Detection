import cv2
import matplotlib.pyplot as plt

MIN_CAR_WIDTH = MIN_CAR_HEIGHT = 28
MIN_CAR_AREA = 5000
LINE_OFFSET = 10
FPS_DURATION = 2

class BoundingBox:
    def __init__(self, x, y, w, h):
        self.left_x = x
        self.top_y = y
        self.right_x = x + w
        self.bottom_y = y + h
        self.center_x = x + int(w / 2)
        self.center_y = y + int(h / 2)
    
    def get_center(self):
        return self.center_x, self.center_y
    
    def get_left_top(self):
        return self.left_x, self.top_y
    
    def get_right_bottom(self):
        return self.right_x, self.bottom_y
    
    def get_area(self):
        return (self.right_x - self.left_x) * (self.bottom_y - self.top_y)
    
    def corners_position(self):
        return (self.left_x, self.top_y), (self.right_x, self.bottom_y)

# Check if the input bbox is overlapped more than 50% with any bbox in the previous frame 
def is_overlapped(bbox_list, input_bbox):
    input_area = (input_bbox.right_x - input_bbox.left_x) * (input_bbox.bottom_y - input_bbox.top_y)
    
    for box in bbox_list:
        x_left = max(box.left_x, input_bbox.left_x)
        y_top = max(box.top_y, input_bbox.top_y)
        x_right = min(box.right_x, input_bbox.right_x)
        y_bottom = min(box.bottom_y, input_bbox.bottom_y)
        
        if x_right > x_left and y_bottom > y_top:
            overlap_area = (x_right - x_left) * (y_bottom - y_top)
            if overlap_area >= 0.5 * input_area:
                return True
    return False

class BBoxMap:
    def __init__(self):
        self.bbox_map = {}
    
    def add_bbox(self, bbox):
        self.bbox_map[0] = bbox

    def remove(self, index):
        self.bbox_map.pop(index, None)


def start_count():
    video_path = './video.mp4'
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Use 'avc1' if 'mp4v' does not work
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    _, prev_frame = cap.read()
    # _ is the number of channels
    height, width, _ = prev_frame.shape
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.GaussianBlur(prev_frame_gray, (21, 21), 0)

    processed_frame_count = 1
    line_y = int(height * 2 / 3)
    detected_bbox = BBoxMap()
    out = cv2.VideoWriter('output.mp4', fourcc, int(fps / FPS_DURATION), (width, height))
    combined_out = cv2.VideoWriter('combined_output.mp4', fourcc, int(fps / FPS_DURATION), (width*2, height))
    total_car = 0
    line_position = (0, line_y), (width, line_y)
    prev_detected_bbox = []

    while cap.isOpened():
        ret, frame = cap.read()
        processed_frame_count += 1
        if not ret:
            break
        if processed_frame_count % FPS_DURATION != 0:
            continue
        print(f'Total Frames: {min(processed_frame_count/total_frames*100, 100):.2f}%')

        cv2.line(frame, line_position[0], line_position[1], (255,127,0), 3) 

        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        frame_diff = cv2.absdiff(prev_frame_gray, gray)
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Use MORPH_CLOSE to close small holes in the foreground
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(contour) for contour in contours]

        center_list = []
        cur_detected_bbox = []
        for (x, y, w, h) in boxes:
            contour_bbox = BoundingBox(x, y, w, h)
            if contour_bbox.get_area() < MIN_CAR_AREA or w < MIN_CAR_WIDTH or h < MIN_CAR_HEIGHT:
                continue
            
            detected_bbox.add_bbox(contour_bbox)
            # Put area in the frame
            # cv2.putText(frame, "Area: "+str(contour_bbox.get_area()), contour_bbox.get_center(), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),2)
            cv2.rectangle(frame, contour_bbox.get_left_top(), contour_bbox.get_right_bottom(), (0, 255, 0), 2)
            cv2.circle(frame, contour_bbox.get_center(), 4, (0, 0,255), -1)

            detected_bbox_key_list = list(detected_bbox.bbox_map.keys())
            for bbox_index in detected_bbox_key_list:
                bbox = detected_bbox.bbox_map[bbox_index]
                center_y = bbox.get_center()[1]
                if center_y < (line_y + LINE_OFFSET) and center_y > (line_y - LINE_OFFSET) and not is_overlapped(prev_detected_bbox, bbox):
                    total_car += 1
                    cv2.line(frame, line_position[0], line_position[1], (0,127,255), 3) 
                    cv2.circle(frame, contour_bbox.get_center(), 4, (0, 255, 0), -1) 
                    detected_bbox.remove(bbox_index)
                    cur_detected_bbox.append(bbox)
                    center_list.append(contour_bbox.get_center())

        prev_detected_bbox = cur_detected_bbox
        prev_frame_gray = gray
        text = f"Vehicle Count: {str(total_car)}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        start_x = (width - text_size[0]) // 2

        cv2.putText(frame, text, (start_x, int(height * 0.15)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
        combined_frame = cv2.hconcat([frame, cv2.merge([thresh, thresh, thresh])])
        combined_out.write(combined_frame)
        out.write(frame)
        if processed_frame_count > 400:
            break

    cap.release()
    out.release()
    combined_out.release()
    return total_car


if __name__ == '__main__':
    total_car = start_count()
    print(f'Total Moving Vehicles Detected: {total_car}')