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
