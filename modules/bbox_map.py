class BBoxMap:
    def __init__(self):
        self.bbox_map = {}
    
    def add_bbox(self, bbox):
        self.bbox_map[0] = bbox

    def remove(self, index):
        self.bbox_map.pop(index, None)

