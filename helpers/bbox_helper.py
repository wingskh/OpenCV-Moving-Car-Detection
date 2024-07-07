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
