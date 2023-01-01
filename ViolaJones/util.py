import numpy as np

# Code from: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    
def non_maximum_supression( regions , half_window , threshold=0.5):
    boxes = np.array(regions)

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats ==> this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indicies
    pick = []

    # boxes = [[x, y, w, h]]
    # grab the coordinates of the bounding boxes
    start_x = boxes[:, 0] - half_window - 1
    start_y = boxes[:, 1] - half_window - 1
    
    end_x   = boxes[:, 0] + half_window
    end_y   = boxes[:, 1 ]+ half_window

    # compute the area of the bounding boxes and sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    area = (end_x - start_x + 1) * (end_y - start_y + 1)
    indicies = np.argsort(end_y)

    # keep looping while some indicies still remain in the indicies list
    while len(indicies) > 0:
        
        # grab the last index in the indicies list and add the index value to the list of picked indicies
        last = len(indicies) - 1
        i = indicies[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        x_maximum = np.maximum(start_x[i], start_x[indicies[:last]])
        y_maximum = np.maximum(start_y[i], start_y[indicies[:last]])
        
        x_minimum = np.minimum(end_x[i]  , end_x  [indicies[:last]])
        y_minimum = np.minimum(end_y[i]  , end_y  [indicies[:last]])

        # compute the width and height of the bounding box
        # how min - max ???!!
        width = np.maximum(0, x_minimum - x_maximum + 1)
        height = np.maximum(0, y_minimum - y_maximum + 1)

        # compute the ratio of overlap
        overlap = (width * height) / area[indicies[:last]]

        # delete all indexes from the index list that have
        indicies = np.delete( indicies , np.concatenate( ( [last] , np.where(overlap > threshold)[0] ) ) )

    # return only the bounding boxes that were picked using the integer data type
    return boxes[pick].astype("int")


