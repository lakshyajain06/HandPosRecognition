import numpy as np
import cv2 as cv

def get_points_from_map(heatmaps, img_size):
    """
    Returns the coordinates of the pixel with the highest intensity on each map\n
    Input: numpy array - heatmap, (original_img_height, original_img_width) \n
    Output: numpy array - coordinate in xy of keypoints. shape (num_heatmaps, 2)
    """
    points = np.zeros((heatmaps.shape[2], 2))
    heatmaps = cv.resize(heatmaps, img_size, interpolation=cv.INTER_CUBIC)

    for i in range(points.shape[0]):
        idx = np.argmax(heatmaps[:, :, i])
        points[i] = np.unravel_index(idx, img_size)
        
        
    points = points[:, ::-1]
    return points.astype(int)
