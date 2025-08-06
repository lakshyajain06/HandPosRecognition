import cv2 as cv
import numpy as np

def visualize_combined_map(img, heat_map, ax):

    heat_map = np.max(heat_map, axis=2)
    heat_map = cv.resize(heat_map, img.shape[:-1], interpolation=cv.INTER_CUBIC)

    heat_map -= heat_map.min()
    if heat_map.max() != 0:
        heat_map /= heat_map.max()

    white_img = np.ones((*heat_map.shape, 3))

    rgba_mask = np.dstack((white_img, heat_map))

    ax.imshow(img)
    ax.imshow(rgba_mask)
    ax.axis('off')

def visualize_plotted_points(img, points, ax):

    ax.imshow(img)

    points = points.T
    ax.scatter(points[0][0], points[1][0], color='green', s=50)
    ax.scatter(points[0][1:5], points[1][1:5], color='red', s=50)
    ax.scatter(points[0][5:9], points[1][5:9], color='blue', s=50)
    ax.scatter(points[0][9:13], points[1][9:13], color='red', s=50)
    ax.scatter(points[0][13:17], points[1][13:17], color='blue', s=50)
    ax.scatter(points[0][17:21], points[1][17:21], color='red', s=50)
    
    ax.axis('off')