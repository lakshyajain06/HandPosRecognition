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