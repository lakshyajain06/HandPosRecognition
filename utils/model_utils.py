import tensorflow as tf
import numpy as np
import cv2 as cv

def get_points_from_map(heatmaps, img_size):
    """
    Returns the coordinates of the pixel with the highest intensity on each map\n
    Input: numpy array - heatmap, (original_img_height, original_img_width) \n
    Output: numpy array - coordinate in xy of keypoints. shape (num_heatmaps, 2)
    """
    points = np.zeros((heatmaps.shape[2], 2))
    heatmaps = cv.resize(heatmaps, (img_size[1], img_size[0]), interpolation=cv.INTER_CUBIC)

    for i in range(points.shape[0]):
        idx = np.argmax(heatmaps[:, :, i])
        points[i] = np.unravel_index(idx, img_size)
        
        
    points = points[:, ::-1]
    return points.astype(int)

def get_points_from_map_tf(heatmaps, img_size):
    """
    batch version of get_points_from_map()\n
    Input: numpy array - batch of heatmaps, (original_img_height, original_img_width) \n
    Output: numpy array - coordinate in xy of keypoints. shape (batchsize, num_heatmaps, 2)
    """

    resized = tf.image.resize(heatmaps, size=img_size, method='bicubic')

    B, H, W, C = resized.shape

    resized = tf.transpose(resized, [0, 3, 1, 2])
    flat = tf.reshape(resized, [B, C, -1])

    max_idxs = tf.argmax(flat, axis=2, output_type=tf.int32)

    y = max_idxs // W
    x = max_idxs % W

    coords = tf.stack([x, y], axis=2)

    return coords
