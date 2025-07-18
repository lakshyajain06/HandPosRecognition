import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.image as img
import json
from os.path import join

num_joints = 21
input_img_height = 368
input_img_width = 368

output_img_height = 46
output_img_width = 46


synth2_folder = "hand_labels_synth/synth2"

img_path_synth2 = tf.io.gfile.glob("hand_labels_synth/synth2/*.jpg")
point_path_synth2 = tf.io.gfile.glob("hand_labels_synth/synth2/*.json")

img_path_synth3 = tf.io.gfile.glob("hand_labels_synth/synth3/*.jpg")
point_path_synth3 = tf.io.gfile.glob("hand_labels_synth/synth3/*.json")

batch_size = 16
split = int(0.6 * len(img_path_synth2 + img_path_synth3))


def load_img_and_pos_with_num(num):
    imgname = str(num).rjust(8, '0')
    hand_img = img.imread(join(synth2_folder, imgname + ".jpg"))

    with open(join(synth2_folder, imgname + ".json")) as d:
        keypoints = json.load(d)
        hand_pos = np.array(keypoints['hand_pts'])
        is_left = keypoints['is_left']

    return hand_img, hand_pos, is_left

def load_data_from_path(img_path, points_path):

    img_path = img_path.numpy().decode('utf-8')
    points_path = points_path.numpy().decode('utf-8')

    hand_img = img.imread(img_path)

    with open(points_path) as d:
        keypoints = json.load(d)
        hand_pos = np.array(keypoints['hand_pts'])

    return hand_img, hand_pos

def tf_load_data_from_path(img_path, point_path):
    hand_img, hand_pos = tf.py_function(load_data_from_path, [img_path, point_path], [tf.float32, tf.float32])
    hand_img.set_shape([input_img_height, input_img_width, 3])
    hand_pos.set_shape([num_joints, 3])
    return hand_img, hand_pos

def plot_gaussian_bob(map, point, sigma=8.0):
    threshold = 4.6052
    delta = np.sqrt(threshold)

    x_center, y_center = point

    x0 = int(max(0, x_center - delta * sigma))
    y0 = int(max(0, y_center - delta * sigma))

    x1 = int(min(input_img_width, x_center + delta * sigma))
    y1 = int(min(input_img_height, y_center + delta * sigma))

    for y in range(y0, y1):
        for x in range(x0, x1):
            dist = (x - x_center) ** 2 + (y - y_center) **2
            exp = dist / 2.0 / sigma / sigma
            if exp > threshold:
                continue
            map[y][x] = max(map[y][x], np.exp(-exp))
            map[y][x] = min(map[y][x], 1.0)

    return map

def create_heat_map(points, sigma = 0.8):
    heat_maps = np.zeros((points.shape[0] + 1, input_img_height, input_img_width), dtype=np.float32)
    
    for i in range(heat_maps.shape[0] - 1):
        curr_point_x = points[i][0]
        curr_point_y = points[i][1]

        heat_maps[i] = plot_gaussian_bob(np.zeros((input_img_height, input_img_width)), (curr_point_x, curr_point_y))

    heat_maps[-1] = np.clip(1 - np.max(heat_maps[:-1], axis=0), 0.0, 1.0)

    heat_maps = heat_maps.transpose((1, 2, 0))

    heat_maps = cv.resize(heat_maps, (46, 46), interpolation=cv.INTER_AREA)

    return heat_maps

def process_y(x, y):
    y_processed = tf.numpy_function(create_heat_map, [y], tf.float32)
    y_processed.set_shape([output_img_height, output_img_width, num_joints + 1])
    return x, y_processed

def get_data_set():


    all_img_path = sorted(img_path_synth2 + img_path_synth3)
    all_point_path = sorted(point_path_synth2 + point_path_synth3)

    data_size = len(all_img_path)
    

    data = tf.data.Dataset.from_tensor_slices((all_img_path, all_point_path))
    data = data.shuffle(data_size)

    train_data = data.take(split)
    val_data = data.skip(split)

    train_data = train_data.map(lambda img_paths, point_paths: tf_load_data_from_path(img_paths, point_paths))
    train_data = train_data.map(process_y)
    train_data = train_data.batch(batch_size).prefetch(5)

    val_data = val_data.map(lambda img_paths, point_paths: tf_load_data_from_path(img_paths, point_paths))
    val_data = val_data.map(process_y)
    val_data = val_data.batch(batch_size).prefetch(5)

    return train_data, val_data

def get_num_batches():
    return (split + batch_size - 1) // batch_size

def get_points_from_map(heatmaps):
    points = np.zeros((num_joints, 2))
    heatmaps = cv.resize(heatmaps, (input_img_height, input_img_width), interpolation=cv.INTER_CUBIC)

    for i in range(points.shape[0]):
        idx = np.argmax(heatmaps[:, :, i])
        points[i] = np.unravel_index(idx, (input_img_height, input_img_width))
        
        
    points = points[:, ::-1]
    return points.astype(int)