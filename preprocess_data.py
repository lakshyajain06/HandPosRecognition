import tensorflow as tf
import numpy as np
import cv2 as cv
import json

num_joints = 21



class DataLoader():
    def __init__(self, data_folder, batch_size, split_ratio, img_size=(368, 368), output_img_size=(46, 46), img_extension="jpg", point_extension="json"):
        self.img_paths = sorted(tf.io.gfile.glob(data_folder + "*." + img_extension))
        self.point_paths = sorted(tf.io.gfile.glob(data_folder + "*." + point_extension))
        self.img_size = img_size
        self.output_img_size = output_img_size
        self.batch_size = batch_size
        self.split_ratio = split_ratio

    def read_data(self, index):
        """
        Reads the data at the specified index in the file path lists\n
        Input: Index of file path\n
        Output: numpy array - (BGR image, Hand Pose)
        """

        img_path = self.img_paths[index]
        points_path = self.point_paths[index]

        hand_img = cv.imread(img_path) # BGR Image

        with open(points_path) as d:
            keypoints = json.load(d)
            hand_pos = np.array(keypoints['hand_pts'])

        return hand_img, hand_pos
    
    def tf_read_data(self, index):
        """
        Calls the read_data function with tf.py_function\n
        Input: Index of file path\n
        Output: tensors - BGR image, Hand Pose
        """

        hand_img, hand_pos = tf.py_function(self.read_data, [index], [tf.float32, tf.float32])
        hand_img.set_shape([*self.img_size, 3])
        hand_pos.set_shape([num_joints, 3])
        return hand_img, hand_pos
    
    def plot_gaussian_bob(self, map, point, sigma=8.0):
        """
        Creates a gaussian blob (0-1) at the specified x-y cooridinate in the map\n
        Input: numpy array - map, (x, y), (optional) sigma_value\n
        Output: numpy array- map
        """

        threshold = 4.6052
        delta = np.sqrt(threshold)

        x_center, y_center = point

        x0 = int(max(0, x_center - delta * sigma))
        y0 = int(max(0, y_center - delta * sigma))

        x1 = int(min(self.img_size[1], x_center + delta * sigma))
        y1 = int(min(self.img_size[0], y_center + delta * sigma))

        for y in range(y0, y1):
            for x in range(x0, x1):
                dist = (x - x_center) ** 2 + (y - y_center) **2
                exp = dist / 2.0 / sigma / sigma
                if exp > threshold:
                    continue
                map[y][x] = max(map[y][x], np.exp(-exp))
                map[y][x] = min(map[y][x], 1.0)

        return map

    def create_heat_map(self, points, sigma = 0.8):
        """
        Creates a set of heatmaps at the x-y coordinate specified by the points \n
        Input: numpy array of points. shape (N, 2) \n
        Output: numpy array of heatmaps. shape (output_img_height, output_img_height, num_points)
        """

        heat_maps = np.zeros((points.shape[0] + 1, *self.img_size), dtype=np.float32)
        
        for i in range(heat_maps.shape[0] - 1):
            curr_point_x = points[i][0]
            curr_point_y = points[i][1]

            heat_maps[i] = self.plot_gaussian_bob(np.zeros(self.img_size), (curr_point_x, curr_point_y))

        heat_maps[-1] = np.clip(1 - np.max(heat_maps[:-1], axis=0), 0.0, 1.0)

        heat_maps = heat_maps.transpose((1, 2, 0))

        heat_maps = cv.resize(heat_maps, (self.output_img_size), interpolation=cv.INTER_AREA)

        return heat_maps
    
    def tf_process_y(self, x, y):
        """
        Converts the hand coordinates into heatmaps \n
        Input: tensors - img, hand pose \n
        Output: tensors -  img, heatmaps shape (output_img_height)
        """
        y_processed = tf.numpy_function(self.create_heat_map, [y], tf.float32)
        y_processed.set_shape([*self.output_img_size, num_joints + 1])
        return x, y_processed
    
    def get_data_set(self):
        """
        Loads and preproccesses data and returns it as a tensorflow Dataset object \n
        Input: None \n
        Output: tf.data.Dataset train_data, val_data
        """
        data_size = len(self.img_paths)
        idx_list = list(range(data_size))
        

        data = tf.data.Dataset.from_tensor_slices(idx_list)
        data = data.shuffle(data_size)

        split_num = int(self.split_ratio * data_size)

        train_data = data.take(split_num)
        val_data = data.skip(split_num)

        train_data = train_data.map(self.tf_read_data)
        train_data = train_data.map(self.tf_process_y)
        train_data = train_data.batch(self.batch_size).prefetch(5)

        val_data = val_data.map(self.tf_read_data)
        val_data = val_data.map(self.tf_process_y)
        val_data = val_data.batch(self.batch_size).prefetch(5)

        return train_data, val_data

    def get_num_batches(self):
        return (int(self.split_ratio * len(self.img_paths)) + self.batch_size - 1) // self.batch_size
    
    def get_random_data(self):
        img, pose = self.read_data(np.random.randint(len(self.img_paths)))

        maps = self.create_heat_map(pose)

        return img, maps, pose