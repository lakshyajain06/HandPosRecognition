import tensorflow as tf

def create_stage_layer(num):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), padding='same', strides=1, activation='relu', name=f'Mstage{num}_conv_1'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), padding='same', strides=1, activation='relu', name=f'Mstage{num}_conv_2'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), padding='same', strides=1, activation='relu', name=f'Mstage{num}_conv_3'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), padding='same', strides=1, activation='relu', name=f'Mstage{num}_conv_4'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), padding='same', strides=1, activation='relu', name=f'Mstage{num}_conv_5'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same', strides=1, activation='relu', name=f'Mstage{num}_conv_6'),
        tf.keras.layers.Conv2D(filters=22, kernel_size=(1, 1), padding='same', strides=1, name=f'Mstage{num}_conv_7')
    ])

class HandModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.feature_extraction = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x - 0.5),  # normalize input form -0.5 to 0.5
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=1, activation='relu', name='conv1_1'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=1, activation='relu', name='conv1_2'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='max_pool1'),

            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=1, activation='relu', name='conv2_1'),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=1, activation='relu', name='conv2_2'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='max_pool2'),

            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=1, activation='relu', name='conv3_1'),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=1, activation='relu', name='conv3_2'),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=1, activation='relu', name='conv3_3'),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=1, activation='relu', name='conv3_4'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='max_pool3'),

            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=1, activation='relu', name='conv4_1'),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=1, activation='relu', name='conv4_2'),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=1, activation='relu', name='conv4_3'),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=1, activation='relu', name='conv4_4'),

            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=1, activation='relu', name='conv5_1'),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=1, activation='relu', name='conv5_2')
        ])

        self.cpm_start = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=1, activation='relu', name='conv5_3_CPM')

        self.stage_1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=1, activation='relu', name='conv6_1_CPM'),
            tf.keras.layers.Conv2D(filters=22, kernel_size=(3, 3), padding='same', strides=1, name='conv6_2_CPM')
        ])

        self.stage_2 = create_stage_layer(2)
        self.stage_3 = create_stage_layer(3)
        self.stage_4 = create_stage_layer(4)
        self.stage_5 = create_stage_layer(5)
        self.stage_6 = create_stage_layer(6)

    def call(self, x):
        features = self.feature_extraction(x)
        features = self.cpm_start(features)

        stage1 = self.stage_1(features)
        self.stage2 = self.stage_2(tf.keras.layers.Concatenate(axis=3)([features, stage1]))
        self.stage3 = self.stage_3(tf.keras.layers.Concatenate(axis=3)([features, self.stage2]))
        self.stage4 = self.stage_4(tf.keras.layers.Concatenate(axis=3)([features, self.stage3]))
        self.stage5 = self.stage_5(tf.keras.layers.Concatenate(axis=3)([features, self.stage4]))
        self.stage6 = self.stage_6(tf.keras.layers.Concatenate(axis=3)([features, self.stage5]))

        return self.stage6, [self.stage2, self.stage3, self.stage4, self.stage5] # (final belief map, intermediate belief maps)