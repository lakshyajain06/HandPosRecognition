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

        self.normalize = tf.keras.layers.Lambda(lambda x: (tf.cast(x, tf.float32) / 255.0) - 0.5)

        self.feature_extraction = tf.keras.Sequential([
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

        self.attach = tf.keras.layers.Concatenate(axis=3, name='Concat_Prev_stage')

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

        normalized = self.normalize(x)
        # normalized = tf.keras.applications.vgg19.preprocess_input(x)
    
        features = self.feature_extraction(normalized)
        features = self.cpm_start(features)

        stage1 = self.stage_1(features)
        stage2 = self.stage_2(self.attach([features, stage1]))
        stage3 = self.stage_3(self.attach([features, stage2]))
        stage4 = self.stage_4(self.attach([features, stage3]))
        stage5 = self.stage_5(self.attach([features, stage4]))
        stage6 = self.stage_6(self.attach([features, stage5]))

        return stage6, [stage2, stage3, stage4, stage5] # (final belief map, intermediate belief maps)