import tensorflow as tf
import cv2 as cv
import numpy as np

from network import HandModel
from process_data import get_points_from_map

model = HandModel()
dummy_input = np.zeros((1, 368, 368, 3))
model(dummy_input)
model.load_weights("vgg19_pretrain_synth_dataset_epoch35_lr_custom_norm.h5")



def draw_finger(frame, points, start_idx):
    finger_points = [tuple(points[start_idx + i]) for i in range(4)]
    for i in range(3):
        cv.line(frame, finger_points[i], finger_points[i+1], color=(255,0,0), thickness=2)


def draw_hand(frame, points):
    hand_start = tuple(points[0])
    print(hand_start)
    finger_starts = [(4 * i) + 1 for i in range(5)]
    for i in finger_starts:
        cv.line(frame, hand_start, tuple(points[i]), color=(255,0,0), thickness=2)
        draw_finger(frame, points, i)



# img = cv.imread('Photo from 2025-07-15 14-54-47.941512.jpeg')
# rgb_frame = np.expand_dims(cv.cvtColor(img, cv.COLOR_BGR2RGB), axis=0)
# heat_maps, _ = model(rgb_frame)

# points = get_points_from_map(heat_maps[0].numpy())

# draw_hand(img, points)

# cv.imshow('video', img)

# cv.waitKey(0)


capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 368)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 368)
while True:
    isTrue, frame = capture.read()

    rgb_frame = np.expand_dims(cv.cvtColor(frame, cv.COLOR_BGR2RGB), axis=0)
    heat_maps, _ = model(rgb_frame)
    

    points = get_points_from_map(heat_maps[0].numpy())

    draw_hand(frame, points)

    cv.imshow('video', frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()