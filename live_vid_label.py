import tensorflow as tf
import cv2 as cv
import numpy as np

from network import HandModel
from utils.model_utils import get_points_from_map

model = HandModel()
dummy_input = np.zeros((1, 368, 368, 3))
model(dummy_input)
model.load_weights("good progress models/vgg_19_multibootstap_epoch25/curr_vgg19_train_epoch_25.h5")



def draw_finger(frame, points, start_idx, color):
    finger_points = [tuple(points[start_idx + i]) for i in range(4)]
    for i in range(4):
        cv.circle(frame, finger_points[i], 5, color, -1)


def draw_hand(frame, points):
    hand_start = tuple(points[0])
    cv.circle(frame, hand_start, 5, (0, 255, 0), -1)
    print(hand_start)
    finger_starts = [(4 * i) + 1 for i in range(5)]
    for i in finger_starts:
        color = (0, 0, 255) if (i - 1) % 8 == 0 else (255,0,0)
        draw_finger(frame, points, i, color)



# img = cv.imread('test_photo.jpeg')
# rgb_frame = np.expand_dims(img, axis=0)
# heat_maps, _ = model(rgb_frame)

# points = get_points_from_map(heat_maps[0].numpy(), (368, 368))

# draw_hand(img, points)

# cv.imshow('video', img)

# cv.waitKey(0)

size = 100


capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, size)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, size)
while True:
    isTrue, frame = capture.read()

    frame = cv.resize(frame, (368, 368), interpolation=cv.INTER_LINEAR)
    rgb_frame = np.expand_dims(frame, axis=0)
    heat_maps, _ = model(rgb_frame)
    

    points = get_points_from_map(heat_maps[0].numpy(), (368, 368))

    draw_hand(frame, points)

    cv.imshow('video', frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()