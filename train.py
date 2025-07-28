import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from sys import stdout

from network import HandModel
from preprocess_data import SynthDataLoader, MultiBootStrapDataLoader
from utils.visualize_data import visualize_combined_map
from utils.model_utils import get_points_from_map

def compute_loss(y_pred, intermediate_pred, y):
    loss_object = tf.keras.losses.MeanSquaredError()
    final_loss = loss_object(y, y_pred)
    total_loss = final_loss
    for i in range(len(intermediate_pred)):
        total_loss += loss_object(y, intermediate_pred[i])
    
    return final_loss, total_loss

@tf.function
def train_step(x, y, model, optimizer):
    with tf.GradientTape() as tape:
        y_pred, y_intermediate = model(x)
        fin_loss, total_loss = compute_loss(y_pred, y_intermediate, y)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return fin_loss, total_loss

def train(model, train_data, num_epochs, num_batches, optimizer, train_loss, loss_history, checkpoint_dir):

    # Setup checkpointing
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

    # Restore if available
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(f"Restored from checkpoint: {ckpt_manager.latest_checkpoint}")
    else:
        print("Training from scratch.")

    old_loss = 0
    for epoch in range(num_epochs):
        train_loss.reset_state()

        for batch_num, (x_batch, y_batch) in enumerate(train_data):
            fin_loss, total_loss = train_step(x_batch, y_batch, model, optimizer)
            train_loss(total_loss)
            print(f"\rProgress: {batch_num + 1}/{num_batches}", end='', flush=True)

        print()

        curr_loss = train_loss.result()

        print(f"Epoch {epoch + 1} ----- Loss: {curr_loss}")
        loss_history.append(curr_loss)


        # save model if loss went down
        if curr_loss < old_loss:
            model.save(f"curr_vgg19_train_epoch_{epoch + 1}.h5")

        old_loss = curr_loss

        # end training if reaches accuracy is great enough
        if curr_loss < 0.0001:
            print("Finished training, ended early")
            break


def main(num_epochs = 25):
    model = HandModel() # instantiate model

    # build graph to allow for transfering weights
    dummy_input = tf.random.normal([1, 368, 368, 3])
    _ = model(dummy_input)

    # load pretrained VGG19 model weights up to layer CONV_4_4
    copy_model = tf.keras.applications.VGG19(False, input_shape=(368, 368, 3), pooling='avg')
    for i in range(1, 16):
        model.feature_extraction.layers[i - 1].set_weights(copy_model.layers[i].get_weights())

    # loading the training and validation dataset 
    loader = MultiBootStrapDataLoader("data/hand143_panopticdb/", 16, 0.8)
    train_data, val_data = loader.get_data_set()
    num_batches = loader.get_num_batches()

    # list of losses after each epoch
    loss_history = []

    # define learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0001,
        decay_steps=(loader.get_num_batches() * num_epochs),  # total_epochs × steps_per_epoch
        alpha=0.0
    )

    # define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    # define create loss object to calculate loss per epoch
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    train(model, train_data, num_epochs, num_batches, optimizer, train_loss, loss_history, "checkpoints")

    np.save("good progress models/vgg_19_multibootstap_epoch25/loss_history", np.array(loss_history))
 

