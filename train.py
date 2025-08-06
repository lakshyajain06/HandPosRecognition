import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import wandb

from sys import stdout

from network import HandModel
from preprocess_data import SynthDataLoader, MultiBootStrapDataLoader
from utils.visualize_data import visualize_combined_map
from utils.model_utils import get_points_from_map, get_points_from_map_tf

@tf.function
def compute_loss(y_pred, intermediate_pred, y):
    loss_object = tf.keras.losses.MeanSquaredError()
    final_loss = loss_object(y, y_pred)
    total_loss = final_loss
    for i in range(len(intermediate_pred)):
        total_loss += loss_object(y, intermediate_pred[i])
    
    return final_loss, total_loss

@tf.function
def compute_mpjpe(predicted_map, true_map):
    predicted_pos = get_points_from_map_tf(predicted_map)
    true_pos = get_points_from_map_tf(true_map)
    error = tf.norm(tf.cast(predicted_pos, tf.float32) - tf.cast(true_pos, tf.float32), axis=-1)
    return tf.reduce_mean(error)

@tf.function
def train_step(x, y, model, optimizer):
    with tf.GradientTape() as tape:
        y_pred, y_intermediate = model(x)
        fin_loss, total_loss = compute_loss(y_pred, y_intermediate, y)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    accuracy = compute_mpjpe(y_pred, y)

    return fin_loss, total_loss, accuracy

@tf.function
def test_step(x, y, model):
    y_pred, y_intermediate = model(x)
    fin_loss, total_loss = compute_loss(y_pred, y_intermediate, y)

    accuracy = compute_mpjpe(y_pred, y)

    return fin_loss, total_loss, accuracy

def train(model, train_data, val_data, num_epochs, num_batches, optimizer, train_loss, test_loss, train_accuracy, test_accuracy, logger, checkpoint_dir):

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
        train_accuracy.reset_state()
        test_loss.reset_state()
        test_accuracy.reset_state()

        # run train steps for the epoch
        for batch_num, (x_batch, y_batch) in enumerate(train_data):
            fin_loss, total_loss, accuracy = train_step(x_batch, y_batch, model, optimizer)
            train_loss(total_loss)
            train_accuracy(accuracy)
            print(f"\rTrain Progress: {batch_num + 1}/{num_batches}", end='', flush=True)

        print()

        # run test steps for the epoch
        for batch_num, (x_batch, y_batch) in enumerate(val_data):
            fin_loss, total_loss, accuracy = test_step(x_batch, y_batch, model)
            test_loss(total_loss)
            test_accuracy(accuracy)
            print(f"\rTest Progress: {batch_num + 1}/{num_batches}", end='', flush=True)

        print()

        curr_train_loss = train_loss.result()
        curr_train_accuracy = train_accuracy.result()
        curr_test_loss = test_loss.result()
        curr_test_accuracy = test_accuracy.result()

        print(f" ----- Epoch {epoch + 1} -----")
        print(f"Train Loss: {curr_train_loss}, Train Accuracy: {curr_train_accuracy}")
        print(f"Test Loss: {curr_test_loss}, Test Accuracy: {curr_test_accuracy}")
        
        logger.log({
            "train_accuracy": curr_train_accuracy, 
            "train_loss": curr_train_loss,
            "test_accuracy": curr_test_accuracy, 
            "test_loss": curr_test_loss
        })

        # save model if loss went down
        if curr_train_loss < old_loss:
            ckpt_manager.save()

        old_loss = curr_train_loss

        # end training if reaches accuracy is great enough
        if curr_train_loss < 0.0001:
            print("Finished training, ended early")
            break


def main(num_epochs = 20):

    # Start a new wandb run to track this script.
    logger = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="jlakshya06-new-york-university",
        # Set the wandb project where this run will be logged.
        project="HandPose",
        # Track hyperparameters and run metadata.
        config={
            "inital_learning_rate": 0.0001,
            "learning_schedule": "Cosine Decay",
            "optimizer": "Adam",
            "dataset": "CMU Panoptic Studio Hands by Multiview Bootstrapping",
            "epochs": 20,
        },
    )

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

    # define learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0001,
        decay_steps=(loader.get_num_batches() * num_epochs),  # total_epochs × steps_per_epoch
        alpha=0.0
    )

    # define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    # define loss and accuracy objects
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

    train(
        model, 
        train_data, val_data, 
        num_epochs, num_batches, 
        optimizer, 
        train_loss, test_loss, 
        train_accuracy, test_accuracy,
        logger, 
        "checkpoints"
    )
    logger.alert(
        title="Training Finished",
        text="Finished training successfully"
    )
    logger.finish()
    
 
if __name__ == "__main__":
    main(1)