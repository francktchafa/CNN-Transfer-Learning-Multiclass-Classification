"""
Implementing a Residual Network with 50 layers to train a state-of-the-art neural network for image classification.
This code  is to recognize sign language digits from 0 to 9.
"""

import os
import numpy as np
import tensorflow as tf
import h5py
from tensorflow.keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, Flatten, BatchNormalization,
                                     Conv2D, AveragePooling2D, MaxPooling2D, Dropout)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# BUILDING A RESIDUAL NETWORK
# Here is the picture of the residual network being built
rn_img_path = os.path.join(os.getcwd(), "images", "resnet50.png")
rn_image = plt.imread(rn_img_path)
plt.imshow(rn_image)
plt.axis('off')
plt.title("ResNet50")
plt.show()

# Note: The following figure describes the architecture of the residual network. "ID BLOCK" in the diagram stands for
# "Identity block" while "CONV BLOCK" stands for "Convolutional block." And "ID BLOCK x3" means stacking 3 ID BLOCKS.
# Here is a look at the ID BLOCK and CONV BLOCK

idblock_img_path = os.path.join(os.getcwd(), "images", "idblock3.png")
convblock_img_path = os.path.join(os.getcwd(), "images", "convblock.png")

fig, ax = plt.subplots(2, 1)
ax[0].imshow(plt.imread(idblock_img_path))
ax[0].set_title('Identity block')
ax[0].axis('off')
ax[1].imshow(plt.imread(convblock_img_path))
ax[1].set_title('Convolutional block')
ax[1].axis('off')
fig.text(0.05, 0.02, 'Note: Skip connection "skips over" 3 layers instead of 2 layers.')
plt.show()


# Building the various blocks
# Identity block
def identity_block(X, f, filters, initializer=random_uniform, training=None):
    """
    Implementation of the identity block as defined in the figure

    Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        initializer -- initializer for the kernel weights of the convolutional layers
        training -- boolean, training mode (True) or testing (False)

    Returns:
        X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """

    # Retrieve Filters for each CONV layer
    F1, F2, F3 = filters

    # Save the input value for later addition (skip connection)
    X_shortcut = X

    # First component of the main path
    X = Conv2D(filters=F1, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer)(X)
    X = BatchNormalization(axis=3)(X, training=training)  # Normalize the output along the channels axis
    X = Activation('relu')(X)

    # Second component of the main path
    X = Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding='same', kernel_initializer=initializer)(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    # Third component of the main path
    X = Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer)(X)
    X = BatchNormalization(axis=3)(X, training=training)

    # Add the shortcut value to the main path and apply the final ReLU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


# Convolutional block
def convolutional_block(X, f, filters, s=2, initializer=glorot_uniform, training=None):
    """
    Implementation of the convolutional block as defined in the figure

    Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        s -- Integer, specifying the stride to be used
        initializer -- initializer for the kernel weights of the convolutional layers
        training -- boolean, training mode (True) or testing (False)

    Returns:
        X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
    """

    # Retrieve Filters for each CONV layer
    F1, F2, F3 = filters

    # Save the input value for later addition (skip connection)
    X_shortcut = X

    # Main path
    # First component of the main path
    X = Conv2D(filters=F1, kernel_size=1, strides=(s, s), padding='valid', kernel_initializer=initializer)(X)
    X = BatchNormalization(axis=3)(X, training=training)  # Normalize the output along the channels axis
    X = Activation('relu')(X)

    # Second component of the main path
    X = Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding='same', kernel_initializer=initializer)(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    # Third component of the main path
    X = Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer)(X)
    X = BatchNormalization(axis=3)(X, training=training)

    # Shortcut path
    X_shortcut = Conv2D(filters=F3, kernel_size=1, strides=(s, s), padding='valid',
                        kernel_initializer=initializer)(X_shortcut)

    X_shortcut = BatchNormalization(axis=3)(X_shortcut, training=training)

    # Combining path and passing it through a ReLU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


# Building the residual network
def ResNet50(input_shape=(64, 64, 3), num_class=10, training=None):
    """
    Stage-wise implementation of the popular ResNet50 architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE

    Arguments:
        input_shape -- shape of the images of the dataset
        num_class -- integer, number of classes
        training -- boolean, training mode (True) or testing (False)

    Returns:
        model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=glorot_uniform)(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1, training=training)
    X = identity_block(X, 3, [64, 64, 256], training=training)
    X = identity_block(X, 3, [64, 64, 256], training=training)

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2, training=training)
    X = identity_block(X, 3, [128, 128, 512], training=training)
    X = identity_block(X, 3, [128, 128, 512], training=training)
    X = identity_block(X, 3, [128, 128, 512], training=training)

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2, training=training)
    X = identity_block(X, 3, [256, 256, 1024], training=training)
    X = identity_block(X, 3, [256, 256, 1024], training=training)
    X = identity_block(X, 3, [256, 256, 1024], training=training)
    X = identity_block(X, 3, [256, 256, 1024], training=training)
    X = identity_block(X, 3, [256, 256, 1024], training=training)

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2, training=training)
    X = identity_block(X, 3, [512, 512, 2048], training=training)
    X = identity_block(X, 3, [512, 512, 2048], training=training)

    # Final Step
    X = AveragePooling2D()(X)
    X = Flatten()(X)
    # X = Dropout()(X, training=training)
    X = Dense(num_class, activation='softmax', kernel_initializer=glorot_uniform)(X)

    # Create model
    resnet_model = Model(inputs=X_input, outputs=X)

    return resnet_model


# LOADING  THE DATASET
def load_datasets(data_path):
    """
    Load datasets from an HDF5 file and convert to NumPy arrays.

    Arguments:
        data_path -- string, path to the HDF5 file containing the datasets

    Returns:
        Xtrain, Xtest, ytrain, ytest -- NumPy arrays
    """
    with h5py.File(data_path, 'r') as hf:
        Xtrain = hf['X_train'][:]  # Convert to NumPy array
        Xtest = hf['X_test'][:]    # Convert to NumPy array
        ytrain = hf['y_train'][:]  # Convert to NumPy array
        ytest = hf['y_test'][:]    # Convert to NumPy array

    return Xtrain, Xtest, ytrain, ytest


dataset_path = os.path.join(os.getcwd(), 'datasets', 'kfold_dataset.h5')  # Define save directory
X_data, X_test, y_data, y_test = load_datasets(dataset_path)

# Print shapes to verify
print(f"X_train shape: {X_data.shape}, y_train shape: {y_data.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

if np.max(X_data) > 1:
    X_data = X_data / 255.
    X_test = X_test / 255.

# Visualize
plt.imshow(X_data[1])
plt.title(f"Class {y_data[1]}")
plt.show()


# TRAINING THE MODEL
# Create TensorFlow dataset objects
the_classes = np.unique(y_data)
num_classes = len(the_classes)
image_shape = X_data.shape[1:]
print("The classes in the dataset:", the_classes)
print("Input image shape:", image_shape)  # Output: (64, 64, 3)

# One-hot encoding
Y_data = np.eye(num_classes)[y_data]
Y_test = np.eye(num_classes)[y_test]

# Verification
print("Y_train shape (one-hot encoded): ", Y_data.shape)

# Define your model
resnet = ResNet50(input_shape=image_shape, num_class=num_classes, training=True)
print(resnet.summary())

# Define KFold cross-validator
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

# Initialize variables to store cross-validation results
cv_scores, all_history = [], []
best_cv_score = 0
best_model_weights = None

# Training with KFold cross-validation
for train_index, val_index in kfold.split(X_data):
    X_train, X_val = tf.gather(X_data, train_index), tf.gather(X_data, val_index)
    Y_train, Y_val = tf.gather(Y_data, train_index), tf.gather(Y_data, val_index)

    # Initialize and compile the model inside the loop
    model = resnet
    model.compile(optimizer=Adam(learning_rate=0.00015), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model_path = os.path.join(os.getcwd(), "models", "best_model.h5")
    check_point = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1)
    callbacks = [early_stopping, check_point]

    # Train the model
    history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_val, Y_val),
                        callbacks=callbacks)

    all_history.append(history.history)

    # Evaluate the model on the validation set and store the results
    scores = model.evaluate(X_val, Y_val, verbose=0)
    cv_scores.append(scores)

    # Track the best model
    if scores[1] > best_cv_score:
        best_cv_score = scores[1]
        best_model_weights = model.get_weights()


# Select the best model
best_model = resnet
best_model.set_weights(best_model_weights)

# Save the best model weights
best_model.save_weights(os.path.join(os.getcwd(), "models", "kfold_model_weights.h5"))

print(f"Validation scores: {cv_scores}")
print(f"Best validation accuracy: {best_cv_score}")


def plot_metrics(histories, metric):
    for i, hist in enumerate(histories):
        plt.plot(hist[metric], label=f"Fold {i + 1} Train {metric.title()}")
        plt.plot(hist[f"val_{metric}"], label=f"Fold {i + 1} Validation {metric.title()}")
    plt.xlabel("Epochs")
    plt.ylabel(metric.title())
    plt.legend()
    plt.show()


plot_metrics(all_history, "loss")
plot_metrics(all_history, "accuracy")


# MODEL PERFORMANCE
pretrained_model = resnet
pretrained_model.compile(optimizer=Adam(learning_rate=0.00015), loss='categorical_crossentropy', metrics=['accuracy'])
pretrained_model.load_weights(os.path.join(os.getcwd(), "models", "kfold_trained_model_weights.h5"))
# best_model.save(os.path.join(os.getcwd(), "models", "best_model.h5"))

# tf.keras.backend.set_learning_phase(False)
pretrained_model.trainable = False
test_hist = pretrained_model.evaluate(X_test, Y_test)

print("Test loss: ", round(test_hist[0], 3))
print("Test Accuracy: ", round(test_hist[1], 3))




