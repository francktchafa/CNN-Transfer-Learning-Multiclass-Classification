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
from tensorflow.keras.layers import RandomRotation, RandomContrast, RandomBrightness
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


# LOAD  THE DATASET
def load_datasets(data_path):
    """
    Load datasets from an HDF5 file and convert to NumPy arrays.

    Arguments:
        data_path -- string, path to the HDF5 file containing the datasets

    Returns:
        Xtrain, Xval, Xtest, ytrain, yval, ytest -- NumPy arrays
    """
    with h5py.File(data_path, 'r') as hf:
        Xtrain = hf['X_train'][:]  # Convert to NumPy array
        Xval = hf['X_val'][:]      # Convert to NumPy array
        Xtest = hf['X_test'][:]    # Convert to NumPy array
        ytrain = hf['y_train'][:]  # Convert to NumPy array
        yval = hf['y_val'][:]      # Convert to NumPy array
        ytest = hf['y_test'][:]    # Convert to NumPy array

    return Xtrain, Xval, Xtest, ytrain, yval, ytest


dataset_path = os.path.join(os.getcwd(), 'datasets', 'dataset.h5')  # Define save directory
X_train, X_val, X_test, y_train, y_val, y_test = load_datasets(dataset_path)

# Print shapes to verify
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Visualize
plt.imshow(X_train[1])
plt.title(f"Class {y_train[1]}")
plt.show()

# Create TensorFlow dataset objects
the_classes = np.unique(y_train)
num_classes = len(the_classes)
image_shape = X_train.shape[1:]
print("The classes in the dataset:", the_classes)
print("Input image shape:", image_shape)  # Output: (64, 64, 3)

# One-hot encoding
Y_train = np.eye(num_classes)[y_train]
Y_test = np.eye(num_classes)[y_test]
Y_val = np.eye(num_classes)[y_val]


def data_augmenter():
    '''
    Create a Sequential model composed of multiple augmentation layers.
    Returns:
        tf.keras.Sequential
    '''

    data_augmentation = tf.keras.models.Sequential([
        RandomRotation(0.02),
        RandomContrast(0.2),
        RandomBrightness(0.2)
    ])

    return data_augmentation


# Instantiate the data augmenter
augmenter = data_augmenter()


# Function to augment images
def augment_image(img, labl):
    """
    Applies augmentation to the given image and normalizes it to the [0, 1] range.

    Arguments:
        img -- input image tensor.
        labl -- the label associated with the image.

    Returns:
        tuple -- A tuple containing the augmented and normalized image tensor and the original label.
    """

    img = augmenter(img)
    img = tf.cast(img, tf.float32) / 255.0  # Normalize the image to the [0, 1] range
    return img, labl


# Testing the data augmenter
plt.figure(figsize=(10, 10))
first_image, first_label = X_train[0], Y_train[0]

for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = augment_image(first_image, first_label)
    plt.imshow(augmented_image[0])
    plt.axis('off')
plt.show()


# Building datasets for efficiency
# Cache: Store in memory to avoid repeated loading and processing.
# Shuffle: Present data in random order to improve generalization and prevent overfitting.
# Batch: Group data into batches of 32 for efficient processing.
# Prefetch: Overlap data loading with training to enhance efficiency.

AUTOTUNE = tf.data.experimental.AUTOTUNE
m = X_train.shape[0]  # Training samples
minibatch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.cache().shuffle(buffer_size=m).batch(minibatch_size).prefetch(AUTOTUNE)
train_dataset = train_dataset.map(lambda a, b: augment_image(a, b))


def print_max_pixel_intensity(dataset, type_dataset=None):
    """
    Function to get the first batch from the dataset, convert to numpy,
    and print the maximum pixel intensity of the images.

    Arguments:
    dataset -- tf.data.Dataset, the dataset to process
    type_dataset -- string specifying whether the dataset is train, validation, or test set
    """

    image, label = next(iter(dataset.take(1)))
    image_np = image.numpy()
    max_value = np.max(image_np)

    # Print the maximum value to make sure max pixel intensity is 1
    print(f"{type_dataset.title()} dataset max pixel intensity: {max_value}")


# Training set has been normalized to range [0,1] in augment_image(). Normalizing test and validation sets
if np.max(X_val) > 1:
    X_val = X_val / 255.
    X_test = X_test / 255.

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_dataset = val_dataset.cache().shuffle(buffer_size=len(X_val)).batch(minibatch_size).prefetch(buffer_size=AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
test_dataset = test_dataset.shuffle(buffer_size=len(X_test)).batch(minibatch_size)

print_max_pixel_intensity(train_dataset, "train")
print_max_pixel_intensity(val_dataset, "validation")
print_max_pixel_intensity(test_dataset, "test")


# BUILD A RESIDUAL NETWORK
# Here is the picture of the residual network being built
resnet_architecture_path = os.path.join(os.getcwd(), "images", "resnet50.png")
architecture = plt.imread(resnet_architecture_path)
plt.imshow(architecture)
plt.axis('off')
plt.title("ResNet50")
plt.show()

# Note: The following figure describes the architecture of the residual network. "ID BLOCK" in the diagram stands for
# "Identity block" while "CONV BLOCK" stands for "Convolutional block." And "ID BLOCK x3" means stacking 3 ID BLOCKS.
# Here is a look at the ID BLOCK and CONV BLOCK
idblock_architecture_path = os.path.join(os.getcwd(), "images", "idblock3.png")
convblock_architecture_path = os.path.join(os.getcwd(), "images", "convblock.png")

fig, ax = plt.subplots(2, 1)
ax[0].imshow(plt.imread(idblock_architecture_path))
ax[0].set_title('Identity block')
ax[0].axis('off')
ax[1].imshow(plt.imread(convblock_architecture_path))
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
    # X = Dropout(rate=0.2)(X, training=training)
    X = Dense(num_class, activation='softmax', kernel_initializer=glorot_uniform)(X)

    # Create model
    resnet_model = Model(inputs=X_input, outputs=X)

    return resnet_model


# TRAIN THE MODEL
# Define the model
model = ResNet50(input_shape=image_shape, num_class=num_classes, training=True)
print(model.summary())

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0008), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks to be used during training
# Stop training metric has stopped improving.
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Model checkpoint to save the best model
model_path = os.path.join(os.getcwd(), "models", "best_model.h5")
check_point = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1)

# Reduce learning rate when a metric has stopped improving.
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.125, patience=10, min_lr=1e-5)
callbacks = [early_stopping, check_point]
# callbacks = [early_stopping, check_point, reduce_lr]

# Train the model
history = model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=callbacks)


def plot_history(hist):
    """
    Plots the training and validation accuracy and loss history.

    Arguments:
        history -- keras.callbacks.History object that contains training history
    """

    # Extract training and validation accuracy from history
    accuracy = hist.history['accuracy']
    val_accuracy = hist.history['val_accuracy']

    # Extract training and validation loss from history
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    # Set the figure size for the plots
    plt.figure(figsize=(8, 8))

    # Plot Training and Validation Loss
    plt.subplot(2, 1, 1)
    plt.plot(loss, '-o', label='Training Loss')
    plt.plot(val_loss, '-o', label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')

    # Plot Training and Validation Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(accuracy, '-o', label='Training Accuracy')
    plt.plot(val_accuracy, '-o', label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')

    # Display the plots
    plt.tight_layout()
    plt.show()


plot_history(history)


# EVALUATE THE MODEL
best_model_path = model_path
pretrained_model = load_model(best_model_path)
pretrained_model.trainable = False
# tf.keras.backend.set_learning_phase(False)

print(pretrained_model.evaluate(train_dataset))
print(pretrained_model.evaluate(val_dataset))

test_hist = pretrained_model.evaluate(X_test, Y_test)
print(f"Model Accuracy: {test_hist[1]: .2f}")
