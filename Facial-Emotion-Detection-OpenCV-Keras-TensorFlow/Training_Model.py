from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.
# It allows for easy and fast prototyping, supports convolutional networks and recurrent networks, and runs seamlessly on CPU and GPU.

# TensorFlow is an open-source machine learning framework developed by Google, widely used for building and training deep learning models.
# It provides a flexible platform for building machine learning models, including neural networks, and is designed for both research and production use.

# OpenCV is an open-source computer vision and machine learning software library that provides a common infrastructure for computer vision applications.

train_data_dir = 'HRI-Internship-Project-25/Facial-Emotion-Detection-OpenCV-Keras-TensorFlow/dataset/train/' # Directory containing training images
validation_data_dir = 'HRI-Internship-Project-25/Facial-Emotion-Detection-OpenCV-Keras-TensorFlow/dataset/test/'# Directory containing validation images (test dataset)

# Data augmentation, which helps to prevent overfitting by generating more diverse training data from the existing images.
# This includes random transformations such as rotation, zoom, shear, and horizontal flipping.
# Overfitting occurs when the model learns the training data too well, including its noise and outliers, which can lead to poor generalization on unseen data.
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values to [0, 1]
    rotation_range=30, # Randomly rotate images by up to 30 degrees 
    shear_range=0.3,  # Randomly shear images by 30%
    zoom_range=0.3, # Randomly zoom images by up to 3
    horizontal_flip=True, # Randomly flip images horizontally
    fill_mode='nearest' # Fill in new pixels with the nearest pixel value
)

validation_datagen = ImageDataGenerator(rescale=1./255) # Only rescale train images

# Create generators for training dataset
train_generator = train_datagen.flow_from_directory(
    train_data_dir, # Directory containing training images
    target_size=(48, 48), # Resize images to 48x48 pixels
    color_mode='grayscale', # Convert images to grayscale
    batch_size=32, # Number of images to process in a batch
    class_mode='categorical', # Use categorical labels for multi-class classification
    shuffle=True # Shuffle the training data
)

# Create a validation generator to load images from the validation directory (test dataset)
# This generator will not apply data augmentation, only rescaling.
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir, # Directory containing validation images
    target_size=(48, 48), # Resize images to 48x48 pixels
    color_mode='grayscale', # Convert images to grayscale
    batch_size=32, # Number of images to process in a batch
    class_mode='categorical', # Use categorical labels for multi-class classification
    shuffle=True # Shuffle validation data
)

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] # List of class labels for the dataset

img, label = next(train_generator) # Get a batch of images and labels from the training generator

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)), # First convolutional layer, the input shape is (48, 48, 1) for grayscale images, where 1 is the number of channels
    # 32 filters of size 3x3, using ReLU activation function
    # ReLU activation function is used to introduce non-linearity

    Conv2D(64, kernel_size=(3, 3), activation='relu'), # Second convolutional layer
    MaxPooling2D(pool_size=(2, 2)), # First max pooling layer, which reduces the spatial dimensions of the output
    Dropout(0.1), # Dropout layer to prevent overfitting, it randomly sets 10% of the input units to 0 during training

    Conv2D(128, kernel_size=(3, 3), activation='relu'), # Third convolutional layer
    MaxPooling2D(pool_size=(2, 2)), # Third max pooling layer
    Dropout(0.1),

    Conv2D(256, kernel_size=(3, 3), activation='relu'), # Fourth convolutional layer
    MaxPooling2D(pool_size=(2, 2)), # Fourth max pooling layer
    Dropout(0.1),

    Flatten(), # Flatten the output from the convolutional layers, which converts the 2D matrix into a 1D vector
    # This is necessary before passing the data to fully connected layers, as they expect 1D input.
    Dense(512, activation='relu'), # Fully connected layer with 512 neurons, dense layer is a layer where each neuron is connected to every neuron in the previous layer
    Dropout(0.2), # Dropout layer to prevent overfitting

    Dense(len(class_labels), activation='softmax') # Output layer with softmax activation for multi-class classification, it has as many neurons as there are classes in the dataset
    # So there are 7 neurons for the 7 classes in the `class_labels` list.
    # Softmax activation function is used to convert the output into probabilities for each class that sum to 1.
    # If you had only two classes, you could use a single neuron with sigmoid activation instead.
])

model.compile(
    optimizer='adam', # Adam optimizer for training, it is an adaptive learning rate optimization algorithm that is widely used in deep learning
    # It combines the advantages of two other popular optimizers, AdaGrad and RMSProp.
    # Adam is generally a good choice for most problems and is often the default optimizer used in Keras.
    # It adapts the learning rate for each parameter based on the first and second moments of the gradients.
    # It is efficient and works well with large datasets and high-dimensional parameter spaces.
    loss='categorical_crossentropy', # Loss function for multi-class classification
    # A loss function measures how well the model's predictions match the true labels by calculating the difference between the output probabilities given by the model and the true labels.
    # Categorical crossentropy is used when the target variable is one-hot encoded (i.e., each class is represented by a binary vector).
    # It is suitable for multi-class classification problems where each input belongs to one of several classes.
    # It calculates the cross-entropy loss between the true labels and the predicted probabilities.
    metrics=['accuracy'] # Metric to evaluate the model's performance, it measures the accuracy of the model's predictions.
)

print(model.summary()) # Print the model summary to see the architecture and number of parameters

# Train the model using the training and validation generators
train_path = 'HRI-Internship-Project-25/Facial-Emotion-Detection-OpenCV-Keras-TensorFlow/dataset/train/'
test_path = 'HRI-Internship-Project-25/Facial-Emotion-Detection-OpenCV-Keras-TensorFlow/dataset/test/'

# In the following code, we will calculate the number of training and test images in the respective directories.
# This is useful for understanding the size of the dataset and for setting up the training process.
num_train_imgs = 0 # Initialize the number of training images
for root, dirs, files in os.walk(train_path): # Walk through the training directory
    # os.walk generates the file names in a directory tree by walking the tree either top-down or bottom-up.
    # For each directory in the tree rooted at directory top (including top itself), it yields a 3-tuple (dirpath, dirnames, filenames).
    # dirpath is a string, the path to the directory.
    num_train_imgs += len(files)

num_test_imgs = 0 # Initialize the number of test images
for root, dirs, files in os.walk(test_path):
    num_test_imgs = len(files)

epochs = 100 # Number of epochs for training, an epoch is one complete forward and backward pass of all the training examples.
# An epoch means the model looks at every training image once.
# The data is split into small groups called batches.
# For each batch, the model makes predictions and learns from its mistakes.
# After going through all batches (the whole dataset), one epoch is complete.
# Training usually runs for several epochs to help the model improve.
# The number of epochs is a hyperparameter that you can tune based on the performance of the model on the validation set.
# A higher number of epochs can lead to better performance, but it can also lead to overfitting if the model learns the training data too well.

history = model.fit(
    train_generator, # The generator will yield batches of images and labels, which will be used to train the model. 
    # This was created above using the `ImageDataGenerator` class, which applies data augmentation to the training images.
    steps_per_epoch=num_train_imgs // 32, # Number of batches (32, as it was set before) to draw from the training generator per epoch
    epochs=epochs, # Number of epochs to train the model
    validation_data=validation_generator, # Validation data generator
    validation_steps=num_test_imgs // 32 # Number of batches to draw from the validation generator per epoch
)

model.save("emotion_model_100_epochs.h5") # Save the trained model to a file, so it can be used later for predictions without retraining
