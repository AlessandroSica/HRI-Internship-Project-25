import os # os module provides a way of using operating system dependent functionality like reading or writing to the file system.
import librosa # librosa is a Python library for audio and music analysis, providing tools for feature extraction, audio processing, and more.
import numpy as np # numpy is a library for numerical computing in Python, providing support for arrays, matrices, and mathematical functions.

# Path to the dataset
dataset_path = 'ravdess_data'

# End-to-end function to load the dataset and extract features
def extract_features(file_path, n_mfcc=13): # Function to extract features from an audio file
    signal, sample_rate = librosa.load(file_path, sr=22050) # Load the audio file with a sample rate of 22050 Hz
    mfccs = librosa.feature.mfcc(y= signal, sr=sample_rate, n_mfcc=n_mfcc) # Extract MFCC features
    # MFCCs (Mel-Frequency Cepstral Coefficients) are features we extract from audio to help a computer
    # understand speech more like a human does. They capture the overall shape of how energy is distributed
    # across different frequencies in our voice, which changes depending on what emotion or word is spoken.
    # To calculate MFCCs, we first split the audio into small time windows (like tiny audio snapshots),
    # then we transform each window into its frequency content using the Fourier Transform.
    # We pass these frequencies through filters spaced according to the Mel scale, which mimics how
    # human ears hear sound (we hear differences better at low frequencies than at high ones).
    # Next, we take the log of the energy in each filter to simulate how we perceive loudness.
    # Finally, we apply the Discrete Cosine Transform (DCT) to make the features compact and smooth.
    # The result is a small set of numbers (MFCCs) that summarize the key audio characteristics,
    # which we can then use for things like emotion detection or speech recognition.

    mfccs_processed = np.mean(mfccs.T, axis=0) # Compute the mean of the MFCC features across time frames. Mean Pooling is used to reduce the dimensionality of the feature set.
    # Mean pooling is a technique where we take the average of the features over time to create a single, fixed-size representation.
    return mfccs_processed # Return the processed MFCCs features

# Batch Processing for Dataset Preparation
# A batch is a group of samples processed together in one go, which is efficient for training models.
features = [] # List to store the extracted features
labels = [] # List to store the corresponding labels

for file_name in os.listdir(dataset_path): # Loop through each file in the dataset directory
    if file_name.endswith('.wav'): # Check if the file is a .wav audio file
        emotion_label = file_name.split('-')[2] # Extract the emotion label from the filename (assuming the label is in the third position)
        file_path = os.path.join(dataset_path, file_name) # Get the full path of the audio file
        mfcc = extract_features(file_path) # Extract features from the audio file as mfccs
        features.append(mfcc) # Append the extracted features to the list
        labels.append(emotion_label) # Append the corresponding label to the list

# Convert the lists to numpy arrays for easier manipulation
x = np.array(features) # Convert the features list to a numpy array
y = np.array(labels) # Convert the labels list to a numpy array

# Building the Emotion Classification Model---------------------------------------------------------------------------------------------------

# Step 1: Split the dataset into training and testing sets

# x = list of MFCC feature arrays-----
# y = list of emotion labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
# This line splits the dataset into training and testing sets.
# 'x' contains the input features (e.g., MFCCs), and 'y' contains the labels (e.g., emotions).
# 80% of the data will go to training (x_train, y_train), and 20% to testing (x_test, y_test).
# 'random_state=42' ensures that the split is reproducible every time you run the code. 
# The order of the audio should randomize in the test and train split, this is done by default.
# 'stratify=y' makes sure the class distribution (e.g., number of samples per emotion) is the same
# in both the training and testing sets, which helps create a balanced and fair evaluation.

# Step 2: choose the right model-----

# Comparing two models RandomForestClassifier and Support Vector Classifier (SVC)

# Import the RandomForestClassifier, a machine learning model that uses multiple decision trees for classification
from sklearn.ensemble import RandomForestClassifier
# Import the Support Vector Classifier (SVC), a model that finds the best boundary to separate different classes
from sklearn.svm import SVC

# Initialize models
# Initialize a Random Forest classifier with a fixed random state for reproducibility
rf_model = RandomForestClassifier(random_state=42)
# Initialize a Support Vector Machine (SVM) classifier:
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
# - kernel='rbf' sets the Radial Basis Function kernel, which helps the model handle non-linear relationships by mapping data into higher-dimensional space.
# - probability=True enables the model to output probability estimates for each class, useful for tasks like confidence scoring or combining with other models.
# - random_state=42 ensures that the training process is reproducible by fixing the randomness in the algorithm (e.g., for internal shuffling or tie-breaking).

# Step 3: Tune and Validate with Cross-Validation-----

# Import GridSearchCV for hyperparameter tuning and StratifiedKFold for stratified cross-validation
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Define the hyperparameter grid for Random Forest:
# 'n_estimators' controls the number of trees in the forest
# 'max_depth' limits how deep each tree can grow (None means unlimited depth)
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}

# Define the hyperparameter grid for SVM:
# 'C' is the regularization parameter (higher values fit training data more closely)
# 'gamma' defines the influence of each support vector (low values = far reach)
svm_params = {
    'C': [1, 10, 100],
    'gamma': ['scale', 0.01, 0.1]
}

# Create a StratifiedKFold object for cross-validation:
# It splits the dataset into 10 folds while preserving the percentage of samples
# for each class (stratification), which is especially important for imbalanced data
cv = StratifiedKFold(n_splits=10)

# Create a GridSearchCV object for Random Forest:
# - Tests every combination of hyperparameters from 'rf_params'
# - Uses 10-fold stratified cross-validation (cv)
# - Evaluates each combination based on accuracy
# - Runs in parallel using all CPU cores (n_jobs=-1)
rf_grid = GridSearchCV(rf_model, rf_params, cv=cv, scoring='accuracy', n_jobs=-1)

# Create a GridSearchCV object for SVM with similar setup
svm_grid = GridSearchCV(svm_model, svm_params, cv=cv, scoring='accuracy', n_jobs=-1)

# Fit both grid searches on the training data to find the best hyperparameters
rf_grid.fit(x_train, y_train)
svm_grid.fit(x_train, y_train)








