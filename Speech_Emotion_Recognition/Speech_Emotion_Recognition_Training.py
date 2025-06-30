import os # os module provides a way of using operating system dependent functionality like reading or writing to the file system.
import librosa # librosa is a Python library for audio and music analysis, providing tools for feature extraction, audio processing, and more.
import numpy as np # numpy is a library for numerical computing in Python, providing support for arrays, matrices, and mathematical functions.
from sklearn.model_selection import train_test_split # Importing train_test_split from scikit-learn for splitting datasets
# train_test_split is a utility function from scikit-learn that allows you to split your dataset into training and testing sets, which is essential for evaluating machine learning models.
import joblib

# Path to the dataset
dataset_path = 'HRI-Internship-Project-25/Speech_Emotion_Recognition/Dataset_Speech_Emotion_Recognition'

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

# Recursively scan through the dataset directory to find and process all .wav files
for root, dirs, files in os.walk(dataset_path):
    # os.walk() traverses the directory tree rooted at `dataset_path`
    # It yields a tuple: (root - current directory path, dirs - subdirectories, files - filenames)
    for file_name in files:
        # Iterate through each file in the current folder
        if file_name.endswith('.wav'):
            # Process only files that end with '.wav' (audio files)
            # This ensures we skip over any irrelevant files
            file_path = os.path.join(root, file_name)
            # Combine the current root folder with the file name to get the full file path
            emotion_label = file_name.split('-')[2]
            # Assuming the filename format is consistent (e.g., '03-01-05-01-02-01-12.wav')
            # Split the filename by dashes and extract the third item (index 2)
            # This typically encodes the emotion label
            mfcc = extract_features(file_path)
            # Extract the MFCC features from the audio file using your custom function
            features.append(mfcc)
            # Store the feature array for this audio sample into the list of all features
            labels.append(emotion_label)
            # Store the corresponding emotion label into the list of all labels

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

# Step 4: Evaluation of the model using confusion matrices-----

# Import the accuracy_score function to compute the classification accuracy
# Import confusion_matrix to generate confusion matrices
# Import ConfusionMatrixDisplay to visualize the confusion matrices
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# Import the matplotlib library to create visual plots
import matplotlib.pyplot as plt

# A confusion matrix is a performance evaluation tool for classification models.
# It shows how many predictions fall into four categories:
# - True Positives (TP): Correctly predicted positive class
# - True Negatives (TN): Correctly predicted negative class
# - False Positives (FP): Incorrectly predicted positive class (actually negative)
# - False Negatives (FN): Incorrectly predicted negative class (actually positive)

# For binary classification, the confusion matrix is a 2x2 table:
#                 Predicted
#                |  Pos  |  Neg
#         -------|-------|------
#     Actual Pos |  TP   |  FN
#     Actual Neg |  FP   |  TN

# For multiclass classification, it's an N x N matrix:
# - Rows represent the actual classes
# - Columns represent the predicted classes
# - The diagonal contains correct predictions
# - Off-diagonal values show misclassifications

# It allows calculation of metrics such as:
# - Accuracy = (TP + TN) / total
# - Precision = TP / (TP + FP)
# - Recall = TP / (TP + FN)
# - F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

# Extract the best Random Forest model found during GridSearchCV
best_rf = rf_grid.best_estimator_
# GridSearchCV is a tool from sklearn that automates the process of tuning hyperparameters.
# It takes a model (e.g., Random Forest) and a dictionary of parameters to test, 
# then trains and evaluates the model on different combinations using cross-validation.
# It selects the parameter combination that performs best on the validation sets.
# The trained model using the best parameter combination can be accessed using .best_estimator_.

# Extract the best Support Vector Machine (SVM) model found during GridSearchCV
best_svm = svm_grid.best_estimator_

# Use the best Random Forest model to predict labels for the test feature set (x_test)
rf_preds = best_rf.predict(x_test)

# Use the best SVM model to predict labels for the test feature set (x_test)
svm_preds = best_svm.predict(x_test)

# Calculate the accuracy of the Random Forest model by comparing predicted vs. true labels
rf_acc = accuracy_score(y_test, rf_preds)

# Calculate the accuracy of the SVM model by comparing predicted vs. true labels
svm_acc = accuracy_score(y_test, svm_preds)

# Print the accuracy of the Random Forest model to the console
print('Random Forest Accuracy', rf_acc)

# Print the accuracy of the SVM model to the console
print('SVM Accuracy', svm_acc)

# Plotting results

# Create a figure with 1 row and 2 columns of subplots to display both confusion matrices side-by-side
# Set the figure size to 12 inches wide by 5 inches tall
fig, axes = plt.subplots(1, 2, figsize=(12, 5)) 

# Generate and display the confusion matrix for the Random Forest model
# - Compute the confusion matrix comparing true vs. predicted labels
# - Use class labels inferred from the best RF model
# - Plot it on the first subplot (axes[0]) with a green color scheme
ConfusionMatrixDisplay(
    confusion_matrix(y_test, rf_preds),  # Compute confusion matrix for RF predictions
    display_labels=best_rf.classes_      # Use label names from the classifier
).plot(ax=axes[0], cmap='Greens')        # Display on left axis with green colors

# Set a title for the first subplot to indicate it shows the Random Forest confusion matrix
axes[0].set_title('Random Forest Confusion Matrix')

# Generate and display the confusion matrix for the SVM model
# - Compute the confusion matrix comparing true vs. predicted labels
# - Use class labels inferred from the best SVM model
# - Plot it on the second subplot (axes[1]) with a blue color scheme
ConfusionMatrixDisplay(
    confusion_matrix(y_test, svm_preds),  # Compute confusion matrix for SVM predictions
    display_labels=best_svm.classes_      # Use label names from the classifier
).plot(ax=axes[1], cmap='Blues')          # Display on right axis with blue colors

# Set a title for the second subplot to indicate it shows the SVM confusion matrix
axes[1].set_title('SVM Confusion Matrix')

# Adjust layout so subplots don’t overlap and everything fits nicely in the figure
plt.tight_layout()

# Display the entire plot window showing both confusion matrices
plt.show()

# Results after training:
# Random Forest Accuracy 0.9583333333333334
# SVM Accuracy 0.8993055555555556

joblib.dump(best_rf, 'Speech_random_forest_model.pkl') # Save the model as a .pkl file










