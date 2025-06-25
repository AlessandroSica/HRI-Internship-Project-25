import pandas as pd # Importing pandas for data manipulation, to analyze or CSV file.
# Pandas is a powerful library for data manipulation and analysis, providing data structures like DataFrames that are ideal for handling structured data.
from sklearn.model_selection import train_test_split # Importing train_test_split from scikit-learn for splitting datasets
# train_test_split is a utility function from scikit-learn that allows you to split your dataset into training and testing sets, which is essential for evaluating machine learning models.
from sklearn.pipeline import make_pipeline # Importing make_pipeline from scikit-learn for creating a machine learning pipeline.
# In programming, a pipeline refers to a sequence of processing steps, where the output of one step is the input to the next. It's a way of structuring data processing in a clear, modular, and efficient manner.
from sklearn.preprocessing import StandardScaler # Importing StandardScaler for feature scaling. It standardizes features by removing the mean and scaling to unit variance.
# So it makes the data have a mean of 0 and a standard deviation of 1, which is important for many machine learning algorithms to perform well.
from sklearn.linear_model import LogisticRegression, RidgeClassifier # Importing LogisticRegression and RidgeClassifier for classification tasks.
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # Importing RandomForestClassifier and GradientBoostingClassifier for ensemble learning methods.
# All four machine learning classifiers are used for classification tasks, where the goal is to predict a categorical label based on input features.

df = pd.read_csv('coords.csv', delimiter=';') # Creating an empty DataFrame to store the landmark data. 
# The DataFrame will be used to store the coordinates of the landmarks detected by MediaPipe Holistic.
# We do this so we can work easily with the data using pandas, a powerful data manipulation library in Python. 

x = df.drop('class', axis=1) # Dropping the 'class' column from the DataFrame to create the feature set (X).
# Drop is meant to remove the 'class' column from the DataFrame, which contains the labels for the emotions.
# Features are the input variables used to train a machine learning model.

y = df['class'] # Extracting the 'class' column from the DataFrame to create the target variable (y).
# So keeping only the 'class' column, which contains the labels for the emotions, to create the target variable.
# The target variable is the output variable that the model will learn to predict based on the features.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234) # Splitting the dataset into training and testing sets.
# x_train and y_train will be used to train the model, while x_test and y_test will be used to evaluate its performance.
# The set is split into 70% training data and 30% testing data, with a random state for reproducibility.    
# Random state ensures that the split is reproducible, meaning that every time you run the code, you will get similar split of data.

pipelines = { # Creating a dictionary to store different machine learning pipelines.
    'lr': make_pipeline(StandardScaler(), LogisticRegression()), # Logistic Regression pipeline with feature scaling.
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()), # Ridge Classifier pipeline with feature scaling.
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()), # Random Forest Classifier pipeline with feature scaling.
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()) # Gradient Boosting Classifier pipeline with feature scaling.
}

fit_models = {} # Creating an empty dictionary to store the fitted (trained) models.
for algo, pipeline in pipelines.items(): # Looping through the pipelines dictionary (created above, which contains 4 different models) to fit each model.
    model = pipeline.fit(x_train, y_train) # Fitting the pipeline to the training data, or training each of the 4 models to the training data (x_train and y_train).
    fit_models[algo] = model # Storing the fitted (trained) model in the fit_models dictionary with the algorithm name as the key, (alo for algorithm).

# First, we create an empty dictionary called 'fit_models'. This dictionary will be used to store all the models after they are trained (also known as "fitted").
# We then loop through each item in the 'pipelines' dictionary. Each item in this dictionary represents a machine learning algorithm (like Random Forest, SVM, etc.)
# and its corresponding pipeline. A pipeline is a sequence of steps that includes data preprocessing (like scaling or transforming the data)
# and the machine learning model itself. This makes it easier to manage and repeat the same steps for different models.
# Inside the loop, we train each model using the pipeline’s .fit() method, which takes the training data (x_train and y_train) and teaches the model
# how to make predictions based on that data. This process adjusts the internal parameters of the model to best match the patterns in the data.
# After the training is done, we store the resulting trained model in the 'fit_models' dictionary.
# We use the name of the algorithm (e.g., "Random Forest", "Logistic Regression") as the key, and the trained model as the value.
# This allows us to keep track of all the different models we trained and easily access or compare them later.
