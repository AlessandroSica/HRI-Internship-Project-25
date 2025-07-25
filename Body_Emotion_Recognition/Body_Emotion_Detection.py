import mediapipe as mp # Face detection library, MediaPipe Holistic module provides a holistic approach to human pose, face, and hand detection. Whenever you want to use MediaPipe, you need to import the mediapipe library first. 
# Whenever you then call the variable mp, it refers to the mediapipe library, which is a collection of tools and models for computer vision tasks, including face detection, pose estimation, and hand tracking.
import cv2 # OpenCV library for computer vision tasks, such as image processing and video capture.
import csv # This module is used to write data to a CSV file.
import os # This module is used to interact with the operating system, such as creating directories and checking if files exist.
import numpy as np # NumPy is used for numerical operations, such as creating arrays and manipulating data.
import pandas as pd # Importing pandas for data manipulation, to analyze or CSV file.
import pickle # Importing pickle to save the trained models to disk for later use.  

with open('Body_fit_models_v2_lr.pkl', 'rb') as f:  # Opening the file 'fit_models.pkl' in read-binary mode to load the trained models.
    model = pickle.load(f)  # Loading the trained model from the file.

mp_drawing = mp.solutions.drawing_utils # Drawing utilities for visualizing landmarks and connections.
mp_holistic = mp.solutions.holistic # Holistic module for detecting face, pose, and hand landmarks. Imported from MediaPipe library.

cap = cv2.VideoCapture(1) # Open the default camera (0) for video capture.

# Initialize the MediaPipe Holistic model with default parameters.
with mp_holistic.Holistic(
    min_detection_confidence=0.5, # Minimum confidence threshold for face detection.
    min_tracking_confidence=0.5 # Minimum confidence threshold for landmark tracking.
) as holistic: # Use the holistic model in a context manager to ensure proper resource management.
    
    # Creating the CSV file to store the landmark data.
    # In order to do it we need to do the same steps that we do inside the loop once, so to initialize it. As we can't put it into the loop or otherwise it will create a new file every time the loop runs.
    #----------------------------------------------------------------------------------------------------
    ret, frame = cap.read() # Read a frame from the camera to initialize the model.
    if not ret: # Check if the frame was successfully captured.
        print("Failed to grab initial frame.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the frame to RGB format for MediaPipe processing.
    image.flags.writeable = False # Set the image to non-writeable for faster processing.
    results = holistic.process(image) # Make detections and track landmarks in the image.
    image.flags.writeable = True

    if results.face_landmarks and results.pose_landmarks:  # Check if both face and pose landmarks are detected.
        num_coords = len(results.face_landmarks.landmark) + len(results.pose_landmarks.landmark) # Get the total number of landmarks detected for both face and pose.
        landmark = ['class'] # Initialize a list to store landmark names, starting with 'class' for the name of the column where all the emotion classes will be stored, such as 'happy', 'sad', etc.
        for val in range(1, num_coords + 1): # Create a list of landmark names for each coordinate (x, y, z) for each landmark, and one for visibility (v) of each landmark.
            landmark += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

        if not os.path.exists('coords.csv'): # Check if the CSV file already exists.
            with open('coords.csv', mode='w', newline='') as f: # If it doesn't exist, create it and write the header row.
                csv_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL) # Write the header row to the CSV file.
                csv_writer.writerow(landmark) # This will create the CSV file with the header row containing the names of the columns: 'class', 'x1', 'y1
    #----------------------------------------------------------------------------------------------------

    while cap.isOpened(): # Loop until the camera is closed.
        ret, frame = cap.read() # Read a frame from the camera.
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert the frame to RGB format for MediaPipe processing.
        # When using MediaPipe, the input image should be in RGB format.
        # OpenCV captures images in BGR format, so we need to convert it.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # Set the image to non-writeable for faster processing.
        # Make detections and track landmarks in the image.
        results = holistic.process(image)
        image.flags.writeable = True # Set the image back to writeable after processing.
        
        # Convert the image back to BGR format for OpenCV display.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Draw face landmarks on the image.
        
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION, # Draw face landmarks using the FACEMESH_TESSELATION connections.
            # The FACEMESH_TESSELATION is a predefined set of connections for face landmarks
            # mp_holistic.FACE_CONNECTIONS wasn't working, so I used mp.solutions.face_mesh.FACEMESH_TESSELATION instead.
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(200, 0, 0), thickness=1, circle_radius=1), # Color of the dots (landmarks). It's BGR format, so (0, 255, 0) is green.
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1) # Color of the lines (connections between landmarks).
        ) 
        
        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # Draw left hand landmarks on the image.
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )

        # Draw right hand landmarks on the image.
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )

        num_coords = len(results.face_landmarks.landmark) + len(results.pose_landmarks.landmark)
        # num_coords stores the total number of landmarks detected for both face and pose, not including hands
        
        # Creating the CSV file to store the landmark data.

        landmark = ['class'] # Initialize a list to store landmark names, starting with 'class' for the name of the column where all the emotion classes will be stored, such as 'happy', 'sad', etc.
        # This CSV file will be used to label the data for training a machine learning model later.
        
        for val in range(1, num_coords+1):
            #landmark += [f'x{val}', f'y{val}', f'z{val}', f'v{val}'] # equivalent to the following line, but using f-strings for better readability.
            landmark += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)] # Create a list of landmark names for each coordinate (x, y, z) for each landmark, and one for visibility (v) of each landmark.
            # The visibility (v) indicates whether the landmark is visible in the frame (1) or not (0).
            # Note that the visibility of all the landmarks of the face is 0 by default for some reason.
            # This list stores all of the names of the columns of CSV file: 'x1', 'y1'... One for every coordinate to see how they vary overtime

        class_name = 'Very Happy' # Define the class name for the current emotion, this will be used to label the data in the CSV file.
        # You can change this to any emotion class you want to label the data with, such as 'sad', 'angry', etc.

        # Export landmarks to CSV file.
        try:
            # Extract pose landmarks.
            pose = results.pose_landmarks.landmark if results.face_landmarks else [] # Extract pose landmarks from the results.
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            # Put all the pose landmarks coordinates into a single row array, without any other information, for every landmark in pose.
            # Flatten the array to make it a single row, where each landmark's x, y, z coordinates and visibility are stored sequentially.
            # Flatten collapses an array of any shape into a one-dimensional array.

            # Extract face landmarks.
            face = results.face_landmarks.landmark if results.face_landmarks else [] # Extract face landmarks from the results, if they exist.
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            row = pose_row + face_row # Combine pose and face landmarks into a single row.

            # Make detections.
            x = pd.DataFrame([row]) # Create a DataFrame using pandas from the row of landmarks, which contains the coordinates and visibility of each landmark.
            body_language_class = model.predict(x)[0] # Predict the body language class using the trained model. Predict, as in the model will predict the emotion class based on the landmarks detected in the current frame.
            # 0 is the index of the predicted class, as the classes are ordered based on their accuracy given the input, so the one with index 0 is the most accurate prediction.
            body_language_prob = model.predict_proba(x)[0] # Get the probabilities of class with index 0 for the current frame.
            print(body_language_class, body_language_prob) # Print the predicted class and its probability.

            if results.pose_landmarks and results.face_landmarks: # Check if both pose and face landmarks are detected.
                try:
                # Grab ear coordinate (x, y), in order to use it as a reference on where to print the text on the frame to show the predicted emotion.
                    left_ear = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR] # Get the coordinates of the left ear landmark, from the data gathered by the holistic model and stored in results.
                    coords = tuple(np.multiply( # OpenCV expects a tuple of coordinates in the format (x, y) for drawing text on the frame.
                        np.array((left_ear.x, left_ear.y)),
                        [640, 480]  # Multiply the coordinates by the width and height of the frame, using numpy, to get the actual pixel values.
                    ).astype(int)) # It has to be converted to integers, as the coordinates are in float format.
            
                    cv2.rectangle(image, (coords[0], coords[1] + 5), (coords[0] + len(body_language_class) * 20, coords[1] - 20), (245, 117, 16), -1) # Draw a rectangle around the left ear landmark to highlight it.
                    cv2.putText(image, body_language_class, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # Draw the text reading the predicted emotion class on the frame at the coordinates of the left ear landmark.
                except Exception as e:
                    print("Drawing failed:", e) # Handle any exceptions that may occur during the drawing process.

            # Get status box
            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1) # Draw a rectangle at the top of the frame to display the status box. -1 means the rectangle will be filled with the color (245, 117, 16).

            # Display the predicted emotion class in the status box.
            cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA) # Title that says class
            cv2.putText(image, body_language_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # Display the predicted emotion class in the status box.
            # Using split to only show the first word of the class, in case it has multiple words, such as 'Very Happy', it will only show 'Very'.

            # Display the predicted probability in the status box.
            cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA) # Title that says Prob
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # Display the predicted probability in the status box.
            # The np.argmax function returns the index of the maximum value in the array, rounded to two decimal places, which corresponds to the predicted class with the highest probability. This is then converted to a string and displayed on the frame.

        except:
            pass # Handle any exceptions that may occur during the export process.

        cv2.imshow('Holistic Model Detection', image) # Display the original frame in a window.

        if cv2.waitKey(10) & 0xFF == ord('q'): # If the 'q' key is pressed, break the loop and exit.
            break   

cap.release() # Release the camera resource.
cv2.destroyAllWindows() # Close all OpenCV windows.