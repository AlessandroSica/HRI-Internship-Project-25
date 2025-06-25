import mediapipe as mp # Face detection library, MediaPipe Holistic module provides a holistic approach to human pose, face, and hand detection. Whenever you want to use MediaPipe, you need to import the mediapipe library first. 
# Whenever you then call the variable mp, it refers to the mediapipe library, which is a collection of tools and models for computer vision tasks, including face detection, pose estimation, and hand tracking.
import cv2 # OpenCV library for computer vision tasks, such as image processing and video capture.

import csv # This module is used to write data to a CSV file.
import os # This module is used to interact with the operating system, such as creating directories and checking if files exist.
import numpy as np # NumPy is used for numerical operations, such as creating arrays and manipulating data.

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
        print(results) # Print the results for debugging purposes.
        
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
            row.insert(0, class_name) # Insert the class name at the beginning of the row.

            with open('coords.csv', mode='a', newline='') as f: # a means append mode, so it will add new data to the end of the file without overwriting existing data.
                csv_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row) # Write the row to the CSV file.
                #newline    

        except:
            pass # Handle any exceptions that may occur during the export process.

        cv2.imshow('Holistic Model Detection', image) # Display the original frame in a window.

        if cv2.waitKey(10) & 0xFF == ord('q'): # If the 'q' key is pressed, break the loop and exit.
            break   

cap.release() # Release the camera resource.
cv2.destroyAllWindows() # Close all OpenCV windows.
