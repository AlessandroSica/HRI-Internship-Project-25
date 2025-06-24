import mediapipe as mp # Face detection library, MediaPipe Holistic module provides a holistic approach to human pose, face, and hand detection.
import cv2 # OpenCV library for computer vision tasks, such as image processing and video capture.

mp_drawing = mp.solutions.drawing_utils # Drawing utilities for visualizing landmarks and connections.
mp_holistic = mp.solutions.holistic # Holistic module for detecting face, pose, and hand landmarks. Imported from MediaPipe library.

cap = cv2.VideoCapture(1) # Open the default camera (0) for video capture.

# Initialize the MediaPipe Holistic model with default parameters.
with mp_holistic.Holistic(
    min_detection_confidence=0.5, # Minimum confidence threshold for face detection.
    min_tracking_confidence=0.5 # Minimum confidence threshold for landmark tracking.
) as holistic: # Use the holistic model in a context manager to ensure proper resource management.

    while cap.isOpened(): # Loop until the camera is closed.
        ret, frame = cap.read() # Read a frame from the camera.
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert the frame to RGB format for MediaPipe processing.
        # When using MediaPipe, the input image should be in RGB format.
        # OpenCV captures images in BGR format, so we need to convert it.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make detections and track landmarks in the image.
        results = holistic.process(image)
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

        cv2.imshow('Holistic Model Detection', image) # Display the original frame in a window.

        if cv2.waitKey(10) & 0xFF == ord('q'): # If the 'q' key is pressed, break the loop and exit.
            break   

cap.release() # Release the camera resource.
cv2.destroyAllWindows() # Close all OpenCV windows.
