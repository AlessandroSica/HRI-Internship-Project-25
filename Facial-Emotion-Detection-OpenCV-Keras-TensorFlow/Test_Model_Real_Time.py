import cv2 # OpenCV for image processing
import numpy as np # NumPy for numerical operations
from keras.models import load_model # Load the trained Keras model

model = load_model('HRI-Internship-Project-25/Facial-Emotion-Detection-OpenCV-Keras-TensorFlow/emotion_model_30epochs.h5') # Load the pre-trained model

# Load a pre-trained face detection model from OpenCV.
# The 'haarcascade_frontalface_default.xml' file contains data 
# that helps the program recognize human faces in images.
# This classifier uses the Haar Cascade method to detect faces 
# by looking for specific patterns (like eyes, nose, mouth, etc.).
faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

label_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'} # Emotion labels

video = cv2.VideoCapture(1)  # 0 = use default webcam
# If you want to use a video file instead of the webcam, you can replace 0 with the path to the video file, e.g., 'path/to/video.mp4'nnn

while True:
    ret, frame = video.read() # Read a frame from the video capture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the frame to grayscale, as the input in our model uses grayscale images
    faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3) # Detect faces in the frame

    for (x, y, w, h) in faces: # Loop through detected faces
        sub_face_img = gray[y:y+h, x:x+w] # Extract the face region of interest (ROI) from the grayscale image
        resized= cv2.resize(sub_face_img, (48, 48))  # Resize the face ROI to 48x48 pixels, as required by the model input
        normalized = resized / 255.0  # Normalize pixel values to the range [0, 1], in the same way as during training, in the other python script
        reshaped = np.reshape(normalized, (1, 48, 48, 1)) # Reshape the image to match the input shape of the model (1, 48, 48, 1), where 1 is the batch size (one image at a time), 48x48 is the image size, and 1 is the number of channels (grayscale)
        result = model.predict(reshaped)  # Predict the emotion using the model, this is a vector of probabilities for each class
        label = np.argmax(result, axis=1)[0]  # Get the index of the class with the highest predicted probability from the result vector
        # select axis=1 to get the index of the maximum value along the specified axis, and [0] to get the first element of the resulting array (since we are processing one image at a time there is only one element in the array)
        print(label) # Print the predicted label index
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1) # Draw a rectangle around the detected face in the original frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2) # Draw a thicker rectangle around the detected face
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1) # Draw a filled rectangle above the face rectangle for the label background
        cv2.putText(frame, label_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) # Put the predicted emotion label text above the face rectangle

    cv2.imshow('Emotion Detection', frame) # Display the frame with detected faces and emotions
    k=cv2.waitKey(1) # Wait for a key press
    if k==ord('q'):
        break # Exit the loop if 'q' is pressed

video.release() # Release the video capture object
cv2.destroyAllWindows() # Close all OpenCV windows

