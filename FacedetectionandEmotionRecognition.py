import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Load ONNX emotion model
session = ort.InferenceSession("emotion-ferplus-8.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
emotion_labels = ['neutral', 'happiness', 'surprise', 'anger', 'disgust', 'fear', 'sadness']

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    try:
        # Convert frame to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = max(int(bbox.xmin * w), 0)
                y = max(int(bbox.ymin * h), 0)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Avoid going out of bounds
                x2 = min(x + width, w)
                y2 = min(y + height, h)

                face = frame[y:y2, x:x2]
                if face.shape[0] == 0 or face.shape[1] == 0:
                    print("Warning: Face crop is empty, skipping frame.")
                    continue

                # Resize & normalize face
                face_resized = cv2.resize(face, (64, 64))
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                face_normalized = face_gray.astype(np.float32) / 255.0
                face_input = np.expand_dims(face_normalized, axis=0)     # Add channel dim
                face_input = np.expand_dims(face_input, axis=0)          # Add batch dim


                # Emotion prediction
                outputs = session.run([output_name], {input_name: face_input})[0]
                
                exp_logits = np.exp(outputs[0] - np.max(outputs[0]))  # for numerical stability
                probs = exp_logits / np.sum(exp_logits)

                top_indices = probs.argsort()[-3:][::-1]
                emotion = emotion_labels[top_indices[0]]  # define it from top prediction

                for i, idx in enumerate(top_indices):
                    label = f"{emotion_labels[idx]}: {probs[idx]:.2f}"
                    cv2.putText(frame, label, (x, y - 10 - i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


                # Draw box & label
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            print("No face detected.")

        # Show frame
        cv2.imshow("Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Runtime error:", str(e))
        break

cap.release()
cv2.destroyAllWindows()
