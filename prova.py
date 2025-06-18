import cv2
import mediapipe as mp
from fer import FER

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
emotion_detector = FER(mtcnn=True)  # use mtcnn for better face detection inside FER

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces with MediaPipe
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, bw, bh = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Draw green box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            # Crop face for emotion detection (with bounds checking)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            face_roi = frame[y1:y2, x1:x2]

            # Use FER to detect emotions on cropped face
            if face_roi.size != 0:
                emotions = emotion_detector.detect_emotions(face_roi)
                if emotions:
                    # Get dominant emotion
                    dominant_emotion, score = emotion_detector.top_emotion(face_roi)
                    if dominant_emotion:
                        cv2.putText(frame, f'{dominant_emotion}', (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Face & Emotion Recognition', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
