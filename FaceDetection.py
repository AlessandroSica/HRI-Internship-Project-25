import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

# Initialize Face Detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results = face_detection.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.detections:
        for detection in results.detections:
            # Get bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)

            # Draw green rectangle box (BGR format: (0,255,0))
            cv2.rectangle(image, bbox, (0, 255, 0), 2)

            # Optional: draw keypoints (e.g., eyes, nose) with circles
            for keypoint in detection.location_data.relative_keypoints:
                x = int(keypoint.x * iw)
                y = int(keypoint.y * ih)
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # green circles

    cv2.imshow('MediaPipe Face Detection (Green Box)', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
