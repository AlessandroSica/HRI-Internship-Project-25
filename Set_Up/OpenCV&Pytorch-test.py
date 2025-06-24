import cv2
import torch

# Test OpenCV video capture
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    cv2.imshow("Test Frame", frame)
    cv2.waitKey(1000)
cap.release()
cv2.destroyAllWindows()

# Test PyTorch tensor creation
x = torch.randn(3, 3)
print("PyTorch tensor:", x)