import cv2
from ultralytics import YOLO

model_path = r"C:\Users\PC\PycharmProjects\GarbageClassification\runs\detect\train-4\weights\best.pt"
model = YOLO(model_path)

image_path = "sample/glass.jpeg"

results = model(image_path)

image_with_box = results[0].plot()

cv2.imshow("Boxed image", image_with_box)

cv2.waitKey(0)
cv2.destroyAllWindows()