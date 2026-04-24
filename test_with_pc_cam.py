import cv2
from ultralytics import YOLO

model_path = r"runs\detect\train-4\weights\best.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture(0)

if cap.isOpened():
    while cap.isOpened():
        is_success, frame = cap.read()

        if is_success:
            results = model(frame, conf=0.4)
            boxed_result = results[0].plot()
            screen_shot = cv2.resize(boxed_result, (1000, 750))
            cv2.imshow("Garbage Detector", screen_shot)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
