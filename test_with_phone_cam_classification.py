import cv2
from ultralytics import YOLO

model_path = r"runs\classify\garbage_classifier_v2-6\weights\best.pt"
model = YOLO(model_path)

camera_url = "http://192.168.1.114:8080/video"
cap = cv2.VideoCapture(camera_url)

while cap.isOpened():

    is_success, frame = cap.read()

    if is_success:
        results = model(frame)
        annotated_frame = frame.copy()

        probs = results[0].probs
        top1_index = probs.top1
        confidence = float(probs.top1conf)

        class_name = results[0].names[top1_index]

        text = f"{class_name}: {confidence:.2f}"

        cv2.putText(annotated_frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        screen_shot = cv2.resize(annotated_frame, (1000, 750))
        cv2.imshow("Garbage Classifier", screen_shot)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()