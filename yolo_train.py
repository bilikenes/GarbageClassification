from ultralytics import YOLO


def train_yolo():
    model = YOLO('yolov8m.pt')
    model.train(
        data=r"C:\Users\PC\Desktop\garbage_dataset\yolo_dataset\data.yaml",
        epochs=100,
        imgsz=640,
        device=0,
        batch=16,
        workers=4,
        patience=20,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5
    )


if __name__ == '__main__':
    train_yolo()
