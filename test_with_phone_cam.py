import cv2  # openCV kütüphanesinin eklenmesi
from ultralytics import YOLO  # ultralytics kütüphanesinden YOLO sınıfının eklenmesi

model_path = r"runs\detect\garbage_classifier_v3\weights\best.pt"  # modelin projedeki dosya konumu
model = YOLO(model_path)  # eğittiğimiz modelin yüklenmesi

camera_url = "http://192.168.1.114:8080/video"  # kullanacağımız kameranın IP adresi
cap = cv2.VideoCapture(camera_url)  # kameranın anlık yayınını alma

while cap.isOpened():

    is_success, frame = cap.read()  # kameranın anlık görüntüsünü okuma

    if is_success:
        results = model(frame, conf=0.4)  # eğittiğimiz model, o anki görüntüyü tahmin eder. Eğer %40'tan daha büyük bir güven varsa tahmin eder
        boxed_result = results[0].plot()  # tahmin edilen nesneyi dikdörtgen içine alır
        screen_shot = cv2.resize(boxed_result, (1000, 750))  # dikdörtgen içine alınmış görseli ekrana yansıtmak için tanımlama yaptık
        cv2.imshow("Garbage Detector", screen_shot)  # openCV kütüphanesiyle anlık olarak ekrana yansıttık

        if cv2.waitKey(1) & 0xFF == ord('q'):  # eğer yayın açıkken 'Q' tuşuna basarsak sonlanır
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()