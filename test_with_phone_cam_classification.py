import cv2  # openCV kütüphanesinin yüklenmesi
from ultralytics import YOLO  # ultralytics kütüphanesinden YOLO sınıfının eklenmesi

model_path = r"runs\classify\garbage_classifier_v2-6\weights\best.pt"  # eğittiğimiz modelin projedeki dosya konumu
model = YOLO(model_path)  # eğittiğimiz modelin yüklenmesi

camera_url = "http://192.168.1.114:8080/video"  # kullanacağımız kameranın IP adresi
cap = cv2.VideoCapture(camera_url)  # kameranın anlık yayınını alma

while cap.isOpened():

    is_success, frame = cap.read()  # kameranın anlık karesini alma

    if is_success:
        results = model(frame)  # modelin yaptığı tahmin
        annotated_frame = frame.copy()  # kameradaki anlık görüntünün kopyası

        probs = results[0].probs  # modelin yaptığı tahminlerin ihtimal listesi
        top1_index = probs.top1  # kategoriler arasında ihtimali en yüksek olanın seçilmesi
        confidence = float(probs.top1conf)  # seçilen kategorinin güven seviyesi

        class_name = results[0].names[top1_index]  # seçilen kategorinin sınıf adı

        text = f"{class_name}: {confidence:.2f}"  # ekranda gözükecek yazı

        cv2.putText(annotated_frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)  # openCV ile ekrandaki yayına modelin tahminini yazma

        screen_shot = cv2.resize(annotated_frame, (1000, 750))  # ekranın anlık görüntüsü
        cv2.imshow("Garbage Classifier", screen_shot)  # ekranın anlık görüntüsünü gösterme

        if cv2.waitKey(1) & 0xFF == ord('q'):  # yayın açıkken 'Q' tuşuna basılınca program kapanır
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()