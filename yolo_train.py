from ultralytics import YOLO


def train_yolo():
    model = YOLO('yolo26n.pt')
    model.train(
        data=r"C:\Users\PC\Desktop\garbage_dataset\yolo_dataset\data.yaml",  # kullandığımız veri setinin bilgisayarımızdaki konumu
        epochs=100,  # modelin veri setini kaç kere tarayacağı
        imgsz=640,  # modelin tüm resimleri ayarlayacağı ortak boyut
        device='cpu',  # eğitim aşamasında kullanılacak donanım (ekran kartı için 0 yazılmalı)
        batch=16,  # veri setini kaçarlı gruplar halinde ayrılacağı
        workers=4,  # modele resimleri gönderecek işlemci işçileri
        patience=20,  # eğer 20 tur boyunca başarı sabit kalırsa durdurmaya yarar
        hsv_h=0.015,  # resmin renk ayarlarıyla %1.5 oranında oynar
        hsv_s=0.7,  # resmin canlılık ayarlarıyla %70 oranında oynar
        hsv_v=0.4,  # resmin parlaklığıyla %40 oranında oynar
        degrees=10.0,  # resmi rastgele -10 ile +10 derece arasında yamultur
        translate=0.1,  # resmi merkezden dört yöne kaydırır
        scale=0.5,  # resmi %50 oranında büyültüp küçültür
        fliplr=0.5  # resme aynalama efekti uygular
    )


if __name__ == '__main__':
    train_yolo()
