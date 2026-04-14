import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import time

def process_carpet(input_path, output_path, target_colors=8, target_width=640, target_height=1386):
    print("İşlem başlatıldı...")
    start_time = time.time()
    
    # 1. Fotoğrafı Oku
    img = cv2.imread(input_path)
    if img is None:
        print("Hata: Görüntü okunamadı!")
        return
        
    print(f"Orijinal Boyut: {img.shape}")
    
    # K-Means ile Renk İndirgeme (Hızlı olması için MiniBatchKMeans kullanıyoruz)
    print(f"Renkler {target_colors} adede indirgeniyor...")
    # İşlemi hızlandırmak için sadece renk indirgemeyi LAB uzayında yapacağız
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Pikselleri düzleştir
    pixels = lab_img.reshape(-1, 3)
    
    # KMeans Kümeleme
    kmeans = MiniBatchKMeans(n_clusters=target_colors, max_iter=50, batch_size=3072, random_state=42, n_init="auto")
    kmeans.fit(pixels)
    
    # Etiketleri ve Merkezleri alıp resmi yeniden oluştur
    new_pixels = kmeans.cluster_centers_[kmeans.labels_]
    new_pixels = np.clip(new_pixels.astype("uint8"), 0, 255)
    
    # Önceki şekle dönüştür ve tekrar BGR uzayına al
    reduced_lab = new_pixels.reshape(lab_img.shape)
    reduced_img = cv2.cvtColor(reduced_lab, cv2.COLOR_LAB2BGR)
    
    # Ölçeklendirme ve Pikselleştirme (Nearest Neighbor - Bulanıklaşmayı Önler)
    print(f"Hedef Boyuta Ölçeklendiriliyor: {target_width}x{target_height}...")
    scaled_img = cv2.resize(reduced_img, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    
    # Kaydet
    cv2.imwrite(output_path, scaled_img)
    print(f"İşlem tamamlandı! Sure: {time.time() - start_time:.2f} saniye")
    print(f"Çıktı kaydedildi: {output_path}")

if __name__ == "__main__":
    base_dir = "/Users/mbpstyle/Desktop/adsız klasör/"
    process_carpet(
        input_path=base_dir + "1.AŞAMA.jpg",
        output_path=base_dir + "OTOMASYON_SONUCU.bmp",
        target_colors=8,
        target_width=640,
        target_height=1386
    )
