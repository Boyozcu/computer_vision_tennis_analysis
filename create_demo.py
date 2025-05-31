import cv2
import numpy as np
from analyzer import TennisCourtAnalyzer


def create_demo_video():
    """Demo video oluştur - hem orijinal hem 2D görüntüyü yan yana göster"""
    # Video dosyasını aç
    video_path = 'data/tennis.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Hata: Video dosyası açılamadı: {video_path}")
        return
    
    # Video özelliklerini al
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Çıktı video boyutları
    if width > 1200:
        scale = 1200 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        new_width = width
        new_height = height
    
    # 2D kort boyutu
    court_width = 600
    court_height = 300
    
    # Kombinasyon video boyutu (yan yana)
    combined_width = new_width + court_width + 20  # 20 piksel boşluk
    combined_height = max(new_height, court_height)
    
    # Video yazıcı
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/demo_output.mp4', fourcc, fps, (combined_width, combined_height))
    
    # Analyzer'ı başlat
    analyzer = TennisCourtAnalyzer()
    
    print(f"Demo video oluşturuluyor... Çıktı: output/demo_output.mp4")
    print(f"Video boyutu: {combined_width}x{combined_height}, FPS: {fps}")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Frame'i yeniden boyutlandır
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Frame'i işle
        result_frame, court_2d = analyzer.process_frame(frame)
        
        # 2D kortu uygun boyuta getir
        court_2d_resized = cv2.resize(court_2d, (court_width, court_height))
        
        # Kombinasyon frame oluştur
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Orijinal videoyu sol tarafa yerleştir
        y_offset = (combined_height - new_height) // 2
        combined_frame[y_offset:y_offset+new_height, 0:new_width] = result_frame
        
        # 2D kortu sağ tarafa yerleştir
        court_y_offset = (combined_height - court_height) // 2
        court_x_start = new_width + 20
        combined_frame[court_y_offset:court_y_offset+court_height, 
                      court_x_start:court_x_start+court_width] = court_2d_resized
        
        # Başlıklar ekle
        cv2.putText(combined_frame, 'Original Video', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_frame, '2D Court View', (court_x_start + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Frame'i dosyaya yaz
        out.write(combined_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:  # Her 30 frame'de bir rapor
            print(f"İşlenen frame sayısı: {frame_count}")
    
    # Temizlik
    cap.release()
    out.release()
    print(f"Demo video tamamlandı! Toplam {frame_count} frame işlendi.")


if __name__ == "__main__":
    create_demo_video()
