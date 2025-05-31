import cv2
import numpy as np
import time
from analyzer import TennisCourtAnalyzer


def test_single_frame():
    """Tek frame test ederek sonuçları göster"""
    # Video dosyasını aç
    cap = cv2.VideoCapture('data/tennis.mp4')
    
    if not cap.isOpened():
        print("Video dosyası açılamadı!")
        return
    
    # 100. frame'e git (daha ilginç olabilir)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame = cap.read()
    
    if not ret:
        print("Frame okunamadı!")
        return
    
    # Frame'i boyutlandır
    height, width = frame.shape[:2]
    if width > 1000:
        scale = 1000 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
    
    print(f"Test frame boyutu: {frame.shape}")
    
    # Analyzer'ı başlat
    analyzer = TennisCourtAnalyzer()
    
    # Frame'i işle
    start_time = time.time()
    result_frame, court_2d = analyzer.process_frame(frame)
    process_time = time.time() - start_time
    
    print(f"İşleme süresi: {process_time:.3f} saniye")
    
    # Kort tespiti sonucu
    if analyzer.court_corners is not None:
        print("✓ Kort başarıyla tespit edildi")
        print(f"Kort köşeleri: {analyzer.court_corners}")
    else:
        print("✗ Kort tespit edilemedi")
    
    # Debug bilgileri için HSV analizi
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Yeşil piksel sayısı
    green_mask = cv2.inRange(hsv, analyzer.green_lower, analyzer.green_upper)
    green_pixels = cv2.countNonZero(green_mask)
    total_pixels = frame.shape[0] * frame.shape[1]
    green_ratio = green_pixels / total_pixels
    
    print(f"Yeşil piksel oranı: {green_ratio:.3f} ({green_pixels}/{total_pixels})")
    
    # Beyaz piksel sayısı
    white_mask = cv2.inRange(hsv, analyzer.white_lower, analyzer.white_upper)
    white_pixels = cv2.countNonZero(white_mask)
    white_ratio = white_pixels / total_pixels
    
    print(f"Beyaz piksel oranı: {white_ratio:.3f} ({white_pixels}/{total_pixels})")
    
    # Oyuncu tespiti
    court_mask = None
    if analyzer.court_corners is not None:
        court_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(court_mask, [analyzer.court_corners.astype(np.int32)], 255)
    
    players = analyzer.detect_players(frame, court_mask)
    print(f"Tespit edilen oyuncu sayısı: {len(players)}")
    
    if players:
        for i, (px, py) in enumerate(players):
            print(f"  Oyuncu {i+1}: ({px}, {py})")
    
    # Top tespiti
    ball = analyzer.detect_ball(frame, court_mask)
    if ball:
        print(f"✓ Top tespit edildi: {ball}")
    else:
        print("✗ Top tespit edilemedi")
    
    # Sonuçları görüntüle
    cv2.imshow('Original Frame', frame)
    cv2.imshow('HSV', hsv)
    cv2.imshow('Green Mask', green_mask)
    cv2.imshow('White Mask', white_mask)
    cv2.imshow('Result', result_frame)
    cv2.imshow('2D Court', court_2d)
    
    print("\nTuş kontrolları:")
    print("- Herhangi bir tuş: Sonraki test")
    print("- ESC: Çıkış")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cap.release()


def performance_test():
    """Performans testi - FPS hesaplama"""
    cap = cv2.VideoCapture('data/tennis.mp4')
    
    if not cap.isOpened():
        print("Video dosyası açılamadı!")
        return
    
    analyzer = TennisCourtAnalyzer()
    
    frame_count = 0
    start_time = time.time()
    process_times = []
    
    print("Performans testi başlatılıyor... (100 frame)")
    
    while frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Frame'i boyutlandır
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # İşleme süresini ölç
        frame_start = time.time()
        result_frame, court_2d = analyzer.process_frame(frame)
        frame_time = time.time() - frame_start
        
        process_times.append(frame_time)
        frame_count += 1
        
        if frame_count % 10 == 0:
            avg_time = np.mean(process_times[-10:])
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"Frame {frame_count}: Ortalama FPS = {fps:.1f}")
    
    total_time = time.time() - start_time
    avg_process_time = np.mean(process_times)
    avg_fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
    
    print(f"\n=== Performans Sonuçları ===")
    print(f"Toplam süre: {total_time:.2f} saniye")
    print(f"İşlenen frame sayısı: {frame_count}")
    print(f"Ortalama frame işleme süresi: {avg_process_time:.4f} saniye")
    print(f"Ortalama FPS: {avg_fps:.1f}")
    print(f"En hızlı frame: {min(process_times):.4f} saniye")
    print(f"En yavaş frame: {max(process_times):.4f} saniye")
    
    cap.release()


def accuracy_test():
    """Doğruluk testi - manuel kontrol için"""
    cap = cv2.VideoCapture('data/tennis.mp4')
    
    if not cap.isOpened():
        print("Video dosyası açılamadı!")
        return
    
    analyzer = TennisCourtAnalyzer()
    
    print("Doğruluk testi - Manuel inceleme")
    print("Kontroller:")
    print("- SPACE: Bir sonraki frame")
    print("- R: Rastgele frame'e git") 
    print("- ESC: Çıkış")
    
    frame_num = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video sona erdi!")
            break
        
        # Frame'i boyutlandır
        height, width = frame.shape[:2]
        if width > 1000:
            scale = 1000 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # İşle
        result_frame, court_2d = analyzer.process_frame(frame)
        
        # Bilgileri göster
        court_status = "✓ Tespit edildi" if analyzer.court_corners is not None else "✗ Tespit edilemedi"
        
        # Oyuncu sayısı
        court_mask = None
        if analyzer.court_corners is not None:
            court_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(court_mask, [analyzer.court_corners.astype(np.int32)], 255)
        
        players = analyzer.detect_players(frame, court_mask)
        ball = analyzer.detect_ball(frame, court_mask)
        
        # Başlık ekle
        info_text = f"Frame: {frame_num}/{total_frames} | Kort: {court_status} | Oyuncu: {len(players)} | Top: {'✓' if ball else '✗'}"
        cv2.putText(result_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Accuracy Test', result_frame)
        cv2.imshow('2D Court', court_2d)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            frame_num += 1
            continue
        elif key == ord('r'):  # R - rastgele frame
            random_frame = np.random.randint(0, total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
            frame_num = random_frame
            print(f"Rastgele frame: {random_frame}")
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Ana test menüsü"""
    print("=== Tenis Video Analizi Test Menüsü ===")
    print("1. Tek frame test (detaylı analiz)")
    print("2. Performans testi (FPS)")
    print("3. Doğruluk testi (manuel inceleme)")
    print("4. Tümünü çalıştır")
    
    choice = input("Seçiminizi yapın (1-4): ").strip()
    
    if choice == '1':
        test_single_frame()
    elif choice == '2':
        performance_test()
    elif choice == '3':
        accuracy_test()
    elif choice == '4':
        print("Tüm testler çalıştırılıyor...\n")
        print("1. Tek frame test:")
        test_single_frame()
        print("\n2. Performans testi:")
        performance_test()
        print("\n3. Doğruluk testi:")
        accuracy_test()
    else:
        print("Geçersiz seçim!")


if __name__ == "__main__":
    main()
