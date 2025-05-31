import cv2
import numpy as np
from analyzer import TennisCourtAnalyzer


class TennisAnalyzerCalibrator:
    """Tenis analyzer parametrelerini kalibre etmek için interaktif araç"""
    
    def __init__(self):
        self.analyzer = TennisCourtAnalyzer()
        self.current_frame = None
        
        # Trackbar değerleri
        self.green_h_min = 40
        self.green_h_max = 80
        self.green_s_min = 40
        self.green_s_max = 255
        self.green_v_min = 40
        self.green_v_max = 255
        
        self.white_h_min = 0
        self.white_h_max = 180
        self.white_s_min = 0
        self.white_s_max = 30
        self.white_v_min = 200
        self.white_v_max = 255
        
        self.player_area_min = 500
        self.player_area_max = 5000
        self.ball_area_min = 5
        self.ball_area_max = 200
        
    def create_trackbars(self):
        """Kalibrasyon trackbar'larını oluştur"""
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        
        # Yeşil renk kontrolları
        cv2.createTrackbar('Green H Min', 'Controls', self.green_h_min, 180, self.update_params)
        cv2.createTrackbar('Green H Max', 'Controls', self.green_h_max, 180, self.update_params)
        cv2.createTrackbar('Green S Min', 'Controls', self.green_s_min, 255, self.update_params)
        cv2.createTrackbar('Green S Max', 'Controls', self.green_s_max, 255, self.update_params)
        cv2.createTrackbar('Green V Min', 'Controls', self.green_v_min, 255, self.update_params)
        cv2.createTrackbar('Green V Max', 'Controls', self.green_v_max, 255, self.update_params)
        
        # Beyaz renk kontrolları
        cv2.createTrackbar('White H Min', 'Controls', self.white_h_min, 180, self.update_params)
        cv2.createTrackbar('White H Max', 'Controls', self.white_h_max, 180, self.update_params)
        cv2.createTrackbar('White S Min', 'Controls', self.white_s_min, 255, self.update_params)
        cv2.createTrackbar('White S Max', 'Controls', self.white_s_max, 255, self.update_params)
        cv2.createTrackbar('White V Min', 'Controls', self.white_v_min, 255, self.update_params)
        cv2.createTrackbar('White V Max', 'Controls', self.white_v_max, 255, self.update_params)
        
        # Alan kontrolları
        cv2.createTrackbar('Player Area Min', 'Controls', self.player_area_min, 10000, self.update_params)
        cv2.createTrackbar('Player Area Max', 'Controls', self.player_area_max, 10000, self.update_params)
        cv2.createTrackbar('Ball Area Min', 'Controls', self.ball_area_min, 500, self.update_params)
        cv2.createTrackbar('Ball Area Max', 'Controls', self.ball_area_max, 500, self.update_params)
    
    def update_params(self, val):
        """Trackbar değerlerini güncelle"""
        self.green_h_min = cv2.getTrackbarPos('Green H Min', 'Controls')
        self.green_h_max = cv2.getTrackbarPos('Green H Max', 'Controls')
        self.green_s_min = cv2.getTrackbarPos('Green S Min', 'Controls')
        self.green_s_max = cv2.getTrackbarPos('Green S Max', 'Controls')
        self.green_v_min = cv2.getTrackbarPos('Green V Min', 'Controls')
        self.green_v_max = cv2.getTrackbarPos('Green V Max', 'Controls')
        
        self.white_h_min = cv2.getTrackbarPos('White H Min', 'Controls')
        self.white_h_max = cv2.getTrackbarPos('White H Max', 'Controls')
        self.white_s_min = cv2.getTrackbarPos('White S Min', 'Controls')
        self.white_s_max = cv2.getTrackbarPos('White S Max', 'Controls')
        self.white_v_min = cv2.getTrackbarPos('White V Min', 'Controls')
        self.white_v_max = cv2.getTrackbarPos('White V Max', 'Controls')
        
        self.player_area_min = cv2.getTrackbarPos('Player Area Min', 'Controls')
        self.player_area_max = cv2.getTrackbarPos('Player Area Max', 'Controls')
        self.ball_area_min = cv2.getTrackbarPos('Ball Area Min', 'Controls')
        self.ball_area_max = cv2.getTrackbarPos('Ball Area Max', 'Controls')
        
        # Analyzer parametrelerini güncelle
        self.analyzer.green_lower = np.array([self.green_h_min, self.green_s_min, self.green_v_min])
        self.analyzer.green_upper = np.array([self.green_h_max, self.green_s_max, self.green_v_max])
        self.analyzer.white_lower = np.array([self.white_h_min, self.white_s_min, self.white_v_min])
        self.analyzer.white_upper = np.array([self.white_h_max, self.white_s_max, self.white_v_max])
        
        # Eğer frame varsa yeniden işle
        if self.current_frame is not None:
            self.process_and_display()
    
    def process_and_display(self):
        """Mevcut frame'i işle ve sonuçları göster"""
        if self.current_frame is None:
            return
        
        # HSV dönüştür
        hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
        
        # Yeşil maske
        green_mask = cv2.inRange(hsv, self.analyzer.green_lower, self.analyzer.green_upper)
        
        # Beyaz maske
        white_mask = cv2.inRange(hsv, self.analyzer.white_lower, self.analyzer.white_upper)
        
        # Ana analiz
        result_frame, court_2d = self.analyzer.process_frame(self.current_frame)
        
        # Sonuçları göster
        cv2.imshow('Original', self.current_frame)
        cv2.imshow('Green Mask', green_mask)
        cv2.imshow('White Mask', white_mask)
        cv2.imshow('Result', result_frame)
        cv2.imshow('2D Court', court_2d)
    
    def save_parameters(self):
        """Kalibre edilmiş parametreleri dosyaya kaydet"""
        params = {
            'green_lower': self.analyzer.green_lower.tolist(),
            'green_upper': self.analyzer.green_upper.tolist(),
            'white_lower': self.analyzer.white_lower.tolist(),
            'white_upper': self.analyzer.white_upper.tolist(),
            'player_area_min': self.player_area_min,
            'player_area_max': self.player_area_max,
            'ball_area_min': self.ball_area_min,
            'ball_area_max': self.ball_area_max
        }
        
        with open('calibration_params.txt', 'w') as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
        
        print("Parametreler calibration_params.txt dosyasına kaydedildi!")
    
    def run_calibration(self):
        """Kalibrasyon arayüzünü çalıştır"""
        # Video dosyasını aç
        cap = cv2.VideoCapture('data/tennis.mp4')
        
        if not cap.isOpened():
            print("Video dosyası açılamadı!")
            return
        
        # İlk frame'i al
        ret, frame = cap.read()
        if not ret:
            print("Video frame'i okunamadı!")
            return
        
        # Frame'i yeniden boyutlandır
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        self.current_frame = frame
        
        # Trackbar'ları oluştur
        self.create_trackbars()
        
        # İlk işleme
        self.process_and_display()
        
        print("Kalibrasyon Modu:")
        print("- Trackbar'ları kullanarak parametreleri ayarlayın")
        print("- SPACE: Bir sonraki frame'e geç")
        print("- S: Parametreleri kaydet")
        print("- ESC: Çıkış")
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE - sonraki frame
                ret, frame = cap.read()
                if ret:
                    # Frame'i yeniden boyutlandır
                    if width > 800:
                        frame = cv2.resize(frame, (new_width, new_height))
                    self.current_frame = frame
                    frame_count += 1
                    print(f"Frame {frame_count}/{total_frames}")
                    self.process_and_display()
                else:
                    print("Video sona erdi, başa dönüyor...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
            elif key == ord('s'):  # S - parametreleri kaydet
                self.save_parameters()
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    calibrator = TennisAnalyzerCalibrator()
    calibrator.run_calibration()
