import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class TennisCourtAnalyzer:
    def __init__(self):
        # Tenis kortu boyutları (metre cinsinden)
        self.court_width = 23.77  # 78 feet
        self.court_height = 10.97  # 36 feet
        
        # 2D görüntü boyutları (piksel)
        self.court_2d_width = 600
        self.court_2d_height = 300
        
        # Renk aralıkları (HSV)
        self.green_lower = np.array([40, 40, 40])
        self.green_upper = np.array([80, 255, 255])
        
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 30, 255])
        
        # Oyuncu renk aralıkları (daha geniş renk yelpazesi)
        self.player_colors = [
            # Mavi tonları
            (np.array([100, 50, 50]), np.array([130, 255, 255])),
            # Kırmızı tonları
            (np.array([0, 50, 50]), np.array([10, 255, 255])),
            (np.array([170, 50, 50]), np.array([180, 255, 255])),
            # Sarı tonları
            (np.array([20, 50, 50]), np.array([30, 255, 255])),
        ]
        
        # Kort köşe noktaları (varsayılan)
        self.court_corners = None
        
    def detect_court(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Tenis kortunu tespit et ve köşe noktalarını bul"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Yeşil renk maskesi
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Morfolojik işlemler
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # Konturları bul
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # En büyük konturu seç (kort olması muhtemel)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Kontur alanı yeterince büyük mü?
        if cv2.contourArea(largest_contour) < frame.shape[0] * frame.shape[1] * 0.1:
            return None
            
        # Kontur yaklaşımı
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Dikdörtgen köşe noktalarını bul
        if len(approx) >= 4:
            # En dış köşe noktalarını bul
            hull = cv2.convexHull(largest_contour)
            hull_points = hull.reshape(-1, 2)
            
            # Köşe noktalarını sırala (sol-üst, sağ-üst, sağ-alt, sol-alt)
            corners = self._order_corners(hull_points)
            self.court_corners = corners
            return corners
            
        return None
    
    def _order_corners(self, points: np.ndarray) -> np.ndarray:
        """Köşe noktalarını doğru sırada düzenle"""
        # Minimum bounding rectangle kullanarak köşeleri bul
        rect = cv2.minAreaRect(points)
        corners = cv2.boxPoints(rect)
        corners = np.int0(corners)
        
        # Köşeleri sırala: sol-üst, sağ-üst, sağ-alt, sol-alt
        center_x = np.mean(corners[:, 0])
        center_y = np.mean(corners[:, 1])
        
        # Her köşenin merkeze göre pozisyonunu belirle
        ordered_corners = np.zeros((4, 2), dtype=np.float32)
        
        for corner in corners:
            if corner[0] < center_x and corner[1] < center_y:  # Sol-üst
                ordered_corners[0] = corner
            elif corner[0] >= center_x and corner[1] < center_y:  # Sağ-üst
                ordered_corners[1] = corner
            elif corner[0] >= center_x and corner[1] >= center_y:  # Sağ-alt
                ordered_corners[2] = corner
            else:  # Sol-alt
                ordered_corners[3] = corner
                
        return ordered_corners
    
    def detect_players(self, frame: np.ndarray, court_mask: np.ndarray = None) -> List[Tuple[int, int]]:
        """Oyuncuları tespit et"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        players = []
        
        # Her renk aralığı için oyuncu ara
        for lower, upper in self.player_colors:
            mask = cv2.inRange(hsv, lower, upper)
            
            # Kort maskesi varsa uygula
            if court_mask is not None:
                mask = cv2.bitwise_and(mask, court_mask)
            
            # Morfolojik işlemler
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Konturları bul
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # Oyuncu boyutuna uygun alan kontrolü
                if 500 < area < 5000:
                    # Merkez noktasını hesapla
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Çok yakın oyuncuları birleştir
                        is_duplicate = False
                        for px, py in players:
                            if abs(cx - px) < 50 and abs(cy - py) < 50:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            players.append((cx, cy))
        
        # En fazla 2 oyuncu döndür (en büyük alanları)
        if len(players) > 2:
            # Kontur alanlarına göre sırala
            players_with_area = []
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            for px, py in players:
                max_area = 0
                for lower, upper in self.player_colors:
                    mask = cv2.inRange(hsv, lower, upper)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if cv2.pointPolygonTest(contour, (px, py), False) >= 0:
                            area = cv2.contourArea(contour)
                            max_area = max(max_area, area)
                players_with_area.append((px, py, max_area))
            
            # Alana göre sırala ve en büyük 2'sini al
            players_with_area.sort(key=lambda x: x[2], reverse=True)
            players = [(px, py) for px, py, _ in players_with_area[:2]]
        
        return players
    
    def detect_ball(self, frame: np.ndarray, court_mask: np.ndarray = None) -> Optional[Tuple[int, int]]:
        """Topu tespit et"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Beyaz renk maskesi
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        # Kort maskesi varsa uygula
        if court_mask is not None:
            white_mask = cv2.bitwise_and(white_mask, court_mask)
        
        # Morfolojik işlemler (top küçük olduğu için daha hassas)
        kernel = np.ones((2, 2), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        # Konturları bul
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_ball = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Top boyutuna uygun alan (çok küçük)
            if 5 < area < 200:
                # Daire benzerlği kontrolü
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Daha dairesel objeler tercih edilir
                    if circularity > 0.3:
                        score = circularity * area  # Dairsellik ve boyut kombinasyonu
                        if score > best_score:
                            best_score = score
                            # Merkez noktasını hesapla
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                best_ball = (cx, cy)
        
        return best_ball
    
    def create_2d_court(self) -> np.ndarray:
        """2D kuşbakışı tenis kortu oluştur"""
        court_2d = np.zeros((self.court_2d_height, self.court_2d_width, 3), dtype=np.uint8)
        
        # Kort zemini (koyu yeşil)
        court_2d.fill(0)
        court_2d[:, :] = [34, 139, 34]  # Forest Green
        
        # Kort sınırları (beyaz çizgiler)
        border_thickness = 3
        
        # Dış sınırlar
        cv2.rectangle(court_2d, (border_thickness, border_thickness), 
                     (self.court_2d_width - border_thickness, self.court_2d_height - border_thickness), 
                     (255, 255, 255), border_thickness)
        
        # Orta çizgi
        mid_x = self.court_2d_width // 2
        cv2.line(court_2d, (mid_x, border_thickness), 
                (mid_x, self.court_2d_height - border_thickness), (255, 255, 255), 2)
        
        # Servis çizgileri
        service_line_y1 = int(self.court_2d_height * 0.25)
        service_line_y2 = int(self.court_2d_height * 0.75)
        
        cv2.line(court_2d, (border_thickness, service_line_y1), 
                (self.court_2d_width - border_thickness, service_line_y1), (255, 255, 255), 2)
        cv2.line(court_2d, (border_thickness, service_line_y2), 
                (self.court_2d_width - border_thickness, service_line_y2), (255, 255, 255), 2)
        
        # Servis kutuları orta çizgisi
        cv2.line(court_2d, (mid_x, service_line_y1), 
                (mid_x, service_line_y2), (255, 255, 255), 2)
        
        return court_2d
    
    def transform_to_2d(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """3D nokta koordinatını 2D kort koordinatına dönüştür"""
        if self.court_corners is None:
            return point
        
        # Perspektif dönüşüm matrisi
        src_points = self.court_corners.astype(np.float32)
        dst_points = np.array([
            [0, 0],
            [self.court_2d_width, 0],
            [self.court_2d_width, self.court_2d_height],
            [0, self.court_2d_height]
        ], dtype=np.float32)
        
        # Perspektif dönüşüm matrisi hesapla
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Noktayı dönüştür
        point_array = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_array, M)
        
        x, y = transformed[0][0]
        x = max(0, min(self.court_2d_width - 1, int(x)))
        y = max(0, min(self.court_2d_height - 1, int(y)))
        
        return (x, y)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tek frame'i işle ve hem orijinal hem 2D görüntüyü döndür"""
        # Tenis kortunu tespit et
        court_corners = self.detect_court(frame)
        
        # Kort maskesi oluştur
        court_mask = None
        if court_corners is not None:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [court_corners.astype(np.int32)], 255)
            court_mask = mask
        
        # Oyuncu ve top tespiti
        players = self.detect_players(frame, court_mask)
        ball = self.detect_ball(frame, court_mask)
        
        # Orijinal frame üzerine çizim
        result_frame = frame.copy()
        
        # Kort köşelerini çiz
        if court_corners is not None:
            cv2.polylines(result_frame, [court_corners.astype(np.int32)], True, (0, 255, 0), 3)
        
        # Oyuncuları çiz
        for i, (px, py) in enumerate(players):
            cv2.circle(result_frame, (px, py), 15, (0, 0, 255), -1)
            cv2.putText(result_frame, f'Player {i+1}', (px-20, py-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Topu çiz
        if ball:
            cv2.circle(result_frame, ball, 8, (255, 255, 0), -1)
            cv2.putText(result_frame, 'Ball', (ball[0]-15, ball[1]-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # 2D görüntü oluştur
        court_2d = self.create_2d_court()
        
        # Oyuncu ve top pozisyonlarını 2D'ye dönüştür
        for i, (px, py) in enumerate(players):
            x2d, y2d = self.transform_to_2d((px, py))
            cv2.circle(court_2d, (x2d, y2d), 8, (0, 0, 255), -1)
            cv2.putText(court_2d, f'P{i+1}', (x2d-10, y2d-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if ball:
            x2d, y2d = self.transform_to_2d(ball)
            cv2.circle(court_2d, (x2d, y2d), 4, (255, 255, 0), -1)
        
        return result_frame, court_2d


def main():
    """Ana fonksiyon - Video işleme ve görüntüleme"""
    # Video dosyasını aç
    video_path = 'data/tennis.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Hata: Video dosyası açılamadı: {video_path}")
        return
    
    # Analyzer'ı başlat
    analyzer = TennisCourtAnalyzer()
    
    print("Video işleniyor... ESC tuşuna basarak çıkabilirsiniz.")
    print("Pencereler: 'Original Video' ve 'Court 2D View'")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video sona erdi veya frame okunamadı.")
            break
        
        # Frame'i yeniden boyutlandır (performans için)
        height, width = frame.shape[:2]
        if width > 1200:
            scale = 1200 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Frame'i işle
        result_frame, court_2d = analyzer.process_frame(frame)
        
        # Görüntüleri göster
        cv2.imshow('Original Video', result_frame)
        cv2.imshow('Court 2D View', court_2d)
        
        # ESC tuşu kontrolü
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC tuşu
            break
        elif key == ord('q'):  # Q tuşu
            break
    
    # Temizlik
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()