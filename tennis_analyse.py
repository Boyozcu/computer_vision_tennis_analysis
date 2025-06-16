#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
from collections import deque
from typing import List, Dict, Tuple, Optional, Any

VIDEO_PATH = "tennis.mp4"

VIDEO_WIDTH: Optional[int] = None
VIDEO_HEIGHT: Optional[int] = None
VIDEO_FPS: Optional[int] = None

SLOWDOWN_FACTOR = 1.0 # Videoyu yavaşlatma faktörü, 1.0 normal hız

BLUE_COURT_HSV_LOWER = np.array([100, 20, 30])
BLUE_COURT_HSV_UPPER = np.array([160, 255, 255])

DEFAULT_SOURCE_POINTS = np.array([[390, 218], [885, 218], [1045, 575], [220, 575]], dtype=np.float32)

COURT_PADDING = 75
COURT_LEFT_X = COURT_PADDING
COURT_RIGHT_X = 400 - COURT_PADDING
COURT_TOP_Y_ADJUSTMENT = 20
COURT_TOP_Y = COURT_PADDING + COURT_TOP_Y_ADJUSTMENT
COURT_BOTTOM_Y = 600 - COURT_PADDING

COURT_WIDTH = COURT_RIGHT_X - COURT_LEFT_X
COURT_HEIGHT = COURT_BOTTOM_Y - COURT_TOP_Y
NET_Y = COURT_TOP_Y + COURT_HEIGHT // 2
SERVICE_LINE_OFFSET = int(COURT_HEIGHT * 0.319) # Servis çizgisi ofseti

SKETCH_PLAYER_FILTER_PADDING = 25 # Oyuncu filtresi için dolgu

SKETCH_LINES = [
    ((COURT_LEFT_X, COURT_TOP_Y), (COURT_RIGHT_X, COURT_TOP_Y)), # Üst çizgi
    ((COURT_RIGHT_X, COURT_TOP_Y), (COURT_RIGHT_X, COURT_BOTTOM_Y)), # Sağ dikey çizgi
    ((COURT_RIGHT_X, COURT_BOTTOM_Y), (COURT_LEFT_X, COURT_BOTTOM_Y)), # Alt çizgi
    ((COURT_LEFT_X, COURT_BOTTOM_Y), (COURT_LEFT_X, COURT_TOP_Y)), # Sol dikey çizgi
    ((COURT_LEFT_X, NET_Y), (COURT_RIGHT_X, NET_Y)), # File çizgisi
    ((COURT_LEFT_X, NET_Y - SERVICE_LINE_OFFSET), (COURT_RIGHT_X, NET_Y - SERVICE_LINE_OFFSET)), # Üst servis çizgisi
    ((COURT_LEFT_X, NET_Y + SERVICE_LINE_OFFSET), (COURT_RIGHT_X, NET_Y + SERVICE_LINE_OFFSET)), # Alt servis çizgisi
    ((COURT_LEFT_X + COURT_WIDTH // 2, NET_Y - SERVICE_LINE_OFFSET), # Orta servis çizgisi (üst kısım)
     (COURT_LEFT_X + COURT_WIDTH // 2, NET_Y + SERVICE_LINE_OFFSET)), # Orta servis çizgisi (alt kısım)
    ((int(COURT_LEFT_X + COURT_WIDTH * 0.15), COURT_TOP_Y), # Tekler için sol yan çizgi (koridor)
     (int(COURT_LEFT_X + COURT_WIDTH * 0.15), COURT_BOTTOM_Y)),
    ((int(COURT_RIGHT_X - COURT_WIDTH * 0.15), COURT_TOP_Y), # Tekler için sağ yan çizgi (koridor)
     (int(COURT_RIGHT_X - COURT_WIDTH * 0.15), COURT_BOTTOM_Y)),
]

DESTINATION_POINTS = np.array([
    [COURT_LEFT_X, COURT_TOP_Y],
    [COURT_RIGHT_X, COURT_TOP_Y],
    [COURT_RIGHT_X, COURT_BOTTOM_Y],
    [COURT_LEFT_X, COURT_BOTTOM_Y]
], dtype=np.float32)

GENERAL_PLAYER_MIN_AREA = 700  # daha da küçültülebilir
GENERAL_PLAYER_MAX_AREA = 12000  # daha da büyütülebilir
GENERAL_PLAYER_MIN_ASPECT_RATIO = 0.8 # h/w
GENERAL_PLAYER_MAX_ASPECT_RATIO = 2.5 # h/w
PLAYER_MIN_SOLIDITY = 0.50

BALL_MAX_FRAMES_BEFORE_FULL_RESET = 5 # Top tamamen kaybolmadan önceki maksimum kare sayısı

SKETCH_WIDTH = 400
SKETCH_HEIGHT = 600
COURT_SKETCH_COLOR = (180, 130, 70) # Kort çizimi rengi (açık mavi)

SKETCH_BALL_SMOOTHING_ALPHA = 0.3 # Kuşbakışı görünümde top yumuşatma alfa değeri

def get_video_properties(video_path: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30 # FPS okunamadıysa varsayılan 30

    cap.release()
    return width, height, fps

def point_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    # Bir noktanın bir doğru segmentine olan en kısa mesafesini hesaplar
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0: # Segment bir nokta ise
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    len_sq_AB = dx*dx + dy*dy # Segmentin karesel uzunluğu
    dot_product = (px - x1) * dx + (py - y1) * dy # (P-A) . (B-A)
    t = dot_product / len_sq_AB # Projeksiyon parametresi

    if t < 0: # Projeksiyon segmentin A ucunun gerisinde
        closest_x, closest_y = x1, y1
    elif t > 1: # Projeksiyon segmentin B ucunun ilerisinde
        closest_x, closest_y = x2, y2
    else: # Projeksiyon segmentin üzerinde
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

class CourtDetector:
    def __init__(self, source_points: np.ndarray, dest_points: np.ndarray,
                 video_w: int, video_h: int):
        self.source_points = source_points
        self.destination_points = dest_points
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_homography_matrix: Optional[np.ndarray] = None
        self.width = video_w
        self.height = video_h
        self.ball_trajectory = deque(maxlen=30) # Top yörüngesi için deque
        self.ball_search_roi: Optional[Tuple[int,int,int,int]] = None # Top arama ROI'si
        self.last_mapped_ball_sketch_pos: Optional[Tuple[int, int]] = None # Kuşbakışındaki son top konumu
        self._compute_homography()

    def find_court_corners_from_blue_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        # Mavi kort maskesinden kort köşelerini bulur
        mask = self.detect_court_area_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        approx_corners = cv2.approxPolyDP(largest_contour, 0.018 * perimeter, True) # Köşe tespiti için epsilon

        if len(approx_corners) == 4:
            pts = approx_corners.reshape(4, 2)
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            ordered_corners = np.zeros((4,2), dtype=np.float32)
            ordered_corners[0] = pts[np.argmin(s)] # Sol üst
            ordered_corners[2] = pts[np.argmax(s)] # Sağ alt
            ordered_corners[1] = pts[np.argmin(diff)] # Sağ üst
            ordered_corners[3] = pts[np.argmax(diff)] # Sol alt
            return ordered_corners
        return None

    def update_source_points(self, new_source_points: np.ndarray) -> None:
        self.source_points = new_source_points
        self._compute_homography()

    def _compute_homography(self) -> None:
        # Homografi matrisini hesaplar
        if self.source_points is not None and len(self.source_points) == 4:
            matrix, _ = cv2.findHomography(self.source_points, self.destination_points)
            self.homography_matrix = matrix
            if matrix is not None:
                try:
                    self.inverse_homography_matrix = np.linalg.inv(matrix)
                except np.linalg.LinAlgError: # Matris tekil ise
                    self.inverse_homography_matrix = None
            else:
                self.inverse_homography_matrix = None

    def detect_court_area_mask(self, frame: np.ndarray) -> np.ndarray:
        # Kort alanını HSV renk uzayında tespit eder
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        court_mask = cv2.inRange(hsv_frame, BLUE_COURT_HSV_LOWER, BLUE_COURT_HSV_UPPER)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_OPEN, kernel) # Gürültü giderme
        court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_CLOSE, kernel) # Boşlukları doldurma
        return court_mask

    def map_to_birdseye(self, point: Optional[Tuple[float, float]]) -> Optional[Tuple[int, int]]:
        # Bir noktayı kuşbakışı görünüme haritalar
        if self.homography_matrix is None or point is None:
            return None
        try:
            point_np = np.array([[point]], dtype=np.float32)
            mapped_point_np = cv2.perspectiveTransform(point_np, self.homography_matrix)
            if mapped_point_np is not None and mapped_point_np.shape[1] > 0:
                return (int(mapped_point_np[0][0][0]), int(mapped_point_np[0][0][1]))
        except (ValueError, TypeError):
            pass
        return None

    def map_from_birdseye(self, point: Optional[Tuple[float, float]]) -> Optional[Tuple[int, int]]:
        # Bir kuşbakışı noktasını orijinal görünüme haritalar
        if self.inverse_homography_matrix is None or point is None:
            return None
        try:
            point_np = np.array([[point]], dtype=np.float32)
            mapped_point_np = cv2.perspectiveTransform(point_np, self.inverse_homography_matrix)
            if mapped_point_np is not None and mapped_point_np.shape[1] > 0:
                return (int(mapped_point_np[0][0][0]), int(mapped_point_np[0][0][1]))
        except (ValueError, TypeError):
            pass
        return None

    def create_sketch_frame(self, players: List[Dict[str, Any]],
                           ball: Optional[Dict[str, Any]],
                           ball_trajectory: deque) -> np.ndarray:
        # Kuşbakışı kort çizimini oluşturur
        sketch = np.ones((SKETCH_HEIGHT, SKETCH_WIDTH, 3), dtype=np.uint8) * 255 # Beyaz arka plan
        cv2.rectangle(sketch,
                      (COURT_LEFT_X, COURT_TOP_Y),
                      (COURT_RIGHT_X, COURT_BOTTOM_Y),
                      COURT_SKETCH_COLOR,
                      -1) # Kort rengi
        line_color = (255, 255, 255) # Beyaz çizgiler
        line_thickness = 2
        for p1, p2 in SKETCH_LINES:
            cv2.line(sketch, p1, p2, line_color, line_thickness)

        player_colors = [(0, 0, 255), (255, 0, 0)] # P1: Kırmızı, P2: Mavi
        mapped_player_bboxes_for_ball_occlusion = [] # Topun oyuncu içinde olup olmadığını kontrol için
        player_positions_on_sketch_for_lines = [] # Oyuncular arası çizgi için

        for player_idx, player_data in enumerate(players):
            if not player_data or player_data.get('confidence', 0.0) < 0.3: # Düşük güvenli oyuncuları atla
                continue

            original_mapped_pos = self.map_to_birdseye(player_data['foot_pos'])
            if original_mapped_pos is None:
                continue

            # Oyuncunun Y pozisyonunu kort sınırları içinde tut
            clamped_y = max(min(original_mapped_pos[1], COURT_BOTTOM_Y - 5), COURT_TOP_Y + 5)
            current_player_sketch_pos = (original_mapped_pos[0], clamped_y)

            player_positions_on_sketch_for_lines.append({'id': player_data.get('id'), 'pos': current_player_sketch_pos, 'data': player_data})

            player_color = player_colors[player_data.get('id', 0)]
            # Oyuncu gölgesi ve daire çizimi
            cv2.circle(sketch, (current_player_sketch_pos[0]+3, current_player_sketch_pos[1]+3), 12, (100,100,100), -1, cv2.LINE_AA) # Gölge
            cv2.circle(sketch, current_player_sketch_pos, 10, player_color, -1, cv2.LINE_AA)
            cv2.circle(sketch, current_player_sketch_pos, 10, (0,0,0), 2, cv2.LINE_AA) # Kenarlık
            label = f"P{player_data.get('id', -1)+1}"
            cv2.putText(sketch, label, (current_player_sketch_pos[0]-10, current_player_sketch_pos[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(sketch, label, (current_player_sketch_pos[0]-10, current_player_sketch_pos[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            if 'bbox' in player_data: # Oyuncu bbox'ını haritala (top gizlenmesi için)
                px, py, pw, ph = player_data['bbox']
                # Bbox'ın alt-orta noktasını ve üst-orta noktasını haritala
                # Bu, perspektiften dolayı tam bir dikdörtgen haritalamasından daha iyi olabilir
                foot_pt = self.map_to_birdseye((px + pw/2, py + ph))
                head_pt = self.map_to_birdseye((px + pw/2, py))
                if foot_pt and head_pt:
                    # Basit bir elips veya dikdörtgen varsayımı yapılabilir
                    # Burada topun oyuncu tarafından gizlenip gizlenmediğini anlamak için bir alan oluşturuyoruz
                    # Daha doğru bir yaklaşım için oyuncunun şeklini daha iyi modellemek gerekir
                    est_w = 20 # Kuşbakışında oyuncu için tahmini genişlik
                    mapped_player_bboxes_for_ball_occlusion.append(
                        (min(foot_pt[0], head_pt[0]) - est_w//2, min(foot_pt[1], head_pt[1]),
                         est_w, abs(foot_pt[1] - head_pt[1]))
                    )


        if len(player_positions_on_sketch_for_lines) == 2: # İki oyuncu varsa aralarına çizgi çiz
            p1_line_data = next((p for p in player_positions_on_sketch_for_lines if p.get('id') == 0), None)
            p2_line_data = next((p for p in player_positions_on_sketch_for_lines if p.get('id') == 1), None)
            if p1_line_data and p2_line_data:
                cv2.line(sketch, p1_line_data['pos'], p2_line_data['pos'], (0, 255, 0), 1, cv2.LINE_AA) # Yeşil çizgi

        final_mapped_ball_pos = None
        if ball and 'center' in ball:
            mapped_ball_center = self.map_to_birdseye(ball['center'])
            ball_occluded_by_player = False
            if mapped_ball_center:
                for (mbx, mby, mbw, mbh) in mapped_player_bboxes_for_ball_occlusion:
                    if mbx <= mapped_ball_center[0] <= mbx + mbw and \
                       mby <= mapped_ball_center[1] <= mby + mbh:
                        ball_occluded_by_player = True
                        break
            
            if mapped_ball_center is not None and not ball_occluded_by_player:
                if self.last_mapped_ball_sketch_pos is not None: # Yumuşatma uygula
                    draw_x = int(self.last_mapped_ball_sketch_pos[0] * (1 - SKETCH_BALL_SMOOTHING_ALPHA) + mapped_ball_center[0] * SKETCH_BALL_SMOOTHING_ALPHA)
                    draw_y = int(self.last_mapped_ball_sketch_pos[1] * (1 - SKETCH_BALL_SMOOTHING_ALPHA) + mapped_ball_center[1] * SKETCH_BALL_SMOOTHING_ALPHA)
                    final_mapped_ball_pos = (draw_x, draw_y)
                else:
                    final_mapped_ball_pos = mapped_ball_center
                self.last_mapped_ball_sketch_pos = final_mapped_ball_pos
            else: # Top haritalanamadı veya oyuncu tarafından gizlendi
                self.last_mapped_ball_sketch_pos = None
                final_mapped_ball_pos = None
        else: # Top tespiti yok
            self.last_mapped_ball_sketch_pos = None
            final_mapped_ball_pos = None

        if final_mapped_ball_pos is not None:
            # Topu çiz (gölge, ana renk, vurgu, kenarlık)
            cv2.circle(sketch, (final_mapped_ball_pos[0]+1, final_mapped_ball_pos[1]+1), 4, (100,100,100), -1, cv2.LINE_AA) # Gölge
            cv2.circle(sketch, final_mapped_ball_pos, 5, (100, 255, 255), -1, cv2.LINE_AA) # Ana renk (açık sarı)
            cv2.circle(sketch, final_mapped_ball_pos, 3, (0, 255, 255), -1, cv2.LINE_AA)   # Vurgu (parlak sarı)
            cv2.circle(sketch, final_mapped_ball_pos, 3, (0,0,0), 1, cv2.LINE_AA)        # Kenarlık

        # Top yörüngesi (kullanılmıyor ama gelecekte eklenebilir)

        # Topun hedefini gösteren çizgi (isteğe bağlı)
        if final_mapped_ball_pos and player_positions_on_sketch_for_lines:
            if len(player_positions_on_sketch_for_lines) == 2 and ball_trajectory and len(ball_trajectory) >=2:
                # Basit yön tahmini (son iki kuşbakışı top konumuna göre)
                last_sketch_pos = None
                prev_sketch_pos = None
                
                # Yörüngede geçerli son iki noktayı bul
                valid_sketch_points = [item[0] for item in ball_trajectory if item[0] is not None]
                if len(valid_sketch_points) >= 2:
                    last_ball_orig_coord = valid_sketch_points[-1]
                    prev_ball_orig_coord = valid_sketch_points[-2]
                    
                    last_sketch_pos_candidate = self.map_to_birdseye(last_ball_orig_coord)
                    prev_sketch_pos_candidate = self.map_to_birdseye(prev_ball_orig_coord)

                    if last_sketch_pos_candidate and prev_sketch_pos_candidate:
                        # Sadece çizimdeki son top konumu ile (yumuşatılmış)
                        # veya doğrudan yörüngeden gelen haritalanmış noktalarla çalışabiliriz.
                        # final_mapped_ball_pos zaten yumuşatılmış olduğu için onu kullanalım.
                        # Bir önceki adımda yumuşatılmamış haritalanmış pozisyona ihtiyacımız var.
                        
                        # Bu kısım daha karmaşık bir yörünge analizi gerektirebilir.
                        # Şimdilik, topun genel Y hareketine göre hedef belirleyelim.
                        sketch_ball_dir_y = 0
                        if self.last_mapped_ball_sketch_pos and final_mapped_ball_pos and \
                           self.last_mapped_ball_sketch_pos[1] != final_mapped_ball_pos[1]: # Y'de hareket var mı?
                            sketch_ball_dir_y = final_mapped_ball_pos[1] - self.last_mapped_ball_sketch_pos[1]

                        target_player_pos = None
                        if sketch_ball_dir_y < -1: # Top yukarı gidiyor (Y azalıyor)
                            target_player_pos = min(player_positions_on_sketch_for_lines, key=lambda p: p['pos'][1])['pos']
                        elif sketch_ball_dir_y > 1: # Top aşağı gidiyor (Y artıyor)
                            target_player_pos = max(player_positions_on_sketch_for_lines, key=lambda p: p['pos'][1])['pos']
                        
                        if target_player_pos:
                            cv2.line(sketch, final_mapped_ball_pos, target_player_pos, (255, 100, 255), 1, cv2.LINE_AA) # Mor hedef çizgisi


        cv2.putText(sketch, "BASELINE", (COURT_LEFT_X + 10, COURT_TOP_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1, cv2.LINE_AA)
        cv2.putText(sketch, "BASELINE", (COURT_LEFT_X + 10, COURT_BOTTOM_Y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1, cv2.LINE_AA)

        return sketch

    def is_court_centered(self, frame: np.ndarray) -> bool:
        # Kortun kamera görüntüsünde merkezlenip merkezlenmediğini kontrol eder (replay tespiti için)
        min_center_fill_ratio = 0.7 # Merkez ROI'nin ne kadarının kortla dolu olması gerektiği
        min_overall_court_area_ratio = 0.20 # Kortun toplam frame alanına oranı (minimum)
        max_overall_court_area_ratio = 0.75 # Kortun toplam frame alanına oranı (maksimum)
        min_aspect_ratio = 1.2 # Kortun sınırlayıcı kutusunun en-boy oranı (genişlik/yükseklik)
        max_aspect_ratio = 2.8

        mask = self.detect_court_area_mask(frame)
        h, w = mask.shape

        # Merkez ROI doluluk kontrolü
        center_x1, center_x2 = int(w * 0.25), int(w * 0.75)
        center_y1, center_y2 = int(h * 0.25), int(h * 0.75)
        center_mask_roi = mask[center_y1:center_y2, center_x1:center_x2]
        total_center_roi_area = (center_x2 - center_x1) * (center_y2 - center_y1)
        actual_center_fill_ratio = 0.0
        if total_center_roi_area > 0:
            filled_center_area = cv2.countNonZero(center_mask_roi)
            actual_center_fill_ratio = filled_center_area / total_center_roi_area

        # Genel alan ve en-boy oranı kontrolü
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overall_area_check = False
        aspect_ratio_check = False

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            frame_area = w * h
            actual_overall_area_ratio = contour_area / frame_area if frame_area > 0 else 0

            x_rect, y_rect, rect_w, rect_h = cv2.boundingRect(largest_contour)
            actual_aspect_ratio = rect_w / rect_h if rect_h > 0 else 0

            overall_area_check = min_overall_court_area_ratio < actual_overall_area_ratio < max_overall_court_area_ratio
            aspect_ratio_check = min_aspect_ratio < actual_aspect_ratio < max_aspect_ratio

        if actual_center_fill_ratio > min_center_fill_ratio and overall_area_check and aspect_ratio_check:
            return True # Kort merkezlenmiş ve makul boyutta

        return False # Kort merkezlenmemiş veya boyutu/oranı uygun değil

class PlayerDetector:
    def __init__(self, video_w: int, video_h: int, court_detector: CourtDetector, player_search_roi: Optional[Tuple[int, int, int, int]] = None):
        self.width = video_w
        self.height = video_h
        self.court_detector = court_detector
        self.player_search_roi = player_search_roi # Oyuncu arama ROI'si

        # Yasak alanlar listesi (x1, y1, x2, y2) formatında
        self.forbidden_areas = [
            # Mevcut yasaklı alanlar
            (378, 293, 454, 300),
            (637, 269, 644, 456),
            (638, 299, 766, 301),
            (767, 297, 838, 301),
            (843, 296, 904, 300),
            (836, 296, 845, 306),
            (823, 270, 837, 304),
            (636, 268, 833, 277),
            (902, 292, 980, 352),
            (638, 447, 896, 454),
            (875, 415, 894, 448),
            (867, 387, 875, 417),
            (847, 343, 856, 350),
            (296, 289, 380, 297),
            (452, 271, 641, 274),
            (382, 448, 650, 450),
            (334, 571, 941, 576),
            (646, 313, 904, 317),
            (647, 335, 896, 340),
            # Yeni eklenen yasaklı alanlar
            (236, 572, 346, 576),
            (968, 3, 1310, 447),
            (539, 106, 747, 131),
            (4, 2, 350, 120),
            (6, 123, 243, 718),
            (241, 604, 307, 666),
            (455, 301, 649, 301),
            (426, 341, 435, 351),
            (299, 297, 375, 346),
            (474, 221, 796, 223),
            (795, 222, 870, 222),
            (846, 102, 912, 139),
            (915, 49, 966, 157),
            (408, 10, 984, 98),
            (448, 1, 960, 11)
        ]

        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=300,  # Geçmiş kare sayısını azalttık (daha hızlı adaptasyon için)
            varThreshold=16,  # Varyans eşiğini düşürdük (daha hassas hareket tespiti)
            detectShadows=True  # Gölgeleri tespit etmeyi açtık
        )
        self.min_solidity = PLAYER_MIN_SOLIDITY # Oyuncu konturu için minimum doluluk oranı
        self.absolute_last_valid_player_state: Dict[int, Optional[Dict[str, Any]]] = {0: None, 1: None} # Oyuncuların son geçerli durumları

    def reset_bg_subtractor(self):
        # Arka plan çıkarıcıyı sıfırlar
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)
        self.absolute_last_valid_player_state = {0: None, 1: None}

    def _is_bbox_inside_roi(self, bbox: Tuple[int,int,int,int], roi: Optional[Tuple[int,int,int,int]]) -> bool:
        # Bir sınırlayıcı kutunun ROI içinde olup olmadığını kontrol eder
        if not roi: return True # ROI yoksa her zaman içindedir
        x, y, w, h = bbox
        rx1, ry1, rx2, ry2 = roi
        return x >= rx1 and y >= ry1 and (x + w) <= rx2 and (y + h) <= ry2

    def _is_point_in_forbidden_area(self, x: int, y: int) -> bool:
        """Bir noktanın yasak alanlarda olup olmadığını kontrol eder"""
        for x1, y1, x2, y2 in self.forbidden_areas:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def _is_bbox_intersects_forbidden_area(self, bbox: Tuple[int,int,int,int]) -> bool:
        """Bir sınırlayıcı kutunun yasak alanlarla kesişip kesişmediğini kontrol eder"""
        x, y, w, h = bbox
        # Bbox'ın köşe noktalarını kontrol et
        corners = [
            (x, y),           # Sol üst
            (x + w, y),       # Sağ üst
            (x, y + h),       # Sol alt
            (x + w, y + h)    # Sağ alt
        ]
        # Herhangi bir köşe yasak alandaysa veya bbox'ın merkezi yasak alandaysa
        center_x = x + w // 2
        center_y = y + h // 2
        return any(self._is_point_in_forbidden_area(cx, cy) for cx, cy in corners) or \
               self._is_point_in_forbidden_area(center_x, center_y)

    def detect_players(self, frame: np.ndarray) -> List[Optional[Dict[str, Any]]]:
        # Oyuncuları tespit eder
        roi_x1, roi_y1, roi_x2, roi_y2 = self.player_search_roi if self.player_search_roi else (0, 0, self.width, self.height)

        frame_for_fg = frame[roi_y1:roi_y2, roi_x1:roi_x2] if self.player_search_roi else frame.copy()
        if frame_for_fg.size == 0: return [None, None] # ROI geçersizse

        # Öğrenme oranını artırdık (daha hızlı adaptasyon)
        fg_mask_roi = self.fgbg.apply(frame_for_fg, learningRate=0.05)

        # Morfolojik işlemleri güncelledik
        # Daha küçük açma çekirdeği (gürültüyü daha az filtrele)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # Daha büyük kapama çekirdeği (hareketleri daha iyi birleştir)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        
        # Morfolojik işlemleri güncelledik
        fg_mask_roi_processed = cv2.morphologyEx(fg_mask_roi, cv2.MORPH_OPEN, open_kernel, iterations=1)
        fg_mask_roi_processed = cv2.morphologyEx(fg_mask_roi_processed, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        
        # Ek olarak, hareket alanlarını genişletmek için dilate işlemi ekledik
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask_roi_processed = cv2.dilate(fg_mask_roi_processed, dilate_kernel, iterations=1)

        contours, _ = cv2.findContours(fg_mask_roi_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        player_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x_roi, y_roi, w, h = cv2.boundingRect(contour)

            # ROI koordinatlarını tam frame koordinatlarına çevir
            x, y = x_roi + roi_x1, y_roi + roi_y1

            # Sınırlayıcı kutunun tamamının ROI içinde olup olmadığını kontrol et
            if not self._is_bbox_inside_roi((x, y, w, h), self.player_search_roi):
                continue

            # Yasak alan kontrolü
            if self._is_bbox_intersects_forbidden_area((x, y, w, h)):
                continue

            foot_pos_x = x + w // 2
            foot_pos_y = y + h

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Doluluk oranı kontrolünü biraz gevşettik
            if solidity < self.min_solidity * 0.8:  # %20 daha toleranslı
                continue

            aspect_ratio_cand = h / w if w > 0 else 0 # Yükseklik / Genişlik

            # Alan kontrolünü biraz gevşettik
            if not (GENERAL_PLAYER_MIN_AREA * 0.8 <= area <= GENERAL_PLAYER_MAX_AREA * 1.2):
                continue

            candidate = {
                'bbox': (x, y, w, h),
                'center': (x + w // 2, y + h // 2),
                'foot_pos': (foot_pos_x, foot_pos_y),
                'area': area,
                'solidity': solidity,
                'aspect_ratio': aspect_ratio_cand,
                'confidence': solidity, # Güven skoru olarak doluluk oranı
                'y_pos_for_sort': foot_pos_y # Sıralama için Y pozisyonu
            }
            player_candidates.append(candidate)

        # Oyuncuları kortun üst ve alt yarısına göre ayır
        # ROI'nin dikey ortasını referans al
        division_line_y_in_roi = (roi_y2 - roi_y1) // 2
        division_line_y_full_frame = roi_y1 + division_line_y_in_roi

        top_half_candidates = sorted(
            [p for p in player_candidates if p['foot_pos'][1] <= division_line_y_full_frame],
            key=lambda x: x['area'], reverse=True
        )
        bottom_half_candidates = sorted(
            [p for p in player_candidates if p['foot_pos'][1] > division_line_y_full_frame],
            key=lambda x: x['area'], reverse=True
        )

        detected_players: Dict[int, Optional[Dict[str, Any]]] = {0: None, 1: None} # 0: alt oyuncu, 1: üst oyuncu

        # Alt yarıdaki oyuncu (ID 0)
        for candidate in bottom_half_candidates:
            if (GENERAL_PLAYER_MIN_AREA <= candidate['area'] <= GENERAL_PLAYER_MAX_AREA and
                GENERAL_PLAYER_MIN_ASPECT_RATIO <= candidate['aspect_ratio'] <= GENERAL_PLAYER_MAX_ASPECT_RATIO):
                detected_players[0] = candidate
                break

        # Üst yarıdaki oyuncu (ID 1)
        for candidate in top_half_candidates:
            if (GENERAL_PLAYER_MIN_AREA <= candidate['area'] <= GENERAL_PLAYER_MAX_AREA and
                GENERAL_PLAYER_MIN_ASPECT_RATIO <= candidate['aspect_ratio'] <= GENERAL_PLAYER_MAX_ASPECT_RATIO):
                detected_players[1] = candidate
                break

        final_players: List[Optional[Dict[str, Any]]] = [None, None]
        for p_id in [0, 1]: # 0: Alt Oyuncu, 1: Üst Oyuncu
            current_detection = detected_players[p_id]
            last_known_valid_state = self.absolute_last_valid_player_state[p_id]

            if current_detection: # Bu karede oyuncu bulundu
                current_detection['id'] = p_id
                current_detection['missed_frames'] = 0
                final_players[p_id] = current_detection
                self.absolute_last_valid_player_state[p_id] = current_detection.copy() # Son geçerli durumu güncelle
            elif last_known_valid_state and self._is_bbox_inside_roi(last_known_valid_state['bbox'], self.player_search_roi):
                # Oyuncu bu karede bulunamadı ama son bilinen durumu ROI içinde ve geçerli
                ghost_player = last_known_valid_state.copy()
                ghost_player['missed_frames'] = ghost_player.get('missed_frames', 0) + 1
                ghost_player['confidence'] = max(0.2, ghost_player.get('confidence', 0.5) * 0.85) # Güveni daha yavaş azalt
                if ghost_player['missed_frames'] < 5: # Maksimum 5 kare hayalet göster (10'dan 5'e düşürüldü)
                     final_players[p_id] = ghost_player
                else: # Çok uzun süre kayıpsa sıfırla
                    self.absolute_last_valid_player_state[p_id] = None
                    final_players[p_id] = None # Veya varsayılan yer tutucu
            else: # Oyuncu bulunamadı ve geçerli son durum yok (veya ROI dışında)
                self.absolute_last_valid_player_state[p_id] = None
                # İsteğe bağlı: Varsayılan bir konumda yer tutucu oyuncu oluşturulabilir
                # roi_center_x = roi_x1 + (roi_x2 - roi_x1) // 2
                # ph_y = roi_y1 + int((roi_y2 - roi_y1) * (0.80 if p_id == 0 else 0.20)) # Alt/üst için varsayılan Y
                # placeholder = { ... 'confidence': 0.1 ... }
                # final_players[p_id] = placeholder
                final_players[p_id] = None


        return final_players


class BallDetector:
    def __init__(self, video_w: int, video_h: int, court_points: np.ndarray,
                 court_detector: CourtDetector, # CourtDetector örneği
                 ball_search_roi: Optional[Tuple[int,int,int,int]] = None):
        self.width = video_w
        self.height = video_h
        self.court_points = court_points # Kullanılmıyor olabilir, court_detector daha önemli
        self.court_detector = court_detector # Kort bilgileri ve haritalama için
        self.ball_search_roi = ball_search_roi

        # Yasak alanlar listesi (x1, y1, x2, y2) formatında
        self.forbidden_areas = [
            # Mevcut yasaklı alanlar
            (378, 293, 454, 300),
            (637, 269, 644, 456),
            (638, 299, 766, 301),
            (767, 297, 838, 301),
            (843, 296, 904, 300),
            (836, 296, 845, 306),
            (823, 270, 837, 304),
            (636, 268, 833, 277),
            (902, 292, 980, 352),
            (638, 447, 896, 454),
            (875, 415, 894, 448),
            (867, 387, 875, 417),
            (847, 343, 856, 350),
            (296, 289, 380, 297),
            (452, 271, 641, 274),
            (382, 448, 650, 450),
            (334, 571, 941, 576),
            (646, 313, 904, 317),
            (647, 335, 896, 340),
            (560, 190, 722, 197),
            # Yeni eklenen yasaklı alanlar
            (236, 572, 346, 576),
            (456, 683, 815, 706),
            (968, 3, 1310, 447),
            (539, 106, 747, 131),
            (4, 2, 443, 157),
            (6, 123, 243, 718),
            (241, 604, 307, 666),
            (455, 301, 649, 301),
            (426, 341, 435, 351),
            (299, 297, 375, 346),
            (474, 221, 796, 223),
            (795, 222, 870, 222),
            (846, 102, 912, 139),
            (915, 49, 966, 157),
            (408, 10, 984, 98),
            (448, 1, 960, 11)
        ]

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

        self.kalman = cv2.KalmanFilter(4, 2) # 4 durum (x,y,vx,vy), 2 ölçüm (x,y)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32) # Sabit hız modeli
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05 # İşlem gürültüsü
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5 # Ölçüm gürültüsü

        self.lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.old_gray: Optional[np.ndarray] = None
        self.p0: Optional[np.ndarray] = None # Optical Flow için önceki noktalar
        self.is_tracking = False
        self.frames_since_last_seen = 0

        self.ball_trajectory = deque(maxlen=30) # (merkez, durum) çiftlerini saklar
        self.last_positions = deque(maxlen=10) # Son MOG tespitleri (ortalama için)

        # Parametreler
        self.min_ball_confidence_mog = 70 # MOG'dan gelen aday için minimum güven (0-100)
        self.min_ball_circularity = 0.5
        self.min_ball_area = 3
        self.max_ball_area = 250
        self.max_line_proximity_distance = 60 # Oyuncular arası hayali çizgiye yakınlık
        self.line_proximity_bonus = 25
        self.avg_position_bonus = 25 # Ortalama MOG pozisyonuna yakınlık bonusu
        self.max_avg_distance = 120 # Ortalama pozisyona maksimum uzaklık (bonus için)

        self.court_line_penalty_distance = 8  # Kort çizgisine yakınlıkta ceza için piksel mesafesi (kuşbakışı)
        self.court_line_penalty_amount = 15   # Kort çizgisi cezası (puandan düşülür)
        self.net_penalty_distance = 15        # Fileye yakınlıkta ceza için piksel mesafesi
        self.net_penalty_amount = 10          # File cezası
        self.prev_player_penalty_amount = 50  # Önceki karede oyuncu içindeyse ceza

        # Oyuncu kutusu yakınlık cezaları
        self.player_box_proximity_distance = 30  # Oyuncu kutusuna yakınlık mesafesi
        self.player_box_proximity_penalty = 40   # Oyuncu kutusuna yakınlık cezası
        self.player_box_top_penalty = 60        # Oyuncu kutusunun üstünde olma cezası
        self.player_box_top_distance = 20       # Oyuncu kutusunun üstünde olma mesafesi

        self.max_predict_frames = 5           # Kaç kare boyunca tahmin yapılacak (MOG/OF yoksa)
        self.last_velocity: Optional[Tuple[float, float]] = None # Son hız (dx, dy)
        self.prev_player_bboxes: Optional[List[Tuple[int,int,int,int]]] = None # Önceki karedeki oyuncu bbox'ları
        self.recent_player_bboxes = deque(maxlen=60) # Son 60 karedeki oyuncu bbox'ları (her eleman bir liste)

    def reset_bg_subtractor(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.last_positions.clear()
        self.ball_trajectory.clear()
        self.old_gray = None
        self.p0 = None
        self.is_tracking = False
        self.frames_since_last_seen = 0
        self.last_velocity = None
        self.prev_player_bboxes = None
        self.recent_player_bboxes.clear()

    def _get_average_position(self) -> Optional[Tuple[float, float]]:
        """Son pozisyonların ortalamasını hesaplar"""
        if not self.last_positions: return None
        valid_positions = [pos for pos in self.last_positions if pos is not None]
        if not valid_positions: return None
        avg_x = sum(x for x, y in valid_positions) / len(valid_positions)
        avg_y = sum(y for x, y in valid_positions) / len(valid_positions)
        return (avg_x, avg_y)

    def _is_point_in_forbidden_area(self, x: int, y: int) -> bool:
        """Bir noktanın yasak alanlarda olup olmadığını kontrol eder"""
        for x1, y1, x2, y2 in self.forbidden_areas:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def _calculate_player_box_penalty(self, x: int, y: int, players: Optional[list] = None) -> int:
        """Oyuncu kutularına yakınlık ve üstünde olma cezalarını hesaplar"""
        if not players:
            return 0

        total_penalty = 0
        for player in players:
            if not player or 'bbox' not in player:
                continue

            px, py, pw, ph = player['bbox']
            
            # Oyuncu kutusunun üstünde olma kontrolü
            if (px - self.player_box_top_distance <= x <= px + pw + self.player_box_top_distance and
                py - self.player_box_top_distance <= y <= py):
                total_penalty += self.player_box_top_penalty
                continue  # Üstteyse diğer cezaları hesaplamaya gerek yok
            
            # Oyuncu kutusuna yakınlık kontrolü
            # Kutu genişletilmiş alanı
            expanded_x1 = px - self.player_box_proximity_distance
            expanded_y1 = py - self.player_box_proximity_distance
            expanded_x2 = px + pw + self.player_box_proximity_distance
            expanded_y2 = py + ph + self.player_box_proximity_distance
            
            if (expanded_x1 <= x <= expanded_x2 and
                expanded_y1 <= y <= expanded_y2):
                # Mesafeye göre ceza hesapla (yakınlık arttıkça ceza artar)
                dist_x = min(abs(x - expanded_x1), abs(x - expanded_x2))
                dist_y = min(abs(y - expanded_y1), abs(y - expanded_y2))
                dist = min(dist_x, dist_y)
                penalty_ratio = 1.0 - (dist / self.player_box_proximity_distance)
                total_penalty += int(self.player_box_proximity_penalty * penalty_ratio)

        return total_penalty

    def _find_ball_candidate_mog(self, frame_roi: np.ndarray, players: Optional[list] = None) -> Tuple[Optional[Tuple[float, float]], int]:
        # MOG ile top adayı bulur, skor (0-100) döndürür
        roi_x_offset = self.ball_search_roi[0] if self.ball_search_roi else 0
        roi_y_offset = self.ball_search_roi[1] if self.ball_search_roi else 0

        fgmask = self.bg_subtractor.apply(frame_roi, learningRate=0.02)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_candidates = []
        avg_pos_full_frame = self._get_average_position() # Tam çerçeve koordinatlarında

        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_ball_area <= area <= self.max_ball_area): continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * area / (perimeter**2)
            if circularity < self.min_ball_circularity: continue

            x_r, y_r, w, h = cv2.boundingRect(contour)
            aspect = w / float(h) if h > 0 else 0
            if not (0.4 <= aspect <= 2.5): continue # Çok ince veya çok kalın değil

            (cx_roi, cy_roi), _ = cv2.minEnclosingCircle(contour)
            cx_full, cy_full = int(cx_roi + roi_x_offset), int(cy_roi + roi_y_offset)

            # Yasak alan kontrolü
            if self._is_point_in_forbidden_area(cx_full, cy_full):
                continue

            # Son 60 karedeki oyuncu bbox'larında elenme kontrolü
            is_in_recent_player_bbox = False
            for bbox_list_for_a_frame in self.recent_player_bboxes:
                for px, py, pw, ph_player in bbox_list_for_a_frame:
                    if px <= cx_full <= px + pw and py <= cy_full <= py + ph_player:
                        is_in_recent_player_bbox = True
                        break
                if is_in_recent_player_bbox: break
            if is_in_recent_player_bbox: continue

            # Skorlama
            circ_score = int(circularity * 40)
            asp_score = int((1.0 - abs(1.0 - aspect)) * 30) # 1'e yakın aspect daha iyi
            area_s = int((area / self.max_ball_area) * 30)
            base_confidence = circ_score + asp_score + area_s

            # Oyuncular arası hayali çizgiye yakınlık bonusu
            line_prox_bonus = 0
            if players and len(players) == 2:
                p1_data = players[0]
                p2_data = players[1]
                if p1_data and p2_data and p1_data.get('confidence',0) > 0.5 and p2_data.get('confidence',0) > 0.5:
                    p1c, p2c = p1_data['center'], p2_data['center']
                    dist_to_line = point_segment_distance(cx_full, cy_full, p1c[0], p1c[1], p2c[0], p2c[1])
                    if dist_to_line < self.max_line_proximity_distance:
                        line_prox_bonus = int(self.line_proximity_bonus * (1 - dist_to_line / self.max_line_proximity_distance))

            # Ortalama MOG pozisyonuna yakınlık bonusu
            avg_pos_bonus_val = 0
            if avg_pos_full_frame:
                dist_to_avg = np.sqrt((cx_full - avg_pos_full_frame[0])**2 + (cy_full - avg_pos_full_frame[1])**2)
                if dist_to_avg < self.max_avg_distance:
                    avg_pos_bonus_val = int(self.avg_position_bonus * (1 - dist_to_avg / self.max_avg_distance))

            # Kort çizgisi ve file cezaları
            court_l_penalty, net_l_penalty = 0, 0
            mapped_ball_sketch = self.court_detector.map_to_birdseye((cx_full, cy_full))
            if mapped_ball_sketch:
                min_dist_to_court_line = float('inf')
                for line_pts in SKETCH_LINES:
                    (x1,y1), (x2,y2) = line_pts
                    # File çizgisini atla (ayrıca ele alınacak)
                    if NET_Y in [y1, y2] and abs(x1-COURT_LEFT_X) < 1 and abs(x2-COURT_RIGHT_X) < 1 : # Bu file çizgisi
                        net_dist = point_segment_distance(mapped_ball_sketch[0], mapped_ball_sketch[1], x1,y1,x2,y2)
                        if net_dist < self.net_penalty_distance:
                            net_l_penalty = self.net_penalty_amount
                        continue # Diğer çizgiler için devam et
                    
                    d = point_segment_distance(mapped_ball_sketch[0], mapped_ball_sketch[1], x1,y1,x2,y2)
                    if d < min_dist_to_court_line: min_dist_to_court_line = d
                
                if min_dist_to_court_line < self.court_line_penalty_distance:
                    court_l_penalty = self.court_line_penalty_amount

            # Önceki karede oyuncu bbox'ı içindeyse ceza
            prev_player_box_penalty = 0
            if self.prev_player_bboxes:
                for prev_pb_x, prev_pb_y, prev_pb_w, prev_pb_h in self.prev_player_bboxes:
                    if prev_pb_x <= cx_full <= prev_pb_x + prev_pb_w and \
                       prev_pb_y <= cy_full <= prev_pb_y + prev_pb_h:
                        prev_player_box_penalty = self.prev_player_penalty_amount
                        break

            # Oyuncu kutusu yakınlık ve üstünde olma cezaları
            player_box_penalty = self._calculate_player_box_penalty(cx_full, cy_full, players)
            
            total_confidence = min(100, max(0, base_confidence + line_prox_bonus + avg_pos_bonus_val - 
                                          court_l_penalty - net_l_penalty - prev_player_box_penalty - player_box_penalty))
            
            if total_confidence >= self.min_ball_confidence_mog: # MOG için iç eşik
                 valid_candidates.append(((cx_roi, cy_roi), total_confidence))

        if valid_candidates:
            valid_candidates.sort(key=lambda x: x[1], reverse=True) # En yüksek güvenli olanı seç
            best_pos_roi, best_score = valid_candidates[0]
            self.last_positions.append((best_pos_roi[0] + roi_x_offset, best_pos_roi[1] + roi_y_offset))
            return best_pos_roi, int(best_score)
        else:
            self.last_positions.append(None)
            return None, 0

    def detect_and_track(self, frame: np.ndarray, players: Optional[list] = None) -> Optional[Dict[str, Any]]:
        # 0. Kurulum
        roi_x_offset, roi_y_offset = 0, 0
        frame_roi = frame
        if self.ball_search_roi:
            x1, y1, x2, y2 = self.ball_search_roi
            x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(self.width, x2), min(self.height, y2)
            if x1c < x2c and y1c < y2c:
                frame_roi = frame[y1c:y2c, x1c:x2c]
                roi_x_offset, roi_y_offset = x1c, y1c
            else: # Geçersiz ROI
                self.ball_trajectory.append((None, "INVALID_ROI"))
                return None
        if frame_roi.size == 0:
            self.ball_trajectory.append((None, "EMPTY_ROI_FRAME"))
            return None

        frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

        # Oyuncu bilgilerini güncelle
        current_frame_player_bboxes = []
        if players:
            for p_data in players:
                if p_data and 'bbox' in p_data:
                    current_frame_player_bboxes.append(p_data['bbox'])
        self.recent_player_bboxes.append(current_frame_player_bboxes) # deque'ye ekle
        # self.prev_player_bboxes bir önceki çağrıdan kalmalı, _find_ball_candidate_mog'da kullanılacak
        # Bu çağrının sonunda güncellenecek.

        # 1. MOG Tespiti
        mog_measurement_roi, mog_score = self._find_ball_candidate_mog(frame_roi, players) # mog_score 0-100

        # 2. Optical Flow
        of_point_roi, of_successful = None, False
        if self.is_tracking and self.old_gray is not None and self.p0 is not None and self.p0.size > 0:
            try:
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
                if st is not None and st[0][0] == 1 and p1 is not None:
                    of_point_roi = (p1[0][0][0], p1[0][0][1])
                    of_successful = True
            except cv2.error: of_successful = False

        # 3. Kalman Tahmini
        kalman_state = self.kalman.predict()
        kalman_predicted_pos_roi = (kalman_state[0,0], kalman_state[1,0])

        # 4. En iyi ölçümü ve kaynağını belirle
        final_measurement_roi: Optional[Tuple[float,float]] = None
        source_of_ball = "NONE"
        current_ball_score = 0 # 0-100 ölçeğinde

        # Öncelik: MOG > OF > Hız Tahmini > Kalman Durum Tahmini
        if mog_measurement_roi is not None: # MOG bir şey buldu (iç eşiğini zaten geçti)
            final_measurement_roi = mog_measurement_roi
            source_of_ball = "MOG"
            current_ball_score = mog_score
            self.frames_since_last_seen = 0
        elif of_successful and of_point_roi is not None:
            dist_of_kalman = np.linalg.norm(np.array(of_point_roi) - np.array(kalman_predicted_pos_roi))
            if dist_of_kalman < 30: # OF, Kalman'a çok uzak değilse
                final_measurement_roi = of_point_roi
                source_of_ball = "OF"
                current_ball_score = 75 # OF için sabit güven
                self.frames_since_last_seen = 0
        
        # Eğer MOG veya OF ile ölçüm varsa Kalman'ı düzelt ve hızı güncelle
        if final_measurement_roi is not None: # MOG veya OF başarılı
            self.kalman.correct(np.array([[final_measurement_roi[0]], [final_measurement_roi[1]]], dtype=np.float32))
            # Hızı güncelle
            if self.ball_trajectory and self.ball_trajectory[-1][0] is not None:
                prev_ball_x_full, prev_ball_y_full = self.ball_trajectory[-1][0]
                current_ball_x_full = final_measurement_roi[0] + roi_x_offset
                current_ball_y_full = final_measurement_roi[1] + roi_y_offset
                self.last_velocity = (current_ball_x_full - prev_ball_x_full, current_ball_y_full - prev_ball_y_full)
            elif self.last_velocity is None: # İlk hız için
                 self.last_velocity = (kalman_state[2,0], kalman_state[3,0]) if kalman_state is not None else (0,0)

        elif self.is_tracking and self.frames_since_last_seen < self.max_predict_frames: # MOG/OF yok, tahmin et
            self.frames_since_last_seen += 1
            predicted_using_velocity = False
            if self.last_velocity and self.ball_trajectory and self.ball_trajectory[-1][0] is not None:
                prev_ball_x_full, prev_ball_y_full = self.ball_trajectory[-1][0]
                pred_x_full_vel = prev_ball_x_full + self.last_velocity[0]
                pred_y_full_vel = prev_ball_y_full + self.last_velocity[1]
                final_measurement_roi = (pred_x_full_vel - roi_x_offset, pred_y_full_vel - roi_y_offset)
                source_of_ball = "VEL_PRED"
                current_ball_score = 65
                predicted_using_velocity = True
            
            if not predicted_using_velocity: # Hız tahmini yoksa Kalman durumunu kullan
                final_measurement_roi = kalman_predicted_pos_roi
                source_of_ball = "KALMAN_PRED"
                current_ball_score = 60
            # Tahminler için Kalman zaten predict() ile ilerledi.
            # Hızı Kalman durumundan güncel tutmaya çalış
            if self.last_velocity is None and kalman_state is not None: # Eğer hız hiç set edilmemişse
                 self.last_velocity = (kalman_state[2,0], kalman_state[3,0])
        else: # Kayıp
            self.is_tracking = False
            self.frames_since_last_seen = 0
            self.last_velocity = None
            self.ball_trajectory.append((None, "LOST"))
            self.p0 = None
            self.old_gray = frame_gray.copy()
            self.prev_player_bboxes = current_frame_player_bboxes # Bir sonraki _find_ball_candidate_mog için
            return None

        # 5. Seçilen final_measurement_roi'yi işle
        if final_measurement_roi is None:
            self.ball_trajectory.append((None, "NO_MEASURE")) # Bu duruma gelinmemeli
            self.prev_player_bboxes = current_frame_player_bboxes
            return None

        self.old_gray = frame_gray.copy()
        self.p0 = np.array([[[final_measurement_roi[0], final_measurement_roi[1]]]], dtype=np.float32)
        self.prev_player_bboxes = current_frame_player_bboxes # Bir sonraki _find_ball_candidate_mog için

        final_cx = final_measurement_roi[0] + roi_x_offset
        final_cy = final_measurement_roi[1] + roi_y_offset

        # ROI sınır kontrolü
        if self.ball_search_roi:
            bs_x1, bs_y1, bs_x2, bs_y2 = self.ball_search_roi
            if not (bs_x1 <= final_cx <= bs_x2 and bs_y1 <= final_cy <= bs_y2):
                self.ball_trajectory.append((None, f"OUT_OF_ROI ({source_of_ball})"))
                self.is_tracking = False # ROI dışına çıkarsa takibi bırak
                return None # ROI dışındaysa gösterme

        final_reported_confidence = current_ball_score / 100.0

        if final_reported_confidence < 0.60: # Ana güven eşiği
            self.ball_trajectory.append((None, f"LOW_CONF ({source_of_ball}, {final_reported_confidence:.2f})"))
            # Takip devam edebilir (frames_since_last_seen yönetir), ama bu karede gösterme
            return None
            
        self.is_tracking = True
        self.ball_trajectory.append(((final_cx, final_cy), source_of_ball))
        
        return {
            'center': (int(final_cx), int(final_cy)),
            'radius': 5, # Sabit yarıçap
            'status': source_of_ball,
            'confidence': final_reported_confidence
        }

class TennisAnalyzer:
    def __init__(self, video_path: str):
        global VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS
        self.video_path = video_path
        VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS = get_video_properties(self.video_path)
        if VIDEO_WIDTH is None: sys.exit("Video özellikleri alınamadı.")

        self.cap: Optional[cv2.VideoCapture] = None
        self.out_main: Optional[cv2.VideoWriter] = None
        self.out_sketch: Optional[cv2.VideoWriter] = None
        self.current_source_points = DEFAULT_SOURCE_POINTS.copy()

        self.court_detector = CourtDetector(self.current_source_points, DESTINATION_POINTS, VIDEO_WIDTH, VIDEO_HEIGHT)
        # BallDetector ve PlayerDetector başlatılması run_analysis'e taşındı, çünkü court_detector'ın ROI'si orada belirleniyor
        self.player_detector: Optional[PlayerDetector] = None
        self.ball_detector: Optional[BallDetector] = None
        self.replay_mode = False

    def setup_video_writers(self, output_dir: str = "output") -> None:
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_fps = (VIDEO_FPS / SLOWDOWN_FACTOR) if VIDEO_FPS and VIDEO_FPS > 0 else (30.0 / SLOWDOWN_FACTOR)

        self.out_main = cv2.VideoWriter(os.path.join(output_dir, 'output_analyzed.mp4'), fourcc, output_fps, (VIDEO_WIDTH, VIDEO_HEIGHT))
        self.out_sketch = cv2.VideoWriter(os.path.join(output_dir, 'output_sketch.mp4'), fourcc, output_fps, (SKETCH_WIDTH, SKETCH_HEIGHT))

    def replay_state_reset(self):
        # Yeniden oynatma modundan çıkarken dedektörleri sıfırla
        # CourtDetector'ın kendisi sıfırlanmaz, sadece homografisi güncellenebilir
        # self.court_detector = CourtDetector(...) # Gerekirse, ama genellikle source_points güncellenir
        if VIDEO_WIDTH is None or VIDEO_HEIGHT is None: return # VGuard

        self.player_detector = PlayerDetector(VIDEO_WIDTH, VIDEO_HEIGHT, self.court_detector, player_search_roi=self.court_detector.ball_search_roi)
        if self.court_detector.ball_search_roi: # ROI varsa BallDetector'ı başlat
            self.ball_detector = BallDetector(VIDEO_WIDTH, VIDEO_HEIGHT, self.current_source_points, self.court_detector, self.court_detector.ball_search_roi)
        else: # ROI henüz belirlenmemişse (ilk karede olabilir)
            self.ball_detector = None

    def run_analysis(self, process_full_video: bool = True, process_30sec: bool = True) -> None:
        if not process_full_video:
            print("Video analizi için process_full_video True olmalıdır")
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Hata: Video dosyası açılamadı {self.video_path}")
            return

        ret, first_frame = self.cap.read()
        if not ret or VIDEO_WIDTH is None or VIDEO_HEIGHT is None : # VGuard
            print("Hata: İlk kare okunamadı veya video özellikleri eksik")
            self.cleanup()
            return

        # Sabit ROI (önceden belirlenmiş)
        self.court_detector.ball_search_roi = (int(VIDEO_WIDTH * 0.20), int(VIDEO_HEIGHT * 0.15),
                                                int(VIDEO_WIDTH * 0.85), int(VIDEO_HEIGHT * 0.90))
        print(f"Kullanılan sabit ROI: {self.court_detector.ball_search_roi}")

        auto_court_corners = self.court_detector.find_court_corners_from_blue_mask(first_frame)
        if auto_court_corners is not None:
            self.current_source_points = auto_court_corners
            self.court_detector.update_source_points(self.current_source_points)
            print("Kort köşeleri otomatik olarak algılandı.")
        else:
            print("Varsayılan kort köşeleri kullanılıyor.")

        def process_video(output_prefix: str):
            # Dedektörleri başlat/tekrar başlat
            self.replay_state_reset()
            
            # Video yazıcıları ayarla
            if not os.path.exists("output"): os.makedirs("output")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_fps = (VIDEO_FPS / SLOWDOWN_FACTOR) if VIDEO_FPS and VIDEO_FPS > 0 else (30.0 / SLOWDOWN_FACTOR)
            
            self.out_main = cv2.VideoWriter(os.path.join("output", f'{output_prefix}_analyzed.mp4'), 
                                          fourcc, output_fps, (VIDEO_WIDTH, VIDEO_HEIGHT))
            self.out_sketch = cv2.VideoWriter(os.path.join("output", f'{output_prefix}_sketch.mp4'), 
                                            fourcc, output_fps, (SKETCH_WIDTH, SKETCH_HEIGHT))

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Videoyu başa sar
            
            print(f"\n{output_prefix} analizi başlıyor...")
            cv2.namedWindow("Tennis Analysis", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Bird's Eye View", cv2.WINDOW_NORMAL)

            frame_count = 0
            auto_court_update_interval = 30
            replay_frames_counter = 0
            min_frames_for_replay_trigger = 5

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: break

                if frame_count % auto_court_update_interval == 0:
                    new_corners = self.court_detector.find_court_corners_from_blue_mask(frame)
                    if new_corners is not None and len(new_corners) == 4:
                        self.current_source_points = new_corners
                        self.court_detector.update_source_points(self.current_source_points)

                court_is_centered_now = self.court_detector.is_court_centered(frame)

                if not court_is_centered_now:
                    replay_frames_counter += 1
                    if replay_frames_counter >= min_frames_for_replay_trigger and not self.replay_mode:
                        print("Yeniden oynatma algılandı! Analiz duraklatıldı.")
                        self.replay_mode = True
                        if self.player_detector: self.player_detector.reset_bg_subtractor()
                        if self.ball_detector: self.ball_detector.reset_bg_subtractor()
                else:
                    replay_frames_counter = 0
                    if self.replay_mode:
                        print("Yeniden oynatma bitti, analiz devam ediyor.")
                        self.replay_mode = False
                        self.replay_state_reset()

                if self.replay_mode:
                    display_frame_replay = frame.copy()
                    cv2.putText(display_frame_replay, "REPLAY DETECTED - ANALYSIS PAUSED", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
                    cv2.imshow("Tennis Analysis", display_frame_replay)
                    blank_sketch_replay = np.ones((SKETCH_HEIGHT, SKETCH_WIDTH, 3), dtype=np.uint8) * 220
                    cv2.putText(blank_sketch_replay, "REPLAY", (SKETCH_WIDTH//2 - 50, SKETCH_HEIGHT//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
                    cv2.imshow("Bird's Eye View", blank_sketch_replay)
                    if self.out_main: self.out_main.write(display_frame_replay)
                    if self.out_sketch: self.out_sketch.write(blank_sketch_replay)
                else:
                    if not self.player_detector or not self.ball_detector:
                        self.replay_state_reset()
                        if not self.player_detector or not self.ball_detector:
                            if cv2.waitKey(1) & 0xFF == ord('q'): break
                            frame_count += 1
                            continue

                    display_frame = frame.copy()
                    players = self.player_detector.detect_players(display_frame)
                    ball = self.ball_detector.detect_and_track(frame, players)

                    self._draw_overlays(display_frame, players, ball)
                    sketch_frame = self.court_detector.create_sketch_frame(players, ball, 
                                                                         self.ball_detector.ball_trajectory if self.ball_detector else deque())

                    cv2.imshow("Tennis Analysis", display_frame)
                    cv2.imshow("Bird's Eye View", sketch_frame)
                    if self.out_main: self.out_main.write(display_frame)
                    if self.out_sketch: self.out_sketch.write(sketch_frame)

                delay = int(1000 / (VIDEO_FPS / SLOWDOWN_FACTOR)) if VIDEO_FPS and VIDEO_FPS > 0 else 30
                if cv2.waitKey(max(1, delay)) & 0xFF == ord('q'):
                    break
                frame_count += 1
                if VIDEO_FPS and frame_count % VIDEO_FPS == 0:
                    print(f"İşlenen kare: {frame_count} ({frame_count // VIDEO_FPS} saniye)...", end='\r')

            print(f"\n{output_prefix} analizi tamamlandı. Toplam işlenen kare: {frame_count}.")
            self.cleanup()

        # Tüm videoyu işle
        process_video("full")

    def _draw_overlays(self, frame: np.ndarray, players: List[Optional[Dict[str, Any]]],
                      ball: Optional[Dict[str, Any]]) -> None:
        # Yasaklı alanları çizme kodu kaldırıldı

        # Oyuncuları çiz
        for player_data in players:
            if not player_data or player_data.get('confidence', 0.0) < 0.3: continue
            
            x, y, w, h = player_data['bbox']
            player_id = player_data.get('id', -1)
            confidence = player_data.get('confidence', 0.0)

            # Modern renk paleti
            if player_id == 0:  # P1
                base_color = (0, 0, 255)  # Kırmızı
                highlight_color = (0, 50, 255)  # Parlak kırmızı
            else:  # P2
                base_color = (255, 0, 0)  # Mavi
                highlight_color = (255, 50, 0)  # Parlak mavi

            # Güven skoruna göre renk ayarı
            if confidence < 0.7:
                base_color = tuple(int(c * (0.7 + 0.3 * confidence)) for c in base_color)
                highlight_color = tuple(int(c * (0.7 + 0.3 * confidence)) for c in highlight_color)

            # Kalınlık ayarları - daha ince çizgiler
            thickness = 2 if confidence >= 0.6 else 1
            
            if w > 0 and h > 0:
                # Gölge efekti (daha hafif)
                shadow_offset = 3
                cv2.rectangle(frame, 
                            (x+shadow_offset, y+shadow_offset), 
                            (x+w+shadow_offset, y+h+shadow_offset), 
                            (0, 0, 0), 1)
                
                # Yarı şeffaf dolgu
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x+w, y+h), base_color, -1)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                
                # İnce dış çerçeve
                cv2.rectangle(frame, (x, y), (x+w, y+h), base_color, 1)
                
                # Köşe vurguları (daha ince)
                corner_length = min(w, h) // 5
                # Sol üst köşe
                cv2.line(frame, (x, y), (x+corner_length, y), highlight_color, 1)
                cv2.line(frame, (x, y), (x, y+corner_length), highlight_color, 1)
                # Sağ üst köşe
                cv2.line(frame, (x+w-corner_length, y), (x+w, y), highlight_color, 1)
                cv2.line(frame, (x+w, y), (x+w, y+corner_length), highlight_color, 1)
                # Sol alt köşe
                cv2.line(frame, (x, y+h-corner_length), (x, y+h), highlight_color, 1)
                cv2.line(frame, (x, y+h), (x+corner_length, y+h), highlight_color, 1)
                # Sağ alt köşe
                cv2.line(frame, (x+w-corner_length, y+h), (x+w, y+h), highlight_color, 1)
                cv2.line(frame, (x+w, y+h), (x+w, y+h-corner_length), highlight_color, 1)

            # Etiket için modern tasarım
            label = f"P{player_id + 1}"
            
            # Etiket arka planı
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness)[0]
            
            # Etiket arka planı (yarı şeffaf ve kutu renginde)
            label_bg_height = label_size[1] + 8
            label_bg_width = label_size[0] + 16
            
            # Etiket arka planı
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (x, y-label_bg_height), 
                         (x+label_bg_width, y), 
                         base_color, -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Etiket alt çizgisi (ince)
            cv2.line(frame, 
                    (x, y-label_bg_height), 
                    (x+label_bg_width, y-label_bg_height), 
                    highlight_color, 1)
            
            # Etiket yazısı (beyaz renk)
            cv2.putText(frame, label, 
                       (x+8, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Topu çiz
        if ball and 'center' in ball and ball.get('confidence', 0.0) >= 0.6:
            ball_center = ball['center']
            ball_radius = ball.get('radius', 5)
            ball_conf = ball.get('confidence', 0.0)
            ball_status = ball.get('status', '')

            # Modern top renkleri
            if ball_conf > 0.85:
                ball_color = (0, 255, 100)  # Parlak yeşil
                glow_color = (0, 255, 150)  # Parlak yeşil parıltı
            elif ball_conf > 0.7:
                ball_color = (0, 200, 255)  # Turuncu
                glow_color = (0, 220, 255)  # Parlak turuncu
            else:
                ball_color = (0, 150, 255)  # Koyu turuncu
                glow_color = (0, 170, 255)  # Turuncu parıltı

            # Top gölgesi
            shadow_offset = 2
            cv2.circle(frame, 
                      (ball_center[0]+shadow_offset, ball_center[1]+shadow_offset), 
                      ball_radius+2, (0, 0, 0), -1, cv2.LINE_AA)

            # Top parıltısı
            cv2.circle(frame, ball_center, ball_radius+1, glow_color, -1, cv2.LINE_AA)
            
            # Ana top
            cv2.circle(frame, ball_center, ball_radius, ball_color, -1, cv2.LINE_AA)
            
            # Top vurgusu
            highlight_pos = (ball_center[0]-ball_radius//2, ball_center[1]-ball_radius//2)
            cv2.circle(frame, highlight_pos, ball_radius//3, (255, 255, 255), -1, cv2.LINE_AA)
            
            # Top kenarlığı
            cv2.circle(frame, ball_center, ball_radius, (0, 0, 0), 1, cv2.LINE_AA)

            # Top durumu (isteğe bağlı)
            if ball_status:
                status_text = f"{ball_status}"
                status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                
                # Durum arka planı
                overlay = frame.copy()
                cv2.rectangle(overlay, 
                            (ball_center[0]+10, ball_center[1]-status_size[1]-5),
                            (ball_center[0]+status_size[0]+15, ball_center[1]-5),
                            (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Durum yazısı
                cv2.putText(frame, status_text,
                           (ball_center[0]+12, ball_center[1]-7),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, ball_color, 1)

    def cleanup(self) -> None:
        if self.cap: self.cap.release()
        if self.out_main: self.out_main.release()
        if self.out_sketch: self.out_sketch.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"Hata: Video dosyası bulunamadı: {VIDEO_PATH}")
        print("Lütfen 'tennis.mp4' dosyasını kod ile aynı dizine yerleştirin veya VIDEO_PATH değişkenini güncelleyin.")
    else:
        analyzer = TennisAnalyzer(VIDEO_PATH)
        try:
            analyzer.run_analysis(process_full_video=True)
        except Exception as e:
            print(f"Analiz sırasında bir hata oluştu: {e}")
            import traceback
            traceback.print_exc()
        finally:
            analyzer.cleanup() 