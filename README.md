# Tenis Video Analizi - OpenCV ile Oyuncu ve Top Tespiti

Bu proje, tenis videolarında geleneksel bilgisayar görüşü tekniklerini kullanarak oyuncu ve top tespiti yapar ve gerçek zamanlı 2D kuşbakışı görüntü oluşturur.

## Özellikler

- **Tenis kortu tespiti**: Yeşil alan segmentasyonu ve köşe noktalarını bulma
- **Oyuncu tespiti**: Renk bazlı kontur analizi ile maksimum 2 oyuncu tespiti
- **Top tespiti**: Beyaz renk filtresi ve daire benzerlği analizi ile küçük top tespiti
- **Perspektif dönüşümü**: 3D video görüntüsünden 2D kuşbakışı görüntüye dönüştürme
- **Gerçek zamanlı görüntüleme**: İki pencerede eş zamanlı video ve 2D görüntü

## Teknolojiler

- **OpenCV**: Bilgisayar görüşü işlemleri
- **NumPy**: Sayısal hesaplamalar
- **Python**: Ana programlama dili

## Kurulum

```bash
pip install opencv-python numpy matplotlib
```

## Kullanım

1. Video dosyanızı `data/tennis.mp4` olarak yerleştirin
2. Programı çalıştırın:

```bash
python analyzer.py
```

3. İki pencere açılacak:
   - **Original Video**: Tespit edilen oyuncu ve topların işaretlendiği orijinal video
   - **Court 2D View**: Gerçek renklerde ve ölçekli 2D kuşbakışı tenis kortu

## Kontroller

- **ESC** veya **Q**: Programdan çıkış

## Algoritma Detayları

### 1. Tenis Kortu Tespiti
- HSV renk uzayında yeşil renk segmentasyonu
- Morfolojik işlemlerle gürültü temizleme
- Kontur analizi ve en büyük yeşil alanı bulma
- Minimum bounding rectangle ile köşe noktalarını belirleme

### 2. Oyuncu Tespiti
- Çoklu renk aralıklarında (mavi, kırmızı, sarı) arama
- Kontur analizi ve alan filtresi (500-5000 piksel)
- Yakın pozisyonlardaki tespitleri birleştirme
- En büyük 2 alanı oyuncu olarak seçme

### 3. Top Tespiti
- Beyaz renk filtresi (HSV: [0,0,200] - [180,30,255])
- Küçük alan filtresi (5-200 piksel)
- Daire benzerlği analizi (circularity > 0.3)
- En yüksek skora sahip nesneyi top olarak seçme

### 4. Perspektif Dönüşümü
- Kort köşe noktalarından 2D koordinatlara perspektif matrisi hesaplama
- Oyuncu ve top pozisyonlarını 2D korta dönüştürme
- Gerçek kort ölçülerinde (23.77m x 10.97m) görüntüleme

## Dosya Yapısı

```
├── analyzer.py          # Ana analiz kodu
├── README.md           # Bu dosya
├── data/
│   └── tennis.mp4      # Giriş video dosyası
└── output/
    └── output.mp4      # Çıkış video dosyası (isteğe bağlı)
```

## Performans Optimizasyonu

- Giriş videosu 1200 piksel genişliğe yeniden boyutlandırılır
- Morfolojik işlemler optimize edilmiştir
- Kontur analizi alanlarla filtrelenmiştir

## Sınırlamalar

- Açık havada, yeşil zeminde oynanan tenis maçları için optimize edilmiştir
- Kötü aydınlatma koşullarında performans düşebilir
- Oyuncu kıyafetlerinin belirli renklerde olması gerekir
- Top beyaz olmalıdır

## Geliştirme Önerileri

- Tracking algoritmaları eklenebilir
- Kalman filtresi ile tahmin doğruluğu artırılabilir
- Farklı kort türleri için renk kalibrasyonu eklenebilir
- Video kaydetme özelliği eklenebilir