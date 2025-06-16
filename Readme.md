# Tenis Maçı Analiz Aracı

Bu Python programı, tenis maç videolarını izleyip analiz eder. Kortu, oyuncuları ve topu bularak hareketlerini takip eder. Sonuçları hem videonun üzerinde hem de kuşbakışı bir kort çiziminde gösterir.

## Ne Yapar?

*   **Kortu Bulur**: Videodaki tenis kortunu otomatik olarak tanır.
*   **Oyuncuları Tanır**: Oyuncuları bulur ve P1, P2 olarak etiketler.
*   **Topu İzler**: Topu tespit eder ve hareketlerini takip eder.
*   **Kuşbakışı Gösterir**: Oyuncu ve top konumlarını 2D bir kort çiziminde gösterir.
*   **Tekrarları Algılar**: Videodaki tekrar sahnelerini fark edip analizi duraklatır.
*   **Video Kaydeder**: Analiz edilmiş videoyu ve kuşbakışı çizimi ayrı video dosyaları olarak kaydeder.

## Kullanılan Yöntemler

Bu projede, tenis maçı analizi için aşağıdaki klasik bilgisayar görü ve görüntü işleme teknikleri kullanılmıştır:

*   **Renk Segmentasyonu (HSV Tabanlı)**: Kort alanını (özellikle mavi rengi) videoda belirginleştirmek ve maskelemek için HSV renk uzayı kullanılmıştır.
*   **Kontur Analizi**:
    *   Kortun köşelerini, oyuncu ve top adaylarını tespit etmek için `cv2.findContours` ve ilgili fonksiyonlar (`cv2.contourArea`, `cv2.boundingRect`, `cv2.approxPolyDP`, `cv2.convexHull`) kullanılmıştır.
    *   Tespit edilen konturların alanı, en-boy oranı, doluluk (solidity), dairesellik gibi geometrik özellikleri, nesneleri sınıflandırmak ve filtrelemek için değerlendirilmiştir.
*   **Arka Plan Çıkarma (MOG2)**: Videodaki hareketli nesneleri (oyuncular ve top) statik arka plandan ayırmak için `cv2.createBackgroundSubtractorMOG2` algoritması kullanılmıştır.
*   **Homografi ile Perspektif Dönüşümü**: Kortun orijinal videodaki perspektif görünümünü, 2D bir kuşbakışı haritaya dönüştürmek için homografi matrisi hesaplanmış ve `cv2.perspectiveTransform` uygulanmıştır.
*   **Optik Akış (Lucas-Kanade)**: Topun kareler arası hareketini takip etmek ve kısa süreli tespit kayıplarında konumunu tahmin etmek için `cv2.calcOpticalFlowPyrLK` yöntemi kullanılmıştır.
*   **Kalman Filtresi**: Topun hareketini modellemek, ölçümlerdeki gürültüyü filtrelemek ve daha yumuşak bir yörünge elde etmek amacıyla top takibinde kullanılmıştır.
*   **Morfolojik İşlemler**: Gürültüyü azaltmak, nesne sınırlarını düzeltmek ve istenmeyen küçük parçaları temizlemek için `cv2.morphologyEx` (açma ve kapama gibi) operasyonları uygulanmıştır.
*   **Kural Tabanlı Filtreleme ve Skorlama**:
    *   Tespit edilen oyuncu ve top adaylarının geçerliliğini doğrulamak için belirli alan, boyut, en-boy oranı aralıkları gibi kurallar tanımlanmıştır.
    *   Top adayları için, oyunculara yakınlık, kort çizgilerine yakınlık, yasak bölgeler gibi faktörlere dayalı bir güven skoru hesaplanarak en olası top tespiti seçilmiştir.

## Gerekenler

*   Python 3
*   OpenCV (`cv2`)
*   NumPy (`numpy`)

## Kurulum

Gerekli programları yüklemek için:
```bash
pip install opencv-python numpy
```

## Nasıl Kullanılır?

1.  `tennis_analyse.py` dosyasındaki `VIDEO_PATH` değişkenini analiz etmek istediğiniz videonun (`.mp4`) adıyla güncelleyin. (Varsayılan: "tennis.mp4")
2.  Programı çalıştırın:
    ```bash
    python tennis_analyse.py
    ```

Analiz bitince, `output/` klasöründe iki video dosyası oluşur:
*   `full_analyzed.mp4`: Üzerinde analizlerin olduğu ana video.
*   `full_sketch.mp4`: Kuşbakışı kort çizimi videosu.

Analiz sırasında canlı önizlemeyi görebilirsiniz. `q` tuşuna basarak işlemi durdurabilirsiniz.