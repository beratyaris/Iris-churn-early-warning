# Simülatör + Kampanya planlayıcı (+ Modele güvenebilir miyiz?) — ilerleme (2026-10-02)

## 0. Keşif
- [x] Uygulama klasörü: `C:\Users\berat\OneDrive\Desktop\bitirme-projesi` (dashboard-baslat.bat → `streamlit run app.py`).
- [x] Okundu: app.py, iris_ui.py, iris_data.py, train.py, calibrate.py, sayfalar/risk_skoru.py, maliyet.py,
  giris.py, smote_duyarlilik.py, model_karsilastirma.py, smoke_test.py, smote_karsilastirma.py/.json.
- [x] Yedek kuralı: `backup\<ad>_<tarih>_<etiket>.py` → bu iş için `_2026-10-02_simkamp`.
- [x] Yedekler alındı: backup\app_2026-10-02_simkamp.py, backup\iris_data_2026-10-02_simkamp.py,
  backup\sayfalar_risk_skoru_2026-10-02_simkamp.py, backup\smoke_test_2026-10-02_simkamp.py,
  backup\sayfalar_giris_2026-10-02_simkamp.py (giris yalnız ihtiyat; değiştirilmezse aynen kalır).
- [!] ORTAM ENGELİ: bu oturumun korumalı alanında Windows Akıllı Uygulama Denetimi
  `numba\_devicearray.cp313-win_amd64.pyd` yüklemesini engelliyor (CodeIntegrity olay 3077/3118,
  2026-10-02 17:57). `shap` → `numba` içe aktarıldığı için `iris_data`/`train` ve dolayısıyla uygulama,
  smoke_test ve ekran görüntüleri bu oturumda ÇALIŞTIRILAMIYOR. Korumalı alanı kapatma izni reddedildi.
  Dün (2026-10-01) aynı dosya engellenmemişti (olay günlüğünde yalnız bugün var).
  → Kod yazıldı; shap gerektirmeyen hesaplar (eğitim bölmesi, nadirlik, kampanya, kalibrasyon sayıları)
  ayrı betiklerle doğrulandı. Smoke test + ekran görüntüleri kullanıcı ortamında çalıştırılmalı.

## 1. Veri gerçekleri (eğitim bölmesi train.py ile birebir yeniden üretildi)
- train_test_split(test 0,20, stratify, seed 42) → (0,25, stratify, seed 42): eğitim 6.000, validation
  2.000, test 2.000. Test müşteri no sırası test_predictions.csv ile birebir; validation churn sırası
  val_predictions.csv ile birebir.
- Eğitim aralıkları: kredi skoru 350–850, yaş 18–92, kıdem 0–10, bakiye 0–238.387,56, ürün 1–4,
  maaş 90,07–199.992,48. Test setinde 1 müşterinin bakiyesi eğitim maksimumunun üstünde.
- Almanya × bakiye = 0: eğitimde 0 / 1.504 (tüm veride 0 / 2.509). Ürün sayısı 4: eğitimde 39 / 6.000.

## 2. Ortak kod (kopya yok)
- [x] YENİ `iris_karar.py` (saf pandas/numpy): nadirlik kontrolü + kampanya hesabı + maliyet varsayımları
  (DEFAULT_COST_FN=5, DEFAULT_COST_FP=1 maliyet.py'den buraya taşındı; maliyet.py artık buradan alıyor).
- [x] YENİ `iris_musteri.py`: customer_form (eğitim aralıklı), gauge_html (tek ölçek + uyarı satırı),
  shap_waterfall, score/features/profile_from_test_row. risk_skoru.py bunları kullanacak şekilde
  sadeleştirildi (aynı widget anahtarları m_*; elle giriş sınırları artık eğitim verisinden:
  önce 300–900 / 18–100 / 0–300.000 / 1–250.000 idi).
- [x] iris_data.py: train_raw() (bölme yeniden üretimi), input_bounds(), rarity_index().
- [x] maliyet.py yedeği: backup\sayfalar_maliyet_2026-10-02_simkamp.py

## 3. Nadirlik kontrolü — YÖNTEM ve EŞİK
- Her değişken okunur gruba konur: ülke, cinsiyet, ürün sayısı, aktif üye, kredi kartı (değer olarak);
  yaş 18–29/30–39/40–49/50–59/60–69/70+; kredi skoru <500/500–599/600–699/700–799/800+;
  kıdem 0/1–3/4–6/7–9/10; bakiye 0/(0–50.000]/(50–100 bin]/(100–150 bin]/>150 bin;
  maaş ≤50 bin/50–100/100–150/>150 bin.
- Eğitim setinde (6.000) sayım: tek değişken grubu < %1 (60 kişi) → "çok nadir"; 45 değişken çiftinin
  herhangi birinde ortak grup < 10 kişi → "çok nadir"; herhangi bir sayı 0 → "yok".
- Kalibrasyon: gerçek test müşterilerinin %1,65'i (33/2.000) "nadir", 0'ı "yok". Almanya×bakiye 0 → "yok";
  Fransa + 4 ürün → "nadir" (ürün=4: 39 kişi < 60; 4 ürün × maaş 50–100 bin: 4 kişi).

## 4. Sayfalar
- [x] sayfalar/simulator.py ("Model ne derdi?", Karar Destek): sabit not, kaydırıcılar (eğitim aralığı),
  test müşterisinden başlangıç yükleme + "Başlangıca dön", başlangıç→şimdi farkı ve değişen alanlar,
  nadirlik kartı, SHAP şelalesi + okuma ipucu. Para değerleri tam sayıya yuvarlanır; aralık dışı
  değer (1 test müşterisinde bakiye) sınıra çekilip not düşülür.
- [x] sayfalar/kampanya.py (Kampanya Planlayıcı, Karar Destek): veri = test seti (2.000), sıralama = model
  skoru, beklenen terk = kalibre olasılık toplamı. Varsayılanlar: temas 1 (=FP), müşteri değeri 5 (=FN),
  başarı %50 (KAYNAK YOK, etiketli), bütçe = uyarı listesine yeten (769 × 1).
  Standalone doğrulama (xgb, eşik 0,21): uyarı listesi 769 kişi, kalibre toplam 340,4, gerçekte 334;
  tüm test kalibre toplam 412,5 / gerçek 407; varsayılanlarla en iyi N = 360, net +229,6.
- [x] sayfalar/model_guven.py ("Modele güvenebilir miyiz?", Analiz, SMOTE sayfasının yanında).
  KUTULAR: 10 eşit genişlikli kutu (calibrate.py `reliability`, N_BINS=10; boş kutu çizilmez);
  ham ve kalibre eğriler test tahminlerinden aynı fonksiyonla canlı hesaplanır.
  SMOTE'suz XGB: smote_karsilastirma.json yalnız özet (ort. %20,22, Brier 0,1016) tutuyor; müşteri
  düzeyi olasılık yok → eğri değil tek nokta (genel ortalama). Eğri için yeniden eğitim gerekirdi (yapılmadı).
  Brier XGB 0,1041 → 0,1011 (calibration.json). Sıralama/uyarı listesi korunumu sayfada hesaplanıyor.
- [x] app.py: iki sayfa Karar Destek'e, bir sayfa Analiz'e (SMOTE'dan sonra). giris.py hub'ına bağlantı
  EKLENMEDİ (hub tek ekrana sığacak şekilde ayarlı; SMOTE sayfası da hub'da yok) — kullanıcıya soru.

## 5. Smoke test
- [x] smoke_test.py: iris_karar birim kontrolleri, simülatör/kampanya/güven sayfası etkileşim testleri,
  Risk Skoru elle giriş sınırı, 12 sayfa × (koyu, açık) açılış döngüsü.
- [x] py_compile: tüm değişen/yeni dosyalar OK.
- [x] (önce) ÇALIŞTIRILAMADI: `ImportError: numba._devicearray failed to import` (Akıllı Uygulama Denetimi).
- [x] 2026-10-02 noshap işinden sonra (shap/numba kaldırıldı, bkz. tasarim-taslak\noshap\ilerleme.md):
  SMOKE TEST OK, 12 sayfa × açık/koyu. Görsel düzeltmeler (şelale etiketleri, kampanya Plan denetimi,
  güven grafiği göstergesi) noshap kaydında.

## 6. Ekran görüntüleri
- [x] Alındı (noshap işinde): simulator-{acik,koyu}.png, simulator-almanya-bakiye0-{acik,koyu}.png,
  kampanya-{acik,koyu}.png, model-guven-{acik,koyu}.png (bu klasörde).

## 7. Kısayol
- [x] C:\Users\berat\OneDrive\Desktop\Iris.lnk Description: "Churn Dashboard (Streamlit)" → "Iris: churn erken uyarı".
  Hedef/ikon/çalışma klasörü geri okundu, değişmedi.
