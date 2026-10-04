# Banka Müşteri Terki (Churn) Erken Uyarı Sistemi

İstatistik bölümü bitirme projesi. Veri: `data/Bank Customer Churn Prediction.csv` (10.000 müşteri, churn oranı %20.37).

## Kurulum (PowerShell)

```powershell
cd "$env:USERPROFILE\OneDrive\Desktop\bitirme-projesi"
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Çalıştırma

```powershell
# 1) Modelleri eğit, sonuçları artifacts/ klasörüne yaz (~1 dk)
.\.venv\Scripts\python.exe train.py

# 1b) Olasılık kalibrasyonu (validation setinde; birkaç saniye)
.\.venv\Scripts\python.exe calibrate.py

# 2) Dashboard'u aç (tarayıcıda http://localhost:8501)
.\.venv\Scripts\streamlit.exe run app.py

# İsteğe bağlı: tüm sayfaları otomatik çalıştıran duman testi
.\.venv\Scripts\python.exe smoke_test.py
```

## Kısayolla çalıştırma

Masaüstündeki **Churn Dashboard** kısayoluna çift tıklayın (hedefi proje klasöründeki `dashboard-baslat.bat`).

- Dashboard kapalıysa Streamlit'i başlatır (headless, e-posta sorusu çıkmaz), hazır olunca tarayıcıda
  http://localhost:8501 açılır. Kapatmak için konsol penceresini kapatın veya Ctrl+C.
- 8501 zaten dinleniyorsa yeni sunucu açmaz, yalnız tarayıcıyı açar.
- `.venv` yoksa hata verir; önce "Kurulum" adımlarını çalıştırın.
- İsteğe bağlı: `DASHBOARD_PORT` ortam değişkeniyle farklı port, `DASHBOARD_NO_BROWSER=1` ile tarayıcı açmadan çalıştırma.

## Yöntem özeti

- `customer_id` modele girmez. One-Hot: `country` (referans France), `gender` (referans Female).
- Yeni özellikler: `has_balance` = (balance > 0), `balance_salary_ratio` = balance / estimated_salary.
- Bölme (stratified, `random_state=42`): %60 eğitim, %20 validation (yalnız eşik seçimi), %20 test.
- Dengesizlik: imblearn Pipeline içinde SMOTE-NC (`sampling_strategy=0.5`), yalnız eğitim verisine.
  0/1 sütunlar ve ürün sayısı kategorik işaretlenir ki sentetik müşteriler 0.37 gibi geçersiz değerler almasın.
  Neden SMOTE-NC: churn oranı ~%20; düz SMOTE kesikli sütunlarda gerçekte olmayan ara değerler üretir,
  SMOTE-NC kategorikleri en yakın komşuların çoğunluk değerinden alır. `sampling_strategy=0.5` azınlığı
  çoğunluğun yarısına çıkarır (tam dengeleme yerine daha az sentetik veri). Pipeline içinde olduğu için
  validation/test verisine hiç uygulanmaz (sızıntı yok). Yan etkisi: model olasılıkları yukarı kayar
  (kalibre değildir); bu yüzden eşik validation setinde ayrıca seçilir.
- Modeller: Lojistik Regresyon (referans), Random Forest, XGBoost. Tüm pipeline'larda StandardScaler var
  (SMOTE'un komşuluk mesafesi için; ağaçlar ölçekten etkilenmez).
- Eşik: validation setinde F2'yi maksimize eden eşik; test metrikleri 0.50 ve bu eşikle raporlanır.
- McNemar testi (3 ikili karşılaştırma) + Holm düzeltmesi. SHAP: TreeExplainer (RF, XGBoost).

## Olasılık kalibrasyonu (`calibrate.py`)

SMOTE-NC eğitimde churn payını yapay olarak artırdığı için model skorları yukarı kayar. `calibrate.py`
(train.py'den sonra, isteğe bağlı; `train.py`'yi ve çıktılarını değiştirmez):

- Test setinde her model için ortalama skor vs gerçek churn oranı, Brier skoru ve 10 kutulu güvenilirlik eğrisi.
- Fark 3 puandan büyükse kalibratör **yalnız validation setinde** eğitilir. Isotonic ve Platt (sigmoid) validation
  içinde 5 katlı CV Brier skoruyla karşılaştırılır (kalibratörü eğitildiği veride değerlendirmek isotonic'i hep
  kazandırır), düşük olan tüm validation ile yeniden eğitilir. Test yalnız raporlama içindir.
- Sonuç (2026-10-01): üç modelde de fark > 3 puan, üçünde de Platt seçildi.

| Model | Ort. skor (ham) | Gerçek (test) | Ort. olasılık (kalibre) | Brier ham → kalibre |
|---|---|---|---|---|
| Lojistik Regresyon | %29,95 | %20,35 | %20,20 | 0,1465 → 0,1352 |
| Random Forest | %27,10 | %20,35 | %20,37 | 0,1111 → 0,1039 |
| XGBoost | %24,43 | %20,35 | %20,63 | 0,1041 → 0,1011 |

- Çıktılar: `artifacts/calibration.joblib` (kalibratörler), `artifacts/calibration.json` (özet + eğriler).
- **Kullanım kuralı:** karar eşikleri kalibre edilmemiş skorlar üzerinde seçildiği için uyarı listesi ve
  sınıflandırma her zaman *orijinal skor ≥ orijinal eşik* ile yapılır. Kalibre olasılık yalnız "Beklenen churn
  oranı" ve ekranda gösterilen müşteri olasılığı içindir. Platt monoton olduğundan eşiğin kalibre ölçekteki
  karşılığı aynı müşterileri seçer (gösterge üzerindeki eşik işareti bu karşılıktır).
- Dosya yoksa uygulama ham skorlarla çalışır ve bunu belirtir.

## Dashboard (Iris, çok sayfalı)

Giriş noktası yine `app.py` (kısayol/bat değişmedi). Sayfalar `sayfalar/` klasöründe, tümü Plotly ve
`iris_ui.py` tasarım sistemiyle:

- **Giriş** (varsayılan, 1600×900'e kaydırmasız sığar): **Beklenen churn oranı** göstergesi = seçili modelin test
  setindeki (2.000) kalibre olasılıklarının ortalaması (eşikten bağımsız). Gösterge ölçeği 0 – 2 × tarihsel oran
  (tarihsel oran tepede); bantlar: tarihsel oran p₀ için n = 2.000 kişilik grupta %95 aralık
  p₀ ± 1,96·√(p₀(1−p₀)/n) (skor testi / Wilson kabul bölgesi) amber, altı turkuaz, üstü mercan. Ayrı kart:
  **Uyarı listesi** = eşik üstü skorlu test müşterisi sayısı (ör. "769 müşteri, eşik 0,21").
- **Karar Destek:**
  - Genel Bakış: filtreler, KPI kartları (açıklamalar ⓘ ipucunda; fark rozeti yalnız filtre varken), churn kırılımları.
    Çubuk rengi portföy ortalamasına göre: grubun %95 Wilson GA'sı ortalamanın tamamen altında → turkuaz,
    ortalamayı içeriyor → amber (ayırt edilemez), tamamen üstünde → mercan; GA'lar bıyık olarak çizilir.
  - Müşteri Risk Skoru: müşteri no ile arama, uyarı listesinin en yüksek skorlu müşterileri için hızlı seçim,
    elle profil girişi; müşteri göstergesi (kalibre olasılık + eşik işareti, uyarı listesi durumu), müşteri
    bilgileri ve Türkçe etiketli SHAP şelale grafiği (mercan artırır, turkuaz azaltır).
  - Maliyet ve Eşik: FN/FP maliyeti (varsayılan 5:1, göreli tamsayı birim), validation (düz) ve test (noktalı)
    maliyet eğrisi, seçili ve maliyet-optimal eşik işaretleri, kalibre olasılıkla Bayes eşiğinin skor karşılığı,
    seçili eşikte recall/precision/uyarı sayısı/maliyet, **Bu eşiği uygula** düğmesi (ortak eşiği değiştirir),
    tüm modeller × üç eşik tablosu.
- **Analiz:** Model Karşılaştırma (metrik tablosu, ROC/PR, McNemar + Holm ve Türkçe yorum, **Kalibrasyon**
  bölümü) · Açıklanabilirlik (global önem, beeswarm, bağımlılık grafiği) · Confounding (tabakalı oranlar ve
  Wilson GA, çapraz tablo, kutu grafikleri, OR orman grafiği, otomatik yorum kutusu).

Model ve karar eşiği kenar çubuğunda ortaktır (eşik kaydırıcısı Türkçe biçimde "0,21" gösterir).

**Tema:** `.streamlit/config.toml` içinde `[theme.light]` ve `[theme.dark]` tanımlı; varsayılan sistem temasıdır.
Kullanıcı **Tema: Sistem / Açık / Koyu** denetimiyle değiştirir (iç sayfalarda kenar çubuğunun en altı, giriş
sayfasında lacivert panelin sağ üstü; ayrıca sağ üstteki ⋮ menüsü). Streamlit 1.64 yerleşik temayı Python'dan
oturum bazında değiştiremediği için seçim Streamlit'in kendi tarayıcı anahtarına (`stActiveTheme-…-v2`) yazılır
ve sayfa bir kez yenilenir; böylece yerleşik widget'lar da değişir. Seçim tarayıcıda hatırlanır (`iris_tema`
çerezi); model/eşik ayarı yenilemede korunur, sayfa içi filtreler sıfırlanır. `iris_ui.py` etkin temayı `st.context.theme.type` ile okur;
CSS değişkenleri, Plotly şablonu (`iris_light` / `iris_dark`), gösterge ve logo (`logo/final/*-dark|light`) buna
göre seçilir. Animasyonlar yalnız ilk yüklemede ve `prefers-reduced-motion` açıkken kapalı.

Not: Bir alt sayfa adresi (ör. `/maliyet`) doğrudan açıldığında tarayıcı konsolunda 2 adet 404
(`/<sayfa>/_stcore/health`, `/_stcore/host-config`) görünür; Streamlit ön yüzünün taban yolu yoklamasıdır,
ardından kök adrese düşer ve uygulamayı etkilemez. Uygulama içi gezinmede ve ana sayfada görünmez.

Eski arayüzler: `backup/app_2026-10-01_oncesi.py` (tek dosya), `backup/*_2026-10-01_faz2oncesi.*` (faz 1).
Ekran görüntüleri: `tasarim-taslak/faz2/`.

### Maliyet Analizi (maliyete duyarlı eşik)

- Toplam maliyet = FN maliyeti × kaçırılan churn + FP maliyeti × yanlış alarm. Varsayılan FN=5, FP=1
  **varsayımsal** birimdir (kaçırılan müşteri kampanya maliyetinden birkaç kat pahalı kabul edildi); arayüzden değiştirilir.
- Eşik 0.01–0.99 aralığında **validation** tahminleri (`val_predictions.csv`) üzerinde maliyeti minimize edecek
  şekilde seçilir; test seti yalnız raporlama içindir. Maliyet-optimal, F2-optimal ve 0.50 eşikleri test setinde
  (Recall, Precision, FN, FP, toplam maliyet) karşılaştırılır.
- Teorik referans: Bayes eşiği FP/(FP+FN). Kalibre olasılık varsayar; SMOTE-NC olasılıkları kaydırdığı için
  ampirik optimum bu değerden farklı çıkabilir.

### Özgünlük: Confounding Kontrolü

Soru: Almanya'daki yüksek churn ülkeden mi, bakiye yapısından mı? (Veride bakiyesi 0 olan Alman müşteri yok.)
Tam veri (10.000) üzerinde, model tahmini kullanmadan (`analysis.py`):

- Ülke × has_balance çapraz tablosu (sayı, satır %) + ki-kare bağımsızlık testi.
- Tabakalı churn oranları: tümü / balance=0 / balance>0.
- balance>0 içinde bakiye ve bakiye/maaş oranı dağılımı (medyan, IQR, kutu grafiği).
- statsmodels Logit: (a) yalnız ülke dummy'leri (ref. France); (b) ülke + has_balance + merkezlenmiş log(bakiye)
  + yaş + cinsiyet + aktif üyelik + ürün sayısı (1/2/3+; 4 ürünlü müşterilerin hepsi churn olduğundan tam ayrışmayı
  önlemek için 3 ve 4 birleştirildi). Germany OR, %95 Wald GA, p ve kontrol sonrası % değişim gösterilir.
  Birincil hipotez tek katsayı olduğundan çoklu karşılaştırma düzeltmesi yapılmadı.
- Yorum metni sonuçlardan otomatik üretilir. Bulgular ilişkiseldir, nedensellik iddiası değildir; OR'nin
  çökertilemezliği (non-collapsibility) nedeniyle kontrol eklemek confounding olmadan da OR'yi değiştirebilir.

Not: `analysis.py`, `statsmodels.api` yerine alt modülleri içe aktarır; `api` Windows Akıllı Uygulama Denetimi'nin
engellediği bir zaman serisi DLL'ini yüklüyor.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `train.py` | Veri hazırlama, eğitim, eşik, metrikler, McNemar, SHAP; çıktıları `artifacts/`e yazar |
| `app.py` | Streamlit giriş noktası: sayfalar, tema, kenar çubuğu (eğitim yapmaz) |
| `calibrate.py` | Kalibrasyon kontrolü + validation'da kalibratör; `artifacts/calibration.*` yazar |
| `sayfalar/` | Sayfa dosyaları (giriş + 6 iç sayfa) |
| `iris_ui.py` | Iris tasarım sistemi: açık/koyu tema, CSS, Plotly şablonu, KPI kartı, tablo, gösterge |
| `iris_data.py` | Veri/artefakt yükleme, kalibrasyon, Wilson GA, beklenen churn, uyarı listesi, ortak model/eşik |
| `analysis.py` | Confounding analizi ve maliyete duyarlı eşik fonksiyonları (test edilebilir) |
| `smoke_test.py` | Analiz fonksiyonlarını ve tüm sayfaların hatasız çalıştığını kontrol eder |
| `artifacts/` | `models.joblib`, `metrics.csv`, `mcnemar.csv`, `thresholds.json`, `test_predictions.csv`, `val_predictions.csv`, `shap_test.joblib`, `background.joblib`, `calibration.joblib`, `calibration.json` |
