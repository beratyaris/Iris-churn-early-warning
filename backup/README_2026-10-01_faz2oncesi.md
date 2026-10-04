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

## Dashboard (Iris, çok sayfalı)

Giriş noktası yine `app.py` (kısayol/bat değişmedi). Sayfalar `sayfalar/` klasöründe:

- **Giriş** (varsayılan): portföy risk göstergesi + sayfa düğmeleri. Gösterge = seçili modelin karar
  eşiğiyle test setinde (2.000 müşteri) uyarı verilen müşteri payı; bantlar <%33 düşük, %33–66 orta, üstü yüksek.
- **Karar Destek:** Genel Bakış (filtreler, KPI kartları, churn kırılımları) · Müşteri Risk Skoru · Maliyet ve Eşik
- **Analiz:** Model Karşılaştırma · Açıklanabilirlik (SHAP) · Confounding

Model ve karar eşiği kenar çubuğunda ortaktır. Tema: `.streamlit/config.toml`; renkler, CSS, Plotly şablonu
ve bileşenler `iris_ui.py`'de. Eski tek dosyalık arayüz: `backup/app_2026-10-01_oncesi.py`.

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
| `sayfalar/` | Sayfa dosyaları (giriş, genel bakış, eski sekmeler) |
| `iris_ui.py` | Iris tasarım sistemi: renkler, CSS, Plotly şablonu, KPI kartı, gösterge |
| `iris_data.py` | Veri/artefakt yükleme, ortak model/eşik ayarı, portföy riski |
| `iris_legacy.py` | Faz 1'de taşınan eski sekme içerikleri |
| `analysis.py` | Confounding analizi ve maliyete duyarlı eşik fonksiyonları (test edilebilir) |
| `smoke_test.py` | Analiz fonksiyonlarını ve tüm sayfaların hatasız çalıştığını kontrol eder |
| `artifacts/` | `models.joblib`, `metrics.csv`, `mcnemar.csv`, `thresholds.json`, `test_predictions.csv`, `val_predictions.csv`, `shap_test.joblib`, `background.joblib` |
