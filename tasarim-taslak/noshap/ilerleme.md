# shap/numba'dan bağımsızlık (noshap) — ilerleme (2026-10-02)

Neden: Akıllı Uygulama Denetimi `numba\_devicearray` .pyd yüklemesini engelliyor; `import shap` → numba
zinciri uygulamayı tamamen açılmaz kılıyor. SAC kapatılmayacak. Çözüm: SHAP'i shap paketi olmadan,
önceki ayarlarla matematiksel olarak aynı yöntemle hesaplamak. Model / eşik / kalibrasyon değişmez.

## 1. Eşdeğerlik kontrolü (değişiklikten ÖNCE) — GEÇTİ
- shap kullanımı (uygulama + betikler): yalnız `iris_data.py` (import, get_explainer, explain_row,
  shap.Explanation) ve `train.py` (import, TreeExplainer xgb/rf → shap_test.joblib). Sayfalar shap'i
  doğrudan çağırmıyor; grafikler zaten Plotly (shap.plots kullanılmıyor). Global sayfalar
  `artifacts/shap_test.joblib` (train.py'nin kaydettiği değerler) okuyor.
- Ayarlar: XGB ve RF `shap.TreeExplainer(model)` arka plan YOK → tree_path_dependent, model_output raw
  (XGB log-odds, RF olasılık). LR: `LinearExplainer(model, Independent(bg200, max_samples=200))` →
  interventional; shap kaynağı (_linear.py 200, 230, 413): phi = (x − ortalama)·coef, taban = coef·ortalama + b.
- Sayısal karşılaştırma (test 2.000 satır, scaler sonrası aynı matris, aynı sütun sırası):
  - XGB pred_contribs vs shap_test.joblib: maks. |fark| = 0,0; taban (bias) farkı 0,0 (−0,70161).
    Satır toplamı vs output_margin maks. 3,8e-6 (float32).
  - RF (sklearn ağaçları → XGBoost Booster JSON, pred_contribs): maks. |fark| = 1,96e-7; taban farkı 1,3e-8.
    Toplam vs predict_proba maks. 2,5e-7.
  - LR: kayıtlı değer yok; formül shap kaynağından doğrulandı; toplam vs logit maks. 1,6e-15.
  - İlk 5 (XGB): age, products_number, active_member, gender_Male, country_Germany — eski ile aynı sıra.
- Betikler: scratchpad eq_xgb.py / eq_all.py (proje dışı).

## 2. Geçiş
- [x] Yedekler: backup\iris_data_2026-10-02_noshap.py, train_…, requirements_….txt, smoke_test_…, README_….md
- [x] YENİ iris_shap.py (Explanation veri sınıfı, booster_for, rf_to_booster, tree_contribs, linear_contribs, contribs)
- [x] iris_data.py: import shap kaldırıldı; get_explainer → get_booster (st.cache_resource), explain_row → iris_shap
- [x] train.py: import shap kaldırıldı, SHAP bölümü iris_shap.contribs ile (train.py ÇALIŞTIRILMADI, artefakt değişmedi)
- [x] requirements.txt: shap satırı yorumla çıkarıldı; .venv'den paket KALDIRILMADI. README: iris_shap satırı.
- [x] Grafikler zaten Plotly idi (shap.plots kullanılmıyordu) → yeniden çizim gerekmedi; görünür fark yok.
- [x] `-X importtime` + sys.modules: iris_data, iris_musteri, iris_shap, train, calibrate, analysis,
  smote_karsilastirma → shap/numba/llvmlite 0 modül.

## 3. Test
- [x] smoke_test.py: + SHAP bölümü (kayıtlı değerlerle fark, taban+katkı = model çıktısı, 3 model) ve
  sonda shap/numba/llvmlite sys.modules koruması. Sonuç: SMOKE TEST OK (12 sayfa × açık/koyu, ilk denemede).
- [x] streamlit headless (port 8662): /_stcore/health ok; /, /risk_skoru, /simulator, /kampanya,
  /model_guven, /aciklanabilirlik → HTTP 200. Sunucu iş sonunda durduruldu.
- [x] Ekran görüntüleri (Playwright 1.63, sistem Python, headless Edge channel=msedge, 1440 geniş, sistem
  teması prefers-color-scheme ile açık/koyu; betik scratchpad shots.py): sayfada stException / JS hatası yok.
  noshap\risk-skoru-{acik,koyu}.png, noshap\aciklanabilirlik-{acik,koyu}.png,
  simkamp\simulator-{acik,koyu}.png, simkamp\simulator-almanya-bakiye0-{acik,koyu}.png ("eğitim verisinde
  yok; tahmin güvenilir değil" görünüyor), simkamp\kampanya-{acik,koyu}.png, simkamp\model-guven-{acik,koyu}.png
- [x] CodeIntegrity günlüğü: engellenen tek dosya numba\_devicearray.cp313-win_amd64.pyd (son 19:00:20,
  geçişten önce); geçişten sonra yeni engel yok.

## 4. Bugünkü yeni sayfalarda bulunan/düzeltilen görsel hatalar (smoke test hatası çıkmadı)
- iris_musteri.shap_waterfall (Risk Skoru + Simülatör): çubuk dışı değer etiketleri sol kenarda değişken
  adlarıyla çakışıyordu ("Ortalama (başlangıç)−0,70"). Eksen aralığına iki yandan %14 pay. Yedek:
  backup\iris_musteri_2026-10-02_noshap.py
- kampanya.py: "Plan" denetiminde "Müşteri sayısı" kırpılıyordu → ilk sütun 1,05 → 1,4. Yedek: backup\sayfalar_kampanya_…
- model_guven.py: güvenilirlik grafiğinde gösterge eğrilerin üstüne biniyordu → grafiğin altına (yatay);
  y ekseninde 0 etiketi kaldırıldı (köşede x'in "0,0"ıyla çakışıyordu); yükseklik 380 → 440. Yedek: backup\sayfalar_model_guven_…
- Düzeltmelerden sonra smoke test yeniden: SMOKE TEST OK.

## 5. Açık kalanlar
- model_guven ortalama çubuğunda "%20,22" etiketi "Gerçek %20,35" kesikli çizgisine çok yakın (küçük, dokunulmadı).
- Beeswarm renk ölçeği (koyu=düşük, açık=yüksek turkuaz tonları) önceden de Plotly'ydi; değiştirilmedi.
