# Beeswarm renkleri + giriş bağlantıları (girisrenk) — ilerleme (2026-10-02)

## 0. Yedekler
- backup\sayfalar_giris_2026-10-02_girisrenk.py, backup\sayfalar_aciklanabilirlik_2026-10-02_girisrenk.py,
  backup\iris_ui_2026-10-02_girisrenk.py, backup\smoke_test_2026-10-02_girisrenk.py

## 1. Beeswarm renk ölçeği (sayfalar\aciklanabilirlik.py)
- [x] Ölçek: turkuaz #14B8A6 (düşük) → amber #F59E0B (orta) → mercan #FF7A59 (yüksek); iki temada aynı
  (eski: temaya göre tek tonlu turkuaz/lacivert). Renk çubuğu "Düşük / Yüksek" etiketleri korundu.
- [x] 0/1 değişkenler (BINARY) yalnız iki uç: Hayır = turkuaz, Evet = mercan (önce sıra yüzdeliğinden ara ton alıyordu).
- [x] Kart açıklaması yeni renk okumasına göre güncellendi. Diğer grafiklere dokunulmadı.

## 2. Giriş bağlantıları (sayfalar\giris.py + iris_ui._LANDING_CSS)
- [x] Karar Destek: … Maliyet ve Eşik, Model ne derdi?, Kampanya Planlayıcı (app.py menü sırası).
  Analiz: Model Karşılaştırma, Modele güvenebilir miyiz?, Açıklanabilirlik (SHAP), Confounding.
- [x] Sığdırma (ölçülerek): panel aralığı 16→10 px, üst/alt dolgu 30→22 px, grup bağlantıları 50→38 px
  (simge 38→30, min-height/line-height sıfırlandı). Streamlit bağlantı kapsayıcısına −6 px boşluk verdiği için
  grup içi gap 12 px = 4 px görünür aralık. Sunum Modu 50 px, gradyanlı, değişmedi.
- [x] Kısa ekran kademesi `@media (min-width:901px) and (max-height:820px)`: dolgu 14 px, logo 32 px,
  başlık 1,6rem, istatistik kartları sıkı, grup bağlantıları 32 px.
- Ölçüm (Playwright + headless Edge, açık ve koyu aynı):
  | Ekran | önce (bağlantı eklenmeden) | sonra |
  |---|---|---|
  | 1440×900 | sığıyor (içerik sonu 868) | sığıyor (içerik sonu 859, kaydırma yok) |
  | 1366×768 | TAŞIYORDU (914 px) | sığıyor (içerik sonu 738, kaydırma yok) |
  | 1920×1080 | sığıyor | sığıyor (içerik sonu 938) |

## 3. Test
- [x] smoke_test.py: + girişte 10 bağlantı (3 yeni dahil) ve beeswarm açıklaması kontrolü → SMOKE TEST OK.
- [x] Ekran görüntüleri bu klasörde: giris-{1440x900,1366x768,1920x1080}-{acik,koyu}.png,
  aciklanabilirlik-{acik,koyu}.png, beeswarm-{acik,koyu}.png. Sayfada stException / JS hatası yok.

## 4. Not (dokunulmadı)
- Açıklanabilirlik tarayıcıda İLK açılan sayfaysa (doğrudan URL, boş önbellek) Plotly etiketleri Inter fontu
  yüklenmeden ölçüyor; "Müşteri süresi (yıl)" etiketinin M'si 11 px kırpılıyor (Global önem + beeswarm).
  Önce başka sayfa açılınca yok. Önceden de vardı; "başka grafiğe dokunma" nedeniyle düzeltilmedi.
