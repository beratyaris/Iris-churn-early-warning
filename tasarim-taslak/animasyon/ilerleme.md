# Sayfa animasyonları ("dümdüz gelmesin", abartmadan) — ilerleme

## 1. Denetim (değişiklikten önce) — TAMAM

Mevcut animasyonlar (iris_ui.py):
- `_BASE_CSS`: `@keyframes iris-fadeup` (opacity 0→1, translateY 6px→0), `iris-grow`, `iris-fade`.
  - `.iris-head` .5s, kartlar `[class*="st-key-card-"]`/`.st-key-filterbar` .55s, `.iris-kpi` .5s fadeup.
    Hepsi AYNI ANDA başlıyor (gecikme/sıra yok).
  - Plotly: `.barlayer .point path` scaleY büyüme .9s; çubuk metni/shape/annotation .55s gecikmeli fade.
  - `prefers-reduced-motion: reduce` -> tüm animation/transition kapalı. (var)
- `_COUNTUP_JS` + `count()`: HTML'de son değer yazılı (JS yoksa doğru), JS 0'dan sayıyor
  (`toLocaleString('tr-TR')`, ease-out cubic, varsayılan 1100 ms; girişte 1300/1500 ms).
  `kpi_row` ve `html(js=True)` ile ekleniyor. Reduced-motion'da sayma yok.
- `_SVG_ANIM` (sparkline çizgi çizimi 1.2s, minibars çubuk büyümesi .8s, nokta fade).
- `gauge_svg`: ibre -90°→değer, 1.8s .3s gecikme, `cubic-bezier(.3,1.35,.45,1)` (fazla salınım/overshoot = "abartılı"
  sınırında). Yay dolumu yok.
- sunum_modu.py: `.iris-answer` fadeup .5s .15s; `.iris-step` transition.
- giris (_LANDING_CSS): yalnız hover transition'ları; hero/hub için giriş animasyonu yok (gösterge + sayma var).

Sorunlar:
1. Kademeli (staggered) giriş yok; her şey aynı anda.
2. YENİDEN ÇALIŞTIRMADA TEKRAR: animasyonlar koşulsuz. Eşik kaydırıcısı değişince içeriği değişen
   st.html öğeleri (KPI kartları, gösterge+sayı bloğu) yeniden oluşturuluyor -> fadeup + sayma + ibre
   baştan oynuyor; Plotly yeniden çizince çubuk büyümesi tekrar oynuyor.
3. İbre süresi/eğrisi istenen ölçünün dışında (1.8s + overshoot).
4. Plotly çizgi/alan grafikleri (ROC, PR, maliyet eğrisi) için giriş efekti yok (yalnız çubuklar).

## 2. Plan / durum
- [x] Yedekler: backup\iris_ui_, app_, sayfalar_sunum_modu_, sayfalar_giris_ (…_2026-10-01_animasyon.py)
- [x] "Önce" ölçümü (once-*.png, kare1-3 = 0,15/0,45/0,9 sn): kaydırıcı sonrası genel_bakis'te 23,
      maliyet'te 11-14 animasyon yeniden çalıştı, sayılar 0'dan tekrar saydı (MutationObserver), SVG
      (gösterge/mini grafik) yine animasyonluydu. Sorun doğrulandı.
- [x] iris_ui: `begin_page(key)` + `animate()`; app.py `begin_page(page.url_path)`
- [x] `_anim_css(landing)` -> `__ANIM__` yer tutucusu; yalnız animate() iken (tek st.html, içerik değişir)
- [x] Kademeli giriş: `--iris-i` (üst düzey sıra), `--iris-c` (sütun), `--iris-k` (KPI) × 70 ms; .42s, 10px
- [x] Plotly: çubuk büyüme .7s, scatter katmanı soldan sağa açılma .8s, metin .4s; kart gecikmesine bağlı
- [x] Sayma: betik yalnız animate(); 800 ms; kartın animation-delay'i bitince başlar; son metin aynen geri yazılır
- [x] Gösterge: ibre .9s ease-out salınımsız; renkli yay maskeyle .75s süpürme; anim=False -> statik
- [x] sparkline/minibars: `_svg_anim()` yalnız animate()
- [x] giris.py: sayma süreleri 1500/1300 -> 900/800 (yalnız süre)
- [x] Doğrulama
  - py_compile (app, iris_ui, giris) OK; smoke_test.py -> SMOKE TEST OK
  - Headless (port 8632, Edge, 1440x900) açık+koyu, 6 sayfa: sayfa hatası yok. Tek konsol hatası
    Streamlit'in alt sayfa doğrudan URL'sinde `/<sayfa>/_stcore/health|host-config` 404'ü (önceden de var, bizden değil).
  - Kaydırıcı sonrası: iris-* animasyonu 0 (çalışanlar yalnız Streamlit'in kendi opacity/skeleton geçişleri),
    sayıların 0'dan tekrar sayması 0, gösterge SVG'si animasyonsuz; base CSS 12.221 -> 8.920 karakter (animasyon kuralları kalktı).
  - Uygulama içi sayfa geçişi (giriş -> Genel Bakış -> Maliyet -> Genel Bakış): her seferinde 10-20 iris-*
    animasyonu çalıştı. Maliyet'te kaydırıcı 3 kez, Genel Bakış'ta filtre tıklaması: 0.
  - reduced-motion: iris-* animasyonu 0, sayılar doğrudan son değer.
  - Giriş sağ panel yüksekliği 900 px (açık/koyu) — taşma yok; son kareler önceki yerleşimle aynı.
  - Son değerler birebir: "%82,1 | %43,4 | 769 | 800", "10.000 | %20,37 | 2.037 | 769".
  - Görseller: once-*/sonra-* (kare1-3 = açılışın ~0,15/0,45/0,9 sn'si), gosterge-kare1-3 (yay süpürme + ibre,
    sağdaki statik sürümle son kare aynı), *-kaydirici-sonrasi. Sunucu durduruldu.
- Not: sunum_modu.py değişmedi (yedeği silindi). Adım değişince yalnız sayfanın kendi `.iris-answer` belirmesi
  oynar (eskisi gibi); kademeli giriş yalnız sayfa açılışında.
- İŞ TAMAMLANDI.
