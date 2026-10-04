# Sunum Modu adım animasyonu + masaüstü kısayolu — ilerleme (2026-10-01)

## Görev 1 — Sunum Modu adım geçişleri
- [x] Okundu: iris_ui.py (begin_page / animate / _anim_css / _COUNTUP_JS), sunum_modu.py, app.py.
  Sorun: adım değişince begin_page aynı sayfayı görüyor -> animate() False -> animasyon CSS'i/sayma
  betiği basılmıyor; yalnız sayfanın kendi koşulsuz `.iris-answer` fadeup'ı oynuyor.
- [x] Yedekler: backup\iris_ui_2026-10-01_sunumanim.py, backup\sayfalar_sunum_modu_2026-10-01_sunumanim.py
- [x] DOM (headless): `.st-key-card-sunum-N` = stVerticalBlock; çocukları soru öğesi + stLayoutWrapper >
  stHorizontalBlock > 2 stColumn (ana sütunla aynı yapı) -> mevcut _anim_css kökü parametreleştirilebilir.
- [x] iris_ui.py (mevcut mekanizma genişletildi, paralel sistem yok):
  - `replay_entrance(scope)`: geri çağırmada bir kerelik istek bırakır.
  - `begin_page(key)`: isteği tüketir; sayfa değiştiyse tam giriş, aynı sayfada istek varsa
    `_iris_anim=True` + `_iris_anim_scope=scope`. Başka yeniden çalıştırmada istek yok -> animasyon yok.
  - `_anim_css(landing, scope)`: scope verilirse aynı kademeli kurallar (--iris-i/--iris-c/--iris-k × 70 ms,
    .42s, 10px) yalnız kapsayıcının içine; Plotly kuralları da kapsama önekli. `inject_base_css` kapsamı geçirir.
  - Kapsam adıma özgü (`.st-key-card-sunum-<adım>`): kural metni her adımda değişir, öğe yerinde
    güncellense de animasyon yeniden başlar.
- [x] sunum_modu.py: `go_to` adım gerçekten değişince `ui.replay_entrance(...)`; kısa cevap + sayılar
  `ui.html(..., js=True)` (sayma betiği yalnız animate() iken); `countable()` yalnız sayıdan oluşan
  olgu değerlerini `ui.count` ile sayan sayıya çevirir (son metin birebir aynı olduğu doğrulanıyor, değilse
  metin aynen kalır). `.iris-answer`'ın koşulsuz fadeup'ı kaldırıldı (kademeli giriş kapsıyor). İçerik/sayı değişmedi.
- [x] Doğrulama
  - py_compile (iris_ui, sunum_modu, app) OK; smoke_test.py -> "Sunum modu OK (7 adım)", SMOKE TEST OK.
  - Headless Edge (port 8641, CDP, animationstart + MutationObserver sayacı):
    - Sonraki ×4 + Önceki ×1 (adım 1→5→4) + sonra 5→6→7→6: her adımda kartta soru + 2 sütun iris-rise,
      Plotly grow/reveal + fade (9–19 animasyon), olgu sayıları 0'dan sayıyor. Başlık ve adım çipleri: 0.
    - Sunum'da kenar çubuğu eşik kaydırıcısı ×2: 0 animasyon, 0 sayma.
    - Genel Bakış'a uygulama içi geçiş: 127 animasyon (tam giriş); eşik kaydırıcısı ×3 (0,21→0,24): 0, 0.
    - Sunum'a geri dönüş: tam sayfa girişi (yine yalnız açılışta).
    - prefers-reduced-motion: reduce + Sonraki: 0 animasyon, sayılar doğrudan son değer.
    - Son değerler: "%82,1 | %43,4 | 769 / 2.000", "%24,4 | %20,6 | %20,35", "5 : 1 | 800 | 73 · 435",
      "2,49 | 2,67 | 2,31; 3,09", "%82,1 | %80,1 | 0,73".
  - Görseller: adim-gecis-kare1/2.png (~0,25/0,5 sn: grafik açılıyor, sayılar 0,00'dan), adim-gecis-son.png.
  - Sunucu ve headless Edge durduruldu.

## Görev 2 — Masaüstü kısayolu
- [x] Arandı: C:\Users\berat\OneDrive\Desktop (= Windows'un Desktop klasörü), C:\Users\berat\Desktop (yok),
  C:\Users\Public\Desktop. Tek eşleşme bulundu.
- [x] DEĞİŞİKLİKTEN ÖNCE (WScript.Shell ile okundu):
  - Yol: `C:\Users\berat\OneDrive\Desktop\Churn Dashboard.lnk` (1861 bayt, 30.09.2026 00:47)
  - TargetPath: `C:\Users\berat\OneDrive\Desktop\bitirme-projesi\dashboard-baslat.bat`
  - Arguments: (boş)
  - WorkingDirectory: `C:\Users\berat\OneDrive\Desktop\bitirme-projesi`
  - IconLocation: `C:\Windows\System32\imageres.dll,144`
  - WindowStyle: 1 (normal pencere); Hotkey: (yok); Description: `Churn Dashboard (Streamlit)`
- [x] Simge: `logo\final\favicon.ico` yalnız 16/32/48 içeriyor (masaüstü için küçük) ->
  `logo\final\iris-app.ico` üretildi (Pillow, venv): 256/128/64 = icon-512.png'den LANCZOS küçültme,
  48/32/16 = mevcut favicon-48/32/16.png (favicon.ico ile aynı kaynak). 35.328 bayt; yeniden okununca
  6 boyut var, her boyut kaynağıyla piksel farkı 0. (Betik oturum geçici klasöründe, projeye eklenmedi.)
- [x] `C:\Users\berat\OneDrive\Desktop\Iris.lnk` WScript.Shell ile oluşturuldu: TargetPath, Arguments (boş),
  WorkingDirectory, WindowStyle (1), Description eskisinden kopyalandı; IconLocation =
  `...\bitirme-projesi\logo\final\iris-app.ico,0`. Geri okunup doğrulandı (hepsi eşit, hedef ve .ico var).
- [x] Doğrulamadan SONRA eski `Churn Dashboard.lnk` silindi. Diğer kısayollara dokunulmadı
  (masaüstünde Cursor, Discord, MetaTFT, MSI Afterburner, Obsidian, Opera GX aynen duruyor).
- [x] `ie4uinit.exe -show` çalıştırıldı (simge önbelleği; Explorer yeniden başlatılmadı, ayar değişmedi).
  Not: Description (ipucu metni) "Churn Dashboard (Streamlit)" olarak korundu; istenirse "Iris" yapılabilir.
- İŞ TAMAMLANDI.
