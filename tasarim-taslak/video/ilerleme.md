# Iris tanıtım videosu (+ font kırpılma düzeltmesi) — ilerleme (2026-10-02)

## 0. Font kırpılma düzeltmesi — TAMAM
- Yedek: backup\sayfalar_aciklanabilirlik_2026-10-02_video.py
- sayfalar\aciklanabilirlik.py: `label_margin()` — Global önem ve beeswarm'da sol kenar boşluğu en uzun etikete
  göre taban (6,4 px/karakter + 14); automargin gerekirse yine büyütür. Neden: sayfa ilk açılansa Plotly etiketi
  Inter yüklenmeden ölçüyordu → "üşteri süresi (yıl)" (11 px kırpılma).
- Doğrulama: doğrudan URL ile soğuk açılış ×3 ve önce başka sayfa açılarak ×3 → etiket solu +52 px (önce −11).
  Beeswarm renkleri değişmedi. smoke_test: SMOKE TEST OK.

## 1. Araçlar
- Sistem Python'una (`--user`) edge-tts 7.2.8 + imageio-ffmpeg 0.6.0 (ffmpeg 7.1 gyan); `playwright install ffmpeg`.
  .venv'e dokunulmadı. ffmpeg ve Playwright video kaydı Akıllı Uygulama Denetimi'ne TAKILMADI (CodeIntegrity'de yeni engel yok).
- Tuzaklar: ffmpeg'e `-nostdin` şart (yoksa arka planda takılıyor); edge-tts her cümleye ~0,3 sn baş + ~1,8 sn
  son sessizlik ekliyor → s1_ses.py kırpıyor.

## 2. Ses
- Ses: tr-TR-AhmetNeural (kullanıcı tercihi), hız +5%. Örnekler: ornek\ (Ahmet + Emel).
- Sayılar okunuşla: 0,50'de → "sıfır virgül ellide", 0,21 → "sıfır virgül yirmi bir". Altyazıda rakamla.
- Aralar: sahne başı 0,4 · cümle arası 0,45 · sahne sonu 0,7 sn; Sahne 10 en az 7 sn, Sahne 11 en az 6 sn.

## 3. Kayıt + kurgu
- s2_kayit.py: tek sürekli kayıt (1920×1080, açık tema, 25 fps, sahte imleç + tıklama halkası), sahneler
  menüden tıklanarak geçiliyor (sayfa giriş animasyonları oynuyor); kartlar ayrı kayıt (HTML/CSS animasyon, logo).
- s3_kurgu.py: sahne klipleri (görüntü + alt bant + sahne sesi) → birleştir → altyazılı sürüm (libass, SRT).
- Yeniden üretim: `python s1_ses.py` → Iris'i 8662'de başlat → `python s2_kayit.py` → `python s3_kurgu.py`
  (yalnız kart/bant: `python s2_kayit.py kartlar`).

## 4. Çıktılar
- iris-tanitim.mp4 (altyazısız, 2:04,2, 31,3 MB), iris-tanitim-altyazili.mp4 (31,0 MB), iris-tanitim.srt (31 satır).
  H.264 High 1080p 25 fps, AAC 48 kHz stereo, faststart.
- Ara dosyalar: klipler\01..11_sahne.mp4, ses\sahneXX.wav, bant\, kayit\.
- ffmpeg engellenmediği için clipchamp-talimat.md gerekmedi.
- Sonradan: *-tel.mp4 (≤24 MiB) ve koordinatörün *-720.mp4 sürümleri. telaffuz\ : Ahmet telaffuz örnekleri.

## 5. v2 görüntü revizyonu (sessiz önizleme) — TAMAM, ses/final kurgu bekliyor
- Geri bildirim: giriş ekranı çok hızlı geçiyor; Iris daha özenli tanıtılsın, animasyonlar görünsün.
- v1 arşivi: kayit_v1\ (webm kayıtlar, isaretler.json, bant\, zamanlama_v1.json, s2_kayit_v1.py).
- Kayıt yöntemi: CDP screencast (Page.startScreencast, JPEG q90, 1920×1080; boyama oldukça kare, animasyonda
  ~60 kare/sn) → kareler gerçek zaman damgalarıyla ffmpeg concat → kayit\sahne_XX.mp4 (1080p 30 fps, CRF 16,
  TV aralığı). Kareler OneDrive dışında: %TEMP%\iris-video-kareler (+ kayit_ham.json; `s2_kayit.py klip` kayıtsız yeniden üretir).
- Animasyon: her sahne yeni sayfaya gidilerek başlar (begin_page oynatır); segment yeni başlığın DOM'a girdiği
  karede başlar, ardından 3,3 sn el sürülmez. Giriş (01b) için görsel ölçüt: sağ panel lacivert olan ilk kare
  (Streamlit'in kenar çubuklu iskeleti bir kare boyanıyordu).
- Kartlar: logo SVG'si (iris-lockup-b-dark) satır içi canlandırma — yay pathLength/dashoffset ile turkuaz→amber→mercan
  çizilir, göz bebeği belirir, "Iris" harfleri sırayla çizilir; font hazır olana dek animasyon duraklatılır.
- Vurgu: turkuaz çerçeve (hafif ölçekle girer) — uyarı listesi 769, recall %82,1, uyarı listesi 769, eşik çipi 0,21,
  Brier 0,1041 / 0,1011, McNemar kartı, Almanya OR kartı, simülatör nadirlik kartı, net fayda. Yumuşak JS kaydırma.
- Bant: yalnız simge (iris-icon-dark.svg); simülatör uyarı kartı ekranın üst yarısında.
- Çıktı: onizleme-v2-sessiz.mp4 (1280×720 30 fps, ses yok, 2:45,7, 13,55 MiB); zamanlama_v2_taslak.json.
- Uygulama kodu değişmedi. Sunucu durduruldu.
