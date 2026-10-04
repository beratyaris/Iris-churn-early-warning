# Giriş sayfasına "Sunum Modu" düğmesi — ilerleme

- [x] giris.py, app.py, iris_ui.py okundu. Hub düğmeleri `st.page_link` + `.st-key-hub` CSS ile.
- [x] Yedekler: backup\sayfalar_giris_2026-10-01_girisbuton.py, backup\iris_ui_2026-10-01_girisbuton.py
- [x] giris.py: istatistiklerin altına `hub-sunum` kapsayıcısında Sunum Modu page_link
- [x] iris_ui.py: `_LANDING_CSS` içine `.st-key-hub-sunum` vurgulu stil (turkuaz dolgu #0F9488→#0F766E)
- [x] py_compile OK, smoke_test.py OK
- [x] Headless streamlit (port 61237, Edge CDP, 1440x900): giriş açık/koyu hatasız; düğme /sunum_modu'ya gidiyor, hata yok.
- [x] İlk ölçüm: sağ panel 907px (taşma). `.st-key-hub-sunum { margin-top: -.2rem }` ile 900px — tek ekrana sığıyor.
- [x] Ekran görüntüleri: giris-acik.png, giris-koyu.png, tiklama-sonrasi-sunum-modu.png. Sunucu durduruldu. smoke_test tekrar OK.
- İŞ TAMAMLANDI.
