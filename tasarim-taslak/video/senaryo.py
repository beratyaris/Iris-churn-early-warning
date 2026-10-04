"""Iris tanıtım videosu - senaryo (metin kullanıcı onaylı; değiştirmeyin).

Her sahne: başlık bandı + cümleler. Cümle = (ekranda/altyazıda görünen, seslendirilen).
Seslendirilen metinde yalnız sayılar okunuşa göre yazıldı (0,50 -> sıfır virgül elli, 0,21 -> sıfır virgül yirmi bir).
"""
VOICE = "tr-TR-AhmetNeural"  # kullanıcı tercihi (2026-10-02); alternatif "tr-TR-EmelNeural" (örnekler: ornek/)
RATE = "+5%"

SCENES = [
    dict(id=1, kind="kart", band=None, sentences=[
        ("Bu Iris: banka müşterilerinin ayrılma riskini önceden gören bir erken uyarı sistemi.", None),
        ("Çanakkale Onsekiz Mart Üniversitesi İstatistik Bölümü bitirme projesi.", None)]),
    dict(id=2, kind="ekran", band="Genel Bakış · 10.000 müşteri · churn %20,4", sentences=[
        ("On bin müşterilik bir banka verisinde her beş müşteriden biri ayrılmış.", None),
        ("Bu dengesizlik yüzünden 'herkes kalacak' diyen bir model bile yüzde seksen başarılı görünür.", None),
        ("Bu yüzden başarıyı doğrulukla değil, ayrılanları yakalama oranıyla ölçtük.", None)]),
    dict(id=3, kind="ekran", band="Model Karşılaştırma · SMOTE-NC · McNemar + Holm", sentences=[
        ("Lojistik regresyon, random forest ve XGBoost'u karşılaştırdık.", None),
        ("Azınlık sınıfını öğrenmek için eğitimde SMOTE-NC kullandık.", None),
        ("Farkların tesadüf olmadığını McNemar testi ve Holm düzeltmesiyle sınadık.", None)]),
    dict(id=4, kind="ekran", band="Eşik: 0,21 · maliyete duyarlı", sentences=[
        ("Karar eşiğini varsayılan 0,50'de bırakmadık.", "Karar eşiğini varsayılan sıfır virgül ellide bırakmadık."),
        ("Bir müşteriyi kaçırmanın yanlış alarmdan daha pahalı olduğu varsayımıyla eşiği 0,21 seçtik.",
         "Bir müşteriyi kaçırmanın yanlış alarmdan daha pahalı olduğu varsayımıyla eşiği sıfır virgül yirmi bir seçtik."),
        ("Sonuç: test verisinde ayrılan her on müşteriden sekizi önceden işaretleniyor.", None)]),
    dict(id=5, kind="ekran", band="Kalibrasyon · olasılıklar gerçek oranla örtüşüyor", sentences=[
        ("SMOTE olasılıkları şişirir.", None),
        ("Kalibrasyonla, modelin 'yüzde otuz' dediği müşterilerin gerçekten yaklaşık yüzde otuzunun ayrıldığı bir ölçeğe geçtik.", None),
        ("Böylece Iris yalnızca kimin değil, kaç kişinin gideceğini de söyleyebiliyor.", None)]),
    dict(id=6, kind="ekran", band="Açıklanabilirlik · SHAP", sentences=[
        ("Iris yalnızca 'riskli' demiyor, nedenini de gösteriyor.", None),
        ("SHAP ile her müşteride riski hangi özelliklerin artırıp azalttığı görülüyor.", None),
        ("Modelin genelinde en belirleyici değişken yaş.", None)]),
    dict(id=7, kind="ekran", band="Confounding · ilişki, nedensellik değil", sentences=[
        ("Almanya'daki yüksek terk oranının bakiye farkından kaynaklanıp kaynaklanmadığını da test ettik.", None),
        ("Bakiye kontrol edildiğinde fark sürüyor.", None),
        ("Bu bir ilişki, nedensellik iddiası değil.", None)]),
    dict(id=8, kind="ekran", band="Simülatör · eğitimde görülmemiş profil uyarısı", sentences=[
        ("Simülatörde müşteri profili değiştirildiğinde tahmin anında güncelleniyor.", None),
        ("Eğitim verisinde hiç görülmemiş bir profil seçilirse Iris, tahminin güvenilir olmadığını söylüyor.", None)]),
    dict(id=9, kind="ekran", band="Kampanya Planlayıcı · kalibre olasılıklarla net fayda", sentences=[
        ("Kampanya planlayıcı, kalibre olasılıklarla bütçeyi somut bir karara çeviriyor.", None),
        ("Varsayımlar değiştikçe kaç müşteriye ulaşmanın en kârlı olduğu yeniden hesaplanıyor.", None)]),
    dict(id=10, kind="ekran", band="Sunum Modu · tek hikâye", min_dur=7.0, sentences=[
        ("Ve tüm bu adımlar, Sunum Modunda tek bir hikâye olarak anlatılıyor.", None)]),
    dict(id=11, kind="kart", band=None, min_dur=6.0, sentences=[
        ("Iris. Veriden karara.", None)]),
]
# ---------- v2 görüntü planı (sessiz önizleme; ses metni telaffuz seçiminden sonra eklenecek) ----------
# Sıra videodaki sıradır; "01b" yeni giriş ekranı sahnesi. band = alt bant başlığı (None: kart).
SCENES_V2 = [
    dict(id="01", kind="kart", band=None),
    dict(id="01b", kind="ekran", band="Giriş · beklenen churn ve uyarı listesi tek ekranda"),
    dict(id="02", kind="ekran", band="Genel Bakış · 10.000 müşteri · churn %20,4"),
    dict(id="03", kind="ekran", band="Model Karşılaştırma · SMOTE-NC · McNemar + Holm"),
    dict(id="04", kind="ekran", band="Eşik: 0,21 · maliyete duyarlı · recall %82"),
    dict(id="05", kind="ekran", band="Kalibrasyon · Brier 0,1041 → 0,1011"),
    dict(id="06", kind="ekran", band="Risk Skoru ve Açıklanabilirlik · SHAP"),
    dict(id="07", kind="ekran", band="Confounding · ilişki, nedensellik değil"),
    dict(id="08", kind="ekran", band="Simülatör · eğitimde görülmemiş profil uyarısı"),
    dict(id="09", kind="ekran", band="Kampanya Planlayıcı · kalibre olasılıklarla net fayda"),
    dict(id="10", kind="ekran", band="Sunum Modu · tek hikâye"),
    dict(id="11", kind="kart", band=None),
]

# ---------- v2 seslendirme (kullanıcı onaylı, 2026-10-02) ----------
# (ekranda/altyazıda, seslendirilen). Telaffuz yalnız seslendirilen metinde: Iris → Ayris, XGBoost → iks ci bust,
# SMOTE-NC → smot en si, SMOTE → smot, McNemar → mek nimar, SHAP → şap, random forest → rendım forest.
# Sayılar okunuşla: 0,50 → sıfır virgül elli, 0,21 → sıfır virgül yirmi bir.
VOICE_V2 = {
    "01": [("Bu Iris.", "Bu Ayris."),
           ("Adını, Yunan mitolojisinde haber getiren tanrıçadan ve gözün irisinden alıyor: gören ve önceden uyaran.", None),
           ("Banka müşterilerinin ayrılma riskini önceden tespit eden bir erken uyarı sistemi.", None),
           ("Çanakkale Onsekiz Mart Üniversitesi İstatistik Bölümü bitirme projesi.", None)],
    "01b": [("Giriş ekranından tüm analizlere tek tıkla ulaşılıyor.", None)],
    "02": SCENES[1]["sentences"],
    "03": [("Lojistik regresyon, random forest ve XGBoost'u karşılaştırdık.",
            "Lojistik regresyon, rendım forest ve iks ci bust'u karşılaştırdık."),
           ("Azınlık sınıfını öğrenmek için eğitimde SMOTE-NC kullandık.",
            "Azınlık sınıfını öğrenmek için eğitimde smot en si kullandık."),
           ("Farkların tesadüf olmadığını McNemar testi ve Holm düzeltmesiyle sınadık.",
            "Farkların tesadüf olmadığını mek nimar testi ve Holm düzeltmesiyle sınadık.")],
    "04": SCENES[3]["sentences"],
    "05": [("SMOTE olasılıkları şişirir.", "Smot olasılıkları şişirir."),
           ("Kalibrasyonla, modelin 'yüzde otuz' dediği müşterilerin gerçekten yaklaşık yüzde otuzunun ayrıldığı bir ölçeğe geçtik.", None),
           ("Böylece Iris yalnızca kimin değil, kaç kişinin gideceğini de söyleyebiliyor.",
            "Böylece Ayris yalnızca kimin değil, kaç kişinin gideceğini de söyleyebiliyor.")],
    "06": [("Iris yalnızca 'riskli' demiyor, nedenini de gösteriyor.",
            "Ayris yalnızca 'riskli' demiyor, nedenini de gösteriyor."),
           ("SHAP ile her müşteride riski hangi özelliklerin artırıp azalttığı görülüyor.",
            "Şap ile her müşteride riski hangi özelliklerin artırıp azalttığı görülüyor."),
           ("Modelin genelinde en belirleyici değişken yaş.", None)],
    "07": SCENES[6]["sentences"],
    "08": [("Simülatörde müşteri profili değiştirildiğinde tahmin anında güncelleniyor.", None),
           ("Eğitim verisinde hiç görülmemiş bir profil seçilirse Iris, tahminin güvenilir olmadığını söylüyor.",
            "Eğitim verisinde hiç görülmemiş bir profil seçilirse Ayris, tahminin güvenilir olmadığını söylüyor.")],
    "09": SCENES[8]["sentences"],
    "10": SCENES[9]["sentences"],
    "11": [("Iris. Veriden karara.", "Ayris. Veriden karara.")],
}

CLOSING_CARD = "A. Berat Yarış · Çanakkale Onsekiz Mart Üniversitesi · İstatistik Bölümü · Bitirme Projesi"

LEAD, GAP, TAIL = 0.4, 0.45, 0.7   # sahne başı / cümle arası / sahne sonu sessizlik (sn)
# Not: edge-tts her cümleye ~0,3 sn baş + ~1,8 sn son sessizlik ekliyor; s1_ses.py bunları kırpar.
