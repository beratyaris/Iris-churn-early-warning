# Kaynak–Proje Özetleri: 21 Kaynağın Bitirme Projesine Uygulanması

Kaynak liste: `knowledge/concepts/bitirme-literatur-kaynaklari.md` (21 kaynak, 4 grup).
Proje kararları: `knowledge/concepts/bitirme-projesi-churn.md`.
Bu belge, Sonnet alt ajanının 2026-09-30 tarihli web araştırmasına dayanır (WebSearch, WebFetch,
Semantic Scholar API, doğrudan PDF okuma).

**Doğrulama seviyeleri:**
- **[TM]** = tam metni bizzat okundum (PDF üzerinden, alıntı/sayfa verilebilir).
- **[Öz]** = özet/içerik web araması veya API üzerinden **parafraz** olarak elde edildi; kelimesi
  kelimesine alıntı değildir, bu yüzden tırnak içinde verilmemiştir.
- **[Bib]** = yalnızca künye (yazar/başlık/sayfa) doğrulandı, içerik doğrulanamadı.
- **[nottan]** = bu iddia yalnızca mevcut `bitirme-literatur-kaynaklari.md` notunda yazılı; bu
  oturumda bağımsız olarak doğrulanamadı.

**Atıf kuralı:** Bir kaynağın bizzat ürettiği bulgu ile, o kaynağın *başka bir çalışmadan aktardığı*
bulgu ayrı tutulmuştur. İkincil aktarım "X (akt. Y, yıl)" biçiminde işaretlenmiştir.

---

## Grup 1 — Churn + ML

### 1. Tékouabou, Gherghina, Toulni, Mata & Martins (2022)

**Künye:** Tékouabou, S. C. K., Gherghina, Ş., Toulni, H., Mata, P. N., & Martins, J. M. (2022).
Towards explainable machine learning for bank churn prediction using data balancing and
ensemble-based methods. *Mathematics*, 10(14), 2379. https://doi.org/10.3390/math10142379
**Doğrulama:** [Öz] (MDPI sayfası 403 verdi; Semantic Scholar API üzerinden özet elde edildi;
yazar listesi tam olarak bu oturumda doğrulandı — not'taki "Tékouabou vd." kısaltması doğru).

**Desteklediği bileşenler:** modeller, dengesizlik/SMOTE, SHAP, boşluk argümanı (ana benchmark).

**Bulgular:** SMOTE tabanlı dengeleme ile ensemble yöntemler birleştiriliyor; rastgele orman
dengelenmiş veride 0.86 doğruluk/F1-skoru elde ediyor (Tékouabou vd., 2022, [Öz]). Model
yorumlaması Shapley değerleri ve öznitelik önem analizi ile yapılıyor (Tékouabou vd., 2022, [Öz]).
Not'taki "en önemli değişken yaş, en önemsiz HasCrCard" bulgusu bu oturumda bağımsız
doğrulanamadı — **[nottan]**.

**Alıntı:** Okunamadı (yalnızca parafraz özet elde edildi); doğrudan alıntı verilmiyor.

**Tezde kullanım:** Literatür bölümünde "ana benchmark" olarak tanıtılır; Giriş'te SMOTE+ensemble+SHAP
üçlüsünün churn literatüründe standart hale geldiğini göstermek için kullanılabilir. Tékouabou vd.
(2022) SMOTE+RF+SHAP kombinasyonunun literatürde yaygınlığını gösterdiği için, projenin
"bu kombinasyon yaygın, katkı McNemar+Holm+maliyete duyarlı eşikte" argümanının karşılaştırma
noktasıdır.

**Çelişki/ihtiyat:** Hangi veri setinin kullanıldığı (büyüklük, churn oranı) bu oturumda
doğrulanamadı; tam metne erişilemedi, dolayısıyla 0.86 rakamının hangi sınıflandırıcıya ve hangi
veri bölünmesine ait olduğu teyit edilmelidir.

---

### 2. ICBASE 2025 (Zhang, Wang, Li & Wang)

**Künye:** Zhang, H.-Y., Wang, Y.-K., Li, Z., & Wang, X. (2025). Machine learning models for bank
customer churn prediction: A comparative study of LightGBM, CatBoost, and XGBoost. *Proceedings
of the 2025 International Conference on Big Data, Artificial Intelligence and Digital Economy
(ICBASE/BDAIE '25)*. https://doi.org/10.1145/3767052.3767054
**Doğrulama:** [Öz] (ACM sayfası 403 verdi; Semantic Scholar API'den özet ve **yazar adları**
doğrulandı — not'taki "[Öz, yazarlar ⚠]" işareti bu oturumda çözüldü: yazarlar Han-Yuan Zhang,
Yan-Kai Wang, Zexuan Li, Xiaoyin Wang).

**Desteklediği bileşenler:** modeller, boşluk argümanı.

**Bulgular:** 11 model bir banka müşteri veri setinde ön işleme sonrası karşılaştırılıyor; temel
metrik recall (Zhang vd., 2025, [Öz]). LightGBM, CatBoost ve XGBoost churn tahmininde üstün
performans gösteriyor; Gradient Boost, AdaBoost ve Bagging ise dengesiz veriyle başa çıkmada etkili
bulunuyor (Zhang vd., 2025, [Öz]) — **not'taki "en iyi modeller LightGBM ve Bagging" ifadesi kısmen
hatalı/eksik**: Bagging asıl olarak dengesizlikle başa çıkmada öne çıkıyor, genel en iyi performans
LightGBM/CatBoost/XGBoost üçlüsünde. Total_Trans_Amt ve Total_Trans_ct önemli yordayıcılar olarak
belirlenmiş (Zhang vd., 2025, [Öz]).

**Alıntı:** Yok (parafraz özet).

**Tezde kullanım:** Giriş/boşluk argümanı bölümünde kullanılır: 11 model karşılaştırılmasına rağmen
istatistiksel anlamlılık testi kullanılmadığı (not'ta belirtildiği gibi, bu oturumda da teyit
edilemedi çünkü tam metne erişilemedi — **[nottan]** kalıyor) örneği olarak sunulur, ancak bu iddia
tam metinle doğrulanmalı.

**Çelişki/ihtiyat:** "İstatistiksel test kullanılmamış" iddiası tam metinden değil, yalnızca not'tan
geliyor — **[nottan]**, teyit gerekiyor.

---

### 3. Zhu (2024) — ÖNEMLİ: Aynı veri setiyle çalışıyor olabilir

**Künye:** Zhu, H. (2024). Bank customer churn prediction with machine learning methods. *Advances
in Economics, Management and Political Sciences*, 69. https://doi.org/10.54254/2754-1169/69/20230773
**Doğrulama:** [Öz] (Semantic Scholar API'den özet; tek yazar "He Zhu" doğrulandı — not'taki "eşleşme
kesin değil" belirsizliği kısmen çözüldü: yazar tek kişi ve University College London bağlantılı).

**Desteklediği bileşenler:** veri seti (olası çakışma), modeller.

**ÖNEMLİ BULGU:** Arama sırasında bu çalışmanın "ABC Bank" verisini kullandığı doğrulandı ve ayrıca
**"ABC Multistate Bank" adlı, farklı yazarlarca (Hui Shan Hon vd.) yazılmış ayrı bir makale** de
bulundu (`journal.uitm.edu.my`). Projenin kendi veri seti de Kaggle üzerinde "ABC Multinational
Bank" (Gaurav Topre) adıyla anılıyor (bkz. `bitirme-projesi-churn.md`). **Zhu (2024)'ün hangi "ABC
Bank" veri setini kullandığı bu oturumda kesinleştirilemedi** — eğer aynı Kaggle veri setiyse, bu
kaynak projenin doğrudan karşılaştırılabilir bir emsali olur (10.000 satır, aynı churn oranı
aralığı). Bu, teze eklenmeden önce mutlaka doğrulanmalı — **[nottan/⚠, yüksek öncelik]**.

**Bulgular:** Lojistik regresyon ile rastgele orman karşılaştırılıyor; RF daha yüksek doğruluk elde
ediyor; yaş churn ile en güçlü ilişkiyi gösteren değişken, kredi kartı durumu en zayıf ilişkiyi
gösteren değişken (Zhu, 2024, [Öz]) — not'taki "en önemli değişken yaş" bulgusu bu oturumda **teyit
edildi**.

**Alıntı:** Yok (parafraz özet).

**Tezde kullanım:** Eğer veri seti eşleşmesi doğrulanırsa, Bulgular bölümünde doğrudan karşılaştırma
("Zhu (2024) aynı/benzer ABC Bank verisinde RF'nin LR'yi geçtiğini bulmuştur, bu çalışmada da...")
için kullanılabilir; aksi halde Literatür bölümünde genel bir RF>LR örneği olarak kalır.

**Çelişki/ihtiyat:** Veri seti eşleşmesi doğrulanamadı; sayfa/cilt bilgisi ("69") teyit edildi ama
tam metne erişilemedi, dolayısıyla örneklem büyüklüğü ve churn oranı **[nottan]** kalıyor.

---

### 4. Ashraf (2024)

**Künye:** Ashraf, R. (2024). Bank customer churn prediction using machine learning framework.
*Journal of Applied Finance & Banking*, 14(4). (Sayfa aralığı kaynaklar arasında tutarsız görünüyor:
bazı kayıtlarda 1–5, SCIENPRESS PDF künyesinde "65–109" — bkz. Atıf uyarıları.)
**Doğrulama:** [Öz] (WebSearch üzerinden özet; PDF indirildi ama metin katmanı sıkıştırılmış/okunamaz
durumdaydı, güvenilir tam metin çıkarımı yapılamadı).

**Desteklediği bileşenler:** modeller, dengesizlik/SMOTE.

**Bulgular:** ABD'de güneyde büyük bir toplum bankasının gerçek müşteri verisiyle 10 sınıflandırma
modeli ve 5 örnekleme tekniği karşılaştırılıyor (Ashraf, 2024, [Öz]). RF, XGBoost, AdaBoost ve
Bagging genel doğruluk, F-skor ve AUC eğrisinde diğerlerini geçiyor; bu dört sınıflandırıcı için
genel doğruluk beş örnekleme yöntemi genelinde %87–96 arasında değişiyor; AdaBoost, orijinal ve
MTDF örnekleme teknikleriyle doğruluk ve F-skorda öne çıkıyor (Ashraf, 2024, [Öz]). Not'taki
"%87–96 AUC 0.90–0.93" rakamları kısmen teyit edildi (doğruluk aralığı evet, AUC 0.90–0.93 rakamı
bu oturumda bağımsız doğrulanamadı — **[nottan]**).

**Alıntı:** Yok.

**Tezde kullanım:** Literatür bölümünde "gerçek banka verisiyle çok-model karşılaştırma" örneği;
projenin gerçek veri yerine Kaggle verisi kullanma kararının (KVKK/BDDK/etik kurul gerekçesiyle)
karşıtı olarak tartışılabilir — Tartışma bölümünde dış geçerlilik sınırlaması olarak.

**Çelişki/ihtiyat:** Sayfa aralığı kaynaklar arası tutarsız (1–5 vs 65–109); teze girmeden önce
orijinal dergi sayfasından teyit edilmeli.

---

### 5. Karamollaoğlu, Yücedağ & Doğru (2021)

**Künye:** Karamollaoğlu, H., Yücedağ, İ., & Doğru, İ. A. (2021). Customer churn prediction using
machine learning methods: A comparative analysis. *UBMK 2021 (6th International Conference on
Computer Science and Engineering)*, 139–144.
**Doğrulama:** [Öz] (avesis.gazi.edu.tr sayfasından yazar adları ve yöntem listesi doğrulandı).

**Desteklediği bileşenler:** modeller.

**Bulgular:** İki ayrı telekom veri seti (Kaggle kaynaklı) üzerinde 8 yöntem karşılaştırılıyor:
Lojistik Regresyon, K-En Yakın Komşu, Karar Ağaçları, Rastgele Orman, Destek Vektör Makineleri,
AdaBoost, Çok Katmanlı Algılayıcı, Naive Bayes (Karamollaoğlu vd., 2021, [Öz]). Rastgele Orman her
iki veri setinde de en iyi performansı gösteriyor (Karamollaoğlu vd., 2021, [Öz]) — not'taki bulgu
teyit edildi.

**⚠ Düzeltme/çelişki bu oturum içinde:** İlk WebSearch araması "RF + SMOTE-ENN kombinasyonu F1-skorda
%95 doğruluk" gibi bir sayı üretti, ancak doğrudan avesis sayfası WebFetch'inde **bu sayı teyit
edilemedi** ("No specific F1-score or accuracy percentages were provided"). Bu rakam **kullanılmamalı**;
muhtemelen arama motorunun genellemesiydi. **[nottan değil — yanlış/asılsız, atılmalı]**.

**Alıntı:** Yok.

**Tezde kullanım:** Literatür bölümünde RF'nin telekom churn'de tekrarlayan üstünlüğüne bir örnek
daha olarak, kısa biçimde.

**Çelişki/ihtiyat:** F1/doğruluk sayıları doğrulanamadı; UBMK bildiri kitabının orijinal PDF'i
bulunamadı (IEEE Xplore arkasında olabilir).

---

### 6. Kaynar, Tuna, Görmez & Deveci (2017) — TAM METİN OKUNDU

**Künye:** Kaynar, O., Tuna, M. F., Görmez, Y., & Deveci, M. A. (2017). Makine öğrenmesi
yöntemleriyle müşteri kaybı analizi. *Cumhuriyet Üniversitesi İktisadi ve İdari Bilimler Dergisi*,
18(1), 1–14.
**Doğrulama:** **[TM]** — dergipark.org.tr üzerinden tam metin PDF'i doğrudan okundu.

**Desteklediği bileşenler:** modeller.

**Bulgular (sayfa referanslı):**
- Veri seti: açık erişimli bir telekomünikasyon firmasının çağrı merkezi işlem kayıtları; 4.667
  müşteri, her biri için 21 öznitelik (Kaynar vd., 2017, s. 1, 7). Eğitim %75 (3.500), test %25
  (1.167), rastgele bölünmüş (Kaynar vd., 2017, s. 7–8).
- Üç yöntem karşılaştırılıyor: Destek Vektör Makineleri (DVM/SVM), Yapay Sinir Ağları (YSA/ANN,
  çok katmanlı algılayıcı), Naive Bayes (NB) (Kaynar vd., 2017, s. 5–7).
- Test sonuçları (Tablo 2, s. 9): NB — Başarı oranı(A)=0.8715, S=0.8964, P=0.8295, F=0.8616; DVM —
  A=0.7789, S=0.8887, P=0.6983, F=0.7821; YSA — A=0.9135, S=0.8906, P=0.9134, F=0.9018 (Kaynar vd.,
  2017, Tablo 2, s. 9). YSA; başarı oranı, "P" (duyarlılık — formülü TP/(TP+FP), yani aslında
  kesinlik/precision) ve F-ölçütünde en iyi sonucu veriyor; NB ise "S" (hassasiyet — formülü
  TP/(TP+FN), yani aslında geri çağırma/recall) ölçütünde en iyi (Kaynar vd., 2017, s. 9).
- **⚠ Terminoloji uyarısı:** Makalede "hassasiyet (S)" ve "duyarlılık (P)" etiketleri, standart
  Türkçe istatistik kullanımının **tersi** formüllerle tanımlanmış: S = TP/(TP+FN) (aslında
  duyarlılık/recall formülü) ama "hassasiyet" diye adlandırılmış; P = TP/(TP+FP) (aslında
  kesinlik/precision formülü) ama "duyarlılık" diye adlandırılmış (Kaynar vd., 2017, s. 8, Tablo 1
  altındaki formül açıklamaları). Teze aktarılırken bu isim karışıklığına dikkat edilmeli;
  formüllere göre yorumlanmalı, etiketlere göre değil.

**Doğrudan alıntı (Türkçe, s. 9):**
> "Sonuçlar incelendiğinde başarı oranı, duyarlılık ve F ölçütü değerlerinde en iyi skorlar sırası
> ile %91,35, % 91,34 ve % 90,18 olup YSA kullanan model ile elde edilirken, hassasiyet değerinde
> en iyi skor % 89,64 olup NB kullanan model ile elde edilmiştir." (Kaynar vd., 2017, s. 9)

**Tezde kullanım:** Literatür bölümünde, çok-model karşılaştırmasının McNemar/istatistiksel test
olmadan yapıldığı bir örnek olarak (boşluk argümanına dolaylı destek — makale hiçbir anlamlılık
testi raporlamıyor, sadece ham metrik karşılaştırması yapıyor, Kaynar vd. 2017, s. 8–9).

**Çelişki/ihtiyat:** İstatistiksel anlamlılık testi kullanılmamış; tek bir rastgele train/test
bölünmesi kullanılmış (Dietterich 1998'in eleştirdiği yaygın pratiğe tam örnek).

---

## Grup 2 — Dengesizlik & SMOTE

### 7. Chawla, Bowyer, Hall & Kegelmeyer (2002)

**Künye:** Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic
minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
https://doi.org/10.1613/jair.953
**Doğrulama:** [Öz] (jair.org üzerinden WebFetch; sayfa numarası içeren metin çıkarılamadı, yalnızca
özet/yöntem parafrazı elde edildi).

**Desteklediği bileşenler:** dengesizlik/SMOTE-NC, sızıntı (dolaylı — yöntemin eğitim setine
uygulanması ilkesi buradan gelir).

**Bulgular:** Azınlık sınıfı için sentetik örnekler üreten SMOTE, çoğunluk sınıfının alt örneklemesi
ile birleştirildiğinde ROC uzayında yalnız alt örneklemeden daha iyi sınıflandırıcı performansı
sağlıyor (Chawla vd., 2002, [Öz]) — not'taki bulgu teyit edildi.

**ÖNEMLİ EK BULGU — SMOTE-NC:** Ayrı bir arama, orijinal Chawla vd. (2002) makalesinin karma
(sürekli + nominal) öznitelikli veriler için **SMOTE-NC (Synthetic Minority Over-sampling
TEchnique – Nominal Continuous)** yöntemini de tanımladığını ve UCI "Adult" veri setinde test
ettiğini doğruladı (arama sentezi, [Öz] — orijinal makalenin ilgili bölümü bizzat okunmadı, bu
yüzden doğrudan alıntı verilmiyor). **Bu, projenin SMOTE-NC kullanım gerekçesini doğrudan
destekliyor**: "karma veri tipi olduğu için SMOTE-NC" savunması, Chawla vd. (2002)'nin kendi
makalesine dayandırılabilir, ayrı bir SMOTE-NC makalesi aramaya gerek kalmadan.

**Alıntı:** Yok (parafraz).

**Tezde kullanım:** Yöntem bölümünde, SMOTE-NC seçiminin gerekçesi olarak doğrudan Chawla vd.
(2002)'ye atıf yapılabilir — "SMOTE-NC, Chawla vd. (2002) tarafından karma öznitelikli veriler için
tanımlanmış olup, bu çalışmada..." cümlesi kurulabilir.

**Çelişki/ihtiyat:** SMOTE-NC'nin tam olarak hangi bölümde/sayfada tanımlandığı bu oturumda sayfa
numarasıyla doğrulanamadı; teze koymadan önce orijinal PDF'in ilgili bölümü (Adult veri seti kısmı)
okunmalı.

---

### 8. Amin, Anwar, Adnan, Nawaz, Howard, Qadir, Hawalah & Hussain (2016)

**Künye:** Amin, A., Anwar, S., Adnan, A., Nawaz, M., Howard, N., Qadir, J., Hawalah, A. Y. A., &
Hussain, A. (2016). Comparing oversampling techniques to handle the class imbalance problem: A
customer churn prediction case study. *IEEE Access*, 4, 7940–7957.
https://doi.org/10.1109/ACCESS.2016.2619719
**Doğrulama:** [Öz] (Semantic Scholar API'den özet elde edildi).

**Desteklediği bileşenler:** dengesizlik/SMOTE-NC.

**Bulgular:** Telekom churn verisinde altı örnekleme tekniği (SMOTE ve MTDF dahil) ile dört kural
üretme algoritması karşılaştırılıyor; MTDF ve genetik algoritma tabanlı kural üretimi en iyi
performansı gösteriyor (Amin vd., 2016, [Öz]). **Not'taki ifade kısmen düzeltilmeli:** not "SMOTE ve
ADASYN karşılaştırılıyor" diyor, ancak bu oturumda doğrulanan özet ADASYN'i açıkça anmıyor; asıl
karşılaştırma altı örnekleme tekniği + dört kural üretme algoritması arasında ve en iyi kombinasyon
SMOTE değil **MTDF** (Amin vd., 2016, [Öz]).

**Alıntı:** Yok.

**Tezde kullanım:** Yöntem/Tartışma bölümünde, "oversampling seçiminin performansı belirgin şekilde
etkilediği" genel argümanını desteklemek için kullanılabilir, ancak "SMOTE vs ADASYN" çerçevesi
yerine "SMOTE/MTDF/kural-tabanlı yöntemler arası fark" çerçevesiyle ifade edilmeli.

**Çelişki/ihtiyat:** Not'taki "SMOTE ve ADASYN" çerçevesi bu oturumda tam olarak doğrulanamadı;
teyit gerekiyor.

---

### 9. Burez & Van den Poel (2009)

**Künye:** Burez, J., & Van den Poel, D. (2009). Handling class imbalance in customer churn
prediction. *Expert Systems with Applications*, 36(3), 4626–4636.
https://doi.org/10.1016/j.eswa.2008.05.027
**Doğrulama:** [Öz] (ScienceDirect/Semantic Scholar abstract alanı yayıncı tarafından
gizlenmiş; içerik birden fazla ikincil kaynaktan [arama sentezleri, eprints.soton.ac.uk üzerinden
başka bir makalenin bu çalışmaya atfı] tutarlı biçimde derlendi).

**Desteklediği bileşenler:** dengesizlik/SMOTE, boşluk argümanı — **"accuracy yanıltıcı" kararının
temel dayanağı**.

**Bulgular:** Çalışma, alt örnekleme (under-sampling) ile iki makine öğrenmesi algoritmasını
(ağırlıklı rastgele ormanlar ve RUSBoost) yalnız rastgele orman temel durumuna karşı, AUC metriğini
kullanarak karşılaştırıyor; alt örnekleme dengesizlikle başa çıkmada en etkili yöntem olarak
bulunuyor, RUSBoost de etkili ama alt örneklemeden daha zayıf, ağırlıklı rastgele ormanlar ise
yalnızca üç sınıflandırma probleminden birinde (appetency) iyileşme sağlıyor (arama sentezi, [Öz]).
Not'taki "Accuracy yerine AUC ve lift öneriyor" bulgusu bu çerçevede tutarlı görünüyor ancak "lift"
kelimesi bu oturumda bağımsız doğrulanamadı — **[nottan]** kısmı kalıyor.

**Alıntı:** Yok (abstract yayıncı tarafından gizlenmiş, tam metne erişilemedi).

**Tezde kullanım:** Giriş ve Yöntem bölümlerinde, projenin "accuracy kullanılmıyor" kararının temel
literatür dayanağı olarak; "Burez ve Van den Poel (2009), dengesiz churn verilerinde accuracy
yerine AUC gibi metriklerin tercih edilmesi gerektiğini göstermiştir" cümlesi kurulabilir.

**Çelişki/ihtiyat:** Tam metne erişilemediği için "lift" önerisi ve tam sayısal sonuçlar
**[nottan]** düzeyinde kalıyor; DOI ve sayfa aralığı (4626–4636) doğrulandı.

---

### 10. Haddadi, Farshidvard, Silva, Reis & Reis (2024)

**Künye:** Haddadi, S. J., Farshidvard, A., Silva, F. dos S., Reis, J. D. dos, & Reis, M. H. da S.
(2024). Customer churn prediction in imbalanced datasets with resampling methods: A comparative
study. *Expert Systems with Applications*, 246, 123086. https://doi.org/10.1016/j.eswa.2023.123086
**Doğrulama:** [Öz] (Semantic Scholar API + WebSearch sentezi; yazar adları biraz farklı
biçimlerde geçiyor, en güvenilir hâli yukarıdaki — bkz. Atıf uyarıları).

**Desteklediği bileşenler:** dengesizlik/SMOTE.

**Bulgular:** Telekomünikasyon, e-ticaret ve bankacılık sektörlerinden verilerde çeşitli yeniden
örnekleme yöntemleri karşılaştırılıyor; SMOTE genel olarak standart bir yaklaşım olarak
kullanılıyor, SMOTE-Tomek Links ve SMOTE-ENN'nin churn tahmininde uygulanması literatürde sınırlı
bulunuyor (Haddadi vd., 2024, [Öz]). Not'taki "iki aşamalı resampling, AUC'de daha iyi" bulgusu bu
oturumda ayrıntılı biçimde teyit edilemedi — **[nottan]**.

**Alıntı:** Yok.

**Tezde kullanım:** Literatür bölümünde, dengesizlik yönteminin sektörler arası genellenebilirliğine
(telekom+e-ticaret+banka) örnek olarak kullanılabilir.

**Çelişki/ihtiyat:** "İki aşamalı resampling" ve somut AUC rakamları doğrulanamadı.

---

### 11. Boonmunewai, Chantaraksa & Rodjanadid (2024)

**Künye:** Boonmunewai, J., Chantaraksa, T., & Rodjanadid, B. (2024). Evaluating machine learning
methods for solving class imbalance in banking customer data: A comparative study. *KKU Science
Journal*, 52(3), 349–362.
**Doğrulama:** **[Bib]** — tam ve doğru künye (yazarlar, başlık, cilt/sayı/sayfa) TCI-Thaijo dergi
içindekiler sayfasından doğrulandı; ancak makale gövdesine (yalnızca kapak/içindekiler indirildi,
tam metin PDF'i bu oturumda okunamadı) ulaşılamadı.

**Desteklediği bileşenler:** dengesizlik/SMOTE.

**Bulgular:** Not'taki sayılar (**SMOTE+SVM: recall %92.99, F %91.37, AUC %96.4**) bu oturumda
bağımsız doğrulanamadı — **[nottan]**, teze girmeden önce mutlaka tam metinden teyit edilmeli.

**Alıntı:** Yok.

**Tezde kullanım:** Bulgular/Tartışma bölümünde "banka verisinde SMOTE+SVM kombinasyonu yüksek
recall verebilir" örneği olarak **yalnızca not'a dayanarak ve açıkça "doğrulanamamış kaynak" notuyla**
kullanılmalı, ya da tam metin bulunana kadar tezden çıkarılmalı.

**Çelişki/ihtiyat:** İçerik doğrulanamadı; yüksek risk — Atıf uyarıları listesinde.

---

## Grup 3 — XAI & SHAP

### 12. Lundberg & Lee (2017)

**Künye:** Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model
predictions. In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*
(sayfa aralığı kaynaklar arası tutarsız — bkz. Atıf uyarıları).
**Doğrulama:** [Öz] (WebSearch sentezi; not'taki "web'de aranmadı" işareti bu oturumda çözüldü —
artık aranmış durumda, ama yalnızca [Öz] düzeyinde).

**Desteklediği bileşenler:** SHAP.

**Bulgular:** Karmaşık modellerin (ensemble, derin öğrenme) yüksek doğruluk sağladığı ama
yorumlanmasının zor olduğu; mevcut yorumlama yöntemlerinin birbiriyle ilişkisinin belirsiz olduğu
vurgulanıyor; makale "additive feature attribution methods" sınıfını tanımlayıp altı mevcut yöntemi
birleştiriyor ve oyun teorisi sonuçlarına dayanan tek bir çözümü garanti eden SHAP değerlerini
öneriyor (Lundberg & Lee, 2017, [Öz]).

**Alıntı:** Yok (parafraz).

**Tezde kullanım:** Yöntem bölümünde SHAP'in teorik temeli olarak standart atıf; "SHAP, Shapley
değerlerine dayanan birleşik bir öznitelik atfı çerçevesidir (Lundberg & Lee, 2017)" cümlesi.

**Çelişki/ihtiyat:** Sayfa aralığı belirsiz — bazı kaynaklar 4765–4774 (NeurIPS proceedings
standart künyesi), ACM DL kaydı 4768–4777 diyor. Teze koymadan önce proceedings.neurips.cc
üzerinden doğrulanmalı.

---

### 13. Guliyev & Yerdelen Tatoğlu (2021)

**Künye:** Guliyev, H., & Yerdelen Tatoğlu, F. (2021). Customer churn analysis in banking sector:
Evidence from explainable machine learning models. *Journal of Applied Microeconometrics*, 1(2),
85–99. https://doi.org/10.53753/jame.1.2.03
**Doğrulama:** [Öz] (WebSearch sentezi + **ikincil teyit**: Rasool vd. 2025'in [TM] okunan tam
metninde bu kaynak [21] numaralı referans olarak anılıyor: "Guliyev and Tatoğlu [21] applied XGBoost
combined with SHAP to banking churn data, revealing how explainability can bridge the gap between
predictions and strategic customer management" (Rasool vd., 2025, s. 4374) — bu **ikincil bir
aktarım**, yani "Guliyev & Tatoğlu (2021)'in bulgusu, Rasool vd. (2025, s. 4374) tarafından
aktarılmıştır" biçiminde kullanılmalı, doğrudan Guliyev & Tatoğlu'na değil).

**Desteklediği bileşenler:** modeller, SHAP.

**Bulgular:** Bankacılık verisinde SHAP değerleri model değerlendirmesi ve yorumlanabilirliği
desteklemek için kullanılıyor; sonuçlara göre XGBoost churn müşterilerini sınıflandırmada diğer
makine öğrenmesi yöntemlerini geride bırakıyor (Guliyev & Yerdelen Tatoğlu, 2021, [Öz]; akt. Rasool
vd., 2025, s. 4374).

**Alıntı:** Yok (parafraz).

**Tezde kullanım:** Literatür bölümünde "bankacılıkta XGBoost+SHAP" emsali olarak — projeye en yakın
benzer çalışma (not'ta da böyle işaretlenmiş, teyit edildi).

**Çelişki/ihtiyat:** Somut sayısal sonuç (accuracy/F1/AUC) bu oturumda elde edilemedi.

---

### 14. Maan & Maan (2023) — TAM METİN OKUNDU

**Künye:** Maan, J., & Maan, H. (2023). Customer churn prediction model using explainable machine
learning. *International Journal of Computer Science Trends and Technology (IJCST)*, 11(1), 33–38.
(arXiv ön-baskı: arXiv:2303.00960)
**Doğrulama:** **[TM]** — IJCST dergi PDF'i doğrudan okundu (6 sayfa, tam metin).

**Desteklediği bileşenler:** modeller, SHAP.

**Bulgular (sayfa referanslı):**
- Veri seti: kamuya açık bir telekom operatörü müşteri veri seti (Eve Mins, Day Mins, Int'l Plan,
  Account Length gibi öznitelikler) (Maan & Maan, 2023, s. 34, Fig. 1). Eğitim %70 / test %30
  (Maan & Maan, 2023, s. 35). Test setinde destek (support) toplamı 1.000 (878 Not-Churn + 122
  Churn) (Maan & Maan, 2023, Tablo 1, s. 36) — bu oran, Rasool vd. (2025)'in kullandığı 3.333
  kayıtlı "Churn in Telecom's" Kaggle veri setiyle (aynı öznitelik isimleri, benzer oranlar) **aynı
  veya çok benzer bir kamuya açık veri seti olabilir**; kesin teyit edilmedi.
- Dört model karşılaştırılıyor: Lojistik Regresyon, Karar Ağacı, Rastgele Orman, XGBoost
  (Maan & Maan, 2023, Tablo 2, s. 36).
- **Sonuçlar (Tablo 2, s. 36):** LR: doğruluk=0.878, F1=0.153; Karar Ağacı: doğruluk=0.922,
  F1=0.614; Rastgele Orman: doğruluk=0.954, F1=0.796; **XGBoost: doğruluk=0.962, F1=0.836 (en
  iyi)** (Maan & Maan, 2023, Tablo 2, s. 36). **Bu, not'taki "brief'te F1 0.836 / acc 0.962
  değerleri teyit edilemedi" uyarısını çözüyor — sayılar bu oturumda tam metinden DOĞRULANDI.**
- SHAP TreeExplainer ile özellik önemi hesaplanıyor; en önemli öznitelikler grafikte Day Mins,
  Int'l Plan, Intl Mins olarak görünüyor (Maan & Maan, 2023, Fig. 6, s. 37).

**Doğrudan alıntı (s. 36):**
> "XGBOOST Classifier provided the highest F1 score and Accuracy score than other 3 models, thereby
> depicting the best performance among all classifiers." (Maan & Maan, 2023, s. 36)

**⚠ İç tutarsızlık uyarısı:** Makalenin kendi içinde çelişki var: Tablo 1'deki sınıflandırma
raporunda ("Classification Report") Churn sınıfı için recall=0.22 ve precision=0.55 görünüyor
(Maan & Maan, 2023, Tablo 1, s. 36), ama aynı sayfadaki düz metinde "XGBoost ensemble model has the
highest AUC of 0.79 with a recall of 0.83 and precision of 0.54" deniyor (Maan & Maan, 2023, s. 36).
Bu iki rakam seti birbiriyle uyuşmuyor — hangi modelin/eşiğin hangi tabloya ait olduğu makalede net
değil. **Bu kaynağın recall rakamları tezde kullanılmamalı; yalnızca doğruluk ve F1 (Tablo 2)
güvenilir görünüyor.**

**Tezde kullanım:** Literatür/Bulgular bölümünde, XGBoost'un yüksek doğruluk (%96.2) ile düşük churn
recall'ı (Tablo 1'e göre %22) bir arada gösterebildiğine — yani **"accuracy yanıltıcı" argümanına
somut bir ikinci örnek** olarak kullanılabilir (Rasool 2025'teki %94.75/%73.20 örneğine ek olarak).

**Çelişki/ihtiyat:** Yukarıdaki iç tutarsızlık nedeniyle recall/precision/AUC rakamları ihtiyatla
kullanılmalı; yalnızca Tablo 2'deki doğruluk/F1 rakamları güvenilir.

---

### 15. Rasool, Aljawaheri & Karram (2025) — TAM METİN OKUNDU (EN KAPSAMLI KAYNAK)

**Künye:** Rasool, H. A., Aljawaheri, K. K., & Karram, A. A. M. (2025). Explainable customer churn
prediction in telecom using ensemble learning and SHAP analysis. *Mathematical Modelling of
Engineering Problems*, 12(12), 4373–4386. https://doi.org/10.18280/mmep.121226
**Doğrulama:** **[TM]** — IIETA dergi PDF'i doğrudan ve eksiksiz okundu (14 sayfa).

**Desteklediği bileşenler:** dengesizlik/SMOTE, sızıntı önleme, SHAP, boşluk argümanı, (dolaylı)
maliyet/eşik.

**Bulgular (sayfa referanslı):**
- Veri seti: Kaggle "Churn in Telecoms Dataset", ABD'li bir telekom operatörünün 3.333 müşterisi,
  21 öznitelik (Rasool vd., 2025, Bölüm 3.1, s. 4375).
- **SMOTE yalnızca eğitim setine uygulanıyor** — aynı projedeki sızıntı önleme ilkesiyle birebir
  örtüşüyor: "This imbalance is addressed using Synthetic Minority Over-sampling Technique (SMOTE),
  which is applied exclusively to the training set" (Rasool vd., 2025, s. 4377, Bölüm 3.3).
  Dengelenmiş eğitim seti 4.560 kayıt (2.280+2.280), dokunulmamış test seti 667 kayıt
  (Rasool vd., 2025, s. 4377).
- Beş temel sınıflandırıcı (Karar Ağacı, Rastgele Orman, Extra Trees, Gradient Boosting, XGBoost)
  üç ensemble yöntemiyle (Soft Voting, Hard Voting, Stacking) birleştiriliyor (Rasool vd., 2025,
  Bölüm 3.4–3.5, s. 4377–4379).
- **Tekil model sonuçları (Tablo 3, s. 4380):** XGBoost — doğruluk=0.9415, recall=0.7629,
  precision=0.8222, F1=0.7914; Rastgele Orman — doğruluk=0.9130, recall=0.7629, precision=0.6789,
  F1=0.7184.
- **En iyi model — Stacking ensemble (Tablo 5/6, s. 4381):** doğruluk=**%94.75**,
  recall=**%73.20**, precision=**%88.75**, F1=**0.8023**, hata oranı=%5.25. Bu sayılar not'taki
  rakamlarla birebir örtüşüyor ve bu oturumda tam metinden **DOĞRULANDI**.
- **SHAP sonuçları (Bölüm 4.4, s. 4382):** total day minutes SHAP varyansının yaklaşık %22.4'üne
  katkıda bulunuyor (en etkili öznitelik), international plan %17.6, account length %11.2, voice
  mail plan %9.8, customer service calls %7.1 (Rasool vd., 2025, s. 4382).
- **Accuracy'nin yanıltıcılığına dair doğrudan metodolojik açıklama (Bölüm 3.6, s. 4379–4380):**
  yazarlar, dengesiz sınıflandırma görevlerinde yüksek doğruluğun azınlık sınıftaki kötü
  performansı maskeleyebileceğini vurguluyor ve bunun kaynağı olarak Akosa (2017)'yi ikincil
  kaynak gösteriyor — yani bu, Rasool vd. (2025)'in kendi bulgusu değil, Akosa (2017, akt. Rasool
  vd., 2025, s. 4380)'den aktarılan bir genel ilke.
- **Boşluk argümanına destek — istatistiksel karşılaştırma zayıflığı:** Rasool vd. (2025) kendi
  Stacking modelinin önceki çalışmalara (Chang vd. 2024 [18] ve bir başka çalışma [8]) göre
  üstünlüğünü **McNemar gibi eşleştirilmiş bir test yerine, farklı makalelerin F1 skorları arasında
  bağımsız örneklem t-testi (independent sample t-test)** ile sınıyor: "we conducted independent
  sample t-tests comparing our F1-scores with those reported in previous works [1, 8]... p < 0.05"
  (Rasool vd., 2025, Bölüm 5, s. 4383). **Bu, 2025 tarihli bile olsa çok güncel bir çalışmanın dahi
  farklı test setleri üzerinde eğitilmiş modelleri eşleştirilmemiş bir testle karşılaştırdığını,
  yani projenin "aynı test seti üzerinde McNemar + Holm" yaklaşımının metodolojik olarak daha
  sağlam olduğunu gösteren somut, güncel bir örnektir** — boşluk argümanı için güçlü bir kanıt.

**Doğrudan alıntılar:**
> "SMOTE, which is applied exclusively to the training set" (Rasool vd., 2025, s. 4377)

> "Accuracy... may be insufficient on its own in imbalanced settings, where high accuracy can mask
> poor performance on the minority class" (Rasool vd., 2025, s. 4379–4380, Akosa 2017'den aktarılan
> genel ilkeye referansla)

**Tezde kullanım:** Bu, projeye en yakın metodolojik emsal. Yöntem bölümünde "SMOTE'un yalnızca
eğitim setine uygulanması" ilkesinin literatürde standart olduğunu göstermek için (Rasool vd., 2025)
doğrudan kullanılabilir; Giriş/Tartışma'da boşluk argümanını güçlendirmek için "2025 tarihli bir
çalışma dahi model karşılaştırmasını McNemar yerine eşleştirilmemiş t-testle yapıyor" örneği çok
güçlü ve güncel bir kanıt olur.

**Çelişki/ihtiyat:** Hiçbiri tespit edilmedi; bu kaynağın iç tutarlılığı yüksek (Maan & Maan'ın
aksine).

---

### 16. Wang vd. (2024/2025)

**Künye:** Wang, [ön isimler tam metinden alınmadı] vd. (2024/2025). Explainability of highly
associated fuzzy churn patterns in binary classification. arXiv:2410.15827 (sonradan *Expert
Systems*, 2025, https://doi.org/10.1111/exsy.70066 olarak yayımlanmış).
**Doğrulama:** [Öz] (arxiv.org/html/2410.15827v1 üzerinden WebFetch ile içerik parafrazı elde
edildi; yazar tam listesi bu oturumda çıkarılamadı — bkz. Atıf uyarıları).

**Desteklediği bileşenler:** SHAP (sınırlılığı).

**Bulgular:** Makine öğrenmesi, bulanık küme (fuzzy-set) teorisi ve top-k yüksek fayda örüntü
madenciliğini (HUIM) birleştiren "Highly Associated Fuzzy Churn Patterns" (HAFCP) çerçevesi
öneriliyor; amaç, SHAP gibi yöntemlerin tekil öznitelik önemine odaklanmasının ötesine geçip
çok-öznitelikli, yumuşak sınırlı örüntüleri yakalamak (Wang vd., 2024/2025, [Öz]). Yöntem dört
aşamalı: (1) SHAP+XGBoost ile öznitelik önemi, (2) bulanık küme dönüşümü, (3) top-k yüksek fayda
örüntü madenciliği, (4) örüntülerin yeni öznitelik olarak modele eklenmesi (Wang vd., 2024/2025,
[Öz]). Beş gerçek veri seti kullanılıyor (dördü telekom, biri bankacılık/kredi kartı), 3.150–51.047
örnek, 13–57 öznitelik (Wang vd., 2024/2025, [Öz]). **Sonuçlar karışık:** 5. veri setinde büyük
iyileşme (AUC +%10, doğruluk %91.73'e), 4. veri setinde mütevazı iyileşme, 1–3. veri setlerinde
değişken sonuçlar — örüntü etkinliğinin veri setinin özelliklerine bağlı olduğu belirtiliyor (Wang
vd., 2024/2025, [Öz]) — not'taki "sonuçlar karışık" ifadesi teyit edildi.

**Alıntı:** Yok (parafraz).

**Tezde kullanım:** Tartışma/Gelecek Çalışmalar bölümünde, SHAP'in tekil öznitelik sınırlılığına
işaret eden güncel bir çalışma olarak; "SHAP'in çok-değişkenli etkileşimleri tam yakalayamadığı
yönündeki eleştiri güncel literatürde de sürüyor (Wang vd., 2024/2025)" cümlesi kurulabilir.

**Çelişki/ihtiyat:** Yazar tam adları doğrulanamadı; sonuçların "karışık" olması, HAFCP'nin her veri
setinde işe yaramadığını gösteriyor — bu bir sınırlılık, projenin SHAP kullanımına doğrudan bir
tehdit değil ama ihtiyatlı bir dipnot olarak kullanılabilir.

---

## Grup 4 — İstatistiksel kıyaslama & McNemar

### 17. McNemar (1947)

**Künye:** McNemar, Q. (1947). Note on the sampling error of the difference between correlated
proportions or percentages. *Psychometrika*, 12(2), 153–157. https://doi.org/10.1007/BF02295996
**Doğrulama:** [Öz] (arama sonucunda özet metni bulundu; orijinal makalenin PDF'i bu oturumda
açılmadı, bu yüzden aşağıdaki "alıntı" ihtiyatla — tam doğrulanmamış — verilmiştir).

**Desteklediği bileşenler:** McNemar/Holm.

**Bulgular:** Korelasyonlu oranlar arasındaki farkın anlamlılığını değerlendirmek için iki formül
sunuluyor; formüllerden birinin ki-kare eşdeğeri gösteriliyor (McNemar, 1947, [Öz]).

**Olası alıntı (doğrulanmamış, ihtiyatla):** "Two formulas are presented for judging the
significance of the difference between correlated proportions." — bu cümlenin **birebir orijinal
metinden mi yoksa bir özetleme motorundan mı** geldiği bu oturumda kesinleştirilemedi; teze
doğrudan alıntı olarak konmadan önce orijinal Psychometrika makalesinden teyit edilmeli.

**Tezde kullanım:** Yöntem bölümünde McNemar testinin orijinal kaynağı olarak standart atıf;
"McNemar (1947) tarafından önerilen test..." cümlesi.

**Çelişki/ihtiyat:** Tam metin okunmadı; alıntının doğruluğu teyit edilmeli.

---

### 18. Dietterich (1998)

**Künye:** Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised
classification learning algorithms. *Neural Computation*, 10(7), 1895–1923.
https://doi.org/10.1162/089976698300017197
**Doğrulama:** [Öz] (birden fazla bağımsız arama sonucu tutarlı biçimde aynı bilgiyi veriyor; sayfa
aralığı — not'taki "⚠" işareti bu oturumda çözüldü — 1895–1923 olarak doğrulandı).

**Desteklediği bileşenler:** McNemar/Holm — **"neden McNemar" sorusunun doğrudan cevabı**.

**Bulgular:** Makale, bir öğrenme algoritmasının diğerinden üstün olup olmadığını belirlemek için
beş yaklaşık istatistiksel testi inceliyor ve bu testlerin gücünü (farkları tespit etme
yeteneğini) ölçüyor (Dietterich, 1998, [Öz]). Beş testten ikisi — **McNemar testi** ve yeni önerilen
**5×2 çapraz doğrulama t testi** — düşük Tip I hatası ve makul güce sahip bulunuyor (Dietterich,
1998, [Öz]). Dietterich, hesaplama maliyeti nedeniyle algoritmalar yalnızca bir kez
çalıştırılabiliyorsa McNemar testinin, on kez çalıştırılabiliyorsa 5×2cv t testinin kullanılmasını
öneriyor (Dietterich, 1998, [Öz]) — not'taki "tek train/test senaryosunda kabul edilebilir Tip I
hatasına sahip tek test McNemar" bulgusu **tam olarak teyit edildi**.

**Alıntı:** Yok (parafraz; tam metin okunmadı).

**Tezde kullanım:** Yöntem bölümünün en önemli metodolojik dayanağı: "Tek bir train/test bölünmesi
kullanıldığında (bu çalışmadaki gibi), Dietterich (1998) düşük Tip I hataya sahip tek uygun test
olarak McNemar testini önermektedir; bu nedenle model karşılaştırmalarında McNemar testi
kullanılmış, çoklu karşılaştırma için Holm düzeltmesi uygulanmıştır" cümlesi doğrudan kurulabilir.

**Çelişki/ihtiyat:** Tam metin okunmadığı için sayfa-düzeyinde alıntı verilemiyor; ancak bulgu
birden fazla bağımsız kaynakta tutarlı, güven düzeyi yüksek.

---

### 19. Yang, Wang, Song & Li (2023)

**Künye:** Yang, J., Wang, R., Song, Y., & Li, J. (2023). Block-regularized 5×2 cross-validated
McNemar's test for comparing two classification algorithms. arXiv:2304.03990.
**Doğrulama:** [Öz] (arxiv.org üzerinden başlık ve özet doğrulandı).

**Desteklediği bileşenler:** McNemar/Holm — sınırlılık/güç tartışması.

**Bulgular:** 5×2 blok-düzenlenmiş çapraz doğrulamanın (BCV), eğitim setleri arasındaki örtüşen
kayıt sayısını düzenleyerek yüksek kaliteli bir hata oranı kestiricisi ürettiği için algoritma
karşılaştırmasında diğer CV yöntemlerinden üstün olduğu gösteriliyor; yazarlar 5×2 BCV'deki 10
korelasyonlu olabilirlik tablosunu tek bir etkin tabloya sıkıştırıp bunun üzerinden bir McNemar
testi tanımlıyor ve bunun makul Tip I hatası ile umut verici güce sahip olduğunu simülasyon ve
gerçek veri setlerinde gösteriyor (Yang vd., 2023, [Öz]) — not'taki "standart McNemar'ın gücü
düşük" bulgusu, bu makalenin **standart (tek bölünmeli) McNemar'ın sınırlı gücünü aşmak için** yeni
bir yöntem önerdiği bağlamında teyit edildi.

**Alıntı:** Yok (parafraz).

**Tezde kullanım:** Tartışma/Sınırlılıklar bölümünde: "Bu çalışmada tek bir train/test bölünmesi
üzerinde standart McNemar testi kullanılmıştır; Yang vd. (2023) standart McNemar testinin
istatistiksel gücünün sınırlı olabileceğini ve 5×2 blok-düzenlenmiş çapraz doğrulamalı bir
varyantın daha güçlü olduğunu göstermektedir; gelecek çalışmalarda bu yaklaşım benimsenebilir"
cümlesi kurulabilir.

**Çelişki/ihtiyat:** Bu kaynak, projenin kullandığı standart (tek-bölünmeli) McNemar yaklaşımının
bir sınırlılığına işaret ediyor — Dietterich (1998)'in önerisiyle (hesaplama kısıtı varsa McNemar
kullan) çelişmiyor ama onu nüanslıyor; tez savunmasında bu nüans önceden hazırlanmalı.

---

### 20. IJRASET (2022) — KÜNYE DOĞRULANAMADI, YÜKSEK RİSK

**Künye (not'tan):** IJRASET, 10(12), 2022 [yazar bilinmiyor].
**Doğrulama:** **Doğrulanamadı.** Bu oturumda "IJRASET" + "Volume 10" + "Issue XII (Aralık 2022)" +
churn araması, tam olarak eşleşen bir makale **bulamadı**. En yakın aday: "Customer Churn
Prediction in Banking Sector - A Hybrid Approach", IJRASET, **Cilt 10, Sayı XI (Kasım 2022)** —
sayı numarası (XI, not'taki 12 değil) ve DOI (10.22214/ijraset.2022.47541) farklı. Aralık 2022
(Sayı XII) altında bulunan churn-ilişkili makaleler ("Review on Latest Trends and Techniques in
Predictive Analytics", "Medical Insurance Cost Prediction Using Machine Learning") da churn'ün
kendisi değil, dolaylı/farklı konular.

**Desteklediği bileşenler:** boşluk argümanı (düşük prestijli dergi örneği olarak, not'ta da böyle
işaretlenmiş).

**Öneri:** Bu kaynağın teze **hiç girmemesi** önerilir; düşük prestijli bir dergiden, kesin
künyesi doğrulanamayan bir kaynak, hakem sürecinden geçmiş güçlü kaynaklar dururken risk taşır.
Eğer mutlaka kullanılacaksa, önce orijinal brief'teki (`proje-brief-v2.md`) tam DOI/URL bulunup
doğrulanmalı.

**Alıntı:** Yok.

**Tezde kullanım:** Önerilmez; kullanılacaksa yalnızca "düşük prestijli dergilerde de churn
literatürünün yaygın olduğu" notu için, Dietterich (1998)'e **ikincil destek olarak değil**,
bağımsız bir dipnot olarak.

**Çelişki/ihtiyat:** **[Atıf uyarısı — en yüksek risk seviyesi]**, aşağıda tekrar listelenmiştir.

---

### 21. Duchemin & Matheus (2021)

**Künye:** Duchemin, R., & Matheus, R. (2021). Forecasting customer churn: Comparing the
performance of statistical methods on more than just accuracy. *Journal of Supply Chain
Management Science*, 2(3–4). (TU Delft OJS, açık erişim.)
**Doğrulama:** [Öz] (journals.open.tudelft.nl üzerinden WebFetch; makale gövdesi büyük ölçüde
sıkıştırılmış PDF akışı olduğu için yalnızca özet düzeyinde parafraz elde edildi).

**Desteklediği bileşenler:** maliyet/eşik (dolaylı — çoklu metrik değerlendirme felsefesi), boşluk
argümanı.

**ÖNEMLİ DÜZELTME:** Not'ta "Brief'teki 'Duchemin 2020, Erasmus tezi' bilgisi hatalı görünüyor"
deniyordu. Bu oturumda **doğrulandı**: kaynak bir Erasmus tezi değil, **2021 tarihli, TU Delft'in
açık erişimli *Journal of Supply Chain Management Science* dergisinde yayımlanmış bir makale**.
Ronan Duchemin, Rotterdam School of Management (Erasmus Üniversitesi) ile ilişkili; Ricardo Matheus
ise TU Delft, Faculty of Technology, Policy and Management ile ilişkili — derginin "TU Delft"
etiketi buradan geliyor. Yani hem "Erasmus" hem "TU Delft" bağlantısı doğru ama "Erasmus tezi"
çerçevesi yanlış; doğrusu "TU Delft dergisinde yayımlanmış, bir yazarı Erasmus'ta çalışan makale."

**Bulgular:** Çalışma, churn tahmini için istatistiksel yöntemlerin performansını yalnızca doğruluk
ve AUC ile değil, daha geniş bir metrik portföyüyle değerlendiriyor (Duchemin & Matheus, 2021,
[Öz]). En-İyi-En-Kötü Yöntemi (Best-Worst Method, BWM) performans metriklerinin göreli önemini
belirlemek için kullanılıyor; metriklerin göreli önemine ve yöntemlerin performansına dayanarak
toplamsal değer fonksiyonuyla her yöntem için genel bir değer hesaplanıyor (Duchemin & Matheus,
2021, [Öz]). Deneysel analiz, kapsamlı metrik portföyleriyle sıralama yapmanın, yalnızca doğruluk
veya AUC gibi tek bir metriğe dayanan değerlendirmelerden **önemli ölçüde farklı sonuçlar**
verdiğini ortaya koyuyor (Duchemin & Matheus, 2021, [Öz]) — not'taki "çok metrikli değerlendirme,
BWM ile" bulgusu teyit edildi.

**Alıntı:** Yok (parafraz).

**Tezde kullanım:** Tartışma bölümünde, projenin "tek metrik yerine çoklu metrik + maliyete duyarlı
eşik" yaklaşımını destekleyen bir emsal olarak: "Duchemin ve Matheus (2021), churn tahmininde
yöntemlerin yalnızca doğruluk/AUC ile değil çok-kriterli bir çerçeveyle değerlendirilmesi
gerektiğini, aksi halde sıralamaların önemli ölçüde değişebileceğini göstermiştir" cümlesi
kurulabilir. Bu, projenin maliyete duyarlı eşik seçimi kararına dolaylı metodolojik destektir
(BWM'nin kendisi kullanılmıyor ama "tek metrik yetersiz" felsefesi ortak).

**Çelişki/ihtiyat:** BWM bir çok-kriterli karar verme (MCDM) yöntemidir, istatistiksel anlamlılık
testi değildir — bu kaynak McNemar/Holm'e alternatif değil, tamamlayıcı bir felsefe örneğidir; teze
yazılırken bu ayrım netleştirilmeli.

---

## Sentez (güncellenmiş)

- **RF/XGB üstünlüğü:** RF ve XGB, LR'yi tutarlı biçimde geçiyor (Zhu, 2024, [Öz]; Ashraf, 2024,
  [Öz]; Kaynar vd., 2017, [TM] — burada YSA en iyi ama DVM/SVM en kötü); yaş tekrar tekrar en
  önemli değişken olarak çıkıyor (Zhu, 2024, [Öz]; Rasool vd., 2025'te ise en önemli değişken "total
  day minutes", yaş değil — **[TM]**, s. 4382; bu bir çelişki değil, farklı veri setleri/sektörler
  arası doğal bir fark: bankacılıkta yaş, telekomda kullanım yoğunluğu öne çıkabiliyor).
- **Dengesizlik:** Accuracy tek başına yanıltıcı (Burez & Van den Poel, 2009, [Öz]; Rasool vd.,
  2025'teki %94.75 doğruluk / %73.20 recall, [TM], s. 4381; **ayrıca yeni bulgu:** Maan & Maan,
  2023'te %96.2 doğruluk fakat Tablo 1'e göre churn sınıfı recall'ı yalnızca %22, [TM], s. 36 —
  **ikinci somut sayısal örnek**). SMOTE recall ve AUC'yi genelde iyileştiriyor ama en iyi
  kombinasyon veri setine göre değişiyor (Amin vd., 2016'da MTDF SMOTE'u geçiyor, [Öz]).
- **SMOTE-NC:** Chawla vd. (2002)'nin orijinal makalesinde tanımlanmış ve test edilmiş ([Öz]) —
  projenin SMOTE-NC gerekçesi doğrudan bu kaynağa dayandırılabilir.
- **SHAP:** Churn çalışmalarında standart hale gelmiş (Guliyev & Yerdelen Tatoğlu, 2021, [Öz];
  Maan & Maan, 2023, [TM]; Rasool vd., 2025, [TM]). Tekil özellik sınırı var ve güncel çalışmalar
  bunu aşmaya çalışıyor, karışık sonuçlarla (Wang vd., 2024/2025, [Öz]).
- **McNemar:** Dietterich (1998, [Öz]) McNemar'ı tek-bölünmeli senaryoda gerekçelendiriyor; Yang
  vd. (2023, [Öz]) standart McNemar'ın gücünün sınırlı olduğunu ve 5×2 BCV varyantının daha güçlü
  olduğunu gösteriyor — bu bir sınırlılık notu olarak kullanılmalı.
- **Boşluk (güçlendirilmiş):** Taranan churn çalışmalarının hiçbiri McNemar + çoklu test düzeltmesi
  + maliyete duyarlı eşiği birlikte uygulamıyor. **En güçlü kanıt:** Rasool vd. (2025, [TM]) — 2025
  tarihli, SHAP+ensemble+SMOTE kullanan güncel bir çalışma bile, model karşılaştırmasını
  eşleştirilmemiş bir t-testle yapıyor, McNemar kullanmıyor (s. 4383). Katkı iddiası bu ölçekte
  (mütevazı ama somut) kurulmalı.

---

## Matris: Bileşen × Kaynak Numarası

| Bileşen | Destekleyen kaynaklar (numara) |
|---|---|
| Veri seti | 3 (olası çakışma, doğrulanmalı) |
| Modeller (RF/XGB/ensemble karşılaştırması) | 1, 2, 3, 4, 5, 6, 13, 14 |
| Dengesizlik / SMOTE(-NC) | 1, 4, 7, 8, 9, 10, 11, 15 |
| Sızıntı önleme (yalnız eğitim setine SMOTE) | 7, 15 |
| SHAP / açıklanabilirlik | 1, 12, 13, 14, 15, 16 |
| McNemar / Holm | 17, 18, 19, 20 |
| Maliyet / eşik (dolaylı, çok-metrik felsefesi) | 9, 15, 21 |
| Boşluk argümanı (istatistiksel test eksikliği) | 2, 6, 9, 15, 18, 19, 20 |
| Dashboard | *(taranan 21 kaynakta doğrudan dashboard/Streamlit kaynağı yok — bu bileşen için literatür boşluğu var, ayrı aranmalı)* |

## Bölüm başına en güçlü 3–5 kaynak

- **Giriş (motivasyon + boşluk):** 9 (Burez & Van den Poel, 2009 — accuracy yanıltıcı), 15
  (Rasool vd., 2025 — güncel çalışma bile McNemar kullanmıyor), 18 (Dietterich, 1998 — neden
  McNemar), 2 (ICBASE 2025 — 11 model, test yok), 21 (Duchemin & Matheus, 2021 — çok-metrik
  gerekliliği).
- **Literatür:** 1 (Tékouabou vd., 2022 — ana benchmark), 15 (Rasool vd., 2025 — en kapsamlı
  metodolojik emsal), 13 (Guliyev & Yerdelen Tatoğlu, 2021 — bankacılık+SHAP), 3 (Zhu, 2024 —
  olası aynı veri seti), 14 (Maan & Maan, 2023 — doğrulanmış sayısal emsal).
- **Yöntem:** 7 (Chawla vd., 2002 — SMOTE-NC kökeni), 15 (Rasool vd., 2025 — eğitim-seti-yalnız
  SMOTE uygulaması), 18 (Dietterich, 1998 — McNemar gerekçesi), 19 (Yang vd., 2023 — sınırlılık),
  12 (Lundberg & Lee, 2017 — SHAP temeli).
- **Bulgular:** 15 (Rasool vd., 2025 — doğrudan karşılaştırılabilir sayılar), 14 (Maan & Maan,
  2023 — XGBoost sayıları + accuracy/recall çelişkisi örneği), 6 (Kaynar vd., 2017 — çok-metrik
  tablo formatı emsali), 9 (Burez & Van den Poel, 2009 — AUC tercihi gerekçesi).
- **Tartışma:** 15 (Rasool vd., 2025 — boşluk kanıtı + sınırlılıklar), 16 (Wang vd., 2024/2025 —
  SHAP'in sınırı), 19 (Yang vd., 2023 — McNemar'ın gücü), 21 (Duchemin & Matheus, 2021 — çok
  kriterli değerlendirme), 4 (Ashraf, 2024 — gerçek veri vs. Kaggle verisi karşılaştırması).

---

## Atıf uyarıları (teze girmeden önce mutlaka teyit edilmeli)

1. **Kaynak 20 (IJRASET 2022, 10/12):** Tam künye bu oturumda **doğrulanamadı**. En yakın aday
   (Cilt 10, Sayı **XI**, Kasım 2022, "Customer Churn Prediction in Banking Sector - A Hybrid
   Approach") sayı numarası bakımından not'taki "10(12)" ile uyuşmuyor. **Önerilen aksiyon:**
   orijinal `proje-brief-v2.md` dosyasından tam DOI/URL bulunmalı; bulunamazsa bu kaynak tezden
   çıkarılmalı.
2. **Kaynak 3 (Zhu, 2024):** "ABC Bank" veri setinin, projenin kendi Kaggle veri setiyle (ABC
   Multinational Bank / Gaurav Topre) aynı olup olmadığı **doğrulanamadı**. Yüksek önem taşıyor —
   eğer aynıysa doğrudan karşılaştırma yapılabilir, aksi halde yanlış bir "aynı veri seti" iddiası
   riski var.
3. **Kaynak 12 (Lundberg & Lee, 2017):** Sayfa aralığı kaynaklar arası tutarsız (4765–4774 vs
   4768–4777). Teze koymadan önce proceedings.neurips.cc üzerinden kesinleştirilmeli.
4. **Kaynak 4 (Ashraf, 2024):** Sayfa aralığı kaynaklar arası tutarsız (1–5 vs 65–109). SCIENPRESS
   orijinal dergi sayfasından teyit edilmeli.
5. **Kaynak 5 (Karamollaoğlu vd., 2021):** İlk aramada ortaya çıkan "RF+SMOTE-ENN, F1 %95" rakamı
   **ikinci bir doğrudan kaynak taramasında (avesis sayfası) teyit edilemedi** — bu rakam muhtemelen
   arama motorunun halüsinasyonu; **kullanılmamalı**.
6. **Kaynak 11 (Boonmunewai vd., 2024):** Künye doğrulandı ama makale içeriğine (dolayısıyla
   not'taki recall %92.99 / F %91.37 / AUC %96.4 rakamlarına) bu oturumda ulaşılamadı — **[nottan]**
   düzeyinde kalıyor, tam metin bulunmalı.
7. **Kaynak 16 (Wang vd., 2024/2025):** Yazarların tam ad listesi bu oturumda elde edilemedi;
   yalnızca "Wang vd." olarak bırakıldı — teze koymadan önce tam yazar listesi (arXiv sayfasından)
   alınmalı.
8. **Kaynak 17 (McNemar, 1947):** Verilen "abstract" cümlesinin birebir orijinal metin mi yoksa bir
   arama motoru özetlemesi mi olduğu kesinleştirilemedi; doğrudan alıntı olarak kullanılmadan önce
   Psychometrika'daki orijinal sayfadan teyit edilmeli.
9. **Kaynak 14 (Maan & Maan, 2023) iç tutarsızlığı:** Makalenin kendi Tablo 1'i ile düz metindeki
   recall/precision/AUC rakamları birbiriyle çelişiyor (bkz. yukarıda). Tezde yalnızca Tablo 2'deki
   doğruluk/F1 rakamları (doğrulanmış, tutarlı) kullanılmalı; recall/AUC rakamları kullanılmamalı.
10. **Kaynak 8 (Amin vd., 2016) ve Kaynak 10 (Haddadi vd., 2024):** Not'taki çerçeveleme
    ("SMOTE vs ADASYN", "iki aşamalı resampling AUC'de daha iyi") bu oturumda tam olarak
    doğrulanamadı; tam metinlerden teyit edilmeli.
11. **Genel not:** [Öz] düzeyindeki tüm bulgular, WebSearch/Semantic Scholar API'sinin **parafraz**
    çıktılarına dayanır — bunlar kaynağın orijinal İngilizce metninin birebir kopyası değildir.
    Yalnızca **[TM]** işaretli üç kaynakta (6 – Kaynar vd. 2017; 14 – Maan & Maan 2023; 15 – Rasool
    vd. 2025) doğrudan alıntı ve kesin sayfa numarası verilmiştir.
