# Unsupervised Learning, Recommenders, Reinforcement Learning: Ders Notları

> Ders notları (Hafta 1–4), tek dosyada ve **mantığı anlaşılacak** şekilde özetlenmiştir.
> Her bölümde "ne?" kadar **"neden?"** sorusuna da cevap verilir. Tekrar bakarken önce bölüm sonlarındaki **🧠 Neden?** kutularını oku.
>
> **Kaynak:** [Unsupervised Learning, Recommenders, Reinforcement Learning](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning) (DeepLearning.AI & Stanford Online, Andrew Ng). Machine Learning Specialization'ın 3. ve son kursu.
> **Önkoşul:** Önceki iki kursun notları (cost function, gradient descent, logistic regression, sinir ağları, TensorFlow, F1/precision/recall). Burada sürekli referans verilecek.
>
> ℹ️ **Çeviri notu:** Ders transkripti otomatik çevrildiği için bazı terimler yanlıştı (örn. "albüm" = *algoritma*, "işbirlikçi filtreleme" = *collaborative filtering*, "not verme inişi" = *gradient descent*). Bu notlarda doğru terimler kullanıldı; karşılıklar için [Terim sözlüğü](#46-terim-sözlüğü) bölümüne bak.

## İçindekiler

**Hafta 1 – Kümeleme ve anomali tespiti**
1. [Denetimsiz öğrenme ve kümeleme](#1-denetimsiz-öğrenme-ve-kümeleme)
2. [K-means: sezgi](#2-k-means-sezgi)
3. [K-means algoritması](#3-k-means-algoritması)
4. [K-means'in optimizasyon hedefi](#4-k-meansin-optimizasyon-hedefi)
5. [K-means: başlatma ve yerel minimumlar](#5-k-means-başlatma-ve-yerel-minimumlar)
6. [K-means: K'yı seçmek](#6-k-means-kyı-seçmek)
7. [Anomali tespitine giriş](#7-anomali-tespitine-giriş)
8. [Gauss (normal) dağılımı](#8-gauss-normal-dağılımı)
9. [Anomali tespit algoritması](#9-anomali-tespit-algoritması)
10. [Anomali tespit sistemi geliştirmek](#10-anomali-tespit-sistemi-geliştirmek)
11. [Anomali tespiti mi, denetimli öğrenme mi?](#11-anomali-tespiti-mi-denetimli-öğrenme-mi)
12. [Anomali tespitinde özellik seçimi](#12-anomali-tespitinde-özellik-seçimi)

**Hafta 2 – Öneri sistemleri**
13. [Öneri sistemlerine giriş](#13-öneri-sistemlerine-giriş)
14. [Özellikler biliniyorsa: lineer regresyon](#14-özellikler-biliniyorsa-lineer-regresyon)
15. [Collaborative filtering: özellikleri **öğrenmek**](#15-collaborative-filtering-özellikleri-öğrenmek)
16. [Binary (ikili) etiketler: beğeni, tıklama, izleme](#16-binary-ikili-etiketler-beğeni-tıklama-izleme)
17. [Mean normalization (ortalama normalleştirme)](#17-mean-normalization-ortalama-normalleştirme)
18. [TensorFlow ile collaborative filtering](#18-tensorflow-ile-collaborative-filtering)
19. [Benzer öğeleri bulma ve sınırlamalar](#19-benzer-öğeleri-bulma-ve-sınırlamalar)
20. [Content-based (içerik tabanlı) filtreleme](#20-content-based-içerik-tabanlı-filtreleme)
21. [Derin öğrenme ile içerik tabanlı filtreleme](#21-derin-öğrenme-ile-içerik-tabanlı-filtreleme)
22. [Büyük katalog: retrieval ve ranking](#22-büyük-katalog-retrieval-ve-ranking)
23. [Öneri sistemlerinin etik yönü](#23-öneri-sistemlerinin-etik-yönü)

**Hafta 3 – PCA (boyut indirgeme)**
24. [PCA sezgisi](#24-pca-sezgisi)
25. [PCA nasıl çalışır?](#25-pca-nasıl-çalışır)
26. [scikit-learn ile PCA](#26-scikit-learn-ile-pca)

**Hafta 4 – Pekiştirmeli öğrenme**
27. [Pekiştirmeli öğrenme nedir?](#27-pekiştirmeli-öğrenme-nedir)
28. [Mars gezgini örneği: durum, eylem, ödül](#28-mars-gezgini-örneği-durum-eylem-ödül)
29. [Getiri (return) ve indirim faktörü (γ)](#29-getiri-return-ve-indirim-faktörü-gamma)
30. [Politika (π) ve RL'nin amacı](#30-politika-pi-ve-rlnin-amacı)
31. [Markov Karar Süreci (MDP) formalizmi](#31-markov-karar-süreci-mdp-formalizmi)
32. [Durum-eylem değer fonksiyonu Q(s,a)](#32-durum-eylem-değer-fonksiyonu-qsa)
33. [Bellman denklemi](#33-bellman-denklemi)
34. [Rastgele (stokastik) ortamlar](#34-rastgele-stokastik-ortamlar)
35. [Sürekli durum uzayları](#35-sürekli-durum-uzayları)
36. [Ay iniş aracı (Lunar Lander) örneği](#36-ay-iniş-aracı-lunar-lander-örneği)
37. [DQN: Q fonksiyonunu sinir ağıyla öğrenme](#37-dqn-q-fonksiyonunu-sinir-ağıyla-öğrenme)
38. [Daha verimli mimari ve ε-greedy](#38-daha-verimli-mimari-ve-epsilon-greedy)
39. [Mini-batch ve soft update](#39-mini-batch-ve-soft-update)
40. [RL'nin bugünkü durumu](#40-rlnin-bugünkü-durumu)
41. [Uzmanlığın sonu: büyük resim](#41-uzmanlığın-sonu-büyük-resim)

**Ekler**
42. [Büyük resim: tüm hikâye tek akışta](#42-büyük-resim-tüm-hikâye-tek-akışta)
43. [Hızlı formül kartı](#43-hızlı-formül-kartı)
44. [Sorun → çözüm rehberi](#44-sorun--çözüm-rehberi)
45. [Sık yapılan hatalar ve karıştırılan kavramlar](#45-sık-yapılan-hatalar-ve-karıştırılan-kavramlar)
46. [Terim sözlüğü](#46-terim-sözlüğü)
47. [Tam Python kodu](#47-tam-python-kodu)

**Tekrar ve pratik**
48. [Elle hesap örnekleri](#48-elle-hesap-örnekleri)
49. [Kendini test et](#49-kendini-test-et)
50. [Tekrar planı](#50-tekrar-planı)

---

# HAFTA 1 – KÜMELEME VE ANOMALİ TESPİTİ

## 1. Denetimsiz öğrenme ve kümeleme

- **Denetimsiz öğrenmede** etiket ($y$) yoktur, sadece $x$ vardır. Amaç: veride **ilginç bir yapı** bulmak (gruplar, sıradışı örnekler...).
- **Kümeleme (clustering):** Benzer veri noktalarını **gruplara** ayırır. Hangi grubun ne olduğunu **biz söylemeyiz**, algoritma bulur.

| Uygulama | Ne kümelenir? |
|---|---|
| Haber gruplama | Aynı konuyu işleyen haberler |
| Pazar segmentasyonu | Benzer ihtiyaçlı kullanıcılar/müşteriler |
| DNA analizi | Benzer gen ifadesi gösteren bireyler |
| Astronomi | Birbirine yakın gökadalar / yıldızlar |

> 🧠 **Neden?** Etiketlemek pahalıdır, bazen **kaç grup olduğunu** bile bilmeyiz. Denetimsiz öğrenme "veriyi önce anlama" aşamasında çok işe yarar.

---

## 2. K-means: sezgi

**K-means**, en yaygın kümeleme algoritmasıdır. Hedef: $K$ küme bulmak.

Örnek: 30 etiketsiz nokta, $K=2$. Algoritma iki adımı **tekrar tekrar** yapar:

1. **Her noktayı en yakın küme merkezine ata.**
2. **Her küme merkezini, ona atanmış noktaların ortalamasına taşı.**

```
Rastgele iki merkez seç (kırmızı ×, mavi ×)
      ▼
Her noktayı yakın olan merkeze boya (kırmızı / mavi)
      ▼
Merkezleri kendi renkli noktalarının ortalamasına taşı
      ▼
Tekrar boya → tekrar taşı → ... → artık değişmeyene kadar
```

- **Cluster centroid** (küme merkezi): kümenin ortası. $\mu_1, \mu_2, \dots, \mu_K$.
- Birkaç turdan sonra atamalar ve merkezler **değişmez** olur → **yakınsadı**.

> 🧠 **Neden işe yarar?** Merkez taşınınca bazı noktalar **diğer merkeze daha yakın** hale gelir; atamalar değişir; yeni ortalama yine biraz kayar. Bu "birbirini düzeltme" döngüsü kümeleri yerine oturtur.

---

## 3. K-means algoritması

**Notasyon**
- $K$: küme sayısı, $\mu_k$: $k$. kümenin merkezi ($n$ boyutlu vektör, $x$ ile aynı boyutta).
- $c^{(i)}$: $x^{(i)}$ noktasının atandığı kümenin **indeksi** ($1 \dots K$).
- $\mu_{c^{(i)}}$: $x^{(i)}$'nin atandığı kümenin merkezi.

**Algoritma**

```
1. K küme merkezi μ1 ... μK'yı RASTGELE başlat

2. Yakınsayana kadar tekrarla:
     (a) ATAMA:   i = 1..m için   c(i) = en yakın merkezin indeksi
                       c(i) = argmin_k || x(i) − μk ||²
     (b) TAŞIMA:  k = 1..K için   μk = kümeye atanmış noktaların ORTALAMASI
```

- Mesafe için karesini almak, minimumu **değiştirmez**; yaygın kullanım karedir.
- **Boş küme sorunu:** Bir kümeye hiç nokta atanmazsa ortalama $0/0$ olur. Çözüm: o kümeyi **sil** ($K-1$ küme ile devam) veya merkezi **yeniden rastgele başlat**.
- K-means, kümeler **net ayrık görünmese** de işe yarar (örn. tişört bedenleri: S/M/L gibi sürekli bir veriyi gruplara bölmek).

---

## 4. K-means'in optimizasyon hedefi

K-means de aslında bir **cost function** minimize eder (önceki kurslardaki gibi):

$$J(c^{(1)},\dots,c^{(m)},\mu_1,\dots,\mu_K) = \frac{1}{m}\sum_{i=1}^{m}\left\| x^{(i)} - \mu_{c^{(i)}} \right\|^2$$

Her noktanın **kendi küme merkezine uzaklığının karesinin ortalaması**. Bu cost'a **distortion function** da denir.

K-means iki adımda şunu yapar:

| Adım | Neyi minimize eder? | Neyi sabit tutar? |
|---|---|---|
| **Atama** | $J$'yi $c^{(i)}$'lere göre | $\mu_k$'lar sabit |
| **Taşıma** | $J$'yi $\mu_k$'lara göre | $c^{(i)}$'ler sabit |

(Her kümenin ortalaması, o kümedeki noktalara olan kare uzaklıkların toplamını **minimize eden** noktadır.)

**Sonuç:** Her iterasyonda $J$ **ya düşer ya aynı kalır**, **asla artmaz**. Bu yüzden K-means **yakınsar**.

> 🧠 **Debug ipucu:** $J$ bir iterasyonda **artıyorsa**, kodunda **bug** vardır. (Gradient descent'teki learning curve ile aynı mantık.) Ayrıca $J$ yataylaşırsa algoritma yakınsamıştır.

---

## 5. K-means: başlatma ve yerel minimumlar

**Standart başlatma:** $K$ tane **eğitim örneğini rastgele seç**, merkezleri onlara eşitle ($K \le m$ olmalı).

**Sorun:** Rastgele başlatmaya göre K-means **farklı sonuçlara** yakınsayabilir. Bazıları **kötü yerel minimum**dur (örn. iki merkez aynı büyük kümeye düşer, başka küme yarı yarıya bölünür).

**Çözüm: Birçok kez başlat, en iyiyi seç**

```
100 kez tekrarla:                          # genelde 50 - 1000 arası
    1. Rastgele başlat (K tane eğitim örneği seç)
    2. K-means'i çalıştır  →  c, μ
    3. J(c, μ) hesapla
En DÜŞÜK J'yi veren sonucu seç             # (kodu: Bölüm 47, best_of_many)
```

> 🧠 **Neden?** $J$ "tek vadili kase" değildir; farklı başlangıçlar farklı vadilere iner. Çoklu başlatma, **iyi vadiyi bulma şansını** artırır. (Küçük $K$ için fark çok olabilir; büyük veride 50–100 deneme yeterli olur.)

---

## 6. K-means: $K$'yı seçmek

Çoğu kümeleme probleminde doğru $K$ **belirsizdir**. Aynı veriye kimi 2 küme, kimi 4 küme görür, ikisi de haklı olabilir. Çünkü **etiket (doğru cevap) yoktur**.

### Dirsek yöntemi (elbow method)

$K$'nın farklı değerleri için K-means çalıştır, **$J$'yi $K$'ya göre çiz**.
- $K$ arttıkça $J$ **düşer** (daha çok merkez = noktalar merkeze daha yakın).
- Eğri bir noktaya kadar **hızlı** düşüp sonra **yavaşlıyorsa** o nokta "**dirsek**" ($K=3$ gibi).

**Uyarılar:**
- Çoğu gerçek veride net bir dirsek **çıkmaz**.
- **$J$'yi minimize edecek $K$'yı seçme**: $K=m$ olunca $J=0$ olur ama anlamsız.

### Daha iyi yol: Amaca göre seç

K-means çoğu zaman **sonraki bir amaç için** kullanılır. $K$'yı o amaca göre seç.

**Örnek (tişört bedenleri):** $K=3$ (S, M, L) → daha az çeşit, daha ucuz üretim, ama daha kötü uyum. $K=5$ (XS, S, M, L, XL) → daha iyi uyum, ama daha pahalı. Karar **iş hedefine** göre verilir.

> 🧠 **Neden?** Denetimsiz öğrenmede "doğru K" yoktur; **faydalı K** vardır.

---

## 7. Anomali tespitine giriş

**Anomali tespiti:** Etiketsiz **normal** örneklerden öğrenip, **olağandışı** olanı bulmak.

**Örnek: uçak motoru kalite kontrolü.** Her motor için özellikler: $x_1$ = üretilen ısı, $x_2$ = titreşim şiddeti. Yeni bir motor, eğitim setindeki motorlara **benziyorsa normal**, **çok farklıysa şüpheli** (daha sıkı kontrol et).

### Yoğunluk tahmini (density estimation)

1. Normal örneklerden $p(x)$ modelini öğren ("bu $x$ ne kadar olası?").
2. Yeni $x_{test}$ için:

$$p(x_{test}) < \varepsilon \;\Rightarrow\; \text{ANOMALİ} \qquad p(x_{test}) \ge \varepsilon \;\Rightarrow\; \text{normal}$$

- Normal örneklerin yoğun olduğu bölgede $p(x)$ **yüksek**, uzakta **düşük**.
- $\varepsilon$ **küçük bir eşik** (sen seçersin).

**Uygulamalar**

| Alan | Neyi yakalar? |
|---|---|
| **Sahtekârlık tespiti** | Kullanıcı özellikleri: giriş sıklığı, işlem sayısı, yazma hızı... → olağandışı davranış |
| **Üretim** | Normal olmayan motor/telefon/devre kartı |
| **Bilgisayar/veri merkezi izleme** | Bellek kullanımı, disk erişimi, CPU yükü → bozulan makine |

> 🧠 **Neden?** Anomaliler **nadir ve çeşitli**dir; hepsinin etiketli örneğini toplayamazsın. Bu yüzden **"normalin nasıl göründüğünü" öğrenip sapmaları** yakalarız.

---

## 8. Gauss (normal) dağılımı

Anomali tespiti için $p(x)$'i modellemek gerekir. Bunun için **Gauss dağılımı** kullanılır.

$$p(x;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi}\,\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

| Parametre | Anlamı |
|---|---|
| $\mu$ (mü) | **Ortalama**: çanın merkezi |
| $\sigma$ (sigma) | **Standart sapma**: çanın genişliği |
| $\sigma^2$ | **Varyans** |

- Eğrinin **altındaki alan = 1** (toplam olasılık). Bu yüzden **dar** eğri **yüksek** (uzun), **geniş** eğri **alçak** olur.

### Parametreleri veriden tahmin etme

$m$ örnek $x^{(1)},\dots,x^{(m)}$ için:

$$\mu = \frac{1}{m}\sum_{i=1}^{m} x^{(i)} \qquad \sigma^2 = \frac{1}{m}\sum_{i=1}^{m}\left(x^{(i)}-\mu\right)^2$$

(Bunlar maximum likelihood tahminleridir; bazı yerlerde $m$ yerine $m-1$ görürsün, pratikte fark ihmal edilebilir.)

> 🧠 **Neden Gauss?** Basit, iki parametreli ve birçok gerçek özellik zaten yaklaşık çan şeklindedir. Değilse, dönüştürürüz (Bölüm 12).

---

## 9. Anomali tespit algoritması

$n$ özellik $\vec{x}=[x_1,\dots,x_n]$ için, **her özelliği ayrı** Gauss ile modelle ve **olasılıkları çarp**:

$$p(\vec{x}) = \prod_{j=1}^{n} p(x_j;\mu_j,\sigma_j^2)$$

**Algoritma**
1. $n$ özellik seç (anomalide **olağandışı** değer alabilecek olanlar).
2. Eğitim setinden **her özellik için** $\mu_j, \sigma_j^2$ hesapla.
3. Yeni $\vec{x}$ için $p(\vec{x})$ hesapla.
4. $p(\vec{x}) < \varepsilon$ ise **anomali** işaretle.

**Sezgi:** Çarpım kullanıldığı için, **herhangi bir özellik** çok olağandışıysa ($p(x_j)$ küçük), çarpım küçülür ve anomali tespit edilir.

> 🧠 **Neden özellikler bağımsız varsayılıyor?** Çarpma kuralı özelliklerin bağımsız olduğunu varsayar. Gerçekte tam bağımsız olmasalar bile algoritma **pratikte iyi çalışır**.

**Sayısal örnek ($\varepsilon=0.02$):**
- $\vec{x}_{test,1}$: $p \approx 0.4 \;\ge\; 0.02$ → **normal**.
- $\vec{x}_{test,2}$: $p \approx 0.0021 \;<\; 0.02$ → **anomali** (çok küçük).

---

## 10. Anomali tespit sistemi geliştirmek

Bir sorun: **$\varepsilon$ ve özellikleri nasıl seçeriz?** Tek bir sayıyla (real-number evaluation) değerlendirme gerekir. Küçük bir miktar **etiketli** veri işe yarar.

### Etiketli veri kullanımı

- $y=0$: normal, $y=1$: anomali.
- **Eğitim seti:** etiketsiz sayılır (çoğunlukla normal). İçine birkaç anomali karışsa **sorun değil**.
- **CV ve test setleri:** etiketli ($y=0/1$).

**Örnek (uçak motorları):** 10.000 iyi, 20 kusurlu motor.

| Küme | İyi (y=0) | Anomali (y=1) |
|---|---|---|
| Eğitim | **6.000** | (karışabilir) |
| Cross-validation | **2.000** | **10** |
| Test | **2.000** | **10** |

### Süreç

1. Eğitim setinde $p(x)$'i öğren ($\mu_j,\sigma_j^2$).
2. CV setinde farklı $\varepsilon$ dene, **anomalileri** ($y=1$) ne kadar iyi yakaladığına bak.
3. Çok **çarpık** veri olduğu için ölçü: **precision, recall, F1** (accuracy değil, Önceki kurs Hafta 3). Tüm $\varepsilon$ değerleri için en yüksek **F1**'i seç.
4. Test setinde son kez değerlendir.

**Çok az anomali varsa** (örn. 2–3): test seti **atlanabilir**, sadece train + CV kullanılır. Ama bu durumda CV'ye aşırı uyma riski artar.

> 🧠 **Neden etiketli veri "ayar" için?** Anomali tespiti **denetimsiz** modelle çalışır (sadece normal veriden $p(x)$ öğrenir), ama **$\varepsilon$ ve özellik seçimini** etiketli CV setiyle ayarlamak daha doğru sonuç verir.

---

## 11. Anomali tespiti mi, denetimli öğrenme mi?

İkisi de etiketli veriyle (CV'de) çalışabilir. Ne zaman hangisi?

| | **Anomali tespiti** | **Denetimli öğrenme** |
|---|---|---|
| **Pozitif ($y=1$) sayısı** | Çok **az** (0–20 tipik) | **Çok** (büyük sayıda hem pozitif hem negatif) |
| **Negatif ($y=0$) sayısı** | Çok fazla | Çok fazla |
| **Anomali türleri** | **Çok çeşitli**; **gelecekteki anomaliler** daha önce görmediklerine benzemeyebilir | Gelecekteki pozitifler **eğitimdekilere benzer** |
| Algoritma ne öğrenir? | "Normal nasıl görünür?" | "Pozitif nasıl görünür?" |

**Örnekler**

| Uygulama | Seçim | Neden? |
|---|---|---|
| **Finansal sahtekârlık** | Anomali tespiti | Sahtekârlar sürekli **yeni yöntemler** bulur |
| **E-posta spam** | **Denetimli** | Çok sayıda spam örneği, spam türleri **benzer** kalıyor |
| **Üretimde yeni/bilinmeyen kusur** | Anomali tespiti | Daha önce görülmemiş kusurlar |
| **Üretimde bilinen kusur** (örn. çizik) | **Denetimli** | Bilinen kusur türü, yeterli örnek |
| **Veri merkezi makine izleme** | Anomali tespiti | Yeni arıza modları |
| **Hava tahmini, hastalık teşhisi** | **Denetimli** | Çok sayıda etiketli örnek, bilinen sınıflar |

> 🧠 **Neden?** Denetimli algoritma, **gördüğü pozitiflere** benzer olanları iyi yakalar; **hiç görmediği tür** bir anomaliyi kaçırır. Anomali tespiti ise "normale benzemeyen **her şeyi**" yakalar.

---

## 12. Anomali tespitinde özellik seçimi

Denetimli öğrenmeden **daha kritik**, çünkü etiketler yoktur; algoritmanın doğru özelliklere ihtiyacı vardır.

### 1) Özellikleri daha Gauss yap

Özelliğin **histogramına** bak (`plt.hist`). Çan şeklinde değilse **dönüştür**:

| Dönüşüm | Ne zaman? |
|---|---|
| $\log(x)$ veya $\log(x+c)$ | Sağa çarpık (uzun kuyruklu) dağılım |
| $\sqrt{x}$ ($x^{0.5}$), $x^{1/3}$ | Hafif çarpıklık |

**Önemli:** Eğitim, CV ve test setlerine **aynı dönüşümü** uygula.

### 2) Hata analizi

Algoritma **bir anomaliyi kaçırıyorsa** ($p(x)$ büyük çıkıyor): o örneğe bak, **neden** normal göründüğünü anla, onu ayırt edecek **yeni bir özellik** üret.

**Örnek: veri merkezi** (özellikler: bellek kullanımı, disk erişimi, CPU yükü, ağ trafiği). Bazı arızalarda CPU yükü yüksek ve ağ trafiği düşüktür (takılmış bir iş). Tek tek değerler normal görünür ama **oranı** anormaldir. **Yeni özellik** ekle:

$$x_5 = \frac{\text{CPU yükü}}{\text{ağ trafiği}} \qquad x_6 = \frac{(\text{CPU yükü})^2}{\text{ağ trafiği}}$$

### 3) Anomalide anormal değer alan özellikleri seç

Özellik, anomali durumunda **çok büyük veya çok küçük** değer alıyorsa kullanışlıdır.

> 🧠 **Neden?** Anomali tespiti çarpım kullandığı için **tek tek** özelliklerin olağandışılığına bakar. İki özelliğin **kombinasyonu** anormalse onu yakalayacak **kombinasyon özelliği** sen üretmelisin.

---

# HAFTA 2 – ÖNERİ SİSTEMLERİ (RECOMMENDER SYSTEMS)

## 13. Öneri sistemlerine giriş

**Öneri sistemleri**, Amazon, Netflix, Spotify, YouTube gibi şirketlerin **gelirinin büyük bölümünü** etkiler. Akademide az konuşulur ama **pratikte çok değerlidir**.

**Örnek: film puanlama.** 4 kullanıcı, 5 film, puanlar 0–5 (veya henüz puanlanmamış "?").

| Film | Alice | Bob | Carol | Dave |
|---|---|---|---|---|
| Love at Last | 5 | 5 | 0 | 0 |
| Romance Forever | 5 | ? | ? | 0 |
| Cute Puppies of Love | ? | 4 | 0 | ? |
| Nonstop Car Chases | 0 | 0 | 5 | 4 |
| Swords vs. Karate | 0 | 0 | 5 | ? |

**Notasyon**

| Sembol | Anlamı |
|---|---|
| $n_u$ | Kullanıcı sayısı (burada 4) |
| $n_m$ | Film (öğe) sayısı (burada 5) |
| $r(i,j)$ | Kullanıcı $j$ film $i$'yi **puanladıysa 1**, aksi 0 |
| $y^{(i,j)}$ | Kullanıcı $j$'nin film $i$'ye verdiği **puan** (sadece $r(i,j)=1$ ise tanımlı) |
| $m^{(j)}$ | Kullanıcı $j$'nin puanladığı film sayısı |

**Amaç:** Puanlanmamış ("?") yerleri **tahmin etmek**; kullanıcının beğeneceği filmleri **önermek**.

---

## 14. Özellikler biliniyorsa: lineer regresyon

Her filmin özellikleri biliniyor olsun: $x_1$ = romantiklik, $x_2$ = aksiyon ($n=2$).

| Film | $x_1$ (romantik) | $x_2$ (aksiyon) |
|---|---|---|
| Love at Last | 0.9 | 0 |
| Cute Puppies of Love | 0.99 | 0 |
| Nonstop Car Chases | 0.1 | 1.0 |

Her **kullanıcı** için ayrı bir lineer regresyon modeli:

$$\text{kullanıcı } j \text{'nin film } i \text{ için tahmini puanı} = \vec{w}^{(j)}\cdot\vec{x}^{(i)} + b^{(j)}$$

**Sezgi:** $\vec{w}^{(j)}$ kullanıcının **zevk vektörü** (romantik mi seviyor, aksiyon mu?).
Örn. Alice için $\vec{w}=[5,0],\ b=0$, film $\vec{x}=[0.99, 0]$ → tahmin $=5\cdot0.99=4.95$. (Alice romantik seviyor.)

### Cost function (kullanıcı $j$ için)

$$\min_{\vec{w}^{(j)},\,b^{(j)}}\; \frac{1}{2}\sum_{i:\,r(i,j)=1}\left(\vec{w}^{(j)}\cdot\vec{x}^{(i)} + b^{(j)} - y^{(i,j)}\right)^2 + \frac{\lambda}{2}\sum_{k=1}^{n}\left(w_k^{(j)}\right)^2$$

- Sadece **puanlanmış** ($r=1$) filmler toplamda.
- Regularization terimi (önceki kurs).
- **Tüm kullanıcılar için** bunların toplamı alınır (hepsi aynı anda öğrenilir).

> 🧠 **Neden?** Bu, tam olarak **lineer regresyon**: kullanıcı başına bir model. Ama özellikler ($x$) **biliniyorsa** işe yarar. Gerçekte filmlerin özelliklerini bilmeyebiliriz; sıradaki bölüm bunu çözer.

---

## 15. Collaborative filtering: özellikleri **öğrenmek**

**Sorun:** Özellikler ($x^{(i)}$) yoksa? Tersine çevir: **kullanıcı parametreleri $\vec{w},b$ biliniyorsa**, özellikleri $x$'i de **aynı mantıkla öğrenebiliriz.**

**Sezgi:** Alice 5 puan verdiyse $\vec{w}^{(1)}\cdot\vec{x}^{(1)}\approx5$; Bob, Carol, Dave'in puanları da $\vec{x}^{(1)}$'i kısıtlar. $x^{(i)}$'yi, tahminler gerçek puanlara **yaklaşacak** şekilde seç.

### Birleşik cost function

$$J(\vec{w},b,\vec{x}) = \frac{1}{2}\sum_{(i,j):\,r(i,j)=1}\left(\vec{w}^{(j)}\cdot\vec{x}^{(i)}+b^{(j)}-y^{(i,j)}\right)^2 + \frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}\left(w_k^{(j)}\right)^2 + \frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}\left(x_k^{(i)}\right)^2$$

### Gradient descent: $w$, $b$ **ve $x$** birlikte

$$w_k^{(j)} = w_k^{(j)} - \alpha\frac{\partial J}{\partial w_k^{(j)}} \qquad b^{(j)} = b^{(j)} - \alpha\frac{\partial J}{\partial b^{(j)}} \qquad x_k^{(i)} = x_k^{(i)} - \alpha\frac{\partial J}{\partial x_k^{(i)}}$$

**Fark:** Önceki kurslardaki tüm algoritmalarda özellikler **verilmişti**. Burada özellikleri de **parametre gibi öğreniyoruz**.

**Parametre sayısı (örnek):** $n_u=4,\ n_m=5,\ n=2$ → $W$: $4\times2=8$, $b$: 4, $X$: $5\times2=10$ → **22 parametre**.

> 🧠 **Neden "collaborative"?** Bir film için **birden fazla kullanıcının** puanı olduğundan, kullanıcılar birlikte (collaborate) o filmin özelliklerini öğrenmemize yardımcı olur. Aynı şekilde bir kullanıcının **birden fazla film** puanı, o kullanıcının zevkini ortaya çıkarır.

**Özellikler ne anlama gelir?** Algoritma özellikleri kendi seçer; "romantik", "aksiyon" gibi **insanca yorumlanabilir** olmak zorunda değildir. Sadece puanları iyi açıklayan sayılardır.

---

## 16. Binary (ikili) etiketler: beğeni, tıklama, izleme

Birçok sistemde puan yoktur, **ikili** sinyal vardır:

| $y^{(i,j)}$ | Anlamı (örnek) |
|---|---|
| **1** | Kullanıcı etkileşti: filmi sonuna kadar izledi, tıkladı, beğendi, satın aldı |
| **0** | Gösterildi ama etkileşmedi |
| **?** | Hiç gösterilmedi |

**Çözüm:** Lineer regresyon yerine **logistic regression** (önceki kurs):

$$f_{(w,b,x)}(x) = g\left(\vec{w}^{(j)}\cdot\vec{x}^{(i)}+b^{(j)}\right) \quad g(z)=\frac{1}{1+e^{-z}}$$

Bu, $P(y^{(i,j)}=1)$ tahmini olur. Loss (binary cross-entropy):

$$L = -y\log f - (1-y)\log(1-f)$$

$$J(w,b,x) = \sum_{(i,j):\,r(i,j)=1} L\left(f_{(w,b,x)}(\vec{x}^{(i)}),\ y^{(i,j)}\right)$$

> 🧠 **Neden?** Puan yerine "evet/hayır" tahmini yapıyoruz. Logistic regression bu iş için zaten doğru araç (Sinir ağı kursu Bölüm 10–11).

---

## 17. Mean normalization (ortalama normalleştirme)

**Sorun:** Hiç film puanlamamış **yeni kullanıcı** (Eve). Regularization yüzünden $\vec{w}^{(5)}\approx 0,\ b^{(5)}\approx0$ olur → **tüm filmlere 0 puan** tahmin edilir. Bu işe yaramaz.

**Çözüm: Mean normalization**

1. Her **film** (satır) için, puanlanmış puanların ortalaması $\mu_i$ hesapla.
2. Puanlardan ortalamayı çıkar: $y^{(i,j)} - \mu_i$.
3. Modeli bu **normalleştirilmiş** puanlarla eğit.
4. Tahmin yaparken ortalamayı **geri ekle**:

$$\text{tahmin} = \vec{w}^{(j)}\cdot\vec{x}^{(i)} + b^{(j)} + \mu_i$$

**Sayısal örnek:** Bir filmin puanları (Alice 5, Bob 5, Carol 0, Dave 0) → $\mu = 2.5$. Normalleştirilmiş: $[2.5,\ 2.5,\ -2.5,\ -2.5]$. Yeni kullanıcı Eve için $w=0,b=0$ → tahmin $=0+2.5=\mathbf{2.5}$ (o filmin ortalama puanı). Mantıklı bir varsayım!

**Not:** **Satırları (filmleri)** normalleştir. (Sütunları normalleştirmek, yani kullanıcı ortalaması çıkarmak, yeni kullanıcı sorununu çözmez.)

> 🧠 **Neden?** Hiçbir bilgin yoksa, "ortalama puanı" tahmin etmek, "0 puan" tahmin etmekten çok daha makul. Ayrıca algoritmanın **daha hızlı** yakınsamasına da yardımcı olur.

---

## 18. TensorFlow ile collaborative filtering

Collaborative filtering'de $x, w, b$ hepsi parametre. Bu, standart `Dense` katmanlı `model.fit` yapısına **uymaz**. Çözüm: **kendi cost'unu yaz**, TensorFlow **türevleri otomatik** hesaplasın (**auto diff / auto grad**).

### Otomatik türev: `tf.GradientTape`

Basit örnek: $J=(wx-y)^2$, $x=1,\ y=1,\ w=3$ (başlangıç), $\alpha=0.01$:

```python
w = tf.Variable(3.0)
x, y, alpha = 1.0, 1.0, 0.01

for _ in range(30):
    with tf.GradientTape() as tape:        # hesaplamayı "kaydet"
        fwb = w * x
        costJ = (fwb - y) ** 2
    [dJdw] = tape.gradient(costJ, [w])     # türevi otomatik al
    w.assign_add(-alpha * dJdw)            # w = w - α·dJ/dw
```

### Collaborative filtering için eğitim döngüsü (Adam ile)

```python
# X: (n_m x n) film özellikleri, W: (n_u x n) kullanıcı ağırlıkları, b: (1 x n_u)
# Hepsi tf.Variable
def cofi_cost(X, W, b, Y, R, lambda_):
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R     # R: sadece puanlananlar
    return 0.5 * tf.reduce_sum(j ** 2) + (lambda_ / 2) * (tf.reduce_sum(X ** 2) + tf.reduce_sum(W ** 2))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)
for it in range(200):
    with tf.GradientTape() as tape:
        cost = cofi_cost(X, W, b, Ynorm, R, lambda_)
    grads = tape.gradient(cost, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))
```

- `R` matrisi 0/1: puanlanmamış hücreler toplamdan çıkar.
- `Ynorm` = mean-normalized puanlar. Tahmin: `X @ W.T + b + mu`.

> 🧠 **Neden bu önemli?** Bazı problemler hazır katmanlara uymaz. TensorFlow'un asıl gücü: **herhangi bir cost** yaz, türevi (backprop) **otomatik** alır. Bu, derin öğrenmede **kendi modelini** kurmanın temelidir.

---

## 19. Benzer öğeleri bulma ve sınırlamalar

### İlgili öğeleri bulma

Collaborative filtering her film için bir $\vec{x}^{(i)}$ öğrenir. $i$ filmine **benzer** filmler: **uzaklığı en küçük** olanlar.

$$\text{uzaklık}(k, i) = \left\|\vec{x}^{(k)}-\vec{x}^{(i)}\right\|^2 = \sum_{l=1}^{n}\left(x_l^{(k)}-x_l^{(i)}\right)^2$$

En küçük uzaklıklı $k$'lar = ilgili öğeler ("Bunu beğendiyseniz şunlar da hoşunuza gidebilir").

### Sınırlamalar

| Sınırlama | Açıklama |
|---|---|
| **Soğuk başlangıç (cold start)** | **Yeni öğe**: kimse puanlamadı. **Yeni kullanıcı**: az puanı var. İkisi için de **güvenilir tahmin zor** |
| **Yan bilgiyi kullanmaz** | Kullanıcı bilgisi (yaş, cinsiyet, ülke, cihaz) ve öğe bilgisi (tür, yıl, yönetmen) **kullanılmaz** |

Bu sınırlamalar **içerik tabanlı filtreleme** ile çözülür (Bölüm 20).

---

## 20. Content-based (içerik tabanlı) filtreleme

| | **Collaborative filtering** | **Content-based filtering** |
|---|---|---|
| Neye göre önerir? | **Benzer kullanıcıların** puanlarına göre | **Kullanıcı özellikleri ve öğe özelliklerine** göre iyi **eşleşmeyi** bulur |
| Yan bilgi | ❌ | ✅ |

**Veri:** Hâlâ $r(i,j)$ ve $y^{(i,j)}$ var. Ek olarak:

| Kullanıcı özellikleri $x_u^{(j)}$ | Film özellikleri $x_m^{(i)}$ |
|---|---|
| Yaş, cinsiyet, ülke | Yayın yılı |
| İzlediği filmler | Türler |
| Tür başına ortalama puanı | Yorumlar, ortalama puan |

### Fikir

Kullanıcı $j$'nin film $i$'ye puanını **iki vektörün dot product'ı** olarak tahmin et:

$$\text{tahmin} = \vec{v}_u^{(j)}\cdot\vec{v}_m^{(i)}$$

- $\vec{v}_u^{(j)}$: kullanıcının **zevklerini** özetleyen vektör, $\vec{x}_u^{(j)}$'den **hesaplanır**.
- $\vec{v}_m^{(i)}$: filmin **özelliklerini** özetleyen vektör, $\vec{x}_m^{(i)}$'den **hesaplanır**.
- İkisi **aynı boyutta** olmalı (örn. 32). Ama $x_u$ ve $x_m$ **farklı boyutlarda** olabilir.

> 🧠 **Neden $x$'ten $v$'ye?** Ham özellikler ($x_u$, $x_m$) doğrudan karşılaştırılamaz (farklı boyut/anlam). $v$ vektörleri, "**eşleşmeyi ölçmeye uygun ortak uzay**" gibidir.

---

## 21. Derin öğrenme ile içerik tabanlı filtreleme

$\vec{v}_u$ ve $\vec{v}_m$'yi hesaplamak için **sinir ağları** kullan.

```
x_u (kullanıcı özellikleri) ──► [ Kullanıcı ağı ] ──► v_u (32 sayı)  ┐
                                                                    ├─► v_u · v_m  ──► tahmin
x_m (film özellikleri)       ──► [ Film ağı     ] ──► v_m (32 sayı)  ┘
```

- İki ağın **çıkış katmanı aynı boyutta** (örn. 32 birim).
- Birkaç gizli katman (ReLU); çıkış katmanı **linear**.
- **İkili etiket** için sonuna **sigmoid** eklenir: $g(\vec{v}_u\cdot\vec{v}_m)$.

**Cost (ikisi birlikte eğitilir):**

$$J = \sum_{(i,j):\,r(i,j)=1}\left(\vec{v}_u^{(j)}\cdot\vec{v}_m^{(i)} - y^{(i,j)}\right)^2 + \text{sinir ağı regularization}$$

İki ağın parametreleri **aynı cost'u** düşürmek için birlikte öğrenilir (tek gradient descent).

### Benzer filmleri bulma

Collaborative filtering'deki gibi: $\left\|\vec{v}_m^{(k)}-\vec{v}_m^{(i)}\right\|^2$ en küçük olanlar. **Bu, önceden hesaplanabilir** (offline): gece, tüm filmler için benzer filmler listesini üretip sakla.

### TensorFlow'da (özet)

```python
user_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32)                       # v_u
])
item_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32)                       # v_m
])

input_user = tf.keras.layers.Input(shape=(num_user_features,))
vu = tf.linalg.l2_normalize(user_NN(input_user), axis=1)   # birim uzunluğa normalleştir
input_item = tf.keras.layers.Input(shape=(num_item_features,))
vm = tf.linalg.l2_normalize(item_NN(input_item), axis=1)

output = tf.keras.layers.Dot(axes=1)([vu, vm])             # dot product
model = tf.keras.Model([input_user, input_item], output)
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError())
```

(Katman boyutları örnektir.)

> 🧠 **Neden iyi?** Hem **kullanıcı** hem **öğe yan bilgisini** kullanır, **soğuk başlangıç** için daha uygundur (yeni filmin yıl/tür bilgisi vardır). Sinir ağları karmaşık eşleşmeleri öğrenebilir.

---

## 22. Büyük katalog: retrieval ve ranking

Katalogda **binlerce/milyonlarca** öğe olabilir. Her kullanıcı için **hepsine** sinir ağı çalıştırmak **çok yavaş**.

**İki aşamalı yaklaşım:**

| Aşama | Ne yapar? | Örnek |
|---|---|---|
| **1) Retrieval** (aday çıkarma) | **Hızlıca** yüzlerce **makul aday** seç | • Kullanıcının izlediği son 10 film için her birine **benzer 10** film<br>• Kullanıcının en sevdiği 3 türde **en popüler 10** film<br>• Ülkede **en çok izlenen 20** film<br>• Tekrarlananları ve **zaten izlenenleri çıkar** |
| **2) Ranking** (sıralama) | Adaylara **öğrenilmiş modeli** uygula, **tahmini puana göre sırala** | $\vec{v}_u$'yu bir kez hesapla, her aday için $\vec{v}_m$ (önceden hesaplı) ile dot product |

**Retrieval kaç aday getirsin?**
- **Çok aday:** daha iyi öneri kalitesi ama **yavaş**.
- **Az aday:** hızlı ama kaliteli adayı kaçırabilir.
- Karar için **offline deney**: 100 yerine 500 aday getirmek öneri kalitesini yeterince artırıyor mu?

> 🧠 **Neden iki aşama?** Retrieval hızlı ama kaba, ranking yavaş ama hassas. İkisini birleştirmek, **gecikmeyi düşük** tutarken **kaliteyi** korur.

---

## 23. Öneri sistemlerinin etik yönü

Öneri sistemleri çok karlı ama **zararlı** olabilir.

**Hedefin ne olduğu önemlidir:**

| Önerilen şey | Sorun |
|---|---|
| **Kullanıcının en çok beğeneceği öğe** | Makul |
| **Tıklanma olasılığı × reklam teklifi** | Kâr odaklı; zararlı reklamverenlere de alan açabilir |
| **Şirkete en çok kâr getiren ürün** | Kullanıcıya **en uygun** ürün değil |
| **İzleme süresini / etkileşimi maksimize et** | Dikkat çeken ama **zararlı içeriği** (aşırı, kışkırtıcı, komplo teorileri) öne çıkarabilir |

**Örnekler**
- **Reklamlar:** İyi örnek: seyahat sektöründe iyi hizmet veren işletmeler öne çıkar. Kötü örnek: **yüksek faizli kısa vadeli borç (payday loan)** gibi **sömürücü** işletmeler, daha çok tıklanıp daha çok teklif verdiği için **güçlenir**.
- **Etkileşim maksimizasyonu:** **Kışkırtıcı ve aşırıcı** içerik daha uzun izlenir → algoritma onu **yayar**.

**Olası iyileştirmeler**
1. **Zararlı** sektörleri/reklamları **filtrele** (örn. sömürücü kredi ürünleri).
2. **Şeffaf ol:** Kullanıcıya neye göre öneri yapıldığını söyle (kâr mı, ilgi mi?).
3. Zararlı içeriğin **yayılımını azaltacak** sinyaller (tür, güvenilirlik) kullan.
4. **Toplumsal etkiyi** sorgula, sadece KPI'a (tıklama) odaklanma.

> 🧠 **Neden bu derste?** Optimize ettiğin **hedef**, sistemin **davranışını** belirler. "Tıklanmayı maksimize et" dersen, tıklanan ne olursa olsun yayarsın. Mühendisin sorumluluğu hedefi **doğru seçmektir**.

---

# HAFTA 3 – PCA (BOYUT İNDİRGEME)

## 24. PCA sezgisi

**PCA (Principal Components Analysis):** Çok sayıda özelliği **az sayıda yeni özelliğe** indirger, mümkün olduğunca **bilgiyi koruyarak**.

**Neden?**
- **Görselleştirme:** 50 özelliği 2–3'e indirip grafik çizmek (ana kullanım).
- (Eskiden **veri sıkıştırma**, hızlandırma için de kullanılırdı; bugün daha az gerekli.)

**Örnek (ülkeler):** 50 özellik (GDP, kişi başı GDP, yaşam süresi, gelir eşitsizliği, ...). PCA bunu 2 özelliğe indirir. Sonuç, örneğin:
- $z_1$ ≈ **ülkenin toplam ekonomi büyüklüğü**
- $z_2$ ≈ **kişi başı ekonomik faaliyet**

Bu özellikleri 2B grafikte çizip ülkeleri **görsel olarak** karşılaştırabilirsin.

### Sezgi: araba örnekleri

| Özellikler | Ne yapar? |
|---|---|
| $x_1$ = uzunluk (çok değişir), $x_2$ = tekerlek çapı (neredeyse sabit) | Sadece $x_1$ yeterli; $x_2$ bilgi taşımıyor |
| $x_1$ = uzunluk, $x_2$ = yükseklik (**ikisi birlikte** değişiyor: büyük araba hem uzun hem yüksek) | İkisini **birleştiren yeni bir eksen** $z$ ("**araba büyüklüğü**") bul |

**Fikir:** Orijinal $x_1, x_2$ eksenlerine bağlı kalma; veriyi **en iyi yakalayan yeni eksen(ler)** ($z$) bul. Veriyi o eksene **izdüşür**, yeni özellik $z$ olur.

> 🧠 **Neden?** Bazı özellikler **gereksiz** veya **birbirine bağlıdır**. Onları tek bir anlamlı eksene birleştirmek, **az sayıda sayıyla** aynı bilgiyi anlatmaya yarar.

---

## 25. PCA nasıl çalışır?

**Adımlar:**
1. **Önişleme:** **Mean normalization** (her özelliğin ortalamasını çıkar); özellikler çok farklı aralıktaysa **feature scaling** de uygula.
2. **Ekseni seç:** **Verinin varyansını (yayılımını) en çok koruyan** ekseni bul = **ilk principal component**.
3. Veriyi bu eksene **izdüşür (project)**; her noktanın eksendeki koordinatı yeni özellik $z$.

### İzdüşümde varyans neden önemli?

Noktaları bir eksene **dik (perpendicular)** olarak izdüşürünce, noktaların **ne kadar dağılmış** (varyansı yüksek) kaldığı önemlidir.
- Eksen verinin **uzun yönüne** hizalıysa → izdüşümler **birbirinden uzak**, bilgi korunur ✅
- Eksen **kısa yöne** hizalıysa → noktalar üst üste biner, bilgi kaybolur ❌

**PCA, varyansı maksimize eden ekseni seçer.**

### Birden çok eksen

- **1. principal component:** en çok varyans yakalayan eksen.
- **2. principal component:** 1.ye **dik (90°)** olan, kalan varyansı en çok yakalayan eksen.
- 3., 4., ... hepsi birbirine dik.
- Kaç eksen kullanacağın ($n_{components}$) sana bağlı: **2–3** ise görselleştirme.

### ⚠️ PCA ≠ lineer regresyon

| | **Lineer regresyon** | **PCA** |
|---|---|---|
| Etiket $y$ | **Var** (tahmin edilecek) | **Yok** (denetimsiz) |
| Mesafe | **Dikey** (y ekseni yönünde) | **Dik (perpendicular)**, eksene en kısa mesafe |
| Amaç | $y$'yi tahmin et | Veriyi **sıkıştır / görselleştir** |

### Yeniden oluşturma (reconstruction)

$z$ değerlerinden orijinal $x$'e **yaklaşık** geri dönebilirsin (izdüşüm noktasına). Bilgi kaybı vardır; ne kadar bilgi korunduğunu **explained variance** gösterir.

> 🧠 **Neden varyans = bilgi?** Tüm noktalar aynı olsaydı (varyans 0), onları ayırt edemezdin. **Dağılım**, noktaları birbirinden ayıran "bilgi"dir. PCA o dağılımı mümkün olduğunca **korur**.

---

## 26. scikit-learn ile PCA

```python
import numpy as np
from sklearn.decomposition import PCA

X = np.array([[ 1,  1], [ 2,  1], [ 3,  2],
              [-1, -1], [-2, -1], [-3, -2]])       # 6 örnek, 2 özellik

# 1 eksene indir
pca_1 = PCA(n_components=1)
pca_1.fit(X)                                       # fit içinde mean normalization'ı da yapar
print(pca_1.explained_variance_ratio_)             # [0.992]  → varyansın %99.2'si korunuyor

X_trans_1 = pca_1.transform(X)                     # z değerleri (6 sayı), ilki ≈ 1.383
X_reduced_1 = pca_1.inverse_transform(X_trans_1)   # yaklaşık orijinal x'e dönüş
```

**Adımlar**
1. **`fit`:** principal component'leri (eksenleri) bulur (ortalama normalizasyonu da otomatik yapar).
2. **`explained_variance_ratio_`:** her eksen varyansın **yüzde kaçını** açıklıyor? ($n_{components}=2$ için toplam 1.0.)
3. **`transform`:** veriyi yeni eksen(ler)e **izdüşürür** ($z$ değerleri).
4. **`inverse_transform`:** $z$'den orijinal uzaya **yaklaşık** geri döner.

**Pratik ipuçları**
- Genelde **2–3 bileşen** (görselleştirme için).
- `explained_variance_ratio_` toplamına bak: **yeterince yüksek** mi (örn. %90+)?
- Önce **scaling** (özellikler çok farklı ölçekteyse).

> 🧠 **Neden bu kadar kolay?** `fit` yalnızca ana eksenleri bulur; `transform` basit bir matris çarpımıdır. Asıl iş **hangi eksenin kaç bilgi taşıdığını** yorumlamaktır.

---

# HAFTA 4 – PEKİŞTİRMELİ ÖĞRENME (REINFORCEMENT LEARNING)

## 27. Pekiştirmeli öğrenme nedir?

**Reinforcement Learning (RL):** Ajanın (agent) bir **ortamda** **eylemler** yaparak, **ödül (reward)** sinyalinden öğrenmesi.

**Örnek: otonom helikopter**
- Durum (state): helikopterin **konumu, yönü, hızı**
- Eylem (action): **kontrol çubuğunu** hareket ettirme
- Ödül: iyi uçuyorsa **+1**, düşerse/çarparsa **−1000**

**Neden denetimli öğrenme değil?**
Denetimli öğrenmede **her durum için doğru eylemi** ($y$) söylemen gerekir. Helikopter için "hangi anda stick'i tam olarak nereye çekmeli?" **bilinmez**. Onun yerine **ne istediğini** (ödül) söylersin, **nasıl yapacağını** algoritma bulur.

**Ödül ile "köpek eğitimi" benzetmesi:** İyi davranışa ödül (puan/mama), kötüye ceza (negatif puan). Köpeğin ne yapacağını **sen söylemezsin**, sadece davranış sonucunu **değerlendirirsin**.

> 🧠 **Neden güçlü?** "Nasıl" yerine **"ne istediğini"** tanımlarsın. Doğru eylemi bilmediğin karmaşık kontrol problemlerinde (robotlar, oyunlar, helikopter) çok işe yarar.

**RL uygulamaları:** helikopter/robot kontrolü, **fabrika optimizasyonu**, finansal hisse yürütme, **oyunlar** (satranç, Go, video oyunları).

---

## 28. Mars gezgini örneği: durum, eylem, ödül

**Basit model:** 6 duraklı yol; ajan (Mars rover) bir durumda ve **sola veya sağa** gidebilir.

```
 Durum:    1      2      3      4      5      6
 Ödül:   100      0      0      0      0     40
         [terminal]                       [terminal]
```

- **Durum (state) $s$:** ajan şu anda nerede (1–6). **Başlangıç: 4**.
- **Eylem (action) $a$:** ← (sola), → (sağa).
- **Ödül $R(s)$:** bulunduğun **duruma** bağlı. Durum 1'de **100**, durum 6'da **40**, aralarda 0.
- **Terminal durum:** 1 ve 6. Oraya ulaşınca **bölüm biter**, başka ödül yok.

Her adımda bilgi: $(s,\ a,\ R(s),\ s')$ = (mevcut durum, eylem, ödül, **sonraki durum**).

**Örnek akış** (4'ten sola git): $4 \to 3 \to 2 \to 1$ — ödüller: $0,\ 0,\ 0,\ 100$.

---

## 29. Getiri (return) ve indirim faktörü ($\gamma$)

**Return**, bir ödül dizisinin **indirimli toplamıdır**:

$$\text{Return} = R_1 + \gamma R_2 + \gamma^2 R_3 + \gamma^3 R_4 + \dots \quad (\text{terminal duruma kadar})$$

- $\gamma$ (gama) = **discount factor**, 1'e yakın ama **1'den küçük** (örn. 0.9, 0.99, **0.5 örnekte**).
- **Gelecek ödüller daha az değerlidir**: ödülü ne kadar geç alırsan, $\gamma^t$ ile o kadar çarpılır.

**Örnek ($\gamma=0.9$):** 4'ten sola giderek 100'e ulaşırsan:

$$0 + 0.9\cdot0 + 0.9^2\cdot0 + 0.9^3\cdot100 = 72.9$$

**Örnek ($\gamma=0.5$, "hep sola git" politikası)**

| Başlangıç durumu | Return |
|---|---|
| 1 (terminal) | **100** |
| 2 | $0 + 0.5\cdot100 = $ **50** |
| 3 | $0+0.5\cdot0+0.25\cdot100 = $ **25** |
| 4 | $0.125\cdot100 = $ **12.5** |
| 5 | $0.0625\cdot100 = $ **6.25** |
| 6 (terminal) | **40** |

**"Hep sağa git" politikasında** aynı hesapla: 1:100, 2: 2.5, 3: 5, 4: 10, 5: 20, 6: 40.

> 🧠 **Neden indirim?** Pratik: "yarın 100 TL" bugünkü 100 TL'den **daha az** değerlidir. Algoritmik: ajan ödülü **daha çabuk** almaya teşvik edilir (**sabırsızlık**). $\gamma$ küçük → **sabırsız**; 1'e yakın → **sabırlı**.

**Negatif ödüller:** Negatif ödül varsa indirim onu **ertelemeyi** (sonraya bırakmayı) teşvik edebilir. Kaçınılmaz bir ceza geç yaşansa da **daha az acı** verir.

---

## 30. Politika ($\pi$) ve RL'nin amacı

**Politika $\pi$:** Her durumu ($s$) bir eyleme ($a$) eşleyen **fonksiyon**:

$$\pi(s) = a$$

Örnek: Mars rover'da (Bölüm 32'deki) **optimal** politika: durum 2, 3, 4 → ←, durum 5 → →.

Politikaların farklı seçenekleri var:
- **Her zaman en yakın ödüle git**
- **Her zaman daha büyük ödüle git**
- **Her zaman sola git**
- ...

**RL'nin amacı:** **Return'ü maksimize eden politikayı** $\pi$ bulmak.

> 🧠 **Neden "politika"?** Bu terim yaygın ama her zaman en açıklayıcı değil. Sadece "**durum → eylem kuralı**" demek.

---

## 31. Markov Karar Süreci (MDP) formalizmi

Mars gezgini formalizmini özetleyelim. RL problemi şunlardan oluşur:

| Kavram | Mars rover | Helikopter | Satranç |
|---|---|---|---|
| **Durumlar** | 6 durum | konum, yön, hız | tahtadaki taşların konumu |
| **Eylemler** | ←, → | kumanda hareketi | olası hamleler |
| **Ödüller** | 100, 0, 40 | +1 (iyi uçuş), −1000 (kaza) | +1 kazanırsan, −1 kaybedersen, 0 beraberlik |
| **İndirim faktörü $\gamma$** | 0.5 | 0.99 | 0.995 |
| **Return** | indirimli ödül toplamı | | |
| **Politika $\pi$** | durum → eylem | | |

Bu yapıya **Markov Decision Process (MDP)** denir.

**Markov özelliği:** Gelecek **sadece şu anki duruma** bağlıdır, oraya **nasıl gelindiğine** (geçmişe) bağlı değildir.

---

## 32. Durum-eylem değer fonksiyonu $Q(s,a)$

**Q fonksiyonu (state-action value function):**

> $Q(s,a)$ = **$s$ durumundayken $a$ eylemini bir kez yap, sonrasında hep OPTİMAL davran**, bu durumda alacağın **return**.

**Neden dairesel gibi görünüyor?** "Optimal davranışı" bilmen gerekir, ama o da Q'dan çıkar. **Merak etme**, Bellman denklemiyle çözeceğiz (Bölüm 33).

### Q'dan en iyi politika

$$\text{Bir durumda en iyi eylem} = \arg\max_a Q(s,a) \qquad \max_a Q(s,a) = \text{o durumdan alınabilecek en iyi return}$$

**Mars rover Q değerleri ($\gamma=0.5$):**

| Durum | $Q(s,\leftarrow)$ | $Q(s,\rightarrow)$ | En iyi eylem | $\max_a Q$ |
|---|---|---|---|---|
| 1 | 100 | 100 | (terminal) | 100 |
| 2 | **50** | 12.5 | ← | 50 |
| 3 | **25** | 6.25 | ← | 25 |
| 4 | **12.5** | 10 | ← | 12.5 |
| 5 | 6.25 | **20** | → | 20 |
| 6 | 40 | 40 | (terminal) | 40 |

**Sonuç:** Optimal politika: 2,3,4 → **←** ; 5 → **→**.

**$\gamma$'nın etkisi:** $\gamma$'yı 0.5'ten **0.9'a** çıkarırsan rover **daha sabırlı** olur, daha uzak ama **daha büyük** ödül için de gitmeye razı olur.

> 🧠 **Neden Q önemli?** $Q(s,a)$'yı **hesaplayabilirsek**, her durumda **$\arg\max_a Q(s,a)$** diyerek optimal eylemi buluruz. RL problemi "$Q$'yu öğrenme" problemine döner.

---

## 33. Bellman denklemi

Q'yu **özyinelemeli** hesaplamayı sağlar.

**Notasyon:** $s$: şu anki durum, $R(s)$: bu durumun ödülü, $a$: eylem, $s'$: eylem sonrası **sonraki durum**, $a'$: sonraki durumda yapılacak eylem.

$$\boxed{Q(s,a) = R(s) + \gamma\,\max_{a'} Q(s',a')}$$

**Sezgi:**

$$\text{Return} = R_1 + \gamma\left[R_2 + \gamma R_3 + \gamma^2 R_4+\dots\right] = R(s) + \gamma\cdot\underbrace{(\text{sonraki durumdan optimal return})}_{\max_{a'}Q(s',a')}$$

**İki parça:**
1. **Hemen alacağın ödül:** $R(s)$.
2. **Sonraki durumdan alacağın en iyi return**, $\gamma$ ile indirimli.

**Terminal durumlarda:** $Q(s,a)=R(s)$ (başka ödül yok).

### Sayısal örnek ($\gamma=0.5$)

$Q(2,\rightarrow)$: 2'de sağa gidersen 3'e varırsın.

$$Q(2,\rightarrow) = R(2) + 0.5\cdot\max_{a'}Q(3,a') = 0 + 0.5\cdot25 = \mathbf{12.5}$$

$Q(4,\leftarrow) = 0 + 0.5\cdot\max_{a'}Q(3,a') = 0 + 0.5\cdot25 = \mathbf{12.5}$.

$Q(5,\rightarrow) = 0 + 0.5\cdot Q(6)=0+0.5\cdot40 = \mathbf{20}$.

> 🧠 **Neden?** Büyük bir problemi ("tüm gelecek ödül toplamı") **küçük bir probleme** böler: "şimdiki ödül + bir sonraki durumun değeri". Bu **bölerek çözme** fikri, RL algoritmalarının (DQN dahil) temelidir.

---

## 34. Rastgele (stokastik) ortamlar

Gerçekte eylem **her zaman istediğin sonucu vermeyebilir**. Mars rover: sola git dersen **%90** sola gider, **%10** yanlışlıkla **sağa** kayar.

- Aynı politikayı iki kez denersen **farklı ödül dizileri** görürsün (şans).
- Return artık **rastgele bir sayıdır**.

**Hedef değişir:** Return'ü değil, **beklenen return'ü** (expected return = çok kez deneyince **ortalama return**) maksimize et.

**Bellman denklemi ($\text{stokastik}$):**

$$Q(s,a) = R(s) + \gamma\,\mathbb{E}\left[\max_{a'}Q(s',a')\right]$$

($\mathbb{E}$: sonraki durumun rastgeleliği üzerinden **ortalama**.)

**Etki:** Yanlış kayma olasılığı arttıkça (örn. %10 → %40) **Q değerleri düşer** (return daha belirsiz/azalır) ve optimal politika **değişebilir**.

> 🧠 **Neden?** Gerçek dünyada robotlar, tekerlekler kayar, rüzgâr eser. **Sürpriz yaşayan** ortamlara dayanıklı algoritmalar gerekir.

---

## 35. Sürekli durum uzayları

Mars rover'ın **6 ayrık durumu** vardı. Gerçek uygulamalarda durum **sürekli sayılar** olabilir.

**Örnek: robot araç.** Durum: $x$ (konum), $y$ (konum), $\theta$ (yönelim), $\dot{x}$, $\dot{y}$, $\dot{\theta}$ (hızlar)... **Sonsuz sayıda olası durum**.

**Örnek: helikopter.** Durum: konum (x, y, z), **roll, pitch, yaw** (dönüş açıları), bunların hızları → **12 sayı**.

Bu durumlarda **tablo tutamazsın**; Q fonksiyonunu **sinir ağı** ile yaklaşıklarız (Bölüm 37).

---

## 36. Ay iniş aracı (Lunar Lander) örneği

RL'nin popüler deneme alanı: simüle ay iniş aracı. Amaç: **iniş pistine yumuşak iniş yapmak**.

**Eylemler (4):**
- **Hiçbir şey yapma** (yerçekimi çeker)
- **Sol itici** (aracı sağa döndürür)
- **Ana motor** (aşağı doğru itme, yukarı çıkarır)
- **Sağ itici**

**Durum (8 sayı):**

$$s = [x,\ y,\ \dot{x},\ \dot{y},\ \theta,\ \dot{\theta},\ l,\ r]$$

- $x, y$: konum; $\dot{x},\dot{y}$: hızlar; $\theta,\dot{\theta}$: eğim ve dönme hızı
- $l, r$: **sol/sağ bacak yere temas ediyor mu** (0/1)

**Ödül fonksiyonu:**

| Olay | Ödül |
|---|---|
| Piste iniş (ne kadar iyi indiğine göre) | **+100 … +140** |
| Pist merkezine doğru/uzağa hareket | + / − (yönüne göre) |
| **Çarpma (crash)** | **−100** |
| Yumuşak iniş | **+100** |
| Her bacak yere değerse | **+10** |
| Ana motoru çalıştırmak | **−0.3** (adım başına) |
| Yan itici çalıştırmak | **−0.03** (adım başına) |

**Neden yakıt cezası?** Motoru gereksiz çalıştırmasın.

**Discount factor:** $\gamma=\mathbf{0.985}$ (1'e çok yakın; ajan **sabırlı**).

**Amaç:** Return'ü maksimize eden politika $\pi(s)=a$'yı bulmak.

---

## 37. DQN: Q fonksiyonunu sinir ağıyla öğrenme

**Deep Q-Network (DQN):** $Q(s,a)$'yı bir **sinir ağıyla** yaklaşıklar.

### Ağ mimarisi (ilk sürüm)

```
Girdi (12 sayı) ──► 64 birim (ReLU) ──► 64 birim (ReLU) ──► 1 çıkış = Q(s,a)
   │
   └── 8 sayı durum  +  4 sayı one-hot eylem (örn. ana motor = [0,0,1,0])
```

**Karar anında:** Durum $s$ için **4 eylemin hepsini** dene: 4 kez ağı çalıştır → $Q(s,\text{nothing}),\ Q(s,\text{left}),\ Q(s,\text{main}),\ Q(s,\text{right})$ → **en büyüğünü** seç.

### Ağı nasıl eğitiriz? (Bellman'dan eğitim verisi üretme)

Bellman: $Q(s,a) = R(s) + \gamma\max_{a'}Q(s',a')$. Bunu **denetimli öğrenmeye** çevir:

- Girdi $x = (s,a)$
- Hedef $y = R(s) + \gamma\max_{a'}Q(s',a')$

Ajan ortamda **tecrübe toplar**: $(s,a,R(s),s')$ demetleri (tuple). Her demetten bir eğitim örneği üretilir.

### DQN algoritması

```
1. Ağı Q(s,a) için RASTGELE başlat (kötü bir tahmin).

2. Tekrarla:
     • Ay iniş aracını çalıştır (kısmen rastgele eylemlerle), tuple'ları topla: (s, a, R(s), s')
     • En son 10.000 tuple'ı sakla  (REPLAY BUFFER)
     • Eğitim seti oluştur:  x = (s, a),   y = R(s) + γ · max_a' Q(s', a')
         (s' terminal ise  y = R(s))
     • Yeni ağ Q_new'yi, Q_new(s,a) ≈ y olacak şekilde eğit (MSE loss)
     • Q = Q_new                          ← tahmini güncelle
```

**Sezgi:** İlk $y$ değerleri **gürültülü** (ağ rastgele). Ama her turda $y$'ler biraz **daha doğru** olur, Q yavaş yavaş **gerçeğe yaklaşır**. Bellman denklemi "kendini düzelten" bir süreç sağlar.

> 🧠 **Neden "replay buffer"?** Sadece son tecrübeyi değil, **son 10.000** tecrübeyi tutup onlardan örneklersek eğitim daha **kararlı** olur (art arda gelen benzer örnekler yerine karışık örnekler).

---

## 38. Daha verimli mimari ve $\epsilon$-greedy

### Verimli mimari (4 çıkışlı)

İlk sürümde her karar için ağı **4 kez** çalıştırıyorduk. Daha verimli: **tek çalıştırma, 4 çıkış**.

```
Girdi: durum s (8 sayı) ──► 64 (ReLU) ──► 64 (ReLU) ──► 4 çıkış:
        Q(s, nothing), Q(s, left), Q(s, main), Q(s, right)
```

- Karar için **tek ileri geçiş**, 4 değer birden → **en büyüğünü** seç.
- $y = R(s)+\gamma\max_{a'}Q(s',a')$ hesabında **$\max_{a'}$ da tek geçişte** gelir.

### Öğrenirken eylemi nasıl seçeriz? $\epsilon$-greedy

Q hâlâ **kötü** bir tahminse "en iyi görüneni" seçmek, ajanı **yanlış alışkanlığa** hapseder (örn. ana motorun hiç işe yaramadığını sanıp **hiç denememek**).

**$\epsilon$-greedy politikası** (genelde $\epsilon=0.05$):
- Olasılık **$1-\epsilon$ (örn. %95):** **açgözlü (greedy)**, $Q(s,a)$'yı **maksimize eden** eylem (**exploitation**, bilineni kullan).
- Olasılık **$\epsilon$ (örn. %5):** **rastgele** eylem (**exploration**, yeni şey dene).

**Trick: $\epsilon$'u azalt.** Başlangıçta **yüksek** ($\epsilon=1.0$, tamamen rastgele), sonra yavaşça **düşür** ($\epsilon\to0.01$). Çünkü başta Q kötü (keşfet), sonra iyileşir (öğrendiğini kullan).

> 🧠 **Neden?** **Keşif–sömürü ikilemi:** Hep bilineni yaparsan daha iyisini **asla bulamazsın**; hep rastgele yaparsan öğrendiğini **kullanamazsın**. $\epsilon$-greedy ikisini dengeler.

---

## 39. Mini-batch ve soft update

İki iyileştirme: **öğrenmeyi hızlandırır** ve **daha kararlı** yapar.

### 1) Mini-batch

**Sorun:** $m = 100$ milyon örnek varsa, gradient descent'in **her adımında** hepsini toplamak **çok yavaş**.

**Çözüm:** Her adımda **küçük bir rastgele alt küme** ($m'=1000$) kullan.

- Her adım **ucuz** olur → toplamda **çok daha hızlı**.
- Yön biraz **gürültülü** ama **ortalamada** minimuma gider.
- **RL'de:** 10.000'lik **replay buffer**'dan her turda 1000'lik **mini-batch** çek.

### 2) Soft update

**Sorun:** $Q = Q_{new}$ ile ağı **tamamen** değiştirmek riskli: kötü bir eğitim turu ağı **birden bozabilir**.

**Çözüm:** Yeni ağa **kademeli** geç:

$$W = \tau W_{new} + (1-\tau)W \qquad B = \tau B_{new} + (1-\tau)B \qquad (\tau = 0.01)$$

- %1 yeni + %99 eski. Değişim **yavaş** ve **istikrarlı**.
- **Sayısal örnek:** $W=2.0,\ W_{new}=4.0,\ \tau=0.01$ → $W = 0.01\cdot4+0.99\cdot2 = \mathbf{2.02}$.

> 🧠 **Neden?** Mini-batch: **hız**. Soft update: **kararlılık** (yeni ağ **kötüyse** bile eski ağı birden bozmaz). RL eğitimi gürültülüdür; bu "yumuşatma" hileleri çok önemlidir.

---

## 40. RL'nin bugünkü durumu

**Gerçekçi bakış** (yutturmaca yok):

| Durum | Açıklama |
|---|---|
| **Simülasyon ≫ gerçek dünya** | Çoğu RL araştırması **simülasyonda** (video oyunları). Aynı algoritmayı **gerçek robotta** çalıştırmak çok daha **zor** |
| **Uygulama sayısı az** | Denetimli ve denetimsiz öğrenmeye göre **çok daha az** pratik uygulama var |
| **Gelecek potansiyel** | Robotik, otonom kontrol, optimizasyon: heyecan verici ve gelişmeye devam ediyor |

**Çıkarım:** RL güçlü, ama **her problem için doğru araç değil**. Çoğu iş problemi için **denetimli öğrenme** hâlâ ilk tercih.

---

## 41. Uzmanlığın sonu: büyük resim

Bu uzmanlıkta 3 kursta öğrendiklerin:

| Kurs | Ana konular |
|---|---|
| **1. Supervised ML** | Lineer/logistic regresyon, gradient descent, regularization |
| **2. Advanced Learning Algorithms** | Sinir ağları, ML geliştirme pratikleri, karar ağaçları |
| **3. Unsupervised, Recommenders, RL** | K-means, anomali tespiti, öneri sistemleri, PCA, pekiştirmeli öğrenme |

**Yazarın son tavsiyesi:** Öğrendiklerini **uygula**. ML araçları güçlü; **insanlara faydalı şeyler** inşa etmek için kullan.

---

# EKLER

## 42. Büyük resim: tüm hikâye tek akışta

```
DENETİMSİZ ÖĞRENME
  Kümeleme (K-means):   ata → taşı → tekrar ; J=distortion ; çoklu başlatma ; K'yı amaca göre seç
  Anomali tespiti:      p(x)=∏ p(xj;μj,σj²) ; p(x)<ε → anomali ; ε ve özellikler CV (F1) ile
                        Özellik: Gauss'laştır (log, √) ; hata analizi → kombinasyon özellik
  Boyut indirgeme (PCA): mean norm. → varyansı en çok koruyan eksen(ler) → z ; görselleştirme

ÖNERİ SİSTEMLERİ
  Collaborative:        puanlardan w, b, x'i birlikte öğren ; mean normalization (yeni kullanıcı)
  Content-based:        v_u · v_m ; iki sinir ağı ; yan bilgi kullanır ; soğuk başlangıca daha iyi
  Büyük katalog:        retrieval (hızlı adaylar) + ranking (öğrenilmiş model)
  Etik:                 hedefi dikkatli seç (kâr / etkileşim ≠ kullanıcı faydası)

PEKİŞTİRMELİ ÖĞRENME
  Çerçeve:              durum, eylem, ödül R(s), indirim γ, return, politika π  (MDP)
  Değer:                Q(s,a) ; Bellman:  Q(s,a) = R(s) + γ max_a' Q(s',a')
  Stokastik:            beklenen return
  Sürekli durum:        DQN (sinir ağı ile Q) ; replay buffer ; y = R + γ max Q(s',a')
  İyileştirmeler:       ε-greedy (keşif) ; mini-batch (hız) ; soft update (kararlılık)
```

### Hangi algoritma, ne zaman?

| Problem | Algoritma |
|---|---|
| Etiketsiz veride **gruplar** bul | **K-means** |
| **Olağandışı** örnekleri yakala (az/çeşitli anomali) | **Anomali tespiti (Gauss)** |
| Çok özelliği **2–3'e indirip görselleştir** | **PCA** |
| Kullanıcıya **öğe öner** (puan/etkileşim var, yan bilgi yok) | **Collaborative filtering** |
| Öneri + **yan bilgi** (kullanıcı/öğe özellikleri) | **Content-based filtering** |
| **Ardışık karar** (robot, oyun), ödül var, doğru eylem bilinmiyor | **Pekiştirmeli öğrenme (DQN)** |

---

## 43. Hızlı formül kartı

| Konu | Formül |
|---|---|
| K-means atama | $c^{(i)} = \arg\min_k \lVert x^{(i)}-\mu_k\rVert^2$ |
| K-means taşıma | $\mu_k$ = kümedeki noktaların ortalaması |
| K-means cost | $J = \frac{1}{m}\sum\lVert x^{(i)}-\mu_{c^{(i)}}\rVert^2$ |
| Gauss | $p(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/2\sigma^2}$ |
| $\mu,\ \sigma^2$ | $\frac{1}{m}\sum x,\ \ \frac{1}{m}\sum(x-\mu)^2$ |
| Anomali | $p(\vec{x})=\prod_j p(x_j;\mu_j,\sigma_j^2) < \varepsilon$ |
| CF tahmin | $\vec{w}^{(j)}\cdot\vec{x}^{(i)}+b^{(j)}$ ; (+ $\mu_i$ mean norm.) |
| CF cost | $\frac12\sum_{r=1}(\dots)^2+\frac\lambda2\sum w^2+\frac\lambda2\sum x^2$ |
| Binary CF | $f=g(\vec{w}\cdot\vec{x}+b)$ ; log loss |
| Content-based tahmin | $\vec{v}_u\cdot\vec{v}_m$ |
| Benzerlik | $\lVert \vec{x}^{(k)}-\vec{x}^{(i)}\rVert^2$ |
| Return | $R_1+\gamma R_2+\gamma^2R_3+\dots$ |
| Q (optimal) | $Q(s,a)$ = $s$'de $a$ yap, sonra optimal davran |
| Bellman | $Q(s,a)=R(s)+\gamma\max_{a'}Q(s',a')$ |
| Optimal eylem | $\pi(s)=\arg\max_a Q(s,a)$ |
| DQN hedefi | $y=R(s)+\gamma\max_{a'}Q(s',a')$ |
| Soft update | $W=\tau W_{new}+(1-\tau)W,\ \tau=0.01$ |

**Kontrol listesi:**
- ✅ K-means'te $J$ **hiç artmamalı** (artıyorsa bug); **çoklu başlatma** yap
- ✅ Anomali: önce özellikleri **Gauss'laştır**; $\varepsilon$'u **CV'de F1** ile seç (accuracy değil)
- ✅ Öneri: **mean normalization** (yeni kullanıcı için); $r(i,j)=1$ olanlara bak
- ✅ PCA: önce **normalize/ölçekle**; `explained_variance_ratio_`'ya bak
- ✅ RL: **terminal** durumda $y=R(s)$; keşif için **$\epsilon$-greedy**; **soft update**

---

## 44. Sorun → çözüm rehberi

| Gözlem | Olası sebep | Ne yapmalı? |
|---|---|---|
| K-means sonucu **kötü/anlamsız** kümeler | Kötü yerel minimum | **Çoklu rastgele başlatma** (50–1000), en düşük $J$ |
| K-means'te $J$ **artıyor** | **Bug** | Atama/taşıma adımını kontrol et |
| Bir kümeye **hiç nokta düşmedi** | Boş küme | Kümeyi sil veya merkezi yeniden başlat |
| Dirsek **belirsiz** | Veri net kümelenmiyor | Amaca göre $K$ seç (örn. tişört bedenleri) |
| Anomali **kaçırılıyor** ($p(x)$ büyük) | Özellik anomaliyi göstermiyor | **Hata analizi**, kombinasyon özellik (oran vb.) |
| Çok **yanlış alarm** | $\varepsilon$ çok büyük / özellik gürültülü | $\varepsilon$'u küçült, özellik seç, CV'de F1 |
| Özellik çan şeklinde değil | Çarpık dağılım | $\log(x+c)$, $\sqrt{x}$ ile dönüştür (**her sette aynı**) |
| Az pozitif örnek + sürekli yeni anomali | | **Anomali tespiti** (denetimli değil) |
| Yeni kullanıcıya hep **0** öneriyor | Regularization | **Mean normalization** |
| Yeni **film/öğe** için tahmin kötü | **Soğuk başlangıç** | **Content-based** (yan bilgi) |
| Öneri çok **yavaş** | Tüm kataloğa model çalıştırıyorsun | **Retrieval + ranking**, benzerlikleri **offline** hesapla |
| Öneri sistemi **zararlı içerik** yayıyor | Hedef yanlış (etkileşim/kâr) | Filtrele, hedefi/şeffaflığı gözden geçir |
| PCA'da $z$ anlamsız | Özellikler ölçeklenmemiş | **Normalize/ölçekle** |
| PCA'da çok bilgi kaybı | Çok az bileşen | `explained_variance_ratio_`'ya göre bileşen artır |
| DQN **ajan hiç keşfetmiyor** | Hep greedy | **$\epsilon$-greedy** ($\epsilon$ başta yüksek) |
| DQN eğitimi **dengesiz/bozuluyor** | Ağ birden değişiyor | **Soft update** ($\tau=0.01$), **replay buffer** |
| DQN eğitimi **çok yavaş** | Her adımda tüm veri | **Mini-batch** |
| Ajan ödülü **çok geç** alıyor | $\gamma$ çok küçük (sabırsız) | $\gamma$'yı 1'e yaklaştır (örn. 0.985) |
| Simülasyonda iyi, gerçekte kötü | Sim-gerçek farkı | Gerçek koşullarda doğrula |

---

## 45. Sık yapılan hatalar ve karıştırılan kavramlar

### Sık yapılan hatalar

| # | Hata | Neden sorun? | Doğrusu |
|---|---|---|---|
| 1 | K-means'i **tek bir** rastgele başlatmayla çalıştırmak | Kötü yerel minimuma takılabilir | Çoklu başlatma |
| 2 | $K$'yı **$J$'yi minimize edecek** şekilde seçmek | $K$ arttıkça $J$ hep düşer | Dirsek / **amaca göre** |
| 3 | Anomali tespitinde **accuracy** kullanmak | Veri çok çarpık | **Precision/recall/F1** |
| 4 | Eğitim/CV/test'e **farklı dönüşümler** uygulamak | Tutarsız dağılım | **Aynı** dönüşüm |
| 5 | Anomali tespitinde çok anomali örneği için **denetimli** öğrenmeyi atlamak | Bol pozitif varsa denetimli daha iyi | Bölüm 11 tablosu |
| 6 | Collaborative filtering'de $r(i,j)=0$ olanları cost'a katmak | Puansız hücre "0 puan" sayılır | Sadece $r(i,j)=1$ |
| 7 | **Sütun** (kullanıcı) ortalaması çıkarmak | Yeni kullanıcı sorunu çözülmez | **Satır** (film) ortalaması |
| 8 | Tahminde **ortalamayı geri eklemeyi** unutmak | Tahmin yanlış ölçekte | $+\mu_i$ ekle |
| 9 | Content-based'de $v_u$ ve $v_m$ **farklı boyutta** | Dot product tanımsız | Aynı boyut (örn. 32) |
| 10 | PCA'yı **normalize etmeden** çalıştırmak | Büyük ölçekli özellik baskın çıkar | Mean norm. + scaling |
| 11 | PCA'yı lineer regresyonla karıştırmak | Etiket/mesafe farkı | Etiket **yok**, **dik** mesafe |
| 12 | Terminal durumda $y=R(s)+\gamma\max Q$ kullanmak | Terminalden sonrası yok | $y=R(s)$ |
| 13 | Sadece greedy davranmak | Keşif yok | **$\epsilon$-greedy** |
| 14 | $Q=Q_{new}$ ile tamamen değiştirmek | Kararsızlık | **Soft update** |
| 15 | Yüksek ödülü **hemen** vermemek / $\gamma$'yı yanlış seçmek | Ajan sabırsız/sabırlı | $\gamma$'yı probleme göre ayarla |

### Karıştırılan kavramlar

| Kavram A | Kavram B | Fark |
|---|---|---|
| **K-means** | **PCA** | Gruplar bul (küme) ↔ Boyut indir (eksen) |
| **Denetimli** | **Anomali tespiti** | Çok pozitif, benzer ↔ Az pozitif, çeşitli |
| **Collaborative** | **Content-based** | Kullanıcı davranışı ↔ Kullanıcı + öğe **özellikleri** |
| **Retrieval** | **Ranking** | Hızlı aday seçimi ↔ Modelle sıralama |
| **Return** | **Q(s,a)** | Bir ödül dizisinin toplamı ↔ $s$'de $a$ yapıp optimal davranışın return'ü |
| **Politika $\pi$** | **Q** | Durum → eylem ↔ Durum-eylem **değeri** |
| **Exploration** | **Exploitation** | Yeni dene ↔ Bildiğini kullan ($\epsilon$-greedy) |
| **Mini-batch** | **Soft update** | Hız (alt küme) ↔ Kararlılık (yavaş ağ değişimi) |
| **Principal component** | **Orijinal özellik** | Yeni eksen (varyansı koruyan) ↔ Eski özellik ($x_1,x_2$) |
| **Dirsek yöntemi** | **Amaca göre K** | Eğri yorumu ↔ İş hedefi |

---

## 46. Terim sözlüğü

| İngilizce | Transkriptteki çeviri / Türkçe |
|---|---|
| Algorithm | "Albüm" (yanlış çeviri) / **algoritma** |
| Gradient descent | "Not verme inişi", "gradyan inişi" / **gradient descent** |
| Clustering / Cluster centroid | Kümeleme / küme merkezi |
| Distortion function | Bozulma fonksiyonu (K-means cost'u) |
| Elbow method | Dirsek yöntemi |
| Anomaly detection | Anomali (olağandışılık) tespiti |
| Density estimation | Yoğunluk tahmini |
| Gaussian (normal) distribution | Gauss (normal) dağılım |
| Recommender system | "Önerilen/tavsiye sistemi" / **öneri sistemi** |
| Collaborative filtering | "İşbirlikçi/toplu filtreleme" / **ortak (işbirlikçi) filtreleme** |
| Content-based filtering | İçerik tabanlı filtreleme |
| Mean normalization | Ortalama normalleştirme |
| Cold start | Soğuk başlangıç |
| Retrieval / Ranking | Aday çıkarma / sıralama |
| Principal Component Analysis | Ana bileşen analizi |
| Explained variance | Açıklanan varyans |
| Reinforcement learning | **Pekiştirmeli (takviyeli)** öğrenme |
| State / Action / Reward | Durum / eylem / ödül |
| Discount factor ($\gamma$) | İndirim faktörü |
| Return | Getiri / geri dönüş |
| Policy ($\pi$) | Politika |
| State-action value function $Q$ | Durum-eylem değer fonksiyonu |
| Bellman equation | Bellman denklemi |
| MDP | Markov karar süreci |
| Terminal state | Bitiş (terminal) durumu |
| Replay buffer | Tecrübe belleği |
| Epsilon-greedy | Epsilon-açgözlü politika |
| Mini-batch | Mini parti |
| Soft update | Yumuşak güncelleme |
| Exploration / Exploitation | Keşif / sömürü |

---

## 47. Tam Python kodu

> Kodlar ders içeriğini örneklemek için yazılmıştır; sayılar örnektir.

```python
import numpy as np
import tensorflow as tf

# ---------- 1) K-means (Bölüm 3–5) ----------
def find_closest_centroids(X, centroids):
    # her nokta için en yakın merkezin indeksi
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2) ** 2   # (m, K)
    return np.argmin(dists, axis=1)

def compute_centroids(X, idx, K):
    return np.array([X[idx == k].mean(axis=0) for k in range(K)])    # boş küme olmamalı

def run_kmeans(X, K, iters=10):
    centroids = X[np.random.permutation(X.shape[0])[:K]]             # K örnek seç
    for _ in range(iters):
        idx = find_closest_centroids(X, centroids)                   # atama
        centroids = compute_centroids(X, idx, K)                     # taşıma
    return centroids, idx

def distortion(X, centroids, idx):
    return np.mean(np.sum((X - centroids[idx]) ** 2, axis=1))        # J

def best_of_many(X, K, n_init=100):
    best = None
    for _ in range(n_init):
        c, i = run_kmeans(X, K)
        J = distortion(X, c, i)
        if best is None or J < best[0]:
            best = (J, c, i)
    return best                                                       # en düşük J

# ---------- 2) Anomali tespiti (Bölüm 8–10) ----------
def estimate_gaussian(X):
    mu = X.mean(axis=0)
    var = X.var(axis=0)                                               # 1/m varyans
    return mu, var

def p_of_x(X, mu, var):
    p = np.exp(-((X - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
    return np.prod(p, axis=1)                                         # özellikleri çarp

def select_threshold(y_val, p_val):
    best_eps, best_F1 = 0, 0
    for eps in np.linspace(p_val.min(), p_val.max(), 1000):
        pred = (p_val < eps)
        tp = np.sum(pred & (y_val == 1)); fp = np.sum(pred & (y_val == 0)); fn = np.sum(~pred & (y_val == 1))
        if tp == 0: continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        F1 = 2 * prec * rec / (prec + rec)
        if F1 > best_F1: best_eps, best_F1 = eps, F1
    return best_eps, best_F1

# ---------- 3) Collaborative filtering (Bölüm 15–18) ----------
def cofi_cost(X, W, b, Y, R, lambda_):
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
    return 0.5 * tf.reduce_sum(j ** 2) + (lambda_ / 2) * (tf.reduce_sum(X ** 2) + tf.reduce_sum(W ** 2))

def normalize_ratings(Y, R):
    mu = (np.sum(Y * R, axis=1) / (np.sum(R, axis=1) + 1e-12)).reshape(-1, 1)   # film başına ortalama
    return (Y - mu) * R, mu

# (X, W, b tf.Variable ; Ynorm, mu = normalize_ratings(Y, R))
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)
# for it in range(200):
#     with tf.GradientTape() as tape:
#         cost = cofi_cost(X, W, b, Ynorm, R, 1.0)
#     grads = tape.gradient(cost, [X, W, b])
#     optimizer.apply_gradients(zip(grads, [X, W, b]))
# tahmin = X @ W.T + b + mu

# ---------- 4) PCA (Bölüm 26) ----------
from sklearn.decomposition import PCA
X_pca = np.array([[1, 1], [2, 1], [3, 2], [-1, -1], [-2, -1], [-3, -2]])
pca = PCA(n_components=1).fit(X_pca)
print("Açıklanan varyans:", pca.explained_variance_ratio_)           # ≈ [0.992]
Z = pca.transform(X_pca)                                              # z değerleri
X_back = pca.inverse_transform(Z)                                     # yaklaşık geri dönüş

# ---------- 5) Pekiştirmeli öğrenme parçaları (Bölüm 37–39) ----------
def choose_action(q_net, state, epsilon, n_actions=4):
    if np.random.rand() < epsilon:                                    # keşfet
        return np.random.randint(n_actions)
    q_values = q_net(state[None, :]).numpy()[0]                       # tek geçişte 4 Q değeri
    return int(np.argmax(q_values))                                   # sömür

def dqn_targets(q_net, rewards, next_states, dones, gamma=0.985):
    max_next_q = np.max(q_net(next_states).numpy(), axis=1)
    return rewards + gamma * max_next_q * (1 - dones)                 # terminalde y = R(s)

def soft_update(q_net, q_new, tau=0.01):
    for w, w_new in zip(q_net.weights, q_new.weights):
        w.assign(tau * w_new + (1 - tau) * w)                         # W = τ·W_new + (1-τ)·W
```

---

# TEKRAR VE PRATİK

## 48. Elle hesap örnekleri

Formülleri anlamanın en hızlı yolu **kâğıtta** sayıları kendin çıkarmak.

### A) K-means: bir iterasyon

Veri (tek boyut): $x = [1,\ 2,\ 6,\ 7]$, $K=2$, başlangıç $\mu_1=1,\ \mu_2=7$.

**Atama:** 1 → küme 1 ($|1-1|=0$); 2 → küme 1 ($1$ vs $5$); 6 → küme 2 ($5$ vs $1$); 7 → küme 2.
**Taşıma:** $\mu_1=(1+2)/2=1.5$, $\mu_2=(6+7)/2=6.5$.
**Cost:** $J=\frac14\left[(1-1.5)^2+(2-1.5)^2+(6-6.5)^2+(7-6.5)^2\right]=\frac14(0.25\cdot4)=\mathbf{0.25}$.

### B) Anomali tespiti: Gauss

Bir özellik için eğitim verisi: $x=[4,\ 5,\ 5,\ 6]$.
$\mu = 5$, $\sigma^2 = \frac{(1+0+0+1)}{4} = 0.5$.

$p(x)=\frac{1}{\sqrt{2\pi\cdot0.5}}\exp\left(-\frac{(x-5)^2}{2\cdot0.5}\right)$, katsayı $\approx 0.564$.

- $x=5.5$: $0.564\cdot e^{-0.25}=0.564\cdot0.779\approx\mathbf{0.44}$ → $\varepsilon=0.02$ için **normal**.
- $x=8$: $0.564\cdot e^{-9}\approx 0.564\cdot0.000123\approx\mathbf{0.00007}$ → **anomali**.

### C) Öneri: tahmin ve mean normalization

**Tahmin:** $\vec{w}=[5,0],\ b=0,\ \vec{x}=[0.99,\ 0]$ → $5\cdot0.99+0\cdot0=\mathbf{4.95}$.

**Mean normalization:** Bir filmin puanları $[5,5,0,0]$, $\mu=2.5$. Normalize: $[2.5,2.5,-2.5,-2.5]$. Yeni kullanıcı ($w=0,b=0$) → tahmin $=0+2.5=\mathbf{2.5}$.

**Parametre sayısı:** $n_u=4,n_m=5,n=2$ → $W$: 8, $b$: 4, $X$: 10 → **22**.

### D) Return ($\gamma=0.5$)

3. durumdan **hep sağa** giderse: ödüller $0,0,0,40$ (durum 3→4→5→6):

$$0 + 0.5\cdot0 + 0.25\cdot0 + 0.125\cdot40 = \mathbf{5}$$

### E) Bellman ($\gamma=0.5$)

$\max_a Q(4,a) = 12.5$ biliniyorsa: $Q(3,\rightarrow) = 0 + 0.5\cdot12.5 = \mathbf{6.25}$.
Terminal 6'da $Q=40$ ise: $Q(5,\rightarrow) = 0+0.5\cdot40=\mathbf{20}$.

### F) Soft update

$W=2.0$, $W_{new}=4.0$, $\tau=0.01$ → $W=0.01\cdot4+0.99\cdot2=\mathbf{2.02}$.

### G) $\epsilon$-greedy olasılıkları

$\epsilon=0.05$, 4 eylem: greedy eylem seçilme olasılığı **$0.95 + 0.05/4 = 0.9625$**, diğer her biri **$0.05/4=0.0125$**.

---

## 49. Kendini test et

Önce cevabı **kendin yaz**, sonra aç ve karşılaştır. Takıldığın sorunun ait olduğu bölüme dön.

**Hafta 1: K-means ve anomali tespiti**

<details><summary><b>S1.</b> K-means'in iki adımı nedir?</summary>

1) Her noktayı **en yakın** küme merkezine ata, 2) her merkezi, atanmış noktaların **ortalamasına** taşı. Değişmeyene kadar tekrarla. (Bölüm 2–3)
</details>

<details><summary><b>S2.</b> K-means'in cost function'ı nedir? İterasyonlarda $J$ artarsa ne anlama gelir?</summary>

$J=\frac1m\sum\lVert x^{(i)}-\mu_{c^{(i)}}\rVert^2$ (distortion). K-means'te $J$ **asla artmaz**; artıyorsa **bug** vardır. (Bölüm 4)
</details>

<details><summary><b>S3.</b> K-means kötü sonuç veriyorsa ne yaparsın? Neden olur?</summary>

Rastgele başlatma **kötü yerel minimuma** götürebilir. **50–1000 kez** farklı başlatmayla çalıştır, **en düşük $J$**'yi seç. (Bölüm 5)
</details>

<details><summary><b>S4.</b> $K$'yı nasıl seçersin? Neden $J$'yi minimize etmek işe yaramaz?</summary>

Dirsek yöntemi veya (daha iyisi) **amaca göre**. $K$ arttıkça $J$ **her zaman düşer** ($K=m$ ise $J=0$). (Bölüm 6)
</details>

<details><summary><b>S5.</b> Anomali tespit algoritmasında $p(x)$ nasıl hesaplanır? Anomaliyi nasıl işaretlersin?</summary>

Her özellik için Gauss ($\mu_j,\sigma_j^2$), **çarpım**: $p(\vec{x})=\prod p(x_j)$. $p(\vec{x})<\varepsilon$ ise **anomali**. (Bölüm 9)
</details>

<details><summary><b>S6.</b> $x=[4,5,5,6]$ için $\mu$ ve $\sigma^2$ kaçtır?</summary>

$\mu=5$, $\sigma^2=(1+0+0+1)/4=0.5$. (Bölüm 48-B)
</details>

<details><summary><b>S7.</b> Anomali tespit sisteminde $\varepsilon$'u nasıl seçersin? Hangi metriği kullanırsın?</summary>

**Etiketli CV setinde**, farklı $\varepsilon$'larla **F1** (precision/recall) hesapla, en yükseğini seç. **Accuracy kullanma** (çarpık veri). (Bölüm 10)
</details>

<details><summary><b>S8.</b> Ne zaman anomali tespiti, ne zaman denetimli öğrenme?</summary>

Az pozitif + gelecekteki anomaliler **farklı** olabilir → **anomali tespiti**. Çok pozitif + gelecekteki pozitifler **benzer** → **denetimli**. (Bölüm 11)
</details>

<details><summary><b>S9.</b> Özellik Gauss değilse ne yaparsın? Hata analizinde ne yaparsın?</summary>

$\log(x+c)$, $\sqrt{x}$ ile **dönüştür** (her sette aynı). Kaçan anomaliye bakıp onu ayıran **yeni/kombinasyon özellik** üret (örn. CPU/ağ trafiği oranı). (Bölüm 12)
</details>

**Hafta 2: Öneri sistemleri**

<details><summary><b>S10.</b> $r(i,j)$ ve $y^{(i,j)}$ ne demek?</summary>

$r(i,j)=1$: kullanıcı $j$ film $i$'yi puanladı. $y^{(i,j)}$: verdiği **puan** (sadece $r=1$ ise tanımlı). (Bölüm 13)
</details>

<details><summary><b>S11.</b> Collaborative filtering'de özellikleri ($x$) nasıl elde ederiz?</summary>

**Öğreniriz**. Cost'ta $w,b$ ile birlikte $x$ de parametredir; gradient descent üçünü birden günceller. (Bölüm 15)
</details>

<details><summary><b>S12.</b> $n_u=4$, $n_m=5$, $n=2$ için parametre sayısı?</summary>

$W$: $4\cdot2=8$, $b$: 4, $X$: $5\cdot2=10$ → **22**. (Bölüm 48-C)
</details>

<details><summary><b>S13.</b> Mean normalization neden gerekir? Satır mı sütun mu normalleştirilir?</summary>

Yeni kullanıcıya **0** yerine **filmin ortalama puanı** tahmin edilsin diye. **Satır (film)** ortalaması çıkarılır, tahminde **geri eklenir**. (Bölüm 17)
</details>

<details><summary><b>S14.</b> İkili etiketlerde ($1$=etkileşti, $0$=etkileşmedi) cost nasıl değişir?</summary>

$f=g(\vec{w}\cdot\vec{x}+b)$ (sigmoid), loss **binary cross-entropy**. (Bölüm 16)
</details>

<details><summary><b>S15.</b> <code>tf.GradientTape</code> ne işe yarar? Neden collaborative filtering'de lazım?</summary>

Hesaplamayı kaydedip **türevleri otomatik** alır. CF, standart `model.fit` yapısına uymadığı için **kendi cost'unu** yazıp türevi `GradientTape` ile alırız. (Bölüm 18)
</details>

<details><summary><b>S16.</b> Collaborative filtering'in iki sınırlaması nedir?</summary>

**Soğuk başlangıç** (yeni öğe/kullanıcı) ve **yan bilgiyi kullanmama**. (Bölüm 19)
</details>

<details><summary><b>S17.</b> Collaborative ile content-based filtering arasındaki fark?</summary>

CF: **benzer kullanıcıların puanlarına** göre. Content-based: **kullanıcı ve öğe özelliklerine** göre eşleşme ($\vec{v}_u\cdot\vec{v}_m$). (Bölüm 20)
</details>

<details><summary><b>S18.</b> Content-based'de $\vec{v}_u$ ve $\vec{v}_m$ boyutları ne olmalı? $\vec{x}_u$ ve $\vec{x}_m$?</summary>

$\vec{v}_u$ ve $\vec{v}_m$ **aynı boyutta** olmalı (dot product için). $\vec{x}_u$ ve $\vec{x}_m$ **farklı boyutta** olabilir. (Bölüm 20–21)
</details>

<details><summary><b>S19.</b> Retrieval ve ranking nedir? Neden iki aşama?</summary>

Retrieval: **hızlı** makul adaylar. Ranking: adaylara **modeli** uygula, sırala. Tüm kataloğa model çalıştırmak **çok yavaş**. (Bölüm 22)
</details>

<details><summary><b>S20.</b> Öneri sistemlerinde etik sorun nedir? Örnek ver.</summary>

Hedef **kâr/etkileşim** olursa, **sömürücü reklam** (payday loan) veya **kışkırtıcı içerik** yayılabilir. Filtreleme, şeffaflık, doğru hedef seçimi gerekir. (Bölüm 23)
</details>

**Hafta 3: PCA**

<details><summary><b>S21.</b> PCA ne yapar? Ne için kullanılır?</summary>

Çok özelliği **az sayıda yeni eksene** indirir, varyansı **mümkün olduğunca korur**. Ana kullanım: **görselleştirme**. (Bölüm 24)
</details>

<details><summary><b>S22.</b> PCA hangi ekseni seçer? Neden varyans?</summary>

Veriyi izdüşürünce **varyansı en çok koruyan** ekseni (1. principal component). Varyans, noktaları **ayırt eden bilgi**dir. (Bölüm 25)
</details>

<details><summary><b>S23.</b> PCA ile lineer regresyon farkı?</summary>

Regresyonda **$y$ var** ve **dikey** mesafe; PCA'da **$y$ yok** ve **dik (perpendicular)** mesafe. (Bölüm 25)
</details>

<details><summary><b>S24.</b> <code>explained_variance_ratio_</code> 0.992 ne demek?</summary>

Seçilen eksen, orijinal verideki **varyansın %99.2'sini** yakalıyor. (Bölüm 26)
</details>

**Hafta 4: Pekiştirmeli öğrenme**

<details><summary><b>S25.</b> RL neden denetimli öğrenme değildir?</summary>

Her durum için **doğru eylem bilinmez**. Sadece **ödül** (ne istediğimiz) verilir, "nasıl"ı algoritma bulur. (Bölüm 27)
</details>

<details><summary><b>S26.</b> Return nedir? $\gamma=0.5$ ile ödüller $0,0,0,100$ için return?</summary>

$R_1+\gamma R_2+\gamma^2R_3+\dots$ → $0+0+0+0.125\cdot100=\mathbf{12.5}$. (Bölüm 29)
</details>

<details><summary><b>S27.</b> İndirim faktörü $\gamma$ küçükse ajan nasıl davranır?</summary>

**Sabırsız**: yakın ödülü tercih eder. $\gamma$ 1'e yakın → daha **sabırlı**. (Bölüm 29)
</details>

<details><summary><b>S28.</b> $Q(s,a)$ ne demek? Optimal eylemi nasıl bulursun?</summary>

$s$'de $a$ yapıp sonra **optimal davranınca** alacağın return. Optimal eylem: $\arg\max_a Q(s,a)$. (Bölüm 32)
</details>

<details><summary><b>S29.</b> Bellman denklemini yaz ve sezgisini açıkla.</summary>

$Q(s,a)=R(s)+\gamma\max_{a'}Q(s',a')$: **şimdiki ödül + sonraki durumdan optimal return (indirimli)**. (Bölüm 33)
</details>

<details><summary><b>S30.</b> $\max_aQ(4,a)=12.5$, $\gamma=0.5$ ise $Q(3,\rightarrow)$ kaçtır?</summary>

$0+0.5\cdot12.5=\mathbf{6.25}$. (Bölüm 48-E)
</details>

<details><summary><b>S31.</b> Stokastik ortamda hedef nedir?</summary>

**Beklenen return'ü** (çok denemenin ortalaması) maksimize etmek. (Bölüm 34)
</details>

<details><summary><b>S32.</b> DQN'de eğitim hedefi $y$ nedir? Terminal durumda?</summary>

$y=R(s)+\gamma\max_{a'}Q(s',a')$. Terminalde $y=R(s)$. (Bölüm 37)
</details>

<details><summary><b>S33.</b> Neden $\epsilon$-greedy? $\epsilon$ nasıl ayarlanır?</summary>

Hep greedy olursa **keşfetmez**, yanlış tahminde **takılır**. $\epsilon=0.05$ rastgele eylem. Başta **yüksek (1.0)**, sonra **0.01'e** düşür. (Bölüm 38)
</details>

<details><summary><b>S34.</b> Mini-batch ve soft update ne işe yarar?</summary>

Mini-batch: her adımda **alt küme** → **hız**. Soft update: $W=\tau W_{new}+(1-\tau)W$ → **kararlılık**. (Bölüm 39)
</details>

<details><summary><b>S35.</b> $W=2.0$, $W_{new}=4.0$, $\tau=0.01$ için soft update sonucu?</summary>

$0.01\cdot4+0.99\cdot2=\mathbf{2.02}$. (Bölüm 48-F)
</details>

<details><summary><b>S36.</b> RL'nin bugünkü sınırlamaları nelerdir?</summary>

Simülasyonda çalışan algoritma **gerçek robotta zor**; denetimli/denetimsizden **çok az uygulama**; ama potansiyeli yüksek. (Bölüm 40)
</details>

**Kod**

<details><summary><b>S37.</b> Bölüm 47'ye bakmadan <code>find_closest_centroids</code> ve <code>compute_centroids</code> fonksiyonlarını yaz.</summary>

```python
def find_closest_centroids(X, centroids):
    d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2) ** 2
    return np.argmin(d, axis=1)

def compute_centroids(X, idx, K):
    return np.array([X[idx == k].mean(axis=0) for k in range(K)])
```
</details>

---

## 50. Tekrar planı

Aralıklı tekrar, aynı bilgiyi yoğun bir günde okumaktan çok daha kalıcıdır. **Bir haftayı bir günde bitirmeye çalışma.**

| Gün | Ne yap? |
|---|---|
| **Gün 1** | **Hafta 1 – K-means** (Bölüm 1–6). Kendi cümlelerinle anlat. Bölüm 48-A'yı kâğıtta yap |
| **Gün 2** | **Hafta 1 – Anomali tespiti** (Bölüm 7–12). Bölüm 48-B'yi yap |
| **Gün 3** | **Hafta 2 – Collaborative filtering** (Bölüm 13–19). Bölüm 48-C |
| **Gün 4** | **Hafta 2 – Content-based, retrieval/ranking, etik** (Bölüm 20–23) |
| **Gün 5** | **Hafta 3 – PCA** (Bölüm 24–26) |
| **Gün 6** | **Hafta 4 – RL temelleri**: durum, ödül, return, Q, Bellman (Bölüm 27–34). Bölüm 48-D, E |
| **Gün 7** | **Hafta 4 – DQN**: ağ, $\epsilon$-greedy, mini-batch, soft update (Bölüm 35–41). Bölüm 48-F, G |
| **Gün 10** | Bölüm 49'daki soruları **cevaplara bakmadan** çöz. Yanlışları işaretle |
| **Gün 14** | Sadece işaretli yanlışlara dön. Bölüm 47'deki kodu **bakmadan** yaz |
| **Gün 21** | Bölüm 44 (sorun → çözüm) ve 45 (hatalar/karıştırılanlar) tablolarını gözden geçir |
| **Gün 30** | Formül kartını (Bölüm 43) bakmadan boş kâğıda yaz |

> 💡 **Önemli:** Bir bölümü anlamadan sonrakine geçme. Takıldığında önceki kursların notlarına dön (özellikle gradient descent, logistic regression, sinir ağları, precision/recall/F1).

---

*Bu notlar, "Unsupervised Learning, Recommenders, Reinforcement Learning" kursunun (Hafta 1–4) video transkriptlerinden hazırlanmıştır.*
