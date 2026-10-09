# Supervised Machine Learning: Regression and Classification

> Ders notları (Hafta 1–3), tek dosyada ve **mantığı anlaşılacak** şekilde özetlenmiştir.
> Her bölümde "ne?" kadar **"neden?"** sorusuna da cevap verilir. Tekrar bakarken önce bölüm sonlarındaki **🧠 Neden?** kutularını oku.

---

## İçindekiler

**Hafta 1 – Temeller**
1. [ML'e giriş: araç vs. ustalık](#1-mlye-giriş-araç-vs-ustalık)
2. [Supervised Learning, Regression, Classification](#2-supervised-learning-regression-classification)
3. [Unsupervised Learning](#3-unsupervised-learning)
4. [Tek değişkenli lineer regresyon modeli](#4-tek-değişkenli-lineer-regresyon-modeli)
5. [Cost Function (Maliyet Fonksiyonu)](#5-cost-function-maliyet-fonksiyonu)
6. [Cost Function'ın geometrik sezgisi](#6-cost-functionın-geometrik-sezgisi)
7. [Gradient Descent](#7-gradient-descent)
8. [Lineer regresyonda Gradient Descent](#8-lineer-regresyonda-gradient-descent)

**Hafta 2 – Çoklu özellik ve pratik ipuçları**
9. [Multiple Linear Regression](#9-multiple-linear-regression)
10. [Vectorization](#10-vectorization)
11. [Feature Scaling](#11-feature-scaling)
12. [Gradient Descent çalışıyor mu? (Learning curve)](#12-gradient-descent-çalışıyor-mu-learning-curve)
13. [Learning rate (α) seçimi](#13-learning-rate-α-seçimi)
14. [Feature Engineering ve Polynomial Regression](#14-feature-engineering-ve-polynomial-regression)

**Hafta 3 – Classification ve Overfitting**
15. [Classification ve neden lineer regresyon olmaz](#15-classification-ve-neden-lineer-regresyon-olmaz)
16. [Logistic Regression ve Sigmoid](#16-logistic-regression-ve-sigmoid)
17. [Decision Boundary](#17-decision-boundary)
18. [Logistic Regression için Cost Function](#18-logistic-regression-için-cost-function)
19. [Logistic Regression'da Gradient Descent](#19-logistic-regressionda-gradient-descent)
20. [Overfitting ve Underfitting](#20-overfitting-ve-underfitting)
21. [Regularization](#21-regularization)

**Ekler**
22. [Büyük resim: tüm hikâye tek akışta](#22-büyük-resim-tüm-hikâye-tek-akışta)
23. [Hızlı formül kartı](#23-hızlı-formül-kartı)
24. [Terim sözlüğü](#24-terim-sözlüğü)
25. [Tam Python kodu](#25-tam-python-kodu)

---

# HAFTA 1 – TEMELLER

## 1. ML'e giriş: araç vs. ustalık

- Algoritmalar birer **alettir** (çekiç, matkap gibi). Önemli olan aleti bilmek değil, **hangi problemde hangisini kullanacağını** bilmektir.
- Deneyimli ekipler bile yanlış yaklaşım seçince aylarca sonuç alamayabilir.

> 🧠 **Neden?** Körü körüne denemek zaman kaybıdır. Baştan doğru yaklaşımı seçmek (supervised mı unsupervised mı, regression mı classification mı) projenin kaderini belirler. Bu notların amacı o seçimi yapabilmeni sağlamak.

---

## 2. Supervised Learning, Regression, Classification

### Supervised Learning (Gözetimli Öğrenme)

Modele **girdi ($x$)** ve ona karşılık gelen **doğru cevap ($y$, label)** verilir. Model $x \rightarrow y$ eşlemesini öğrenir; sonra hiç görmediği yeni bir $x$ için $y$ tahmin eder.

| Uygulama | Girdi ($x$) | Çıktı ($y$) |
|---|---|---|
| Spam filtresi | E-posta | Spam / değil |
| Konuşma tanıma | Ses kaydı | Metin |
| Makine çevirisi | Kaynak dil | Hedef dil |
| Online reklam | Kullanıcı + reklam | Tıklama olasılığı |
| Görsel denetim | Ürün fotoğrafı | Kusur var / yok |

### Supervised'ın iki türü

| | **Regression** | **Classification** |
|---|---|---|
| Çıktı | **Sonsuz** olası sürekli sayı | **Sonlu** sayıda ayrık sınıf |
| Örnek | Ev fiyatı: 150.000, 183.500 ... | 0 veya 1; kedi / köpek |
| Ara değer? | Var | **Yok** |

Classification kendi içinde ikiye ayrılır:
- **Binary classification:** 2 sınıf (iyi huylu = 0, kötü huylu = 1)
- **Multi-class classification:** 2'den fazla sınıf (Tip 1 kanser, Tip 2 kanser, kanser değil)

Birden fazla özellik (feature) kullanılabilir: tümör boyutu + yaş + hücre şekli vb. Model, sınıfları ayıran **decision boundary**'yi (karar sınırı) öğrenir.

> 🧠 **Neden bu ayrım önemli?** Çıktının *türü* algoritmayı belirler. Fiyat tahmin ediyorsan regression, "evet/hayır" tahmin ediyorsan classification. Yanlış türü seçersen (örn. 0/1 problemine düz çizgi uydurmak) model bozuk çalışır (bkz. Bölüm 15).

---

## 3. Unsupervised Learning

Elinde **sadece $x$** var, **doğru cevap ($y$) yok**. Algoritma verideki gizli yapıyı, benzerlikleri, sıradışılıkları **kendi başına** bulur.

| | Supervised | Unsupervised |
|---|---|---|
| Veri | $(x, y)$ çiftleri | Sadece $x$ |
| Algoritmaya söylenen | "Bu $x$'in cevabı $y$, kuralı öğren" | "Sadece $x$ var, grupları kendin bul" |

### Üç ana tür

| Tür | Ne yapar? | Örnek |
|---|---|---|
| **Clustering** | Benzer noktaları gruplar | Google News'te aynı konudaki haberleri toplama, müşteri segmentasyonu, DNA/gen ifadesi alt tipleri |
| **Anomaly Detection** | Normale uymayan olağandışıyı bulur | Kredi kartı sahtekârlığı |
| **Dimensionality Reduction** | Büyük veriyi bilgi kaybını minimumda tutarak sıkıştırır | Çok özellikli veriyi az özelliğe indirme |

### Hızlı sınıflama alıştırması

| Problem | Tür | Neden? |
|---|---|---|
| Spam filtreleme | Supervised | E-postalar önceden spam / değil etiketli |
| Haber gruplama | Unsupervised | Etiket yok, benzerliğe göre kümelenir |
| Pazar segmentasyonu | Unsupervised | Segmentler önceden tanımlı değil |
| Diyabet teşhisi | Supervised | Hasta etiketleri (diyabetli / değil) var |

> 🧠 **Neden?** Test: *"Elimde doğru cevap etiketleri var mı?"* Varsa supervised, yoksa unsupervised. Bu tek soru çoğu zaman yeterli.

---

## 4. Tek değişkenli lineer regresyon modeli

### Notasyon

| Sembol | Anlamı |
|---|---|
| $x$ | Girdi / **feature** (örn. evin metrekaresi) |
| $y$ | Gerçek hedef / **target** (gerçek satış fiyatı) |
| $\hat{y}$ | Modelin **tahmini** çıktısı |
| $m$ | Eğitim örneği sayısı |
| $(x^{(i)}, y^{(i)})$ | $i$. eğitim örneği (**üs değil, indis**) |

### Model

$$f_{w,b}(x) = wx + b$$

- **$w$ (weight):** doğrunun **eğimi**
- **$b$ (bias):** doğrunun **y eksenini kestiği nokta**
- **Amaç:** Veriye en iyi oturan doğruyu çizecek $w$ ve $b$ değerlerini bulmak.

> 🧠 **Neden $w$ ve $b$ "parametre"?** Model yapısı (düz çizgi) sabit; öğrenilen şey sadece bu iki sayı. Eğitim = bu iki sayıyı veriden bulmak.

---

## 5. Cost Function (Maliyet Fonksiyonu)

"Bu $w, b$ ne kadar kötü?" sorusuna **tek bir sayı** ile cevap verir.

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2$$

Parçaları:

| Parça | Ne işe yarar? |
|---|---|
| $f(x^{(i)}) - y^{(i)}$ | **Hata** (tahmin − gerçek) |
| **Karesi** | Negatifleri pozitif yapar (hatalar birbirini götürmesin) + **büyük hataları daha sert cezalandırır** |
| $\sum$ | Tüm örneklerin hatasını toplar |
| $\frac{1}{m}$ | **Ortalama** alır; veri sayısı arttı diye maliyet yapay büyümesin |
| $\frac{1}{2}$ | Türev alınca üsten gelen **2 ile sadeleşsin** diye (sadece matematiksel kolaylık) |

> 🧠 **Neden bir cost function lazım?** "Bu çizgi iyi mi?" sorusu göze bağlıdır. $J$ bunu **ölçülebilir** yapar. Sonra hedef çok net olur: **$J$'yi minimuma indiren $w, b$'yi bul.**

---

## 6. Cost Function'ın geometrik sezgisi

### A. Tek parametre ($b=0$, yani $f(x)=wx$)

- $J(w)$ grafiği **U şeklinde parabol** olur.
- Çizgi veriye tam oturunca $J \approx 0$ (dip nokta).
- $w$ doğru değerden uzaklaştıkça parabolün kolları **hızla yukarı** çıkar.

### B. İki parametre ($w$ ve $b$)

- $J(w,b)$ **3 boyutlu bir çorba kasesi / hamak** olur.
- **Contour plot (kontür grafiği):** Kaseyi yatay dilimlersek **iç içe elipsler** çıkar.
  - Aynı elips üzerindeki tüm $(w,b)$ çiftlerinin $J$ değeri **eşit**.
  - Elipslerin **merkezi = kasenin dibi = global minimum**.
  - Merkezin $(w,b)$ değerleri = veriye **en iyi uyan doğru**.

<img width="2752" height="1536" alt="Cost function 3D yüzey ve kontür grafiği" src="https://github.com/user-attachments/assets/b73f7ab8-926d-444a-9920-5d2d3b2819bb" />

> 🧠 **Neden bu görsel önemli?** Eğitimi artık bir **"kasenin dibini bulma" oyunu** olarak düşünebilirsin. Gradient Descent tam olarak bu oyunu otomatik oynar.

**Sonraki adım:** Elle (grafikten) aramak karmaşık problemlerde imkânsız → **Gradient Descent**.

---

## 7. Gradient Descent

### Amaç ve kapsam

$J(w,b)$'yi minimize eden $w, b$'yi **sistematik** bulmak.
- Sadece lineer regresyonda değil, **derin öğrenme dahil ML'in her yerinde** kullanılır.
- 2 parametreyle sınırlı değil: $J(w_1,\dots,w_n,b)$ için de çalışır.

### Fikir (dağ/vadi benzetmesi 🏔️)

1. Rastgele bir noktadan başla (lineer regresyonda genelde $w=0, b=0$).
2. Etrafına bak: *"Minik bir adım atsam, hangi yön beni en hızlı aşağı indirir?"* (= en dik iniş yönü)
3. O yöne **küçük adım** at.
4. Yakınsayana (değişim bitene) kadar tekrarla.

### Güncelleme formülü

$$w = w - \alpha \frac{\partial J(w,b)}{\partial w} \qquad b = b - \alpha \frac{\partial J(w,b)}{\partial b}$$

| Sembol | Anlamı |
|---|---|
| `=` | Burada **atama** (kodlardaki gibi), matematiksel eşitlik değil |
| $\alpha$ | **Learning rate** (öğrenme oranı): adım **büyüklüğü**, küçük pozitif sayı (örn. 0.01) |
| $\frac{\partial J}{\partial w}$ | **Türev**: hangi **yöne** gidileceğini söyler |

### ⚠️ Simultaneous (eş zamanlı) güncelleme

Önce **ikisini de eski değerlerle** hesapla, sonra birlikte ata:

```python
tmp_w = w - alpha * dJ_dw     # ikisi de ESKİ w, b ile hesaplanır
tmp_b = b - alpha * dJ_db
w = tmp_w
b = tmp_b
```

❌ Yanlış: `w`'yi güncelleyip sonra `b`'nin türevini **yeni w** ile hesaplamak. Bu farklı bir algoritma olur.

> 🧠 **Neden eş zamanlı?** Bir adım = "şu anki konumdan" bakıp karar vermek. Yarı yolda konumu değiştirip bakmak, o adımın tanımını bozar.

### Türevin mantığı (sezgi)

Türev = o noktada eğriye çizilen **teğetin eğimi**.

| Konum | Türev | $w - \alpha \cdot \text{türev}$ | Hareket |
|---|---|---|---|
| Minimumun **sağında** | Pozitif | $w$ **azalır** | Sola → $J$ düşer ✅ |
| Minimumun **solunda** | Negatif | $w$ **artar** (negatiften çıkarmak = eklemek) | Sağa → $J$ düşer ✅ |
| Minimumda | 0 | Değişmez | Durur |

> 🧠 **Neden formülde eksi işareti var?** Türev **yukarı** çıkış yönünü gösterir. Biz **aşağı** inmek istiyoruz, o yüzden türevi **çıkarırız**. Eksiyi yanlışlıkla artıya çevirmek klasik bug'dır (cost büyür).

### Learning rate ($\alpha$) etkisi

| $\alpha$ | Ne olur? |
|---|---|
| **Çok küçük** 🐢 | Çalışır ama **çok yavaş** |
| **Çok büyük** 🚀 | Minimumu **aşar**, karşı tarafa atlar; cost azalmak yerine artar, **diverge** (ıraksama) olabilir |
| Uygun | Hızlı ve istikrarlı iner |

**Güzel özellik:** $\alpha$ sabit olsa bile, minimuma yaklaştıkça **türev küçülür** → adımlar **otomatik küçülür** → algoritma kendiliğinden yavaşlayıp oturur.

### Local minimum

- Genel cost fonksiyonlarında (sinir ağları) birden fazla vadi olabilir; başlangıç noktasına göre **farklı yerel minimuma** inebilirsin.
- **Lineer regresyonda bu sorun yok** (bkz. Bölüm 8).

---

## 8. Lineer regresyonda Gradient Descent

Parçalar: model $f(x)=wx+b$ + squared error cost + gradient descent.

**Türevler** (kalkülüsle çıkarılır, ezberlemen yeterli):

$$\frac{\partial J}{\partial w} = \frac{1}{m}\sum_{i=1}^{m}\left(f(x^{(i)}) - y^{(i)}\right)x^{(i)} \qquad \frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}\left(f(x^{(i)}) - y^{(i)}\right)$$

> 🧠 **Neden $b$'de $x^{(i)}$ çarpanı yok?** $f = wx + b$ olduğundan $f$'nin $b$'ye göre türevi 1, $w$'ye göre türevi $x$'tir. Bu fark formüle olduğu gibi yansır.
> **Neden cost'ta $\frac{1}{2m}$ vardı?** Karenin türevinden gelen 2, paydaki 2 ile sadeleşir → formüller temiz çıkar.

**Algoritma:** Yakınsayana kadar (eş zamanlı):

$$w = w - \alpha \cdot \frac{1}{m}\sum (f - y)\,x \qquad b = b - \alpha \cdot \frac{1}{m}\sum (f - y)$$

### Neden yerel minimum tuzağı yok?

- Lineer regresyonun squared-error cost'u **convex** (kase şeklinde): **tek bir** minimum var = **global minimum**.
- Uygun $\alpha$ ile gradient descent **her zaman** oraya yakınsar, başlangıç noktası önemsiz.

### Batch Gradient Descent

Her güncelleme adımında **tüm $m$ eğitim örneği** kullanılır (küçük alt küme değil). İleride her adımda veri alt kümesi kullanan varyantlar da var.

---

# HAFTA 2 – ÇOKLU ÖZELLİK VE PRATİK İPUÇLARI

## 9. Multiple Linear Regression

Ev fiyatını sadece metrekareyle değil **birçok özellikle** tahmin et.

| Sembol | Anlamı |
|---|---|
| $n$ | Toplam özellik sayısı |
| $x_j$ | $j$. özellik |
| $\vec{x}^{(i)}$ | $i$. örneğin **tüm özellik vektörü** |
| $x_j^{(i)}$ | $i$. örneğin $j$. özelliği (tek sayı) |

Örnek: $\vec{x}^{(2)} = [1416, 3, 2, 40]$, $x_3^{(2)} = 2$.

**Model:**

$$f(\vec{x}) = w_1x_1 + w_2x_2 + \dots + w_nx_n + b = \vec{w}\cdot\vec{x} + b$$

- $\vec{w}\cdot\vec{x}$ = **dot product**: karşılıklı elemanları çarp, hepsini topla.
- $\vec{w}$ vektör, $b$ **tek sayı**.

**Parametre yorumu** (örnek: $f = 0.1x_1 + 4x_2 + 10x_3 - 2x_4 + 80$, bin $):

| Parametre | Anlamı |
|---|---|
| $b=80$ | Taban fiyat |
| $w_1=0.1$ | Her ekstra feet² → +100 $ |
| $w_2=4$ | Her ekstra oda → +4 bin $ |
| $w_3=10$ | Her ekstra kat → +10 bin $ |
| $w_4=-2$ | Ev 1 yıl yaşlandıkça −2 bin $ (**negatif ağırlık = fiyatı düşürür**) |

> 🧠 **Neden vektör notasyonu?** Uzun toplamı $\vec{w}\cdot\vec{x}+b$ ile kısa yazarsın, üstelik kodda tek satıra (hızlı) dönüşür.

**Gradient descent (çoklu):** Her $w_j$ için ($j=1..n$) ve eş zamanlı:

$$w_j = w_j - \alpha\frac{1}{m}\sum (f(\vec{x}^{(i)}) - y^{(i)})\,x_j^{(i)} \qquad b = b - \alpha\frac{1}{m}\sum (f(\vec{x}^{(i)}) - y^{(i)})$$

Yapı tek özellikle **aynı**; sadece her $w_j$ kendi özelliği $x_j$ ile çarpılan hatayı kullanır.

**Normal equation (dipnot):** $w, b$'yi iteratif değil **tek seferde** (lineer cebirle) bulan alternatif. Sadece lineer regresyona özel, çok özellikte yavaş, pratikte pek kullanılmaz. Mülakatta adı geçerse bu kastedilir.

---

## 10. Vectorization

Döngü yerine **tüm vektör işlemini tek seferde** yapan kütüphane fonksiyonu (NumPy).

```python
# ❌ Elle (n büyüyünce imkânsız)
f = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + b

# ⚠️ for döngüsü (yavaş)
f = 0
for j in range(n):          # 0..n-1 (n dahil DEĞİL)
    f = f + w[j] * x[j]
f = f + b

# ✅ Vectorized (kısa + hızlı)
f = np.dot(w, x) + b
```

Gradient descent güncellemesi de aynı: `w = w - 0.1 * d` (döngüsüz).

> 🧠 **Neden daha hızlı?** Döngü çarpmaları **sırayla** yapar. `np.dot` ise CPU/GPU'nun **paralel donanımını** kullanır: tüm çarpmaları aynı anda yapar, toplamayı da verimli yapar. Binlerce özellikte fark **dakikalar vs. saatler** olur. Vectorization, ML'i büyük veriye ölçeklemenin anahtarıdır.

Not: Matematikte indeks 1'den ($w_1$), Python'da 0'dan ($w[0]$) başlar.

---

## 11. Feature Scaling

Özellikler çok farklı büyüklükteyse gradient descent **yavaşlar**. Çözüm: hepsini benzer aralığa getirmek.

### Neden gerekli? (örnek)

$x_1$ = metrekare (300–2000), $x_2$ = oda sayısı (0–5). Gerçek: 2000 feet², 5 oda, 500 bin $.

| Deneme | $w_1, w_2, b$ | Tahmin | Sonuç |
|---|---|---|---|
| A | 50, 0.1, 50 | ≈ 100 milyon $ | ❌ |
| B | 0.1, 50, 50 | 500 bin $ | ✅ |

**Çıkarım:** Değer aralığı **büyük** olan özelliğin iyi parametresi **küçük**, aralığı **küçük** olanın parametresi **büyük** olur.

> 🧠 **Neden bu yavaşlatır?** $w_1$ küçücük değişse bile (çok büyük $x_1$ ile çarpıldığı için) $J$ çok değişir; $w_2$'nin $J$'yi etkilemesi için büyük değişiklik gerekir. Sonuç: kontür grafiği **uzun ince elips**; gradient descent bu dar vadide **sağa sola zıplayarak** ilerler 🐌. Ölçekleyince kontürler **daireye** yaklaşır, algoritma minimuma **doğrudan** gider 🚀.

### Yöntemler

| Yöntem | Formül | Not |
|---|---|---|
| Max'a bölme | $x / x_{max}$ | En basit |
| Mean normalization | $\dfrac{x-\mu}{x_{max}-x_{min}}$ | Sıfır etrafında ortalar |
| **Z-score** | $\dfrac{x-\mu}{\sigma}$ | Ortalama ve standart sapma ($\sigma$) kullanır |

### Ne kadar ölçekleme yeterli?

Hedef: her özellik yaklaşık **−1 … +1**. Sınırlar esnek.

| Aralık | Ölçekle mi? |
|---|---|
| −3…+3, −0.3…+0.3, 0…3, −2…+0.5 | Gerekmez |
| **−100…+100** | ✅ (çok büyük) |
| **−0.001…+0.001** | ✅ (çok küçük) |
| **98.6…105** (vücut sıcaklığı °F) | ✅ (değerler ~100, fark küçük ama büyük sayılar) |

> 💡 **Altın kural:** Ölçeklemenin neredeyse hiç zararı yok. **Emin değilsen yap.**

---

## 12. Gradient Descent çalışıyor mu? (Learning curve)

**Learning curve:** Her iterasyondan sonra $J$'yi hesapla ve çiz.
- **Yatay eksen: iterasyon sayısı** (eskiden $w$ veya $b$'ydi, bu farklı!)
- **Dikey eksen:** $J$

| Gözlem | Anlam |
|---|---|
| $J$ **her iterasyonda düşüyor** | ✅ Çalışıyor |
| $J$ **bir kez bile artıyor** | ❌ $\alpha$ çok büyük **veya bug** |
| Eğri **düzleşiyor** | ✅ Yakınsadı (converged) |

Yakınsama 30, 1000 ya da 100.000 iterasyon sürebilir; önceden tahmin etmek zor → **eğriye bak**.

**Otomatik test (alternatif):** $\varepsilon$ (örn. 0.001) seç; bir iterasyonda $J$ $\varepsilon$'dan az düşerse yakınsadı say.

> 🧠 **Neden grafiği tercih et?** Doğru $\varepsilon$ seçmek zor. Grafik ise bir şey ters giderse **erkenden uyarı** verir.

---

## 13. Learning rate (α) seçimi

| Grafikte | Sebep | Çözüm |
|---|---|---|
| Cost bazen artıp bazen azalıyor (zıplıyor) | $\alpha$ çok büyük **veya bug** | $\alpha$'yı küçült |
| Cost sürekli artıyor | $\alpha$ çok büyük **veya bug** | $\alpha$'yı küçült, kodu kontrol et |

**Klasik bug:**
```python
w = w + alpha * dj_dw   # ❌ cost'u minimumdan uzaklaştırır
w = w - alpha * dj_dw   # ✅
```

**Debug ipucu 🔧:** Doğru yazılmış gradient descent'te **yeterince küçük $\alpha$ ile cost HER iterasyonda düşmelidir**. Çok küçük $\alpha$ ile bile artıyorsa **kodda hata vardır**. (Çok küçük $\alpha$ sadece debug içindir; gerçek eğitimde verimsiz.)

**İyi α nasıl seçilir?**
1. Değerleri dene: `0.001 → 0.003 → 0.01 → 0.03 → 0.1 → ...` (her biri ~3×).
2. Her biri için birkaç iterasyon çalıştırıp learning curve çiz.
3. **Çok küçük** ve **çok büyük** olanı bul.
4. Cost'u **hızlı ve tutarlı** düşüren **en büyük makul** $\alpha$'yı (veya biraz küçüğünü) seç.

---

## 14. Feature Engineering ve Polynomial Regression

### Feature Engineering

Problem hakkındaki **bilgi/sezgiyle** ham özellikleri dönüştürüp/birleştirerek **yeni özellik** üretmek.

Örnek: arsa genişliği $x_1$, derinliği $x_2$ → **alan** $x_3 = x_1 \cdot x_2$.

$$f(\vec{x}) = w_1x_1 + w_2x_2 + w_3x_3 + b$$

> 🧠 **Neden?** Fiyat belki genişlikten/derinlikten çok **alana** bağlıdır. Yeni özelliği eklersen model hangisinin önemli olduğuna **kendisi karar verir** ($w$'leri öğrenerek). Doğru özellik seçimi performansı çok etkiler.

### Polynomial Regression (eğri uydurma)

Çoklu lineer regresyon + feature engineering = veriye **eğri** uydurma.

| Model | Özellikler | Not |
|---|---|---|
| Düz çizgi | $x$ | Veriye uymuyor |
| Quadratic | $x, x^2$ | Sonunda **aşağı döner** → büyük evin fiyatı düşmesi mantıksız ❌ |
| Cubic | $x, x^2, x^3$ | Daha iyi uyum ✅ |
| Karekök | $x, \sqrt{x}$ | Eğim azalır, **hiç aşağı inmez** ✅ |

### ⚠️ Polinom özelliklerde scaling şart

$x$: 1–1.000, $x^2$: 1–1.000.000, $x^3$: 1–1.000.000.000. Aralıklar devasa farklı → **mutlaka ölçekle.**

Hangi özelliklerin seçileceği: şimdilik "seçeneğin var" bil; model performansını ölçüp karar verme sonraki kursta.

### Scikit-learn

Açık kaynaklı, endüstride yaygın ML kütüphanesi; lineer regresyon birkaç satır. Ama **algoritmaları sıfırdan yazmayı bil**, kütüphaneyi kara kutu olarak çağırma.

---

# HAFTA 3 – CLASSIFICATION VE OVERFITTING

## 15. Classification ve neden lineer regresyon olmaz

**Binary classification:** İki olası çıktı (spam mi? hileli işlem mi? tümör kötü huylu mu?).

| | Negatif sınıf | Pozitif sınıf |
|---|---|---|
| Sözel | hayır / yanlış | evet / doğru |
| Sayısal | $y=0$ | $y=1$ |

"Negatif/pozitif" **kötü/iyi demek değil**, sadece **yokluk (0)** ve **varlık (1)** demek. Hangisine 0 dediğin biraz keyfidir. "Sınıf" = "kategori".

### Neden lineer regresyon kötü?

- Çıktısı 0–1 ile sınırlı değildir (negatif veya 1'den büyük çıkabilir).
- Sağ tarafa **çok büyük bir tümör** örneği eklenince çizgi **kayar**, karar eşiği (0.5) de sağa kayar. Oysa o örnek zaten "kötü huylu"ydu, **hiçbir şeyi değiştirmemeliydi**. Sonuç: daha kötü sınıflandırıcı ❌.

> 🧠 **Neden?** Lineer regresyon "ne kadar yüksek" sorusu için tasarlandı. Uç değerler çizgiyi çeker. Classification'da ise sadece **hangi taraf** önemli. Çözüm: **Logistic Regression**.

⚠️ İsminde "regression" geçse de **classification algoritmasıdır** (tarihsel isimlendirme).

---

## 16. Logistic Regression ve Sigmoid

Veriye **S şeklinde eğri** uydurur; çıktı **her zaman 0–1** arasındadır.

### Sigmoid

$$g(z) = \frac{1}{1+e^{-z}}$$

| $z$ | $g(z)$ |
|---|---|
| Çok büyük pozitif | ≈ **1** |
| **0** | **0.5** |
| Çok büyük negatif | ≈ **0** |

### Model (2 adım)

$$z = \vec{w}\cdot\vec{x} + b \qquad f(\vec{x}) = g(z) = \frac{1}{1+e^{-(\vec{w}\cdot\vec{x}+b)}}$$

### Çıktının yorumu

$f(\vec{x})$ = **$y=1$ olma olasılığı**.
- $f=0.7$ → "%70 kötü huylu".
- $P(y=0) = 1 - 0.7 = 0.3$. İki olasılığın toplamı hep 1.
- Makale gösterimi: $f(\vec{x}) = P(y=1 \mid \vec{x};\vec{w},b)$ (ezber gerekmez).

> 🧠 **Neden sigmoid?** Lineer kısım ($z$) $-\infty$…$+\infty$ arası bir sayı üretir. Sigmoid bunu **olasılığa** (0–1) sıkıştırır. Olasılık da classification için doğal çıktıdır.

---

## 17. Decision Boundary

Olasılığı 0/1'e çevirmek için **eşik** (genelde 0.5):

$$f(\vec{x}) \ge 0.5 \Rightarrow \hat{y}=1 \qquad f(\vec{x}) < 0.5 \Rightarrow \hat{y}=0$$

**Mantık zinciri:**

$$f(\vec{x}) \ge 0.5 \iff g(z) \ge 0.5 \iff z \ge 0 \iff \vec{w}\cdot\vec{x}+b \ge 0$$

> 🧠 **Neden $z \ge 0$?** Çünkü $g(0)=0.5$ ve sigmoid artan bir fonksiyon. Yani 0.5 eşiği ile "$z$ pozitif mi?" sorusu **aynı şeydir**.

**Decision boundary** = $\vec{w}\cdot\vec{x}+b = 0$ (modelin kararsız kaldığı çizgi/eğri).

| Örnek | Sınır | Şekil |
|---|---|---|
| $w_1=w_2=1,\ b=-3$ | $x_1+x_2=3$ | **Düz çizgi** |
| $z = x_1^2 + x_2^2 - 1$ | $x_1^2+x_2^2=1$ | **Çember** (dışı $\hat{y}=1$, içi $\hat{y}=0$) |

**Önemli:** Sadece ham özelliklerle ($x_1, x_2, \dots$) sınır **her zaman düz çizgi**. Eğrisel sınır için **polinom özellikler** eklenmeli.

---

## 18. Logistic Regression için Cost Function

### Neden squared error kullanılmaz?

Squared error + sigmoid → **convex olmayan** ("kıpır kıpır") yüzey, **birçok yerel minimum** → gradient descent takılır ❌.

### Çözüm: Loss function

- **Loss $L$:** **Tek** örnekteki hata.
- **Cost $J$:** Tüm örneklerdeki loss'un **ortalaması**.

$$L = \begin{cases} -\log(f(\vec{x})) & y=1 \\ -\log(1-f(\vec{x})) & y=0 \end{cases}$$

### Sezgi ($y=1$ için)

| Tahmin $f$ | Loss |
|---|---|
| 1'e yakın | ≈ 0 (doğru) |
| 0.5 | orta |
| 0.1 | çok yüksek |
| 0'a yaklaşır | **∞** |

$y=0$ için simetrik: tahmin 1'e yaklaştıkça loss ∞.

> 🧠 **Neden $-\log$?** "**Çok emin ama yanlış**" tahmini sert cezalandırmak ister. $-\log$ tam bunu yapar: yanlışa güven arttıkça ceza **sonsuza** gider. Üstelik toplam cost **convex** olur → gradient descent güvenle global minimuma gider.

### Tek formül (birleştirilmiş)

$y$ sadece 0 veya 1 olduğundan iki durum birleşir:

$$L = -y\log(f) - (1-y)\log(1-f)$$

Kontrol: $y=1$ → ikinci terim kaybolur ✅; $y=0$ → ilk terim kaybolur ✅.

**Cost:**

$$J(\vec{w},b) = -\frac{1}{m}\sum_{i=1}^{m}\Big[y^{(i)}\log f(\vec{x}^{(i)}) + (1-y^{(i)})\log\big(1-f(\vec{x}^{(i)})\big)\Big]$$

İstatistikteki **maximum likelihood estimation** ilkesinden türetilmiştir (detay gerekmez).

---

## 19. Logistic Regression'da Gradient Descent

$$w_j = w_j - \alpha\frac{1}{m}\sum (f(\vec{x}^{(i)})-y^{(i)})\,x_j^{(i)} \qquad b = b - \alpha\frac{1}{m}\sum (f(\vec{x}^{(i)})-y^{(i)})$$

### 🤔 "Bu lineer regresyonla aynı değil mi?"

Formül **aynı görünür**, algoritma **farklı**: fark $f$'nin tanımında.

| | $f(\vec{x})$ |
|---|---|
| Linear regression | $\vec{w}\cdot\vec{x}+b$ |
| Logistic regression | $\text{sigmoid}(\vec{w}\cdot\vec{x}+b)$ |

> 🧠 **Neden formül bu kadar benzer?** Cost'u ($-\log$ loss) türevleyince ortaya çıkan sade ifade tesadüf değil; logistic loss tam bu temiz sonucu verecek şekilde seçilmiş (maximum likelihood).

Eski araçlar burada da geçerli: **learning curve, vectorization, feature scaling**. Eş zamanlı güncelle.

---

## 20. Overfitting ve Underfitting

**Generalization:** Modelin **hiç görmediği yeni örneklerde** de iyi tahmin etmesi. **Asıl amaç budur.**

### Regresyon örneği (ev fiyatı)

| Model | Durum | Terim |
|---|---|---|
| Düz çizgi | Veriye uymaz | **Underfitting = high bias** |
| $x, x^2$ | Uyar, yeni verilerde de iyi | ✅ **Tam doğru** (Goldilocks 🐻) |
| $x, x^2, x^3, x^4$ | Tüm eğitim noktalarından geçer ($J=0$) ama çok dalgalı | **Overfitting = high variance** |

### Terimler

- **High bias:** Model verinin yapısını yakalayamıyor (aşırı güçlü varsayım, örn. "her şey doğrusaldır"). *(ML'de "bias" ayrıca adaletsiz önyargı anlamına da gelir; bu ayrı bir konu.)*
- **High variance:** Model her eğitim örneğine aşırı uymaya çalışıyor. Eğitim verisi **biraz** değişse öğrenilen fonksiyon **tamamen** değişir.

> 🧠 **Neden overfitting kötü?** Eğitim verisinin **gürültüsünü** de ezberler. Eğitimde mükemmel, gerçek hayatta kötü olur. Underfitting ise eğitimde bile kötü: model fazla basit.

Classification'da da aynı: düz çizgi sınır (underfit), elips (iyi), aşırı kıvrımlı sınır (overfit).

### Overfitting'i azaltmanın 3 yolu

| # | Yöntem | Açıklama | Dezavantaj |
|---|---|---|---|
| 1 | **Daha fazla veri** | Model daha az dalgalı fonksiyon öğrenir. **Bir numaralı araç.** | Her zaman mümkün değil |
| 2 | **Daha az özellik** (feature selection) | Çok özellik + az veri = overfit riski | Atılan özelliklerdeki bilgi kaybolur |
| 3 | **Regularization** | Parametreleri **küçültmeyi** teşvik eder, özellik atmaz | — |

> 🧠 **Neden parametreleri küçültmek işe yarar?** Overfit modellerde $w$'ler genelde **çok büyük** olur (eğri sert kıvrılsın diye). Küçük $w$ = daha yumuşak, daha basit fonksiyon. Bir $w$'yi tam 0 yapmak = özelliği atmak (sert); regularization bunu **nazikçe** yapar.

---

## 21. Regularization

### Cost function'a ceza ekleme

Sezgi: $w_3, w_4$ küçük olsun istiyorsan maliyete $+1000w_3^2 + 1000w_4^2$ ekle; minimize etmenin tek yolu bunları ≈ 0 tutmak olur. Hangi özelliğin önemli olduğunu bilmiyorsan **tüm $w_j$'leri** biraz cezalandır:

$$J(\vec{w},b) = \underbrace{\frac{1}{2m}\sum (f(\vec{x}^{(i)})-y^{(i)})^2}_{\text{veriye uyum}} + \underbrace{\frac{\lambda}{2m}\sum_{j=1}^{n} w_j^2}_{\text{regularization terimi}}$$

- **$\lambda$ (lambda):** Regularization parametresi, **sen seçersin** ($\alpha$ gibi).
- İki terim de $\frac{1}{2m}$ ile ölçekli → veri büyüse bile aynı $\lambda$ çalışmaya devam etme eğilimi.
- Geleneksel olarak **sadece $w_1..w_n$** düzenlenir, **$b$ düzenlenmez** (fark yaratmaz).

> 🧠 **Neden iki terim?** Bir **çekişme** var: 1. terim modeli veriye **iyi uymaya**, 2. terim $w$'leri **küçük tutmaya** iter. $\lambda$ bu ikisi arasındaki dengeyi ayarlar.

### $\lambda$'nın etkisi

| $\lambda$ | Sonuç |
|---|---|
| **0** | Ceza yok → dalgalı eğri → **overfitting** |
| **Çok büyük** (örn. $10^{10}$) | Tüm $w_j \approx 0$ → $f \approx b$ → **yatay düz çizgi → underfitting** |
| **Orta** | Tüm özellikler kalır, eğri düzgün ve makul ✅ |

### Regularized linear regression: Gradient descent

$$w_j = w_j - \alpha\left[\frac{1}{m}\sum (f(\vec{x}^{(i)})-y^{(i)})\,x_j^{(i)} + \frac{\lambda}{m}w_j\right] \qquad b = b - \alpha\frac{1}{m}\sum (f(\vec{x}^{(i)})-y^{(i)})$$

($b$ için değişiklik yok, çünkü düzenlenmiyor.)

**Yeniden yazım (sezgi):**

$$w_j = w_j\left(1 - \alpha\frac{\lambda}{m}\right) - \alpha\frac{1}{m}\sum (f - y)\,x_j$$

- İkinci kısım = olağan gradient descent adımı.
- Birinci kısım: her iterasyonda $w_j$, **1'den biraz küçük** bir sayıyla çarpılır → **$w_j$ biraz küçülür**.
- Örnek: $\alpha=0.01,\ \lambda=1,\ m=50$ → çarpan $= 1 - 0.0002 = 0.9998$.

> 🧠 **Neden "weight decay" denir?** Her adımda $w$'yi hafifçe aşağı çekiyor (çürütüyor). Veri onu büyütmek istemedikçe $w$ küçük kalır.

Aynı fikir **logistic regression'a** da uygulanır (sıradaki konu).

---

# EKLER

## 22. Büyük resim: tüm hikâye tek akışta

```
Problem tipi?            → Supervised (etiket var) / Unsupervised (yok)
   Supervised:           → Regression (sürekli sayı) / Classification (kategori)

Regression yolu:
  Model     f(x) = w·x + b
  Ölçü      J(w,b) = squared error cost   (modelin ne kadar kötü olduğu)
  Optimize  Gradient Descent              (J'nin dibine otomatik in)
  Hızlandır Vectorization + Feature Scaling + iyi α (learning curve ile kontrol)
  Esnet     Feature Engineering / Polynomial (düz çizgi → eğri)

Classification yolu:
  Model     f(x) = sigmoid(w·x + b)       (olasılık, 0–1)
  Karar     f ≥ 0.5 → 1 ; sınır: w·x + b = 0
  Ölçü      Log loss (squared error convex olmadığı için kullanılmaz)
  Optimize  Gradient Descent (formül aynı görünür, f farklı)

Sorun çıkarsa:
  Underfit (high bias)    → daha karmaşık model / daha çok özellik
  Overfit (high variance) → daha çok veri / daha az özellik / REGULARIZATION (λ)
```

---

## 23. Hızlı formül kartı

| Konu | Formül |
|---|---|
| Linear model | $f = \vec{w}\cdot\vec{x}+b$ |
| Squared error cost | $J = \frac{1}{2m}\sum(f-y)^2$ |
| GD güncellemesi | $w_j = w_j - \alpha\,\partial J/\partial w_j$ |
| Linear gradient | $\partial J/\partial w_j = \frac{1}{m}\sum(f-y)x_j$ , $\partial J/\partial b = \frac{1}{m}\sum(f-y)$ |
| Z-score | $(x-\mu)/\sigma$ |
| Sigmoid | $g(z)=1/(1+e^{-z})$ |
| Logistic model | $f = g(\vec{w}\cdot\vec{x}+b)$ |
| Logistic loss | $-y\log f-(1-y)\log(1-f)$ |
| Regularized cost | $J + \frac{\lambda}{2m}\sum w_j^2$ |
| Regularized gradient | $\frac{1}{m}\sum(f-y)x_j + \frac{\lambda}{m}w_j$ |

**Kontrol listesi:**
- ✅ Güncellemeler **eş zamanlı**
- ✅ Eksi işareti: `w = w - alpha * dJ_dw`
- ✅ Özellikleri **ölçekle**; yeni veride **eğitimdeki $\mu,\sigma$** kullan
- ✅ Learning curve **her adımda düşmeli**
- ✅ $b$ **düzenlenmez**

---

## 24. Terim sözlüğü

| İngilizce | Türkçe / Anlam |
|---|---|
| Supervised / Unsupervised | Gözetimli / gözetimsiz öğrenme |
| Feature / Target (label) | Özellik / hedef (etiket) |
| Regression / Classification | Sürekli tahmin / kategori tahmini |
| Cost function | Maliyet fonksiyonu (tüm veri için ortalama hata) |
| Loss | Kayıp (tek örnek için hata) |
| Gradient Descent | Dereceli azalma / eğim inişi |
| Learning rate ($\alpha$) | Öğrenme oranı (adım büyüklüğü) |
| Convergence | Yakınsama |
| Convex | Dışbükey (tek minimumlu kase) |
| Vectorization | Döngüsüz, vektör işlemleriyle hesap |
| Feature scaling | Özellik ölçekleme |
| Feature engineering | Yeni özellik üretme |
| Decision boundary | Karar sınırı |
| Overfitting (high variance) | Aşırı uyum |
| Underfitting (high bias) | Yetersiz uyum |
| Generalization | Genelleme |
| Regularization ($\lambda$) | Düzenlileştirme |

---

## 25. Tam Python kodu

> Kodlar ders içeriğini örneklemek için yazılmıştır; sayılar örnektir.

```python
import numpy as np

# ---------- Ortak yardımcılar ----------
def zscore_normalize_features(X):
    """Her sütunu (x - mu) / sigma ile ölçekler. mu ve sigma'yı da döndür:
    yeni veriyi tahmin ederken AYNI değerlerle ölçeklemek gerekir."""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / sigma, mu, sigma

# ---------- Linear regression (çoklu özellik, vectorized) ----------
def compute_cost(X, y, w, b):
    m = X.shape[0]
    err = X @ w + b - y
    return (1 / (2 * m)) * np.sum(err ** 2)

def compute_gradient(X, y, w, b):
    m = X.shape[0]
    err = X @ w + b - y                  # f(x) - y  (tüm örnekler için)
    dj_dw = (1 / m) * (X.T @ err)        # her özellik için türev
    dj_db = (1 / m) * np.sum(err)
    return dj_dw, dj_db

def gradient_descent(X, y, w, b, alpha, num_iters):
    cost_history = []                    # learning curve için
    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)   # eski w, b ile hesaplandı
        w = w - alpha * dj_dw                         # eksi işareti!
        b = b - alpha * dj_db                         # => eş zamanlı güncelleme
        cost_history.append(compute_cost(X, y, w, b))
    return w, b, cost_history

# Polynomial regression örneği: y = 1 + x^2
x = np.arange(0, 20, 1.0)
y = 1 + x ** 2
X = np.c_[x, x ** 2, x ** 3]                          # özellikler: x, x², x³
X_norm, mu, sigma = zscore_normalize_features(X)      # polinomda ölçekleme şart!
w, b, history = gradient_descent(X_norm, y, np.zeros(3), 0.0, alpha=0.1, num_iters=1000)
print("Cost her adımda azalıyor mu?", all(np.diff(history) <= 0))

# ---------- Logistic regression ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    f = sigmoid(X @ w + b)
    # Not: f tam 0/1 olursa log(0) hatası çıkar; pratikte küçük epsilon eklenir
    return -(1 / m) * np.sum(y * np.log(f) + (1 - y) * np.log(1 - f))

def compute_gradient_logistic(X, y, w, b):
    m = X.shape[0]
    f = sigmoid(X @ w + b)               # lineer regresyondan TEK fark: sigmoid
    err = f - y
    return (1 / m) * (X.T @ err), (1 / m) * np.sum(err)

def gradient_descent_logistic(X, y, w, b, alpha, num_iters):
    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b

def predict(X, w, b, threshold=0.5):
    return (sigmoid(X @ w + b) >= threshold).astype(int)

X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
w, b = gradient_descent_logistic(X_train, y_train, np.zeros(2), 0.0, alpha=0.1, num_iters=10000)
print("Tahminler:", predict(X_train, w, b))

# ---------- Regularization (linear regression) ----------
def compute_cost_linear_reg(X, y, w, b, lambda_):
    m = X.shape[0]
    err = X @ w + b - y
    return (1 / (2 * m)) * np.sum(err ** 2) + (lambda_ / (2 * m)) * np.sum(w ** 2)  # b düzenlenmez

def compute_gradient_linear_reg(X, y, w, b, lambda_):
    m = X.shape[0]
    err = X @ w + b - y
    dj_dw = (1 / m) * (X.T @ err) + (lambda_ / m) * w    # ek terim: (λ/m)·w
    dj_db = (1 / m) * np.sum(err)                         # b için değişmez
    return dj_dw, dj_db
```

---

*Sıradaki konu: Regularization'ın logistic regression'a uygulanması ve Hafta 3 pratik lab'ı.*
