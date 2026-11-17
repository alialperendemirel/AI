# Advanced Learning Algorithms: Ders Notları

> Ders notları (Hafta 1–4), tek dosyada ve **mantığı anlaşılacak** şekilde özetlenmiştir.
> Her bölümde "ne?" kadar **"neden?"** sorusuna da cevap verilir. Tekrar bakarken önce bölüm sonlarındaki **🧠 Neden?** kutularını oku.
>
> **Kaynak:** [Advanced Learning Algorithms](https://www.coursera.org/learn/advanced-learning-algorithms) (DeepLearning.AI & Stanford Online, Andrew Ng)
> **Önkoşul:** [Supervised Machine Learning: Regression and Classification](https://www.coursera.org/learn/machine-learning) notları. Cost function, gradient descent, logistic regression, regularization ve feature scaling bilgisi varsayılır.
>
> ℹ️ **Çeviri notu:** Ders transkripti otomatik çevrildiği için bazı terimler yanlıştı (örn. "ileri pervane" = *forward propagation*, "çapraz tropi" = *cross-entropy*). Bu notlarda doğru terimler kullanıldı; karşılıklar için [Terim sözlüğü](#44-terim-sözlüğü) bölümüne bak.

## İçindekiler

**Hafta 1 – Sinir ağları**
1. [Sinir ağlarına giriş](#1-sinir-ağlarına-giriş)
2. [Tek nöron: talep tahmini örneği](#2-tek-nöron-talep-tahmini-örneği)
3. [Görüntü örneği: yüz tanıma](#3-görüntü-örneği-yüz-tanıma)
4. [Nöron katmanı ve notasyon](#4-nöron-katmanı-ve-notasyon)
5. [Forward propagation (ileri yayılma): rakam tanıma](#5-forward-propagation-ileri-yayılma-rakam-tanıma)
6. [TensorFlow'da tahmin (inference) ve veri temsili](#6-tensorflowda-tahmin-inference-ve-veri-temsili)
7. [TensorFlow ile ağ kurma ve sıfırdan forward prop](#7-tensorflow-ile-ağ-kurma-ve-sıfırdan-forward-prop)
8. [Yapay genel zeka (AGI) hakkında](#8-yapay-genel-zeka-agi-hakkında)
9. [Vektörleştirme ve matris çarpımı](#9-vektörleştirme-ve-matris-çarpımı)

**Hafta 2 – Sinir ağı eğitimi**
10. [TensorFlow ile eğitim: 3 adım](#10-tensorflow-ile-eğitim-3-adım)
11. [Aktivasyon fonksiyonları: sigmoid'in ötesi](#11-aktivasyon-fonksiyonları-sigmoidin-ötesi)
12. [Çok sınıflı sınıflandırma ve Softmax](#12-çok-sınıflı-sınıflandırma-ve-softmax)
13. [Sayısal doğruluk ve from_logits](#13-sayısal-doğruluk-ve-from_logits)
14. [Çoklu etiket (multi-label) sınıflandırma](#14-çoklu-etiket-multi-label-sınıflandırma)
15. [Adam: daha akıllı optimizasyon](#15-adam-daha-akıllı-optimizasyon)
16. [Katman türleri: yoğun (dense) ve evrişimli (convolutional)](#16-katman-türleri-yoğun-dense-ve-evrişimli-convolutional)
17. [(Opsiyonel) Türev, hesaplama grafiği ve backpropagation](#17-opsiyonel-türev-hesaplama-grafiği-ve-backpropagation)

**Hafta 3 – ML uygulama tavsiyeleri**
18. [Sonra ne denemeli? Modeli değerlendirmek](#18-sonra-ne-denemeli-modeli-değerlendirmek)
19. [Model seçimi: train / cross-validation / test](#19-model-seçimi-train--cross-validation--test)
20. [Bias ve variance'ı teşhis etme](#20-bias-ve-varianceı-teşhis-etme)
21. [Regularization ve bias/variance](#21-regularization-ve-biasvariance)
22. [Baseline performans: "iyi" ne demek?](#22-baseline-performans-iyi-ne-demek)
23. [Öğrenme eğrileri (learning curves)](#23-öğrenme-eğrileri-learning-curves)
24. [Sonraki adıma karar verme: özet tablo](#24-sonraki-adıma-karar-verme-özet-tablo)
25. [Bias/variance ve sinir ağları](#25-biasvariance-ve-sinir-ağları)
26. [ML geliştirme döngüsü ve hata analizi](#26-ml-geliştirme-döngüsü-ve-hata-analizi)
27. [Veri eklemek: augmentation, synthesis ve transfer learning](#27-veri-eklemek-augmentation-synthesis-ve-transfer-learning)
28. [Bir ML projesinin yaşam döngüsü](#28-bir-ml-projesinin-yaşam-döngüsü)
29. [Adalet, önyargı ve etik](#29-adalet-önyargı-ve-etik)
30. [Çarpık veri setleri: precision, recall, F1](#30-çarpık-veri-setleri-precision-recall-f1)

**Hafta 4 – Karar ağaçları**
31. [Karar ağacı modeli](#31-karar-ağacı-modeli)
32. [Karar ağacı öğrenme süreci](#32-karar-ağacı-öğrenme-süreci)
33. [Entropi: saflığı ölçmek](#33-entropi-saflığı-ölçmek)
34. [Bilgi kazancı (information gain) ve bölme seçimi](#34-bilgi-kazancı-information-gain-ve-bölme-seçimi)
35. [One-hot kodlama: ikiden fazla değer alan özellikler](#35-one-hot-kodlama-ikiden-fazla-değer-alan-özellikler)
36. [Sürekli değerli özellikler](#36-sürekli-değerli-özellikler)
37. [Regresyon ağaçları](#37-regresyon-ağaçları)
38. [Ağaç toplulukları ve örnekleme (bootstrap)](#38-ağaç-toplulukları-ve-örnekleme-bootstrap)
39. [Random forest (rastgele orman)](#39-random-forest-rastgele-orman)
40. [XGBoost: boosting](#40-xgboost-boosting)
41. [Karar ağacı mı sinir ağı mı?](#41-karar-ağacı-mı-sinir-ağı-mı)

**Ekler**
42. [Büyük resim: tüm hikâye tek akışta](#42-büyük-resim-tüm-hikâye-tek-akışta)
43. [Hızlı formül kartı](#43-hızlı-formül-kartı)
44. [Terim sözlüğü](#44-terim-sözlüğü)
45. [Tam Python kodu](#45-tam-python-kodu)

**Tekrar ve pratik**
46. [Elle hesap örnekleri](#46-elle-hesap-örnekleri)
47. [Sık yapılan hatalar](#47-sık-yapılan-hatalar)
48. [Karıştırılan kavramlar](#48-karıştırılan-kavramlar)
49. [Sorun → çözüm rehberi](#49-sorun--çözüm-rehberi)
50. [Kendini test et](#50-kendini-test-et)
51. [Tekrar planı](#51-tekrar-planı)

---

# HAFTA 1 – SİNİR AĞLARI

## 1. Sinir ağlarına giriş

- Sinir ağları **beyni taklit eden yazılım** yazma motivasyonuyla başladı (1950'ler). 1980–90'larda el yazısı rakam tanıma (posta kodu, çek) ile popülerleşti, 90'ların sonunda gözden düştü, **~2005'ten sonra "deep learning" adıyla** geri geldi.
- Büyük etki sırası: **konuşma tanıma → bilgisayarlı görü** (2012 ImageNet) → **doğal dil işleme** → iklim, tıbbi görüntüleme, reklam, ürün önerisi...
- **Biyolojik nöron:** girdi sinyallerini alır, hesap yapar, çıktıyı (akson) başka nöronlara yollar. **Yapay nöron:** sayı(lar) alır, hesaplar, bir sayı çıkarır.
- Beyin sadece **ilham kaynağıdır**. Beynin gerçekte nasıl öğrendiğini hâlâ bilmiyoruz; bugünkü sinir ağları beyinle çok az ilgilidir.

> 🧠 **Neden şimdi popüler?** Grafiğe bak: yatay eksen **veri miktarı**, dikey eksen **performans**. Klasik algoritmalar (lineer/logistic regresyon) belli bir noktadan sonra **düzleşir**. İnternet, mobil ve dijital kayıtlarla veri arttı; **büyük sinir ağları büyük veriyle yükselmeye devam eder**. GPU gibi hızlı donanım da bunları eğitilebilir kıldı.

---

## 2. Tek nöron: talep tahmini örneği

**Problem:** Bir tişört en çok satanlar arasına girecek mi? ($y=1$ evet, $y=0$ hayır). Girdi: **fiyat** $x$.

Logistic regression uygularsak:

$$a = f(x) = \frac{1}{1+e^{-(wx+b)}}$$

- Çıktıya **$a$** diyoruz (*activation*, "aktivasyon"). Sinir bilimden gelen bir terim: bir nöronun aşağı akıştaki nöronlara ne kadar güçlü sinyal gönderdiği.
- Bu küçük logistic regression birimi = **tek bir yapay nöron**: bir sayı alır, formülü hesaplar, bir sayı (olasılık) verir.

### Çok nöronlu örnek

4 girdi: **fiyat, nakliye maliyeti, pazarlama, malzeme kalitesi**. Gizli katmanda 3 nöron, her biri bir "ara kavramı" tahmin eder:

| Nöron | Neyi tahmin eder? | Hangi girdilere bakar (sezgi) |
|---|---|---|
| 1 | **Uygun fiyatlı mı?** (affordability) | fiyat, nakliye |
| 2 | **Farkındalık** (awareness) | pazarlama |
| 3 | **Algılanan kalite** | fiyat, malzeme kalitesi |

Bu üç sayı **çıktı katmanındaki** nörona gider; o da "en çok satan mı?" olasılığını verir.

**Önemli:** Yukarıdaki "hangi nöron neye bakar" sezgisi sadece anlatım içindir. Gerçekte **her nöron önceki katmanın TÜM çıktılarını alır** ve hangi girdilere ne kadar önem vereceğini ($w$'leri) **veriden öğrenir**.

> 🧠 **Neden bu önemli?** Lineer/logistic regresyonda yeni özellikleri (alan = genişlik × derinlik gibi) **elle üretmek** gerekiyordu. Sinir ağında gizli katman bu özellikleri **kendisi öğrenir**. Bu yüzden sinir ağları "otomatik feature engineering" yapar.

**Terimler:**
- **Input layer** (giriş katmanı): $\vec{x}$
- **Hidden layer** (gizli katman): ara özellikleri hesaplar; eğitim verisinde doğrudan "doğru cevabı" görmeyiz (o yüzden "gizli")
- **Output layer** (çıkış katmanı): son tahmin
- **Activation vector** $\vec{a}$: bir katmanın çıktı sayıları

---

## 3. Görüntü örneği: yüz tanıma

- Bir görüntü, **piksel parlaklık değerleri** (0–255) matrisidir. $1000\times1000$ piksel → $1.000.000$ sayılık bir vektör.
- Yüz tanıma ağı: bu 1 milyonluk vektörü girdi alır, birkaç gizli katmandan geçer, **kişinin kimliğini** tahmin eder.
- Eğitilmiş ağın gizli katmanlarını görselleştirirsen:

| Katman | Ne öğrenir? |
|---|---|
| İlk gizli katman | Çok **kısa kenarlar / çizgi parçaları** |
| İkinci gizli katman | **Yüz parçaları** (göz, burun, ağız köşeleri) |
| Üçüncü gizli katman | **Yüzün genel şekilleri** |

Araba tanıma da aynı: kenarlar → araba parçaları → arabanın geneli.

> 🧠 **Neden?** Erken katmanlar **basit** örüntüleri, sonraki katmanlar **bunları birleştirip karmaşık** örüntüleri yakalar. Bu "kendi kendine özellik öğrenme" derin öğrenmenin gücüdür.

---

## 4. Nöron katmanı ve notasyon

**Katman** = aynı girdiyi alan nöronlar grubu. Çıktısı bir sayı **vektörüdür**.

Katman $l$'deki $j$. nöron:

$$a_j^{[l]} = g\left(\vec{w}_j^{[l]} \cdot \vec{a}^{[l-1]} + b_j^{[l]}\right)$$

| Sembol | Anlamı |
|---|---|
| $[l]$ (köşeli parantez üst simge) | **Katman numarası** |
| $j$ (alt simge) | Katmandaki **nöron numarası** |
| $\vec{a}^{[l-1]}$ | Önceki katmanın çıktısı (bu katmanın girdisi) |
| $\vec{a}^{[0]} = \vec{x}$ | **Giriş katmanı** (Katman 0) |
| $g$ | **Aktivasyon fonksiyonu** (şimdilik sigmoid) |
| $\vec{w}_j^{[l]},\ b_j^{[l]}$ | Katman $l$, nöron $j$'nin parametreleri |

**Katman sayma kuralı:** Giriş katmanı **sayılmaz**. "4 katmanlı ağ" = 3 gizli katman + 1 çıkış katmanı.

**Sayısal örnek (Bölüm 2'deki ağ):** Gizli katmanın 3 nöronu $0.3,\ 0.7,\ 0.2$ üretir → $\vec{a}^{[1]} = [0.3,\ 0.7,\ 0.2]$. Çıkış nöronu bu vektörü alıp tek bir sayı verir, örn. $a^{[2]} = 0.84$. Eşik 0.5 ise tahmin $\hat{y}=1$.

> 🧠 **Neden bu notasyon?** Ağda çok sayıda $w$, $b$, $a$ var. Üst simge **hangi katman**, alt simge **hangi nöron** olduğunu net tutar. Aksi halde karışır.

---

## 5. Forward propagation (ileri yayılma): rakam tanıma

**Forward propagation:** Girdiden başlayıp **soldan sağa** her katmanı hesaplayarak tahmine ulaşma. Tahmin (inference) böyle yapılır.

**Örnek:** $8\times8=64$ pikselli görüntü, **0 mı 1 mi?** (ikili sınıflandırma). Mimari:

```
x (64)  →  Katman 1 (25 nöron)  →  Katman 2 (15 nöron)  →  Katman 3 (1 nöron)  →  tahmin
```

$$\vec{a}^{[1]} = g(W^{[1]}\vec{x} + \vec{b}^{[1]}) \quad \vec{a}^{[2]} = g(W^{[2]}\vec{a}^{[1]} + \vec{b}^{[2]}) \quad a^{[3]} = g(W^{[3]}\vec{a}^{[2]} + b^{[3]})$$

- $\vec{a}^{[1]}$: 25 sayı, $\vec{a}^{[2]}$: 15 sayı, $a^{[3]}$: tek sayı (**skaler**) = $f(\vec{x})$.
- Tahmin: $a^{[3]} \ge 0.5 \Rightarrow \hat{y}=1$, değilse $\hat{y}=0$.

> 🧠 **Neden "forward"?** Hesap girdiden çıktıya **ileri** akar. İleride türevleri hesaplarken **geriye** doğru gideceğiz (backpropagation, Bölüm 17).

---

## 6. TensorFlow'da tahmin (inference) ve veri temsili

**TensorFlow** (ve PyTorch) derin öğrenme için önde gelen çerçevelerdir. Keras, TensorFlow içindeki yüksek seviyeli arayüzdür.

### Örnek: kahve kavurma 🔥☕

Girdiler: **sıcaklık (°C)** ve **süre (dakika)**. Çıktı: iyi kahve mi? ($1$/$0$).

```python
x = np.array([[200.0, 17.0]])                  # 1 örnek, 2 özellik (1x2 matris)

layer_1 = Dense(units=3, activation='sigmoid')  # 3 nöronlu katman
a1 = layer_1(x)                                 # 1x3 çıktı

layer_2 = Dense(units=1, activation='sigmoid')  # çıkış katmanı
a2 = layer_2(a1)

yhat = 1 if a2 >= 0.5 else 0
```

### Veri nasıl temsil edilir?

| Kavram | Açıklama |
|---|---|
| Matris boyutu | **satır × sütun**. $2\times3$ matris = 2 satır, 3 sütun |
| `np.array([200, 17])` | **1B dizi** (vektör), tek köşeli parantez |
| `np.array([[200, 17]])` | **2B dizi** ($1\times2$ matris), çift köşeli parantez |
| TensorFlow'un tercihi | Tüm veriyi **matris (2B)** olarak tutar |
| `tf.Tensor` | TensorFlow'un matris tipi. `a1.numpy()` ile NumPy dizisine çevirirsin |

> 🧠 **Neden hep matris (2B)?** TensorFlow büyük veriyi **verimli hesaplamak** için tasarlandı. Matrisler ve vektörleştirme bunu sağlar (Bölüm 9). Eski Python/NumPy ile TensorFlow arasındaki bu "1B vs 2B" fark kafa karıştırır, o yüzden köşeli parantezlere dikkat et.

---

## 7. TensorFlow ile ağ kurma ve sıfırdan forward prop

### Sequential API

Katmanları tek tek çağırmak yerine **birleştir**:

```python
model = Sequential([
    Dense(units=3, activation='sigmoid'),
    Dense(units=1, activation='sigmoid')
])
model.compile(...)          # Bölüm 10'da
model.fit(X, Y)             # eğit
model.predict(X_new)        # tahmin (forward prop)
```

### Forward prop'u sıfırdan yazmak

Kütüphane olmadan tek bir katmanı şöyle hesaplarsın:

```python
def dense(a_in, W, b, g):
    units = W.shape[1]              # W'nin sütun sayısı = nöron sayısı
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]                 # j. nöronun ağırlıkları (j. sütun)
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    return a_out

def sequential(x, W1, b1, W2, b2):
    a1 = dense(x,  W1, b1, sigmoid)
    a2 = dense(a1, W2, b2, sigmoid)
    return a2
```

**Konvansiyon:** $W$'da **her sütun bir nöronun** ağırlık vektörüdür; $b$ nöron sayısı kadar elemanlı vektördür. Büyük harf ($W, X, Z, A$) **matris**, küçük harf **vektör/skaler**.

> 🧠 **Neden sıfırdan yazmayı bilmeliyim?** Kütüphane sihir değil: çağırdığın `Dense` katmanı, arkada tam olarak bu hesabı yapar. Neyin çalıştığını bilirsen **hata ayıklarken** kaybolmazsın. Ama gerçek projede kendi kodunu değil TensorFlow/PyTorch kullanırsın (daha hızlı, GPU destekli).

---

## 8. Yapay genel zeka (AGI) hakkında

| | **ANI** (Artificial Narrow Intelligence) | **AGI** (Artificial General Intelligence) |
|---|---|---|
| Ne yapar? | **Tek bir dar görev** (akıllı hoparlör, sürücüsüz araç, web arama, fabrika denetimi) | Tipik bir insanın yapabildiği **her şeyi** yapmak |
| Durum | Son yıllarda **muazzam ilerleme**, büyük değer yaratıyor | Ne kadar yaklaştığımız **belirsiz** |

- **Yanlış çıkarım:** "ANI'de büyük ilerleme = AGI'ye de büyük ilerleme". Bu **abartıdır**.
- **Beyni simüle etmek zor:** Gerçek nöron çok karmaşık; sinir bilimi beynin nasıl çalıştığını daha bilmiyor. Basit yapay nöron beyni taklit etmenin çok uzağında.
- **"Tek öğrenme algoritması" hipotezi:** Hayvan deneylerinde, **işitme korteksine** kulak yerine **görüntü** sinyali beslenince o bölge **görmeyi öğrendi** (benzer deneyler dokunma ve diğer duyular için de var). Beynin farklı bölgeleri belki **aynı genel algoritmayı** farklı veriyle çalıştırıyor. Bu algoritmayı bulabilirsek AGI'ye yol açabilir.
- Yazarın tavrı: AGI hayali **ilham verici**, ama yolu **belirsiz ve zor**. Onlarca yıl mı yüzyıllar mı sürer bilmiyoruz.

> 🧠 **Neden bu bölüm var?** Medyadaki AGI abartısına karşı **gerçekçi** ol. Bugün değer yaratan şey ANI. Senin işin: dar görevlerde iyi çalışan sistemler kurmak.

---

## 9. Vektörleştirme ve matris çarpımı

**Neden sinir ağları büyüyebildi?** Forward prop **matris çarpımı** ile çok verimli yapılır. GPU'lar ve paralel CPU işlevleri **büyük matris çarpımlarında** çok hızlıdır. Vektörleştirme olmasaydı derin öğrenme bugünkü ölçeğe ulaşamazdı.

### Dot product ve vektör-matris

- **Dot product:** $\vec{a}\cdot\vec{w}$ = karşılıklı elemanları çarp, topla. Bir sayı verir.
- **Transpoz** ($\vec{a}^T$): sütun vektörü satır vektörüne çevir ve tersi.

### Matris çarpımı kuralı

$A$ ($m\times k$) ile $W$ ($k\times n$) çarpılır → $Z$ ($m\times n$).

> **İç boyutlar eşit olmalı** ($k$ = $k$). Sonuç boyutu dış boyutlardan gelir.

$Z$'nin $(i,j)$ elemanı = $A$'nın **$i$. satırı** ile $W$'nin **$j$. sütununun** dot product'ı.

**Mini örnek:** $A = \begin{bmatrix}1 & 2\end{bmatrix}$ ($1\times2$), $W = \begin{bmatrix}3 & 5\\4 & 6\end{bmatrix}$ ($2\times2$):

$$Z = AW = \begin{bmatrix}1\cdot3+2\cdot4 & 1\cdot5+2\cdot6\end{bmatrix} = \begin{bmatrix}11 & 17\end{bmatrix}$$

### Vektörleştirilmiş katman (tüm nöronlar aynı anda)

```python
def dense(A_in, W, B, g):
    Z = np.matmul(A_in, W) + B   # tüm nöronlar tek matris çarpımıyla
    A_out = g(Z)                 # g eleman bazında uygulanır
    return A_out
```

- Eskiden: `for j in range(units)` döngüsü (yavaş).
- Şimdi: **tek `np.matmul`**. (`A_in @ W` aynı şey.)
- Bu yüzden TensorFlow'da girdi $X$ matrisi: **her satır bir örnek**.

> 🧠 **Neden hızlı?** Önceki kurstaki vectorization ile aynı mantık. Döngü çarpmaları sırayla yapar; `matmul` ise donanımın paralel gücünü kullanıp **hepsini aynı anda** yapar. Bu, hem **bir katmandaki tüm nöronları** hem de (satır satır) **tüm örnekleri** tek seferde hesaplamamızı sağlar.

---

# HAFTA 2 – SİNİR AĞI EĞİTİMİ

## 10. TensorFlow ile eğitim: 3 adım

Yine rakam tanıma ($0$ mı $1$ mi?) örneği. Ağ: 25 → 15 → 1 nöron.

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

# 1) Modeli tanımla (forward prop'u belirle)
model = Sequential([
    Dense(units=25, activation='sigmoid'),
    Dense(units=15, activation='sigmoid'),
    Dense(units=1,  activation='sigmoid')
])

# 2) Loss fonksiyonunu belirleyip derle
model.compile(loss=BinaryCrossentropy())

# 3) Eğit (gradient descent çalıştırır)
model.fit(X, y, epochs=100)
```

| Adım | Lojistik regresyonda | Sinir ağında |
|---|---|---|
| 1. Model | $f = g(\vec{w}\cdot\vec{x}+b)$ | `Sequential([...])` |
| 2. Loss / Cost | log loss | **`BinaryCrossentropy`** (aynı şey) |
| 3. Minimize et | Gradient descent | `model.fit(...)` |

- **Binary cross-entropy** = bizim logistic regression log loss'umuz: $L = -y\log f - (1-y)\log(1-f)$.
- **Epoch:** Tüm eğitim setinin bir kez gradient descent'ten geçmesi. `epochs=100` → 100 iterasyon.
- **Regresyon** problemi ise loss olarak `MeanSquaredError()` kullanılır.
- Cost: $J(\vec{W},\vec{B})$ = tüm katmanların tüm $w, b$'lerinin fonksiyonu = loss'ların ortalaması.

> 🧠 **Neden türevleri biz yazmıyoruz?** `model.fit` içinde **backpropagation** otomatik çalışır: tüm parametreler için $\partial J/\partial w$ hesaplanır, sonra gradient descent güncellemesi yapılır (Bölüm 17).

---

## 11. Aktivasyon fonksiyonları: sigmoid'in ötesi

Şu ana kadar her yerde **sigmoid** kullandık (lojistik regresyondan geldiği için). Başka seçenekler ağı **çok daha güçlü** yapar.

### Üç temel aktivasyon

| Ad | Formül | Çıktı aralığı | Not |
|---|---|---|---|
| **Linear** ("aktivasyon yok") | $g(z)=z$ | $-\infty \dots +\infty$ | |
| **Sigmoid** | $g(z)=\dfrac{1}{1+e^{-z}}$ | $0 \dots 1$ | olasılık |
| **ReLU** (Rectified Linear Unit) | $g(z)=\max(0,z)$ | $0 \dots +\infty$ | negatifse 0, pozitifse $z$ |

**Neden gerekli (motivasyon)?** Talep tahmininde "farkındalık" 0/1 değil, **0'dan çok büyük sayılara kadar** değişebilir. Sigmoid bunu yapamaz; ReLU yapabilir.

### Çıktı katmanı için seçim (hedef $y$'ye göre)

| Problem | $y$ | Çıkış aktivasyonu |
|---|---|---|
| **İkili sınıflandırma** | 0 veya 1 | **Sigmoid** ($P(y=1)$ öğrenir) |
| **Regresyon** ($-/+$ olabilir, örn. hisse değişimi) | pozitif veya negatif sayı | **Linear** |
| **Regresyon** (negatif olamaz, örn. ev fiyatı) | $\ge 0$ | **ReLU** |

### Gizli katmanlar için seçim

> **Varsayılan: ReLU.** Bugün gizli katmanlarda en yaygın tercih.

İki neden:
1. **Hesap daha hızlı:** $\max(0,z)$ basit; sigmoid üstel fonksiyon gerektirir.
2. **Gradient descent daha hızlı:** ReLU yalnızca bir tarafta düz; sigmoid **iki tarafta da düzdür** (doyma). Düz bölgede türev ≈ 0 olduğu için gradient descent **çok yavaşlar**.

```python
Dense(units=25, activation='relu')     # gizli katman
Dense(units=1,  activation='sigmoid')  # ikili sınıflandırma çıkışı
```

> 💡 Gizli katmanlarda sigmoid **neredeyse hiç** kullanılmaz (ikili sınıflandırmanın çıkış katmanı hariç). Ayrıca **leaky ReLU, tanh, swish** gibi başka seçenekler var ama ReLU çoğu zaman yeterli.

### Neden aktivasyon fonksiyonu gerekir? (doğrusal olmama)

Tüm nöronlarda **linear** ($g(z)=z$) kullanırsan:

$$a^{[1]} = w_1 x + b_1 \qquad a^{[2]} = w_2 a^{[1]} + b_2 = (w_2w_1)x + (w_2b_1 + b_2) = wx + b$$

**Sonuç:** Kaç katman olursa olsun ağ **tek bir lineer regresyona** çöker. Çıkış katmanı sigmoid olsa bile tüm ağ **logistic regression'dan fazlasını yapamaz**.

> 🧠 **Neden?** Lineer fonksiyonların birleşimi yine lineerdir. **Doğrusal olmayan** aktivasyon (ReLU) gizli katmanlara karmaşık, eğrisel örüntüleri öğrenme gücü verir. **Kural: Gizli katmanlarda linear aktivasyon kullanma.**

---

## 12. Çok sınıflı sınıflandırma ve Softmax

**Multi-class:** $y$, ikiden fazla **ayrık** değerden birini alır. Örnek: 10 rakamı (0–9) tanımak, 5 hastalıktan birini bulmak, ilaç hapında hangi **kusur** türü olduğunu bulmak.

Yeni araç: **Softmax regression** = logistic regression'ın çok sınıflı genellemesi.

### Logistic'ten softmax'e

Logistic regression aslında **iki** sayı hesaplar: $a_1 = P(y=1)$ ve $a_2 = 1-a_1 = P(y=0)$. Toplamları 1.
> Hızlı test: $P(y=1)=0.71$ ise $P(y=0)=0.29$.

### Softmax formülü ($N$ sınıf)

$$z_j = \vec{w}_j\cdot\vec{x} + b_j \qquad a_j = \frac{e^{z_j}}{\sum_{k=1}^{N} e^{z_k}} = P(y=j)$$

- Hesaplanan $a_1, \dots, a_N$ **hepsi 0–1 arası ve toplamı 1** olur (bir olasılık dağılımı).
- $N=2$ ise softmax, logistic regression ile **aynı** sonucu verir.

**Sayısal örnek ($N=3$):** $z = [2,\ 1,\ 0.1]$

$$e^2=7.389,\ e^1=2.718,\ e^{0.1}=1.105 \quad\Rightarrow\quad \text{toplam}=11.212$$
$$a = [0.659,\ 0.242,\ 0.099] \quad (\text{toplam}=1)$$

### Softmax loss (cost)

$$L(\vec{a}, y) = -\log a_y \qquad (\text{doğru sınıfa verilen olasılığın } -\log\text{'u})$$

- Örnek: doğru sınıf 1 ise $L=-\ln 0.659 = 0.417$.
- Doğru sınıfa verilen olasılık $a_y$ **1'e yaklaşırsa** loss $\to 0$. **0'a yaklaşırsa** loss $\to\infty$.
- $N=2$ için logistic loss'a eşittir.

> 🧠 **Neden $-\log a_y$?** Logistic'teki mantığın aynısı: "**çok emin ama yanlış**" tahmini sert cezalandır. Sadece **doğru sınıfın** olasılığına bakılır; yanlış sınıfların olasılığı zaten toplamın 1 olma kısıtı yüzünden dolaylı etkilenir.

### Sinir ağında softmax

**Çıkış katmanı** $N$ nöronlu ve **softmax aktivasyonlu** olur (10 rakam için 10 nöron).

```python
from tensorflow.keras.losses import SparseCategoricalCrossentropy

model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=10, activation='softmax')     # 10 sınıf
])
model.compile(loss=SparseCategoricalCrossentropy())
model.fit(X, y, epochs=100)
```

- **`SparseCategoricalCrossentropy`:** "Categorical" = sınıflandırma, "sparse" = $y$ **tek bir sınıf numarasıdır** (10 sınıftan sadece biri).

**Önemli fark:** Diğer aktivasyonlarda ($g(z)$ sigmoid/ReLU) her $a_j$ sadece **kendi** $z_j$'sine bağlıdır. Softmax'te her $a_j$ **tüm** $z_1 \dots z_N$'ye bağlıdır (paydadaki toplam yüzünden).

---

## 13. Sayısal doğruluk ve from_logits

Yukarıdaki kod **doğru ama sayısal olarak tam kararlı değil**. Bilgisayarlar sayıları sınırlı bellekte (floating-point) saklar; bu **yuvarlama hatalarına** yol açar.

**Örnek:** $2/10000$'i iki farklı biçimde hesaplamak (örn. $1 + \frac{1}{10000} - (1 - \frac{1}{10000})$) küçük farklı sonuç verir.

Softmax/logistic'te $a = \frac{1}{1+e^{-z}}$ **ara değer olarak** hesaplanırsa, çok büyük/küçük $z$'de yuvarlama hatası büyür. Çözüm: **loss'u $z$ cinsinden yazıp** TensorFlow'a ara adımı atlatmak.

### Logistic regression için (sayısal olarak daha kararlı)

```python
model = Sequential([
    Dense(25, activation='relu'),
    Dense(15, activation='relu'),
    Dense(1,  activation='linear')              # sigmoid YOK, linear
])
model.compile(loss=BinaryCrossentropy(from_logits=True))
```

### Softmax için (sayısal olarak daha kararlı)

```python
model = Sequential([
    Dense(25, activation='relu'),
    Dense(15, activation='relu'),
    Dense(10, activation='linear')              # softmax YOK, linear
])
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))

# Tahmin: model artık z (logit) döndürür, olasılık için softmax uygula
logits = model(X)
probs  = tf.nn.softmax(logits)                  # (ikili için tf.nn.sigmoid)
```

- **Logit** = $z$ değeri (aktivasyondan önceki).
- `from_logits=True`: "Çıkış katmanı $z$ veriyor, loss içinde sigmoid/softmax'i **TensorFlow kendi hesaplasın**."

> 🧠 **Neden?** Mantık aynı, ama TensorFlow $z$'yi doğrudan görünce **daha kararlı bir formül** kullanabilir. Sonuç matematiksel olarak aynı, ama **sayısal hata azalır**. Okunabilirliği az bozulur: çıkış katmanı `linear`, tahminde **elle softmax/sigmoid** uygularsın. Bu yüzden çok sınıflı ağların kodunda `linear` çıkış + `from_logits=True` görürsün.

---

## 14. Çoklu etiket (multi-label) sınıflandırma

Bazen **bir girdi aynı anda birden fazla etikete** sahip olabilir.

| | **Multi-class** | **Multi-label** |
|---|---|---|
| Çıktı | **Tek** etiket, $N$ sınıftan biri | **Birden çok** etiket aynı anda |
| Örnek | Rakam: 0–9'dan **biri** | Bir sokak görüntüsünde **araba var mı? otobüs var mı? yaya var mı?** (hepsi birden olabilir) |
| $y$ | tek sayı | **vektör** (örn. $[1,0,1]$) |
| Çıkış katmanı | **Softmax** (toplam = 1) | **Her çıkış için ayrı sigmoid** |

```python
Dense(units=3, activation='sigmoid')   # 3 bağımsız ikili soru
```

Multi-label'ı, **birden fazla bağımsız ikili sınıflandırma** olarak düşünebilirsin (ayrı ağlar yerine tek ağ, 3 çıkış nöronu).

> 🧠 **Neden softmax değil sigmoid?** Softmax olasılıkları **toplamı 1'e zorlar** (sınıflar birbirini dışlar). Multi-label'da araba **ve** yaya aynı anda olabilir, toplamın 1 olması gerekmez.

---

## 15. Adam: daha akıllı optimizasyon

**Gradient descent** (öğrenme oranı $\alpha$ sabit). Daha hızlı bir alternatif: **Adam** (*Adaptive Moment estimation*).

**Fikir:** Öğrenme oranını **her parametre için otomatik ayarlar.**
- Bir parametre **hep aynı yöne** küçük adımlarla ilerliyorsa → $\alpha$'sını **artır** (daha hızlı git).
- Bir parametre **ileri geri zıplıyorsa** → $\alpha$'sını **azalt**.

**Her parametre ($w_1,\dots,w_n,b$) için ayrı $\alpha_j$** vardır.

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=SparseCategoricalCrossentropy(from_logits=True)
)
```

- **Başlangıç $\alpha$'yı** yine sen verirsin (örn. $10^{-3}$). Adam bunu zamanla ayarlar.
- Adam, gradient descent'e göre genellikle **daha hızlı yakınsar** ve $\alpha$ seçimine **daha az duyarlıdır**. Bugün sinir ağlarında **fiili standart** optimizer.

> 🧠 **Neden?** Learning curve'de gördün: $\alpha$ çok küçükse yavaş, çok büyükse zıplar. Adam bu dengeyi **otomatik** kurmaya çalışır.

---

## 16. Katman türleri: yoğun (dense) ve evrişimli (convolutional)

Şimdiye kadar hep **Dense** katman: her nöron önceki katmandaki **tüm** aktivasyonları alır.

### Convolutional (evrişimli) katman

Her nöron girdinin **yalnızca bir bölgesine** bakar.

- **Görüntü örneği:** İlk nöron görüntünün sol üst küçük dikdörtgenine, ikincisi başka bir dikdörtgene bakar, vb.
- **EKG (zaman serisi) örneği:** Her nöron sinyalin **belirli bir zaman penceresine** bakar (100 zaman noktasından 20'lik pencereler gibi).

**Avantajları:**
1. **Daha hızlı hesap** (her nöronun girdisi az)
2. **Daha az eğitim verisi** gerekir
3. **Aşırı uyuma (overfitting) daha az** yatkın

**Tasarım seçimleri:** pencere boyutu, kaç nöron. Bu mimarilere **Convolutional Neural Network (CNN)** denir. Başka özel katmanlar da var (örn. **transformer** katmanları), ama bu kursta ayrıntıya girilmiyor.

> 🧠 **Neden?** Tüm piksellere bakmak çoğu zaman gereksizdir; yerel örüntüler (kenar, parça) yeterlidir. Mimariyi probleme göre **tasarlamak** performansı artırır.

---

## 17. (Opsiyonel) Türev, hesaplama grafiği ve backpropagation

Bu bölüm **isteğe bağlı**; amaç `model.fit`'in içinde neler olduğuna sezgi kazandırmak.

### Türevin sezgisi

Basit cost: $J(w)=w^2$, $w=3$ → $J=9$.

$w$'yi $\varepsilon=0.001$ artır: $w=3.001 \Rightarrow J=9.006001$. $J$ yaklaşık **6 katı** kadar arttı:

$$\Delta J \approx 6 \cdot \varepsilon \;\Rightarrow\; \frac{\partial J}{\partial w}=6 \quad (\text{analitik: } 2w = 6 \checkmark)$$

**Yorum:** Türev, "$w$'yi küçük bir miktar artırırsam $J$ **kaç katı** artar?" demektir. Büyük türev → $w$'ye duyarlı. Küçük türev → $w$'nin değişmesi $J$'yi pek etkilemez.

### Hesaplama grafiği (computation graph)

Cost hesabı küçük adımlara bölünür; her adım bir **düğüm**. TensorFlow türevleri bu grafikle **otomatik** bulur.

**Örnek:** tek nöron, linear aktivasyon, $x=-2,\ y=2,\ w=2,\ b=8$:

```
w ──► c = w·x ──► a = c + b ──► d = a − y ──► J = ½ d²
```

**Forward (soldan sağa):**

$$c = 2\cdot(-2) = -4 \;\to\; a = -4 + 8 = 4 \;\to\; d = 4-2 = 2 \;\to\; J = \tfrac{1}{2}\cdot 2^2 = 2$$

### Backprop = **sağdan sola** türev hesabı

Zincir kuralıyla **her düğümün türevini bir sonrakinden** hesaplarız:

| Türev | Değer | Neden |
|---|---|---|
| $\partial J/\partial d$ | $d = 2$ | $J=\tfrac12 d^2$ |
| $\partial J/\partial a$ | $2$ | $d=a-y$, türev 1 |
| $\partial J/\partial c$ | $2$ | $a=c+b$, türev 1 |
| $\partial J/\partial b$ | $2$ | $a=c+b$, türev 1 |
| $\partial J/\partial w$ | $2\cdot x = -4$ | $c=wx$, $\partial c/\partial w = x=-2$ |

**Doğrulama:** $w=2.001$ yapınca $J \approx 1.996$ olur ($2-0.004$): türev $-4$ ✓.

### Neden backprop verimli?

$N$ düğüm ve $P$ parametre için her parametrenin türevini ayrı ayrı hesaplamak **~$N\times P$** adım ister. Backprop ise **~$N+P$** adımda **hepsini** verir.

> 🧠 **Neden?** Sağdan sola giderken **ortak ara türevleri yeniden kullanırsın** (örn. $\partial J/\partial a$ hem $b$ hem $w$ için kullanılır). Milyonlarca parametreli ağlar bu sayede eğitilebilir. **TensorFlow/PyTorch** bu grafiği senin için çıkarır ("autodiff").

---

# HAFTA 3 – MAKİNE ÖĞRENİMİ UYGULAMA TAVSİYELERİ

## 18. Sonra ne denemeli? Modeli değerlendirmek

Elinde bir model var, tahminleri **kabul edilemeyecek kadar kötü**. Ne yaparsın? Seçenekler:

- Daha fazla eğitim verisi topla
- Daha az özellik dene
- Ek özellik topla
- Polinom özellikler ekle
- $\lambda$'yı (regularization) azalt
- $\lambda$'yı artır

Rastgele denemek **haftalar/aylar** kaybettirir. **Tanılama (diagnostic)** yap: neyin yanlış olduğunu söyleyen test. Hangi yola yatırım yapacağını bilirsin.

> 🧠 **Neden?** Verimli ekipler algoritmalarını bilmekten çok **"sonraki adımda ne yapayım?"** kararını doğru verir. Aynı işi bir ekip 6 ayda, daha yetenekli bir ekip birkaç haftada yapar.

### Modeli değerlendirme: train/test ayrımı

Veriyi ikiye böl (örn. **%70 eğitim, %30 test**):

$$J_{train}(\vec{w},b) \quad \text{(eğitimde hata)} \qquad J_{test}(\vec{w},b) \quad \text{(testte hata)}$$

- Hata hesabında **regularization terimi yok**: sadece $\frac{1}{2m}\sum (f-y)^2$ (regresyon) veya log loss.
- **Sınıflandırma için alternatif:** hatalı sınıflandırılan örneklerin **oranı** (kesirli hata). $f(x)\ge0.5$ → $\hat{y}=1$; $\hat{y}\neq y$ ise hata say.

**Overfit sinyali:** $J_{train}$ çok **düşük**, $J_{test}$ çok **yüksek**. Model eğitimde ezberledi, yeni veride kötü.

---

## 19. Model seçimi: train / cross-validation / test

**Sorun:** Birçok model arasında (örn. polinom derecesi $d=1,\dots,10$) hangisi? $J_{test}$'e bakıp seçersen, **test setine "uydurmuş" olursun**. $J_{test}$ genelleme hatasının **iyimser** bir tahmini olur.

**Çözüm: veriyi üçe böl**

| Küme | Oran | Ne için? |
|---|---|---|
| **Training set** | ~%60 | Parametreleri ($w,b$) öğren |
| **Cross-validation (dev) set** | ~%20 | **Model seçimi** (derece $d$, $\lambda$, mimari) |
| **Test set** | ~%20 | **Sadece sonda**, nihai genelleme hatası |

$$J_{train},\quad J_{cv},\quad J_{test}$$

**Süreç:**
1. Her aday model ($d=1..10$) için $w,b$'yi **eğitim setinde** öğren.
2. Her birinin **$J_{cv}$**'sini hesapla.
3. **En düşük $J_{cv}$**'ye sahip modeli seç.
4. Seçilen modelin **$J_{test}$**'ini bir kez hesaplayıp genelleme hatası olarak raporla.

> 🧠 **Neden üçüncü küme?** Cross-validation seti **seçim yapmak** için kullanıldığından ona da bir ölçüde "uydurulur". Test seti **hiç dokunulmamış** kalır, bu yüzden **dürüst** (tarafsız) bir ölçüdür. **Test setine bakarak karar verme.**

Bu yaklaşım sinir ağı mimarisi seçiminde de geçerli: farklı mimarileri eğit, $J_{cv}$ en düşük olanı seç, $J_{test}$ ile raporla.

---

## 20. Bias ve variance'ı teşhis etme

Polinom regresyonda (derece $d$) üç durum:

| | **High bias** (underfit) | **Doğru** | **High variance** (overfit) |
|---|---|---|---|
| Model | Çok basit (örn. düz çizgi $d=1$) | Uygun ($d=2$) | Çok karmaşık ($d=4$) |
| $J_{train}$ | **Yüksek** | **Düşük** | **Düşük** |
| $J_{cv}$ | **Yüksek** | **Düşük** | **Yüksek** ($J_{train}$'den çok büyük) |

**Teşhis kuralı:**

| Gözlem | Teşhis |
|---|---|
| $J_{train}$ yüksek | **High bias** |
| $J_{train}$ düşük, $J_{cv} \gg J_{train}$ | **High variance** |
| İkisi de yüksek, $J_{cv} \gg J_{train}$ | **Her ikisi birden** (nadir, sinir ağlarında olabilir) |

**$d$ arttıkça:**
- $J_{train}$ **sürekli düşer** (model karmaşıklaştıkça eğitime daha iyi uyar).
- $J_{cv}$ önce düşer, sonra **tekrar yükselir** (U şekli). En dip = en iyi $d$.

> 🧠 **Neden $J_{train}$'e bakmak yetmez?** $J_{train}$ düşük olsa da model overfit olabilir. **$J_{cv}$ ile $J_{train}$'i birlikte** okumalısın: biri "eğitimde bile kötü mü?", diğeri "yeni veride kötü mü?" sorusunu cevaplar.

---

## 21. Regularization ve bias/variance

Regularization parametresi $\lambda$ ile **karmaşıklık** kontrol edilir.

| $\lambda$ | Etki | Teşhis |
|---|---|---|
| **Çok büyük** | $w \approx 0$, model **düz çizgi** | $J_{train}$ yüksek → **high bias** |
| **Çok küçük (0)** | Ceza yok, dalgalı eğri | $J_{train}$ düşük, $J_{cv}$ yüksek → **high variance** |
| **Orta** | Denge | $J_{train}$ ve $J_{cv}$ ikisi de düşük ✅ |

### $\lambda$'yı nasıl seçersin?

Cross-validation ile:

1. $\lambda$ için aday değerler dene (örn. $0,\ 0.01,\ 0.02,\ 0.04,\ 0.08,\ \dots$, yaklaşık **2 katı** artarak $\sim10$'a kadar).
2. Her $\lambda$ için $w,b$'yi öğren (**regularization'lı** cost ile).
3. Her birinin $J_{cv}$'sini hesapla (**regularization terimi OLMADAN**).
4. **En düşük $J_{cv}$**'ye sahip $\lambda$'yı seç.

**$\lambda$ arttıkça:** $J_{train}$ **artar**, $J_{cv}$ **U şekli** çizer. Bu, derece $d$ grafiğinin **ayna görüntüsüdür**: küçük $\lambda$ = yüksek $d$ (overfit), büyük $\lambda$ = düşük $d$ (underfit).

> 🧠 **Neden?** $\lambda$ bir "karmaşıklık düğmesi". Çok çevirirsen model fazla basitleşir (bias), az çevirirsen fazla karmaşık kalır (variance). Doğru noktayı **veri** söylesin.

---

## 22. Baseline performans: "iyi" ne demek?

$J_{train}=10.8\%$ hata iyi mi, kötü mü? **Referansa göre** değişir. Bu referansa **baseline** denir. Kaynakları:

- **İnsan seviyesi performans** (özellikle ses/görüntü gibi insanın iyi olduğu işlerde)
- **Rakip algoritma** performansı
- **Önceki deneyim / tahmin**

### Örnek: konuşma tanıma (web aramasında)

| Ölçüt | Değer |
|---|---|
| **Baseline** (insan seviyesi hata) | **%10.6** |
| Eğitim hatası $J_{train}$ | %10.8 |
| Çapraz doğrulama hatası $J_{cv}$ | %14.8 |

**İki farkı ayrı ayrı oku:**

| Fark | Değer | Anlam |
|---|---|---|
| $J_{train}$ − Baseline | $10.8 - 10.6 = $ **0.2** | **Küçük** → **bias sorunu yok** |
| $J_{cv}$ − $J_{train}$ | $14.8 - 10.8 = $ **4.0** | **Büyük** → **variance sorunu var** |

> 🧠 **Neden baseline?** İnsan bile bu seslerin %10.6'sını anlamıyorsa (gürültü, kötü kayıt) eğitim hatasının %10.8 olması **mükemmel**dir; %0'a inmeyi beklemek yanlış. Baseline olmadan "yüksek" ve "düşük" **anlamsızdır**.

**Genel teşhis:**
- **Baseline ile $J_{train}$ arası büyük** → **high bias**
- **$J_{train}$ ile $J_{cv}$ arası büyük** → **high variance**

---

## 23. Öğrenme eğrileri (learning curves)

**Öğrenme eğrisi:** $J_{train}$ ve $J_{cv}$'yi **eğitim seti büyüklüğünün** ($m_{train}$) fonksiyonu olarak çiz.

**Genel davranış:**
- $m$ arttıkça **$J_{cv}$ düşer** (daha çok veri = daha iyi genelleme).
- $m$ arttıkça **$J_{train}$ artar** (az örneğe mükemmel uymak kolay, çok örneğe zor).
- Genelde $J_{cv} \ge J_{train}$.

### High bias durumu (düz çizgi uyduruluyor)

- Eğriler bir noktadan sonra **düzleşir** ve **yüksek** kalır.
- Baseline'ın **çok üstünde** düzleşir.
- **Daha fazla veri işe yaramaz!** Çok basit model, ne kadar veri olursa olsun, karmaşık örüntüyü yakalayamaz.

### High variance durumu (4. dereceden polinom, küçük $\lambda$)

- $J_{train}$ düşük, $J_{cv}$ **çok yüksek**; aralarında **büyük boşluk** var.
- $m$ arttıkça **boşluk kapanır**: $J_{cv}$ düşmeye devam eder.
- **Daha fazla veri yardımcı olur!**

> 🧠 **Neden?** Overfit model **veri azlığından** şikâyetçi (çok esnek, az örnek). Underfit model **esneklikten** şikâyetçi (çok basit); veri ekleyince daha iyi bir düz çizgi bulur ama hâlâ düz çizgidir.

**Pratik not:** Öğrenme eğrisi çizmek **hesaplama maliyetlidir** (farklı $m$'ler için eğitim). Çoğu zaman sadece zihinsel bir model olarak kullanılır.

---

## 24. Sonraki adıma karar verme: özet tablo

Başa dön: 6 seçenek. Her biri **hangi sorunu** çözer?

| Deneyeceğin şey | Çözdüğü sorun |
|---|---|
| **Daha fazla eğitim verisi** | **High variance** |
| **Daha az özellik** (feature selection) | **High variance** |
| **$\lambda$'yı artır** | **High variance** |
| **Ek özellik** ekle | **High bias** |
| **Polinom özellik** ekle | **High bias** |
| **$\lambda$'yı azalt** | **High bias** |

**Karar akışı:**

```
J_train, baseline'dan çok mu yüksek?   ── Evet ──►  HIGH BIAS     → ek/polinom özellik, λ'yı azalt,
        │                                                            (sinir ağında: daha büyük ağ)
        Hayır
        ▼
J_cv, J_train'den çok mu yüksek?       ── Evet ──►  HIGH VARIANCE → daha çok veri, daha az özellik, λ'yı artır
        │
        Hayır
        ▼
       Bitti (iyi performans) 🎉
```

> ⚠️ **Yaygın hatalar:**
> - "Overfit olduğu için **daha az veri** kullanayım" → yanlış, daha az veri overfit'i kötüleştirir.
> - High bias iken **daha çok veri toplamak** vakit kaybıdır.

---

## 25. Bias/variance ve sinir ağları

Eski ML'de **bias–variance değiş tokuşu** (trade-off) vardı: model karmaşıklığını artırırsan bias düşer ama variance artar. İkisini aynı anda düşürmek zordu.

**Sinir ağları bu ikilemi büyük ölçüde kırar:**

> **Yeterince büyük bir sinir ağı**, makul büyüklükteki eğitim setlerinde **neredeyse her zaman düşük bias**'a ulaşır (eğitim setine iyi uyar).

### Reçete (sinir ağları için)

```
1. Eğitim setinde iyi mi?   (J_train, baseline'a yakın mı?)
        │ Hayır ──► DAHA BÜYÜK AĞ (daha çok nöron / katman)  ──► tekrar 1
        │ Evet
        ▼
2. Cross-validation setinde iyi mi?   (J_cv, J_train'e yakın mı?)
        │ Hayır ──► DAHA FAZLA VERİ topla                   ──► tekrar 1
        │ Evet
        ▼
       Bitti ✅
```

**Sınırlar:**
- Ağı büyütmek **hesaplamayı** pahalılaştırır (GPU yardımcı olur).
- Veri toplamak her zaman mümkün değildir.

> 🧠 **Neden "büyük ağ + düzgün regularization" asla zarar vermez?** Bir ağı **uygun şekilde regularize edersen**, büyütmek **neredeyse hiç** performansı kötüleştirmez (hesap maliyeti dışında). Büyük ağ yine düşük bias verir, regularization variance'ı kontrol eder. Bu, derin öğrenmenin büyük başarısının nedenlerinden biri.

**TensorFlow'da regularization:**

```python
from tensorflow.keras.regularizers import L2

layer = Dense(units=25, activation='relu', kernel_regularizer=L2(0.01))   # λ = 0.01
```

---

## 26. ML geliştirme döngüsü ve hata analizi

### Yinelemeli döngü

```
Mimariyi seç (model, veri, hiperparametreler)
      ▼
   Eğit
      ▼
Tanıla (bias, variance, hata analizi)
      ▼
   (tekrar mimariyi/veriyi ayarla) ──► ...
```

Gerçek projelerde bu döngüyü **birçok kez** döndürürsün. Örnek: **spam sınıflandırıcı** (e-posta metni → spam mi?). Özellik: en sık 10.000 kelimenin varlığı.

### Hata analizi (error analysis)

Bias/variance'tan sonra **ikinci en önemli** tanı aracı.

**Süreç:** Cross-validation setinde **yanlış sınıflandırılan örneklere elle bak**, ortak temalara göre grupla.

**Örnek:** $m_{cv}=500$, algoritma **100'ünü** yanlış sınıflandırıyor. Bu 100'ü incele:

| Kategori | Sayı |
|---|---|
| **İlaç (farmasötik) spam** | **21** |
| Şifre/kimlik avı (phishing) | **18** |
| Olağandışı e-posta yönlendirme (routing) | 7 |
| Kasıtlı yazım hataları | 3 |
| ... (diğerleri, kategoriler üst üste binebilir) | |

**Sonuç:** İlaç spam'i ve phishing **büyük sorunlar**; kasıtlı yazım hatalarını yakalamak için karmaşık bir algoritma yazsan bile **100 hatadan yalnızca 3'ünü** çözersin. **Önceliği doğru kişiye ver.**

**Çözümler (bulgulara göre):**
- Çok ilaç spam'i → **daha çok ilaç spam verisi** topla, ilaç adı gibi **yeni özellikler** ekle.
- Phishing çok → **URL özellikleri** ekle.

**Sınır:** Hata sayısı çok büyükse (örn. 5000 hata), **rastgele 100–500 örnek** seç ve onları incele. Hata analizi **insanın iyi olduğu** işlerde (spam, görüntü) kolaydır. İnsanın zorlandığı işlerde (örn. reklam tıklaması tahmini) zordur.

> 🧠 **Neden?** Bias/variance **ne tür** bir sorunun olduğunu söyler. Hata analizi ise **tam olarak hangi örnekler** ve **hangi nedenle** yanlış olduğunu gösterir. Birlikte, zamanını **en çok fayda sağlayacak** işe harcamanı sağlar.

---

## 27. Veri eklemek: augmentation, synthesis ve transfer learning

Çoğu zaman **veri eklemek** performansı artırır. Ama "her şeyi" toplamak yerine **akıllıca ekle**.

### 1) Hedefli veri toplama

Hata analizi "ilaç spam'i problem" diyorsa **özellikle ilaç spam'i** topla (rastgele daha çok e-posta değil). Daha ucuz ve etkili.

### 2) Data augmentation (veri çoğaltma)

Mevcut örneklerden **değiştirilmiş yeni örnekler** üret.

- **Görüntü / OCR:** harfi **döndür, büyüt/küçült, kontrastı değiştir, bozulma ekle** (örn. "A" harfinin çarpıtılmış hâlleri). Etiket aynı kalır.
- **Konuşma:** ses klibine **arka plan gürültüsü** (kalabalık, araba sesi), kötü bağlantı ekle.

**Önemli:** Eklenen bozulma, **test setinde karşılaşacağın** gerçek bozulmayı yansıtmalı. **Rastgele anlamsız gürültü** eklemek işe yaramaz.

### 3) Data synthesis (sentetik veri)

**Sıfırdan yapay örnek üret.** Örnek: OCR için bilgisayarındaki **farklı fontlarla** harf/kelime görüntüleri üretmek (renk, kontrast, bulanıklık ayarlayarak).

### Model-merkezli vs. veri-merkezli

Geleneksel araştırma **model-merkezli** ($\to$ algoritmayı geliştir, veri sabit). Pek çok pratik uygulamada **veri-merkezli** yaklaşım (algoritma hazır, **veriyi geliştir**) daha verimli.

### Transfer learning (aktarımlı öğrenme)

Çok az veriniz varsa: **başka bir büyük görevde önceden eğitilmiş** bir ağdan yararlan.

**Adımlar:**
1. **Supervised pretraining:** Büyük bir veri setinde (örn. 1 milyon görüntü, 1000 sınıf) ağı eğit (veya **hazır indir**).
2. Son (çıkış) katmanı **kendi görevine göre değiştir** (örn. 1000 çıkış yerine 10 rakam çıkışı).
3. **Fine-tuning** (ince ayar), iki seçenek:
   - **Seçenek 1:** **Sadece yeni çıkış katmanının** parametrelerini eğit (çok az veri varsa).
   - **Seçenek 2:** **Tüm parametreleri** eğit (biraz daha çok veri varsa), ama ilk değerler önceden eğitilmişten gelir.

**Şart:** Önceden eğitilen görevin **girdi tipi aynı** olmalı (görüntü → görüntü, ses → ses, metin → metin).

> 🧠 **Neden işe yarar?** İlk katmanlar **genel örüntüleri** (kenar, köşe, eğri) öğrenmiştir ve bunlar çok farklı görüntü görevlerinde işe yarar. Sen sadece **görev-spesifik** son kısmı öğretirsin. Pre-trained model indirip fine-tune etmek **yaygın pratiktir** (GPT, BERT, ImageNet modelleri gibi).

---

## 28. Bir ML projesinin yaşam döngüsü

Model eğitmek projenin **küçük bir parçasıdır**:

```
1. Projenin kapsamını belirle   (ne yapacağız? ör. konuşma tanıma)
         ▼
2. Veriyi topla                 (hangi veri? ne kadar?)
         ▼
3. Modeli eğit                  (eğit → tanıla → hata analizi → tekrar)
         ▼
4. Üretime al (deploy)          (gerçek kullanıcılara)
         ▼
5. İzle ve bakım yap            (performans düştü mü? veri değişti mi?)
```

**Deployment:** Model bir **inference sunucusuna** koyulur; mobil uygulama bir **API çağrısı** ile $x$'i yollar, sunucu $\hat{y}$ döndürür.

**Pratik konular:** Yeterli **hesaplama** ve **yazılım mühendisliği** gerekir (ölçekleme, güvenilirlik, logging, **veri kayması** takibi). Veri zamanla değişirse (yeni telaffuzlar, yeni cihazlar) model **güncellenmeli**.

**MLOps:** ML sistemlerinin **üretimde** sistematik olarak kurulması, dağıtılması ve izlenmesi pratiği.

> 🧠 **Neden?** Projede başarısızlık çoğu zaman **model**den değil, **veri**den, **dağıtımdan** veya **izlemeden** gelir. Tek seferlik model değil, **yaşayan bir sistem** kuruyorsun.

---

## 29. Adalet, önyargı ve etik

ML bugün **milyarlarca insanı** etkiliyor. Yanlış kullanım gerçek zarar verir.

**Kötü örnekler:**
- **İşe alım** algoritmaları kadınlara karşı **ayrımcılık** yaptı.
- **Yüz tanıma** sistemleri **koyu tenli** kişilerde çok daha kötü çalıştı; yanlış tutuklamalara yol açtı.
- **Kredi/ipotek onayı**nda azınlık gruplara karşı önyargı.
- **Stereotip** pekiştiren çıktılar.

**Kötü niyetli kullanımlar:** **Deepfake**'ler, spam/sahte içerik üretimi, kullanıcıları **bağımlı kılıp** sömürmek, **gizli gözetim**.

### Pratik rehber

1. **Çeşitli bir ekip** kur (cinsiyet, etnik köken, kültür, deneyim). Farklı bakışlar sorunları **erkenden** yakalar.
2. Alanındaki **literatürü ve standartları** (örn. finans/sağlıkta zaten olan kurallar) araştır.
3. Dağıtmadan önce modeli **farklı alt gruplar (alt popülasyonlar) üzerinde denetle** (audit): ten rengi, cinsiyet, yaş vb. boyutlarda performans eşit mi?
4. **Azaltma planı** hazırla: sorun çıkarsa **eski sisteme geri dönme (rollback)** gibi.
5. **Sistem dağıtıldıktan sonra da izle** ve sorun çıkarsa harekete geç.

**Not:** Etik "kolay" bir kontrol listesi değil; ama **sormak ve çabalamak** gerekir. Kötü bir ML sistemi ciddi zarar verebilir.

> 🧠 **Neden bu derste?** Doğruluk (accuracy) yüksek olsa bile model **bazı gruplara adaletsiz** olabilir. Yaptığın şeyin **toplumsal etkisi** olabilir. Mühendis olarak sorumluluk senin.

---

## 30. Çarpık veri setleri: precision, recall, F1

### Sorun

**Nadir hastalık** örneği: hastaların yalnızca **%0.5'i** hasta. Hiçbir şey öğrenmeyen "**$y=0$ (hep sağlıklı)**" algoritması **%99.5 doğrulukla** çalışır! Gerçek bir öğrenme algoritması **%99 doğruluk** (%1 hata) elde etse, ilk bakışta "daha kötü" görünür ama aslında **gerçekten bir şey öğrenmiştir**. **Doğruluk (accuracy)** çarpık veride **yanıltıcıdır.**

### Confusion matrix (karışıklık matrisi)

| | **Gerçek $y=1$** | **Gerçek $y=0$** |
|---|---|---|
| **Tahmin $\hat{y}=1$** | **TP** (True Positive) | **FP** (False Positive) |
| **Tahmin $\hat{y}=0$** | **FN** (False Negative) | **TN** (True Negative) |

$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN}$$

| Metrik | Soru | Sezgi |
|---|---|---|
| **Precision** (kesinlik) | "Hasta **dedikleri**mizin kaçı **gerçekten** hasta?" | Yanlış alarmı (**FP**) cezalandırır |
| **Recall** (duyarlılık/hatırlama) | "**Gerçek hastaların** kaçını **yakaladık**?" | Kaçırmayı (**FN**) cezalandırır |

**"Hep 0" algoritması:** TP = 0 → **recall = 0** (hiçbir hastayı bulamaz). Bu sorunu ortaya çıkarır.

> İkisi de **yüksek** olmalı. **Biri bile 0'a yakınsa algoritma işe yaramaz.**

### Sayısal örnek

100 test: **TP=15, FP=5, FN=10, TN=70**.

$$P = \frac{15}{15+5} = 0.75 \qquad R = \frac{15}{15+10} = 0.60$$

### Precision–Recall değiş tokuşu (eşik)

Logistic regression: $f(\vec{x}) \ge \text{eşik} \Rightarrow \hat{y}=1$.

| Eşik | Etki | Ne zaman? |
|---|---|---|
| **Yüksek** (örn. 0.7) | **Precision ↑, Recall ↓** (sadece çok emin olunca "hasta" de) | Yanlış alarm **pahalıysa** (örn. ağır tedavi) |
| **Düşük** (örn. 0.3) | **Precision ↓, Recall ↑** (şüphede bile "hasta" de) | Kaçırmak **pahalıysa** (örn. ölümcül hastalık, erken tarama) |

İkisini **birden** artırmak (iki ucu da iyi yapmak) ancak **daha iyi modelle** olur.

### F1 skoru

Birçok algoritma arasında **tek bir sayı** gerekir. **Ortalama almak kötü fikir** (biri çok kötüyken yüksek çıkar). **F1 = harmonik ortalama:**

$$F1 = \frac{2PR}{P+R} = \frac{1}{\tfrac12\left(\tfrac1P + \tfrac1R\right)}$$

Düşük olan değere **daha fazla ağırlık** verir.

| Algoritma | Precision | Recall | Ortalama | **F1** |
|---|---|---|---|---|
| 1 | 0.5 | 0.4 | 0.45 | **0.444** |
| 2 | 0.7 | 0.1 | 0.40 | **0.175** |
| 3 | 0.02 | 1.0 | **0.51** ⚠️ | **0.0392** |

Algoritma 3 "her şeye 1 de" gibi çalışır; **ortalama** onu **en iyi** gösterir ama **F1** doğru şekilde en kötü gösterir.

**Bizim örnek:** $F1 = \frac{2\cdot0.75\cdot0.6}{0.75+0.6} = \frac{0.9}{1.35} = 0.667$.

> 🧠 **Neden harmonik ortalama?** Aritmetik ortalama "biri çok iyi, biri çok kötü"yü **gizler**. Harmonik ortalama **zayıf halkaya** odaklanır: P veya R'den biri düşükse F1 de düşer.

---

# HAFTA 4 – KARAR AĞAÇLARI

## 31. Karar ağacı modeli

**Örnek:** Kedi sınıflandırması. 10 hayvan örneği, **kategorik özellikler**:

| Özellik | Olası değerler |
|---|---|
| **Kulak şekli** | sivri (pointy) / sarkık (floppy) |
| **Yüz şekli** | yuvarlak / yuvarlak değil |
| **Bıyık** | var / yok |

Hedef $y$: kedi (1) / kedi değil (0).

**Karar ağacı** soru soran bir yapıdır:

```
            [Kulak şekli?]            ← KÖK DÜĞÜM (root node)
            /            \
        sivri            sarkık
          │                │
     [Yüz şekli?]     [Bıyık?]        ← KARAR DÜĞÜMLERİ (decision nodes)
      /      \         /     \
  yuvarlak  değil     var    yok
    │         │        │      │
  KEDİ    KEDİ DEĞİL  KEDİ  KEDİ DEĞİL   ← YAPRAK DÜĞÜMLER (leaf nodes) = tahmin
```

- **Kök düğüm:** En üst, ilk soru.
- **Karar düğümü:** Bir özelliğe bakıp dallara ayırır.
- **Yaprak düğüm:** Son tahmin.
- **Tahmin:** Yeni örnek kökten başlar, soruları cevaplayarak yaprağa iner.

Aynı veri için **birçok farklı ağaç** olabilir; öğrenme algoritması **en iyisini** bulmaya çalışır.

> 🧠 **Neden "ağaç"?** Sorular dallanarak ilerler, ters çevrilmiş bir ağaca benzer (kök üstte).

---

## 32. Karar ağacı öğrenme süreci

İki **temel karar:**

### Karar 1: Her düğümde hangi özelliğe göre bölelim?

**Amaç: Saflığı (purity) maksimize etmek.** İdeal: bölme sonrası her dal **tamamen kedi** veya **tamamen kedi değil**.

- Kök düğümde **kulak şekli** seçilirse: sol dal 5 örnek (4 kedi, 1 köpek), sağ dal 5 örnek (1 kedi, 4 köpek).
- **Yüz şekli** seçilirse: sol 7 örnek (4 kedi, 3 köpek), sağ 3 örnek (1 kedi, 2 köpek). Daha **az saf**.
- **Bıyık** seçilirse: sol 4 örnek (3 kedi, 1 köpek), sağ 6 örnek (2 kedi, 4 köpek).

**Hangisi en iyi?** Matematiksel ölçü gerekir → **entropi** ve **bilgi kazancı** (Bölüm 33–34).

### Karar 2: Ne zaman bölmeyi bırakalım?

Durdurma kriterleri (biri yeterli):
1. Bir düğüm **%100 tek sınıf** (saf) olduğunda.
2. Bölmek ağacı **maksimum derinliği** aşırıyorsa. (Derinlik: kökten yaprağa kaç adım.)
3. Bölmenin sağladığı **saflık artışı bir eşiğin altındaysa.**
4. Düğümdeki **örnek sayısı bir eşiğin altındaysa.**

> 🧠 **Neden durdurma kriteri?** Ağaç çok derinse **eğitime ezberler** (**overfit**) ve yeni veride kötü performans verir. Ağacı **küçük** tutmak **genellemeyi** artırır, ayrıca ağaç daha okunaklı olur.

### Öğrenme algoritması (özet, **özyinelemeli**)

```
1. Tüm örneklerle KÖK düğümden başla.
2. Tüm özellikler arasında EN YÜKSEK BİLGİ KAZANCI sağlayanı seç.
3. Veriyi o özelliğe göre iki alt kümeye böl, iki dal oluştur.
4. Her dal için, durdurma kriteri sağlanana kadar 2–3'ü TEKRARLA (aynı algoritma, daha küçük veriyle).
```

---

## 33. Entropi: saflığı ölçmek

**Entropi** bir örnek kümesinin **safsızlığını** (impurity) ölçer.

$p_1$ = kümedeki **pozitif (kedi)** örneklerin oranı, $p_0 = 1-p_1$:

$$H(p_1) = -p_1\log_2(p_1) - p_0\log_2(p_0)$$

(Konvansiyon: $0\log_2 0 = 0$. Logaritma tabanı 2: böylece maksimum 1 olur.)

| Küme | $p_1$ | Entropi $H(p_1)$ | Yorum |
|---|---|---|---|
| 3 kedi, 3 köpek | 0.5 | **1** | **En safsız** (50–50) |
| 5 kedi, 1 köpek | 5/6 ≈ 0.83 | **0.65** | Oldukça saf |
| 2 kedi, 4 köpek | 2/6 ≈ 0.33 | **0.92** | Saf değil |
| 6 kedi (hepsi) | 1 | **0** | **Tam saf** |
| 6 köpek (hepsi) | 0 | **0** | **Tam saf** |

**Eğri şekli:** $p_1=0.5$'te **tepe (1)**, $p_1=0$ ve $p_1=1$'de **0**.

**Elle kontrol** ($p_1 = 0.8$):

$$H(0.8) = -0.8\log_2 0.8 - 0.2\log_2 0.2 = 0.8(0.322) + 0.2(2.322) = 0.258 + 0.464 \approx 0.72$$

> 🧠 **Neden entropi?** Tam karışık küme (50–50) **en belirsiz** (tahmin etmesi en zor) → yüksek entropi. Tam tek sınıf **kesin** → entropi 0. Entropi, "bu kümeden rastgele bir örnek çeksem **ne kadar şaşırırım**?" sorusunun ölçüsüdür. (Logistic loss ile aynı $-\log$ ailesinden.)

---

## 34. Bilgi kazancı (information gain) ve bölme seçimi

**Bilgi kazancı:** Bir özelliğe göre bölünce **entropinin ne kadar azaldığı**.

$$\text{Information Gain} = H(p_1^{\text{kök}}) - \left( w_{sol}\,H(p_1^{sol}) + w_{sağ}\,H(p_1^{sağ}) \right)$$

$w_{sol}, w_{sağ}$ = o dala giden örnek **oranları** (**ağırlıklı ortalama**, çünkü büyük dal daha önemli).

### 10 örnekli kedi veri seti

**Kök:** 5 kedi / 10 örnek → $p_1 = 0.5$ → $H = 1$.

| Bölme özelliği | Sol dal | Sağ dal | Ağırlıklı entropi | **Bilgi kazancı** |
|---|---|---|---|---|
| **Kulak şekli** | 4/5 kedi → $H(0.8)=0.72$ | 1/5 kedi → $H(0.2)=0.72$ | $\frac{5}{10}0.72+\frac{5}{10}0.72=0.72$ | $1-0.72=$ **0.28** |
| **Yüz şekli** | 4/7 kedi → $H=0.99$ | 1/3 kedi → $H=0.92$ | $\frac{7}{10}0.99+\frac{3}{10}0.92\approx0.97$ | $1-0.97=$ **0.03** |
| **Bıyık** | 3/4 kedi → $H=0.81$ | 2/6 kedi → $H=0.92$ | $\frac{4}{10}0.81+\frac{6}{10}0.92\approx0.88$ | $1-0.88=$ **0.12** |

**Kazanan: Kulak şekli (0.28).** Kök düğümde ona göre böleriz.

### Neden ağırlıklı ortalama?

Çok örnekli saf olmayan dal, **az örnekli saf olmayan dala** göre sonuca daha çok zarar verir. Ağırlıklandırma bunu yansıtır.

### Neden "bilgi kazancı" kavramı gerekli?

Durdurma kriterlerinden biri ("kazanç eşiğin altında") için de lazım: bölmenin **çok az fayda** sağladığı yerde **durup ağacı büyütme**.

### Tam algoritma (örnek)

1. Kökte kulak şekline böl → sol dal (5 örnek, 4 kedi), sağ dal (5 örnek, 1 kedi).
2. **Sol dalda** (4 kedi, 1 köpek) tekrar en iyi özelliği ara: **yüz şekli** → iki dal da **tam saf** (entropi 0) → yaprak.
3. **Sağ dalda** (1 kedi, 4 köpek) tekrar ara: **bıyık** → iki dal da tam saf → yaprak.
4. Durdur.

> 🧠 **Neden "açgözlü (greedy)"?** Her düğümde **o anki en iyi** bölmeyi seçeriz; bütün ağaç için **global olarak optimal** olduğunun garantisi yoktur. Ama pratikte çok iyi çalışır.

---

## 35. One-hot kodlama: ikiden fazla değer alan özellikler

Şimdiye kadar özellikler **2 değerliydi**. Ya **kulak şekli** üç değer alırsa: **sivri, oval, sarkık**?

**One-hot encoding:** $k$ değerli bir **kategorik** özelliği **$k$ adet 0/1 özelliğe** çevir.

| Hayvan | Eski: Kulak | → Sivri mi? | Oval mi? | Sarkık mı? |
|---|---|---|---|---|
| 1 | sivri | **1** | 0 | 0 |
| 2 | oval | 0 | **1** | 0 |
| 3 | sarkık | 0 | 0 | **1** |

"Bir tanesi **sıcak (1)**, diğerleri 0" olduğu için **"one-hot"**.

- Artık her yeni özellik **ikili** → mevcut karar ağacı algoritması aynen çalışır.
- **Sadece karar ağaçları için değil**: **sinir ağları, lineer/logistic regresyon** için de kategorik özellikleri sayıya çevirmenin standart yoludur.

> 🧠 **Neden $1,2,3$ (sıralı sayı) yapmıyoruz?** Sayısal kodlama sahte bir **sıra/mesafe** ima eder (sarkık > sivri gibi). One-hot, kategoriler arasında **böyle bir ilişki varsaymaz**.

---

## 36. Sürekli değerli özellikler

Kategorik özelliklerin yanında **sürekli sayılar** (örn. hayvanın **ağırlığı**, kg) da olabilir.

**Fikir:** Sürekli özelliği bir **eşik** ile ikiye böl: "ağırlık ≤ 9 mu?" Sonra yine **bilgi kazancı** ile en iyi eşiği seç.

**Algoritma:**
1. Örnekleri özelliğe göre **sırala**.
2. **Ardışık değerlerin ortalarını** aday eşik al ($m$ örnek için $m-1$ aday).
3. Her aday eşik için **bilgi kazancını** hesapla.
4. **En yüksek** kazançlı eşiği seç.
5. Bu en iyi kazancı diğer özelliklerin kazancıyla karşılaştır; en iyisine böl.

**Örnek (derste):** "Ağırlık ≤ 8" için bilgi kazancı **0.24**, "Ağırlık ≤ 9" için **0.61** → 9 çok daha iyi. Diğer aday eşikler de aynı şekilde denenir ve en yüksek kazançlı olan seçilir.

> 🧠 **Neden?** Sürekli özellik aslında çok sayıda "evet/hayır" sorusuna çevrilebilir ("ağırlık ≤ 8?", "≤ 9?", ...). Algoritma onlardan **en bilgilendirici** olanı seçer.

---

## 37. Regresyon ağaçları

Şimdiye kadar **sınıflandırma** ağaçları. Aynı yapıyla **sayı tahmini** (regresyon) de yapılır. Örnek: hayvanın **ağırlığını** (kg) tahmin et (girdiler: kulak, yüz şekli, bıyık).

**Tahmin:** Yaprağa düşen **eğitim örneklerinin ağırlıklarının ortalaması**.

| Konu | Sınıflandırma ağacı | Regresyon ağacı |
|---|---|---|
| Yaprak çıktısı | Çoğunluk sınıfı | **Ortalama değer** |
| Bölme ölçütü | **Entropi azalması** (bilgi kazancı) | **Varyans azalması** |

**Varyans** (dağılım): bir kümedeki sayıların ortalamadan sapması. Düşük varyans = değerler birbirine yakın (iyi).

$$\text{Varyans azalması} = \text{Var}_{kök} - \left(w_{sol}\text{Var}_{sol} + w_{sağ}\text{Var}_{sağ}\right)$$

Her adımda **en çok varyans azaltan** özellik seçilir. Aynı bilgi kazancı mantığı, entropi yerine varyansla.

> 🧠 **Neden?** Sınıflandırmada "saf" = tek sınıf. Regresyonda "saf" = **değerler birbirine yakın** (düşük varyans). Mantık aynı: dalları mümkün olduğunca homojen yap.

---

## 38. Ağaç toplulukları ve örnekleme (bootstrap)

### Tek karar ağacının zayıflığı

**Veride küçük bir değişikliğe aşırı duyarlı.** 10 örnekten **sadece birini** değiştirirsen, kökteki en iyi bölme tamamen değişebilir ve **tamamen farklı bir ağaç** çıkar. Yani ağaç **dayanıksız** (yüksek varyans).

### Çözüm: Ağaç topluluğu (tree ensemble)

**Çok sayıda ağaç** eğit, **oylamayla** karar ver.
- Yeni örnek her ağaca verilir; her ağaç bir tahmin yapar.
- **En çok oy alan** tahmin seçilir (regresyonda **ortalama**).

> 🧠 **Neden işe yarar?** Tek ağaç küçük değişikliğe hassas, ama **birçok ağacın oyu** her birinin rastgele hatasını **ortalar** → daha **sağlam** tahmin.

### Değiştirerek örnekleme (sampling with replacement)

Her ağaç için **biraz farklı bir eğitim seti** gerekir. Çözüm: orijinal $m$ örnekten **değiştirerek** $m$ örnek çek.

- Bir torbadan **çek → not al → GERİ KOY → tekrar çek.**
- Aynı örnek **birden fazla** çıkabilir, bazı örnekler **hiç** çıkmayabilir.
- Yeni set **orijinaline benzer ama farklı**.

```python
idx = np.random.choice(m, size=m, replace=True)   # replace=True → değiştirerek
X_boot, y_boot = X[idx], y[idx]
```

Bu teknik **bootstrap** olarak da bilinir.

### Bagged decision tree

```
B kez (örn. B = 100) tekrarla:
    1. Değiştirerek örnekleyerek yeni bir eğitim seti oluştur
    2. Bu sette bir karar ağacı eğit
Tahmin: B ağacın oylaması
```

**$B$ seçimi:** Tipik **64–128** civarı. $B$'yi çok artırmak zarar vermez ama **azalan getiri** ve daha fazla hesap.

---

## 39. Random forest (rastgele orman)

Bagged tree'de bir sorun: **tüm ağaçların kökü aynı özellik olabilir** (çünkü kök hep en güçlü özelliği seçer) → ağaçlar birbirine **fazla benzer**.

**Random forest ek rastgelelik ekler:** **Her düğümde**, $n$ özelliğin **rastgele bir alt kümesini** ($k$ özellik) seç, bölmeyi sadece **bunlar arasından** yap.

- Tipik: $k = \sqrt{n}$ (özellik sayısı $n$ **büyükse**).

```
B kez tekrarla:
    1. Değiştirerek yeni eğitim seti oluştur
    2. Ağacı eğit, ama her düğümde sadece k=√n rastgele özelliği değerlendir
Tahmin: B ağacın oylaması
```

> 🧠 **Neden bu daha iyi?** Ağaçlar **birbirinden daha farklı** (çeşitlilik) olur, oy birliği daha **güvenilir** hata yapar. Aynı fikir: "10 benzer uzmandan daha çok **farklı bakış açısına** sahip 10 uzman daha doğrudur."

**Sonuç:** Tek ağaçtan **çok daha sağlam**.

---

## 40. XGBoost: boosting

**Boosting** (güçlendirme): bagged tree'ye benzer, ama bir fark: **sonraki ağaç, önceki ağaçların hatalarına odaklanır.**

**Fikir ("bilinçli pratik"):** Bir şeyi öğrenirken zaten iyi bildiğin kısımları tekrar çalışmak yerine **yapamadığın** kısımlara odaklanırsın.

```
B kez tekrarla:
    1. Değiştirerek yeni set oluştur AMA daha önce eğitilen ağaçların
       YANLIŞ sınıflandırdığı örneklerin seçilme olasılığını ARTIR
    2. Bu sette yeni bir ağaç eğit
```

(Matematiksel ayrıntı gerekmez.)

### XGBoost (eXtreme Gradient Boosting)

Boosting'in en popüler uygulaması:
- **Hızlı, açık kaynak, verimli**
- Bölme ve durdurma için **iyi varsayılan ölçütler** (içinde)
- **Yerleşik regularization** (overfit'i engeller)
- **Makine öğrenimi yarışmalarında** (Kaggle) ve ticari uygulamalarda çok başarılı

```python
from xgboost import XGBClassifier      # sınıflandırma
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from xgboost import XGBRegressor       # regresyon
model = XGBRegressor()
```

**Not:** XGBoost her örneğe **farklı ağırlık** verir (bootstrap yerine), ayrıntı kütüphanede gizli.

> 🧠 **Neden güçlü?** Hatalara **bilinçli odaklanma**, tüm örneklere eşit davranmaktan daha etkili öğrenmeyi sağlar.

---

## 41. Karar ağacı mı sinir ağı mı?

| | **Karar ağaçları / ağaç toplulukları** | **Sinir ağları** |
|---|---|---|
| **Veri türü** | **Tablo (yapılandırılmış) veri**: elektronik tablo gibi; kategorik + sürekli özellikler (ev fiyatı, müşteri verisi) | **Her tür veri**: tablo **ve** yapılandırılmamış (görüntü, ses, video, metin) |
| **Yapılandırılmamış veri** | ❌ Önerilmez | ✅ **En iyisi** |
| **Eğitim hızı** | ✅ **Hızlı** (yinelemeli geliştirme döngüsü hızlanır) | ❌ Yavaş (büyük ağlar saatler/günler) |
| **Yorumlanabilirlik** | ✅ **Küçük ağaçlar** insan tarafından okunabilir (100 ağaçlık topluluk için pek değil) | ❌ Kara kutu |
| **Transfer learning** | ❌ Yok | ✅ Var |
| **Birden çok modeli birleştirme** | Zor | ✅ Birden çok sinir ağını **birlikte eğitmek** kolay (tek gradient descent ile) |

**Pratik öneri:**
- **Tablo veri** → önce **XGBoost** dene (hızlı, güçlü).
- **Görüntü/ses/metin** → **sinir ağı**.
- **Hedef:** ikisini de bil; probleme göre seç.

> 🧠 **Neden?** Ağaçlar "özellik üzerinde soru sor" mantığıyla çalışır, bu yüzden **zaten anlamlı özellikleri** olan tablo verisinde harikadır. Görüntüde tek bir piksel anlamsızdır; piksel **örüntülerini** çıkarmak için derin ağlar gerekir.

---

# EKLER

## 42. Büyük resim: tüm hikâye tek akışta

```
HAFTA 1-2: SİNİR AĞLARI
  Nöron:    a = g(w·x + b)            (g: sigmoid / ReLU / linear)
  Katman:   A_out = g(A_in @ W + B)   (vektörleştirilmiş)
  Ağ:       Katmanlar üst üste (Dense, Conv...) → forward prop
  Çıkış:    Binary → sigmoid | Multi-class → softmax | Regresyon → linear/ReLU
  Gizli:    ReLU (varsayılan); linear KULLANMA
  Eğitim:   model.compile(loss, optimizer=Adam) ; model.fit()   [backprop otomatik]
  Sayısal:  linear çıkış + from_logits=True

HAFTA 3: ML SİSTEMİ GELİŞTİRME
  Veriyi böl:  train (60%) / cv (20%) / test (20%)
  Teşhis:      J_train vs baseline → BIAS ; J_cv vs J_train → VARIANCE
  Düzelt:      High bias → daha büyük model / özellik ekle / λ↓
               High variance → daha çok veri / az özellik / λ↑
  Sinir ağı:   Büyük ağ + (gerekirse) regularization + yeterli veri
  Hata analizi: yanlış örneklere ELLE bak, kategorize et, öncelik ver
  Veri:        augmentation / synthesis / transfer learning
  Çarpık veri: precision, recall, F1  (accuracy değil)
  Döngü:       kapsam → veri → eğit → deploy → izle ; etik/adalet

HAFTA 4: KARAR AĞAÇLARI
  Ağaç:      kök → karar düğümleri → yaprak
  Bölme:     bilgi kazancı = entropi azalması (regresyonda varyans azalması)
  Özellik:   kategorik → one-hot ; sürekli → eşik
  Topluluk:  bagged (bootstrap) → random forest (rastgele özellik) → XGBoost (boosting)
  Seçim:     tablo veri → ağaçlar/XGBoost ; yapılandırılmamış veri → sinir ağı
```

---

## 43. Hızlı formül kartı

| Konu | Formül |
|---|---|
| Nöron aktivasyonu | $a_j^{[l]} = g(\vec{w}_j^{[l]}\cdot\vec{a}^{[l-1]} + b_j^{[l]})$ |
| Vektörleştirilmiş katman | $A_{out} = g(A_{in}W + B)$ |
| Sigmoid | $g(z) = 1/(1+e^{-z})$ |
| ReLU | $g(z) = \max(0, z)$ |
| Softmax | $a_j = e^{z_j}/\sum_k e^{z_k}$ |
| Softmax loss | $L = -\log a_y$ |
| Binary cross-entropy | $L = -y\log f - (1-y)\log(1-f)$ |
| Hata (train/cv/test) | $J_{train},\ J_{cv},\ J_{test}$ (regularization **YOK**) |
| High bias teşhisi | $J_{train}$ − baseline **büyük** |
| High variance teşhisi | $J_{cv}$ − $J_{train}$ **büyük** |
| Precision | $TP/(TP+FP)$ |
| Recall | $TP/(TP+FN)$ |
| F1 | $2PR/(P+R)$ |
| Entropi | $H(p_1) = -p_1\log_2 p_1 - (1-p_1)\log_2(1-p_1)$ |
| Bilgi kazancı | $H(\text{kök}) - \sum w_{dal}H(\text{dal})$ |
| Random forest özellik alt kümesi | $k = \sqrt{n}$ |

**Kontrol listesi:**
- ✅ Gizli katmanlarda **ReLU** (linear değil)
- ✅ Çıkış aktivasyonu **hedefe göre** (sigmoid / softmax / linear)
- ✅ Çok sınıflıda **`from_logits=True`** + linear çıkış (tahminde softmax uygula)
- ✅ **Model seçimini $J_{cv}$ ile** yap, $J_{test}$'e **sonda** bak
- ✅ Hata hesabında **regularization terimi yok**
- ✅ **Baseline** belirle, sonra bias/variance teşhis et
- ✅ Çarpık veride **accuracy değil** precision/recall/F1
- ✅ Kategorik özellikleri **one-hot** yap

---

## 44. Terim sözlüğü

| İngilizce | Transkriptteki çeviri / Türkçe |
|---|---|
| Forward propagation | "İleri pervane", "ileri yayılım" / **ileri yayılma** |
| Backpropagation | "Geri yayılım" / **geri yayılma** |
| Neuron / Unit | Nöron / birim |
| Hidden layer | Gizli katman |
| Activation | Aktivasyon |
| Dense layer | Yoğun katman |
| Convolutional layer | Evrişimli katman |
| Cross-entropy | "Çapraz tropi" / **çapraz entropi** |
| Logit | Aktivasyondan önceki $z$ değeri |
| Epoch | Tüm eğitim setinin bir tam geçişi |
| Optimizer | Optimizasyon algoritması (örn. Adam) |
| Computation graph | Hesaplama grafiği |
| Train / Cross-validation / Test set | Eğitim / çapraz doğrulama / test kümesi |
| Baseline | Karşılaştırma referansı (örn. insan performansı) |
| Bias / Variance | Önyargı / varyans |
| Learning curve | Öğrenme eğrisi |
| Error analysis | Hata analizi |
| Data augmentation | Veri çoğaltma |
| Transfer learning | Aktarımlı öğrenme |
| Precision / Recall | Kesinlik / duyarlılık (hatırlama) |
| Skewed dataset | Çarpık (dengesiz) veri seti |
| Decision tree | Karar ağacı |
| Root / Decision / Leaf node | Kök / karar / yaprak düğüm |
| Entropy | Entropi (safsızlık ölçüsü) |
| Information gain | Bilgi kazancı |
| One-hot encoding | One-hot kodlama |
| Sampling with replacement | Değiştirerek örnekleme (bootstrap) |
| Tree ensemble | Ağaç topluluğu |
| Random forest | Rastgele orman |
| Boosting | Güçlendirme (XGBoost) |
| Structured / Unstructured data | Yapılandırılmış (tablo) / yapılandırılmamış (görüntü, ses, metin) veri |
| ANI / AGI | Dar / genel yapay zeka |

---

## 45. Tam Python kodu

> Kodlar ders içeriğini örneklemek için yazılmıştır; sayılar örnektir.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import L2

# ---------- 1) Sıfırdan forward prop (Bölüm 7, 9) ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dense_loop(a_in, W, b, g):
    """Döngülü katman: W'nin her sütunu bir nöron."""
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        z = np.dot(W[:, j], a_in) + b[j]
        a_out[j] = g(z)
    return a_out

def dense_vec(A_in, W, B, g):
    """Vektörleştirilmiş katman: döngüsüz."""
    return g(np.matmul(A_in, W) + B)

# ---------- 2) İkili sınıflandırma (Bölüm 10–11) ----------
model_bin = Sequential([
    Dense(25, activation='relu'),
    Dense(15, activation='relu'),
    Dense(1,  activation='sigmoid')
])
model_bin.compile(loss=BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
# model_bin.fit(X, y, epochs=100)

# ---------- 3) Çok sınıflı (sayısal olarak kararlı sürüm; Bölüm 12–13) ----------
model_multi = Sequential([
    Dense(25, activation='relu'),
    Dense(15, activation='relu'),
    Dense(10, activation='linear')                 # softmax YOK
])
model_multi.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
# model_multi.fit(X, y, epochs=100)
# logits = model_multi(X_new)
# probs  = tf.nn.softmax(logits)                   # olasılığa çevir
# pred   = np.argmax(probs, axis=1)

# ---------- 4) Regularization (Bölüm 25) ----------
model_reg = Sequential([
    Dense(120, activation='relu', kernel_regularizer=L2(0.1)),
    Dense(40,  activation='relu', kernel_regularizer=L2(0.1)),
    Dense(10,  activation='linear')
])

# ---------- 5) Train / CV / Test bölme (Bölüm 19) ----------
from sklearn.model_selection import train_test_split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.40, random_state=1)
X_cv, X_test, y_cv, y_test     = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=1)

# ---------- 6) Precision / Recall / F1 (Bölüm 30) ----------
def precision_recall_f1(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

# ---------- 7) Entropi ve bilgi kazancı (Bölüm 33–34) ----------
def entropy(p1):
    if p1 == 0 or p1 == 1:
        return 0.0
    return -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)

def information_gain(y_node, y_left, y_right):
    p = lambda y: np.mean(y) if len(y) > 0 else 0.0
    w_left  = len(y_left)  / len(y_node)
    w_right = len(y_right) / len(y_node)
    return entropy(p(y_node)) - (w_left * entropy(p(y_left)) + w_right * entropy(p(y_right)))

# Kulak şekli örneği: 5 kedi / 10 örnek; sol: 4 kedi + 1 köpek; sağ: 1 kedi + 4 köpek
y_node  = np.array([1]*5 + [0]*5)
y_left  = np.array([1]*4 + [0]*1)
y_right = np.array([1]*1 + [0]*4)
print("Bilgi kazancı (kulak şekli):", round(information_gain(y_node, y_left, y_right), 2))  # ≈ 0.28

# ---------- 8) Bootstrap (Bölüm 38) ----------
def bootstrap_sample(X, y):
    m = X.shape[0]
    idx = np.random.choice(m, size=m, replace=True)
    return X[idx], y[idx]

# ---------- 9) Random forest ve XGBoost (Bölüm 39–40) ----------
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt')   # B=100, k=√n
# rf.fit(X_train, y_train)

from xgboost import XGBClassifier
xgb = XGBClassifier()
# xgb.fit(X_train, y_train)
```

---

# TEKRAR VE PRATİK

## 46. Elle hesap örnekleri

Formülleri anlamanın en hızlı yolu **kâğıtta** sayıları kendin çıkarmak.

### A) Parametre sayısı (Bölüm 5'teki ağ)

Ağ: $64 \to 25 \to 15 \to 1$.

| Katman | $W$ boyutu | $b$ | Toplam |
|---|---|---|---|
| 1 | $64\times25 = 1600$ | 25 | **1625** |
| 2 | $25\times15 = 375$ | 15 | **390** |
| 3 | $15\times1 = 15$ | 1 | **16** |
| | | | **2031** |

### B) Tek katmanlı forward prop

$\vec{x}=[1,\ 2]$. Katman 1 (2 nöron, ReLU): $\vec{w}_1=[1,\ 2],\ b_1=-1$ ve $\vec{w}_2=[-1,\ 1],\ b_2=0.5$.

$$z_1 = 1\cdot1+2\cdot2-1 = 4 \to a_1 = \text{ReLU}(4)=4$$
$$z_2 = -1\cdot1+1\cdot2+0.5 = 1.5 \to a_2 = \text{ReLU}(1.5)=1.5$$

Çıkış (sigmoid): $\vec{w}=[0.5,\ -1],\ b=0.2$, girdi $[4,\ 1.5]$:

$$z = 0.5\cdot4 - 1\cdot1.5 + 0.2 = 0.7 \qquad a = \frac{1}{1+e^{-0.7}} \approx 0.668$$

$0.668 \ge 0.5 \Rightarrow \hat{y}=1$.

### C) Softmax (3 sınıf)

$z=[2,\ 1,\ 0.1]$ → $a=[0.659,\ 0.242,\ 0.099]$. Doğru sınıf **2** ise $L=-\ln0.242 \approx 1.419$. Doğru sınıf **1** ise $L=-\ln0.659\approx0.417$.
> **Gözlemle:** Doğru sınıfa verilen olasılık küçüldükçe loss **hızla** artıyor.

### D) Backprop (Bölüm 17)

$x=-2,\ y=2,\ w=2,\ b=8$:
$J=2,\ \partial J/\partial b=2,\ \partial J/\partial w=-4$.

**Kendin dene:** Bir adım gradient descent ($\alpha=0.1$): $w = 2 - 0.1(-4) = 2.4$, $b = 8 - 0.1(2)=7.8$. Yeni $J$ ne olur?
($a = 2.4\cdot(-2)+7.8 = 3$, $d = 3-2 = 1$, $J = 0.5$. Cost $2 \to 0.5$ ✓.)

### E) Bias/variance teşhisi

Baseline %10.6, $J_{train}$ %10.8, $J_{cv}$ %14.8:
$10.8-10.6=0.2$ (küçük → bias sorunu yok), $14.8-10.8=4.0$ (büyük → **variance** sorunu) → **daha çok veri / regularization**.

### F) Precision, Recall, F1

$TP=15,\ FP=5,\ FN=10,\ TN=70$ → $P=0.75,\ R=0.60,\ F1=0.667$.

### G) Entropi ve bilgi kazancı

$H(0.8) = 0.72$. Kulak şekli: $1 - (0.5\cdot0.72 + 0.5\cdot0.72) = 0.28$.

---

## 47. Sık yapılan hatalar

| # | Hata | Neden sorun? | Doğrusu |
|---|---|---|---|
| 1 | **Gizli katmanlarda linear** aktivasyon | Tüm ağ tek lineer regresyona çöker | **ReLU** |
| 2 | Çıkış aktivasyonunu **hedefe göre seçmemek** | Olasılık yerine yanlış aralık | Binary→sigmoid, multi-class→softmax, regresyon→linear/ReLU |
| 3 | Çok sınıflıda softmax **ve** `from_logits=True` birlikte | Softmax iki kez uygulanır | Çıkış **linear** + `from_logits=True` |
| 4 | `from_logits=True` sonrası **olasılık beklemek** | Model logit ($z$) döndürür | `tf.nn.softmax(logits)` uygula |
| 5 | **Multi-label** işe softmax kullanmak | Olasılıkları toplam 1'e zorlar | Her çıkışa ayrı **sigmoid** |
| 6 | Model seçimini **test setiyle** yapmak | Test setine "uydurmuş" olursun | **CV seti** ile seç, test'e sonda bak |
| 7 | $J_{cv}$ hesaplarken **regularization terimini dahil etmek** | Hata yanlış ölçülür | Regularization **hariç** |
| 8 | **Baseline'sız** "yüksek hata" demek | Referanssız anlamsız | Önce baseline belirle |
| 9 | High bias iken **daha çok veri** toplamak | Basit model veriyle iyileşmez | Daha karmaşık model / özellik ekle |
| 10 | Overfit iken **daha az veri** kullanmak | Overfit kötüleşir | Daha çok veri, $\lambda$↑ |
| 11 | Çarpık veride **accuracy**'ye bakmak | "Hep 0" algoritması %99.5 alır | Precision, Recall, F1 |
| 12 | Precision ve Recall'ı **ortalamak** | Biri çok kötüyken yüksek çıkar | **F1** (harmonik ortalama) |
| 13 | Kategorik özelliği $1,2,3$ yapmak | Sahte sıra/mesafe | **One-hot** |
| 14 | Karar ağacını çok derin büyütmek | Overfit | Derinlik/örnek eşiği, **topluluk** |
| 15 | Görüntü/ses verisinde **karar ağacı** | Yapılandırılmamış veri | **Sinir ağı** |
| 16 | Veri artırmada **gerçekçi olmayan** bozulma | Test dağılımını yansıtmaz | Gerçek dünyada göreceğin bozulma |
| 17 | `np.array([200,17])` ile `[[200,17]]`'yi karıştırmak | 1B vs 2B | TensorFlow **2B** ister |
| 18 | Etik/adalet kontrolü **yapmamak** | Gruplara ayrımcılık | Alt gruplarda denetle |

---

## 48. Karıştırılan kavramlar

| Kavram A | Kavram B | Fark |
|---|---|---|
| **Multi-class** | **Multi-label** | Tek etiket ($N$ sınıftan biri, softmax) ↔ birden çok etiket aynı anda (ayrı sigmoidler) |
| **Sigmoid** | **Softmax** | İki sınıf, tek olasılık ↔ $N$ sınıf, toplamı 1 olan olasılıklar |
| **ReLU** | **Sigmoid** | Gizli katman, hızlı, doymaz ↔ ikili çıkış, iki tarafta düz |
| **Cross-validation set** | **Test set** | **Model seçimi** için ↔ **nihai** genelleme ölçümü, son adımda |
| **Bias** | **Variance** | Çok basit (eğitimde kötü) ↔ çok karmaşık (eğitimde iyi, yeni veride kötü) |
| **Precision** | **Recall** | Yanlış alarm (FP) odaklı ↔ kaçırma (FN) odaklı |
| **Entropi** | **Bilgi kazancı** | Bir kümenin safsızlığı ↔ bölmeyle entropinin **azalması** |
| **Bagged tree** | **Random forest** | Sadece örnek rastgeleliği ↔ örnek **+ özellik** rastgeleliği |
| **Random forest** | **XGBoost** | Ağaçlar **bağımsız** ↔ her ağaç öncekilerin **hatalarına odaklanır** |
| **Gradient descent** | **Adam** | Sabit $\alpha$ ↔ parametre başına **uyarlanan** $\alpha$ |
| **Dense layer** | **Convolutional layer** | Her nöron **tüm** girdilere bakar ↔ **bölgesel** bakar |
| **Data augmentation** | **Data synthesis** | Mevcut örneği **değiştir** ↔ **sıfırdan** üret |
| **Forward prop** | **Backprop** | Soldan sağa tahmin ↔ sağdan sola türev |
| **Transfer learning seçenek 1** | **Seçenek 2** | Sadece çıkış katmanını eğit ↔ tüm parametreleri ince ayarla |

---

## 49. Sorun → çözüm rehberi

| Gözlem | Olası sebep | Ne yapmalı? |
|---|---|---|
| $J_{train}$, baseline'dan **çok yüksek** | **High bias** | Daha büyük ağ, özellik/polinom ekle, $\lambda$↓ |
| $J_{train}$ düşük, $J_{cv}$ **çok yüksek** | **High variance** | Daha çok veri, az özellik, $\lambda$↑ |
| Sinir ağı eğitimde iyi değil | Ağ küçük | **Daha büyük ağ** |
| Sinir ağı eğitimde iyi, CV'de kötü | Veri az | **Daha fazla veri** |
| Öğrenme eğrileri **düzleşip yüksek** kalıyor | High bias | Veri eklemek **işe yaramaz**, modeli büyüt |
| Eğrilerde **büyük boşluk** var | High variance | **Daha çok veri** yardımcı olur |
| Accuracy yüksek ama hiç pozitif bulunmuyor | Çarpık veri | Recall/F1 bak, eşiği **düşür** |
| Çok **yanlış alarm** | Precision düşük | Eşiği **yükselt** |
| Önemli vakaları **kaçırıyor** | Recall düşük | Eşiği **düşür** |
| Spam gibi bir hata kategorisi baskın | Hedeflenebilir hata | **Hata analizi** + o kategoriye veri/özellik |
| Çok az etiketli verin var | Veri kıt | **Transfer learning**, augmentation |
| Gerçek dünyada performans düştü | **Veri kayması** | İzle, veriyi güncelle, yeniden eğit |
| Tablo veride zayıf sonuç | Yanlış model türü | **XGBoost** dene |
| Tek karar ağacı kararsız | Yüksek varyans | **Random forest / XGBoost** |
| Eğitim çok yavaş yakınsıyor | $\alpha$ sabit / ReLU yok | **Adam**, ReLU, **feature scaling** |
| Sayısal hata / `nan` | Softmax/sigmoid ara adım | Linear çıkış + **`from_logits=True`** |
| Model bazı gruplarda kötü | Adaletsizlik | Alt gruplarda **denetle**, veriyi dengele |

---

## 50. Kendini test et

Önce cevabı **kendin yaz**, sonra aç ve karşılaştır. Takıldığın sorunun ait olduğu bölüme dön.

**Hafta 1: Sinir ağları**

<details><summary><b>S1.</b> Bir nöronun yaptığı hesabı yaz. $a_j^{[l]}$ ne anlama gelir?</summary>

$a_j^{[l]} = g(\vec{w}_j^{[l]}\cdot\vec{a}^{[l-1]} + b_j^{[l]})$: katman $l$'deki $j$. nöronun aktivasyonu. Önceki katmanın çıktısını alır, ağırlıklı toplam + bias hesaplar, $g$'den geçirir. (Bölüm 4)
</details>

<details><summary><b>S2.</b> "4 katmanlı sinir ağı" kaç gizli katman içerir? (Giriş katmanı sayılır mı?)</summary>

Giriş katmanı **sayılmaz**. 4 katman = 3 gizli + 1 çıkış. (Bölüm 4)
</details>

<details><summary><b>S3.</b> 64 → 25 → 15 → 1 ağında toplam parametre sayısı kaçtır?</summary>

$1625 + 390 + 16 = 2031$. (Bölüm 46-A)
</details>

<details><summary><b>S4.</b> Gizli katman nöronlarını biz mi seçeriz ("fiyat, farkındalık, kalite")? Yoksa ağ mı öğrenir?</summary>

Ağ öğrenir. "Uygun fiyatlı mı" gibi etiketler sadece sezgi içindir; her nöron **tüm girdileri** alır ve $w$'leri veriden öğrenir. Bu, otomatik feature engineering'dir. (Bölüm 2)
</details>

<details><summary><b>S5.</b> $A_{in}$ ($1\times2$) ile $W$ ($2\times3$) çarpılırsa sonuç boyutu ne olur?</summary>

$1\times3$. İç boyutlar (2 ve 2) eşit olmalı; sonuç dış boyutlardan gelir. (Bölüm 9)
</details>

<details><summary><b>S6.</b> Vektörleştirme (matmul) neden döngüden hızlı?</summary>

Paralel donanımı (CPU/GPU) kullanıp tüm çarpmaları aynı anda yapar. (Bölüm 9)
</details>

**Hafta 2: Eğitim**

<details><summary><b>S7.</b> TensorFlow'da bir ağı eğitmenin 3 adımı nedir?</summary>

1) Modeli tanımla (`Sequential`), 2) loss'u belirleyip `compile` et, 3) `fit` ile eğit. (Bölüm 10)
</details>

<details><summary><b>S8.</b> Gizli katmanlarda neden ReLU tercih edilir? Neden linear olmaz?</summary>

ReLU **daha hızlı hesaplanır** ve sigmoid gibi iki tarafta da düz olmadığından gradient descent **daha hızlıdır**. Linear aktivasyonla tüm ağ tek bir lineer regresyona çöker (lineerin lineeri lineerdir). (Bölüm 11)
</details>

<details><summary><b>S9.</b> Çıkış aktivasyonunu nasıl seçersin: (a) spam mi?, (b) hisse senedi değişimi, (c) ev fiyatı?</summary>

(a) **Sigmoid**, (b) **Linear** (±), (c) **ReLU** (≥0). (Bölüm 11)
</details>

<details><summary><b>S10.</b> $z=[2,1,0.1]$ için softmax çıktısı ve toplamı? Doğru sınıf 1 ise loss?</summary>

$[0.659,\ 0.242,\ 0.099]$, toplam 1. $L=-\ln0.659\approx0.417$. (Bölüm 12, 46-C)
</details>

<details><summary><b>S11.</b> Çok sınıflı ağda <code>from_logits=True</code> kullanırken çıkış katmanı aktivasyonu ne olur? Tahmin için ne yaparsın?</summary>

Çıkış katmanı **linear**. Tahminde logit'leri `tf.nn.softmax` ile olasılığa çevirirsin. Amaç: **sayısal yuvarlama hatalarını** azaltmak. (Bölüm 13)
</details>

<details><summary><b>S12.</b> Multi-class ile multi-label arasındaki fark nedir? Çıkış katmanı nasıl değişir?</summary>

Multi-class: tek etiket, **softmax**. Multi-label: aynı anda birden çok etiket, her çıkışa **ayrı sigmoid**. (Bölüm 14)
</details>

<details><summary><b>S13.</b> Adam gradient descent'ten neyle farklıdır?</summary>

Her parametre için **öğrenme oranını otomatik** ayarlar (tutarlı yönde artırır, zıplarsa azaltır). (Bölüm 15)
</details>

<details><summary><b>S14.</b> $J=\tfrac12(wx+b-y)^2$, $x=-2,y=2,w=2,b=8$ için $\partial J/\partial w$ kaçtır?</summary>

$-4$ (hata $d=2$, $\partial J/\partial w = d\cdot x = 2\cdot(-2)$). (Bölüm 17, 46-D)
</details>

**Hafta 3: ML tavsiyeleri**

<details><summary><b>S15.</b> Neden veriyi train / CV / test olarak üçe böleriz?</summary>

CV seti **model seçimi** için, test seti **dürüst** genelleme ölçümü için. Test setine bakarak seçim yaparsan ona "uydurmuş" olursun. (Bölüm 19)
</details>

<details><summary><b>S16.</b> $J_{train}=10.8\%$, baseline $=10.6\%$, $J_{cv}=14.8\%$. High bias mı, high variance mı?</summary>

**High variance.** Baseline–train farkı 0.2 (küçük), CV–train farkı 4.0 (büyük). (Bölüm 22)
</details>

<details><summary><b>S17.</b> $\lambda$ çok büyükse hangi sorun olur? Çok küçükse?</summary>

Çok büyük: $w\approx0$, düz çizgi → **high bias**. Çok küçük: dalgalı eğri → **high variance**. (Bölüm 21)
</details>

<details><summary><b>S18.</b> High bias durumunda daha fazla eğitim verisi yardımcı olur mu?</summary>

**Hayır.** Çok basit model veriyle iyileşmez; eğriler yüksekte düzleşir. High variance'ta yardımcı olur. (Bölüm 23)
</details>

<details><summary><b>S19.</b> Sinir ağları için bias/variance "reçetesi" nedir?</summary>

Eğitimde iyi değil → **daha büyük ağ**. CV'de iyi değil → **daha fazla veri**. Doğru regularization ile büyük ağ genelde zarar vermez. (Bölüm 25)
</details>

<details><summary><b>S20.</b> Hata analizi nasıl yapılır? Örnek ver.</summary>

CV'de yanlış sınıflandırılanlara **elle bak**, ortak temalara grupla, sayısını bul (örn. 100 hatadan 21'i ilaç spam'i, 3'ü yazım hatası). Büyük kategorilere öncelik ver. (Bölüm 26)
</details>

<details><summary><b>S21.</b> Transfer learning'in adımları nelerdir? Hangi şart gerekir?</summary>

Büyük veride **önceden eğitilmiş** ağı al, çıkış katmanını değiştir, (a) sadece çıkış katmanını ya da (b) tüm parametreleri **ince ayarla**. **Girdi tipi aynı** olmalı. (Bölüm 27)
</details>

<details><summary><b>S22.</b> Hastaların %0.5'i hasta. "Herkese sağlıklı de" algoritmasının doğruluğu nedir? Neden sorun?</summary>

%99.5. Hiçbir şey öğrenmedi ama accuracy yüksek; **recall = 0**. Çarpık veride accuracy yanıltıcıdır. (Bölüm 30)
</details>

<details><summary><b>S23.</b> TP=15, FP=5, FN=10 için precision, recall ve F1?</summary>

$P=0.75,\ R=0.60,\ F1=0.667$. (Bölüm 30, 46-F)
</details>

<details><summary><b>S24.</b> Eşiği 0.5'ten 0.7'ye çıkarırsan precision ve recall'a ne olur?</summary>

Precision **artar**, recall **azalır**. (Bölüm 30)
</details>

<details><summary><b>S25.</b> Neden F1, ortalama yerine kullanılır?</summary>

Harmonik ortalama **düşük olana ağırlık** verir. Algoritma 3 (P=0.02, R=1.0) ortalamada 0.51 ile "en iyi" görünür ama F1=0.0392 ile doğru şekilde en kötü olur. (Bölüm 30)
</details>

**Hafta 4: Karar ağaçları**

<details><summary><b>S26.</b> Karar ağacı öğrenirken hangi iki temel karar verilir?</summary>

1) Her düğümde **hangi özelliğe böleceğiz** (saflığı maksimize et), 2) **Ne zaman duracağız** (saf düğüm, max derinlik, kazanç eşiği, örnek sayısı eşiği). (Bölüm 32)
</details>

<details><summary><b>S27.</b> 3 kedi + 3 köpekten oluşan bir kümenin entropisi? Peki 6 kediden oluşan kümenin?</summary>

Önce **1** (en safsız), sonra **0** (tam saf). (Bölüm 33)
</details>

<details><summary><b>S28.</b> Kulak şekli için bilgi kazancını hesapla (kök: 5 kedi/10; sol 4/5, sağ 1/5).</summary>

$1 - (0.5\cdot0.72+0.5\cdot0.72) = \mathbf{0.28}$. (Bölüm 34, 46-G)
</details>

<details><summary><b>S29.</b> 3 değerli bir özelliği (sivri/oval/sarkık) nasıl kodlarsın? Neden?</summary>

**One-hot**: 3 adet 0/1 özellik. Sahte sıra/mesafe ima etmez, ağaç algoritması ikili özelliklerle aynen çalışır. (Bölüm 35)
</details>

<details><summary><b>S30.</b> Sürekli özellikte (ağırlık) en iyi eşiği nasıl bulursun?</summary>

Örnekleri sırala, **ardışık değerlerin ortalarını** aday eşik al, her biri için bilgi kazancını hesapla, **en yükseğini** seç. (Bölüm 36)
</details>

<details><summary><b>S31.</b> Regresyon ağacı neyi tahmin eder, bölmeyi neye göre seçer?</summary>

Yaprağa düşen örneklerin **ortalamasını**. Bölme ölçütü **varyans azalması**. (Bölüm 37)
</details>

<details><summary><b>S32.</b> Tek karar ağacının zayıflığı nedir? Çözüm?</summary>

Veride **küçük değişikliğe aşırı duyarlı** (tamamen farklı ağaç çıkabilir). Çözüm: **ağaç topluluğu** (çok ağaç, oylama). (Bölüm 38)
</details>

<details><summary><b>S33.</b> Değiştirerek örnekleme (sampling with replacement) nedir? Neden kullanılır?</summary>

Torbadan çek-**geri koy**-tekrar çek. Aynı örnek birden fazla çıkabilir. Her ağaç için **farklı ama benzer** eğitim seti üretir. (Bölüm 38)
</details>

<details><summary><b>S34.</b> Random forest, bagged tree'den farkı nedir?</summary>

Ek olarak **her düğümde rastgele $k=\sqrt{n}$ özellik** değerlendirir → ağaçlar daha **çeşitli**. (Bölüm 39)
</details>

<details><summary><b>S35.</b> XGBoost'un temel fikri nedir?</summary>

**Boosting**: Her yeni ağaç, öncekilerin **yanlış yaptığı örneklere** odaklanır ("bilinçli pratik"). Hızlı, yerleşik regularization'lı, çok başarılı. (Bölüm 40)
</details>

<details><summary><b>S36.</b> Hangi durumda karar ağacı/XGBoost, hangisinde sinir ağı?</summary>

**Tablo (yapılandırılmış) veri** → ağaçlar/XGBoost (hızlı). **Görüntü/ses/metin** → sinir ağı. (Bölüm 41)
</details>

**Kod**

<details><summary><b>S37.</b> Bölüm 45'e bakmadan <code>entropy</code> ve <code>information_gain</code> fonksiyonlarını yaz.</summary>

```python
def entropy(p1):
    if p1 == 0 or p1 == 1: return 0.0
    return -p1*np.log2(p1) - (1-p1)*np.log2(1-p1)

def information_gain(y_node, y_left, y_right):
    p = lambda y: np.mean(y) if len(y) else 0.0
    wl, wr = len(y_left)/len(y_node), len(y_right)/len(y_node)
    return entropy(p(y_node)) - (wl*entropy(p(y_left)) + wr*entropy(p(y_right)))
```
</details>

---

## 51. Tekrar planı

Aralıklı tekrar, aynı bilgiyi yoğun bir günde okumaktan çok daha kalıcıdır. **Bir haftayı bir günde bitirmeye çalışma.**

| Gün | Ne yap? |
|---|---|
| **Gün 1** | **Sadece Hafta 1** (Bölüm 1–9). Her bölümden sonra kapat, kendi cümlelerinle anlat. Bölüm 46-A ve B'yi kâğıtta yap |
| **Gün 2** | **Hafta 2** (Bölüm 10–17). Bölüm 46-C ve D'yi yap |
| **Gün 3** | **Hafta 3'ün ilk yarısı** (Bölüm 18–25): train/cv/test, bias/variance. Bölüm 46-E |
| **Gün 4** | **Hafta 3'ün ikinci yarısı** (Bölüm 26–30): hata analizi, transfer learning, precision/recall. Bölüm 46-F |
| **Gün 5** | **Hafta 4** (Bölüm 31–41). Bölüm 46-G'yi yap |
| **Gün 7** | Bölüm 50'deki soruları **cevaplara bakmadan** çöz. Yanlışları işaretle |
| **Gün 10** | Sadece işaretli yanlışlara dön. Bölüm 45'teki kodu **bakmadan** sıfırdan yaz |
| **Gün 14** | Bölüm 47 (hatalar), 48 (karıştırılanlar) ve 49 (sorun → çözüm) tablolarını gözden geçir |
| **Gün 30** | Formül kartını (Bölüm 43) bakmadan boş kâğıda yaz |

> 💡 **Önemli:** Bir bölümü anlamadan sonrakine geçme. Her haftanın bölümleri bir öncekinin üstüne kurulu. Takıldığında önceki kursun notlarındaki ilgili konuya dön (cost function, gradient descent, logistic regression, regularization).

---

*Bu notlar, "Advanced Learning Algorithms" kursunun (Hafta 1–4) video transkriptlerinden hazırlanmıştır.*
