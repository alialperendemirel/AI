# Sınıflandırma, Lojistik Regresyon, Aşırı Uyum ve Düzenlileştirme – Ders Özeti

> Makine Öğrenmesi – Hafta 3 notları.
> Önceki özetler: [Dereceli Azalma](gradient-descent-ozet.md) · [Çoklu Lineer Regresyon ve Vektörleştirme](coklu-lineer-regresyon-ozet.md) · [Özellik Ölçekleme, Öğrenme Oranı, Polinom Regresyon](pratik-ipuclari-ozet.md)
> Terim karşılıkları: **ikili sınıflandırma = binary classification**, **karar sınırı = decision boundary**, **kayıp = loss**, **maliyet = cost**, **aşırı uyum = overfitting**, **yetersiz uyum = underfitting**, **yüksek önyargı = high bias**, **yüksek varyans = high variance**, **düzenlileştirme = regularization**, **genelleme = generalization**.

---

## İçindekiler

1. [Sınıflandırma ve neden lineer regresyon olmaz?](#1-sınıflandırma-ve-neden-lineer-regresyon-olmaz)
2. [Lojistik regresyon ve sigmoid fonksiyonu](#2-lojistik-regresyon-ve-sigmoid-fonksiyonu)
3. [Karar sınırı (decision boundary)](#3-karar-sınırı-decision-boundary)
4. [Lojistik regresyon için maliyet fonksiyonu](#4-lojistik-regresyon-için-maliyet-fonksiyonu)
5. [Basitleştirilmiş kayıp ve maliyet](#5-basitleştirilmiş-kayıp-ve-maliyet)
6. [Lojistik regresyonda dereceli azalma](#6-lojistik-regresyonda-dereceli-azalma)
7. [Aşırı uyum ve yetersiz uyum](#7-aşırı-uyum-ve-yetersiz-uyum)
8. [Aşırı uyumu azaltmanın 3 yolu](#8-aşırı-uyumu-azaltmanın-3-yolu)
9. [Düzenlileştirmeli maliyet fonksiyonu](#9-düzenlileştirmeli-maliyet-fonksiyonu)
10. [Düzenlileştirmeli lineer regresyon](#10-düzenlileştirmeli-lineer-regresyon)
11. [Hızlı tekrar / hatırlatma kartı](#11-hızlı-tekrar--hatırlatma-kartı)
12. [Örnek Python kodu](#12-örnek-python-kodu)

---

## 1. Sınıflandırma ve neden lineer regresyon olmaz?

**Sınıflandırma:** Çıktı `y`, sonsuz sayıda olası sayıdan biri değil, **küçük bir kümeden** (birkaç olası değerden) biridir.

### İkili sınıflandırma (binary classification)

Sadece **iki** olası çıktı vardır. Örnekler:

- E-posta spam mi? (evet / hayır)
- Çevrimiçi finansal işlem hileli mi?
- Tümör kötü huylu (malign) mu?

### Gösterim

| Kullanım | Negatif sınıf | Pozitif sınıf |
|---|---|---|
| Sözel | hayır / yanlış | evet / doğru |
| Sayısal (en yaygın) | `y = 0` | `y = 1` |

- "Negatif" ve "pozitif" **kötü/iyi demek değildir**; sadece **yokluk (0)** ve **varlık (1)** anlamındadır (örn. spam olmayan e-posta = negatif örnek, spam e-posta = pozitif örnek).
- Hangisine 0, hangisine 1 dediğin biraz **keyfidir**.
- "Sınıf" ve "kategori" aynı anlamda kullanılır.

### Lineer regresyon neden iyi değil?

Lineer regresyon veriye düz çizgi uydurur, 0.5 eşiği koyarsan (`< 0.5 → 0`, `≥ 0.5 → 1`) **bazen** çalışır. **Ama:**

- Çıktısı 0–1 ile sınırlı değildir (0'dan küçük veya 1'den büyük olabilir).
- Veri setine **sağ tarafta çok büyük bir tümör örneği** eklediğinde en iyi uyan çizgi **kayar** ve karar eşiği de sağa kayar. Oysa bu örnek zaten "kötü huylu" olduğundan **hiçbir şeyi değiştirmemeliydi**. Sonuç: daha kötü bir sınıflandırıcı. ❌

> Bu yüzden sınıflandırmada **lojistik regresyon** kullanılır.
> ⚠️ İsimde "regresyon" geçse de lojistik regresyon **sınıflandırma** algoritmasıdır (tarihsel bir isimlendirme). Çıktı etiketi 0 veya 1 olan ikili sınıflandırma problemleri için kullanılır.

---

## 2. Lojistik regresyon ve sigmoid fonksiyonu

Lojistik regresyon veriye **S şeklinde bir eğri** uydurur; çıktı her zaman **0 ile 1 arasındadır**. Bu, belki de dünyada en çok kullanılan sınıflandırma algoritmasıdır (uzun süre internet reklamcılığı da bunun küçük bir varyasyonuyla yönetildi).

### Sigmoid (lojistik) fonksiyonu

```
g(z) = 1 / (1 + e⁻ᶻ)          (e ≈ 2.718)
```

| `z` | `g(z)` |
|---|---|
| Çok büyük pozitif (örn. 100) | ≈ **1** (e⁻¹⁰⁰ ≈ 0 olduğundan) |
| **0** | **0.5** (e⁰ = 1 → 1/(1+1)) |
| Çok büyük negatif | ≈ **0** |

Grafikte: 0'a yakın başlar, `z = 0`'da dikey ekseni 0.5'te keser, 1'e doğru yükselir.

### Model: 2 adım

```
Adım 1:  z = w · x + b            (lineer regresyon gibi)
Adım 2:  f(x) = g(z) = 1 / (1 + e^−(w·x + b))
```

### Çıktının yorumu

`f(x)` = verilen `x` için **`y = 1` olma olasılığı**.

- Tümör örneği: `f(x) = 0.7` → model "tümörün kötü huylu olma ihtimali **%70**" diyor.
- `y` ya 0 ya 1 olduğundan: `P(y=0) = 1 − 0.7 = 0.3` (**%30**). İki olasılığın toplamı her zaman 1.
- Makalelerde şöyle yazılabilir: `f(x) = P(y = 1 | x; w, b)` ("`w`, `b` parametreleri verilmişken `x` için `y=1` olasılığı"). Bu gösterimi ezberlemek gerekmiyor.

---

## 3. Karar sınırı (decision boundary)

Olasılığı 0/1 tahmine çevirmek için bir **eşik** kullanılır (en yaygını **0.5**):

```
f(x) ≥ 0.5   →   ŷ = 1
f(x) < 0.5   →   ŷ = 0
```

### Adım adım mantık

```
f(x) ≥ 0.5  ⇔  g(z) ≥ 0.5  ⇔  z ≥ 0  ⇔  w·x + b ≥ 0
```

| Koşul | Tahmin |
|---|---|
| `w·x + b ≥ 0` | `ŷ = 1` |
| `w·x + b < 0` | `ŷ = 0` |

**Karar sınırı** = `w·x + b = 0` olan yer (modelin 0 mı 1 mi diye "nötr" olduğu çizgi/eğri).

### Örnek 1: Doğrusal karar sınırı

İki özellik `x₁`, `x₂`; `w₁ = 1`, `w₂ = 1`, `b = −3`:

```
z = x₁ + x₂ − 3 = 0   →   x₁ + x₂ = 3   (düz çizgi)
```

Çizginin bir tarafı `ŷ = 1`, diğer tarafı `ŷ = 0`.

### Örnek 2: Doğrusal olmayan karar sınırı (polinom özelliklerle)

`z = w₁x₁² + w₂x₂² + b`, `w₁ = w₂ = 1`, `b = −1`:

```
x₁² + x₂² = 1   →   birim çember
```

- Çemberin **dışı** (`x₁² + x₂² ≥ 1`) → `ŷ = 1`
- Çemberin **içi** → `ŷ = 0`

Daha yüksek dereceli polinom terimleri (`x₁x₂`, `x₁²`, ...) ile elips ve çok daha karmaşık sınırlar elde edilebilir.

> **Önemli:** Sadece `x₁, x₂, x₃...` gibi **ham özellikler** kullanırsan karar sınırı **her zaman düz çizgidir**. Eğrisel sınır için polinom özellikler eklemek gerekir.

---

## 4. Lojistik regresyon için maliyet fonksiyonu

### Kare hata maliyeti neden uygun değil?

- Lineer regresyonda kare hata maliyeti **konvekstir** (kase şekli) → tek global minimum.
- Aynı maliyeti `f(x) = sigmoid(w·x + b)` ile kullanırsan maliyet yüzeyi **konveks olmaz** ("kıpır kıpır"), **birçok yerel minimum** çıkar → dereceli azalma takılabilir. ❌

### Çözüm: Yeni bir kayıp fonksiyonu (loss)

- **Kayıp `L`:** **Tek bir** eğitim örneğinde ne kadar iyi/kötü olduğunu ölçer.
- **Maliyet `J`:** Tüm eğitim setindeki kayıpların **ortalamasıdır**.

```
J(w,b) = (1/m) · Σ L( f(x⁽ⁱ⁾), y⁽ⁱ⁾ )       (i = 1..m)
```

Lojistik regresyon için kayıp:

```
y = 1 ise:   L = −log( f(x) )
y = 0 ise:   L = −log( 1 − f(x) )
```

### Sezgi

**`y = 1` durumu** (`−log f`):

| Model tahmini `f` | Kayıp |
|---|---|
| 1'e yakın | ≈ **0** (doğru cevaba yakın) |
| 0.5 | orta |
| 0.1 (tümörün kötü huylu olma ihtimali %10 diyor ama aslında kötü huylu) | **çok yüksek** |
| 0'a yaklaşır | **sonsuza** gider |

**`y = 0` durumu** (`−log(1 − f)`):

| Model tahmini `f` | Kayıp |
|---|---|
| 0'a yakın | ≈ **0** |
| Büyüdükçe | kayıp artar |
| 1'e yaklaşır (örn. %99.9 kötü huylu der ama değildir) | **sonsuza** gider |

> **Mantık:** Tahmin gerçek etiketten ne kadar **uzaksa** ceza o kadar büyük. Özellikle "çok emin ama yanlış" tahmin **çok sert** cezalandırılır.

✅ Bu kayıp seçimiyle toplam maliyet **konveks** olur → dereceli azalma global minimuma güvenle yakınsar (konveks olduğunun ispatı kapsam dışı).

---

## 5. Basitleştirilmiş kayıp ve maliyet

`y` sadece 0 veya 1 olabildiğinden iki durum **tek formülde** yazılabilir:

```
L( f(x), y ) = −y · log( f(x) ) − (1 − y) · log( 1 − f(x) )
```

**Kontrol:**

- `y = 1` → ikinci terim (1−y = 0) kaybolur → `−log f(x)` ✅
- `y = 0` → ilk terim kaybolur → `−log(1 − f(x))` ✅

### Lojistik regresyonun maliyet fonksiyonu

```
J(w,b) = −(1/m) · Σ [ y⁽ⁱ⁾ · log( f(x⁽ⁱ⁾) ) + (1 − y⁽ⁱ⁾) · log( 1 − f(x⁽ⁱ⁾) ) ]
```

(Toplam i = 1'den m'ye.)

- Lojistik regresyonu eğitmek için neredeyse herkesin kullandığı maliyet budur.
- İstatistikteki **maksimum olabilirlik tahmini (maximum likelihood estimation)** ilkesinden türetilmiştir (detayı bu kurs için gerekmiyor).
- Konveks olması güzel bir özelliğidir.

---

## 6. Lojistik regresyonda dereceli azalma

Hedef: `J(w,b)`'yi minimize eden `w`, `b`'yi bulmak.

```
wⱼ = wⱼ − α · (1/m) · Σ ( f(x⁽ⁱ⁾) − y⁽ⁱ⁾ ) · xⱼ⁽ⁱ⁾        (j = 1..n)
b  = b  − α · (1/m) · Σ ( f(x⁽ⁱ⁾) − y⁽ⁱ⁾ )
```

**Eş zamanlı güncelleme** yapılır (önce tüm sağ tarafları hesapla, sonra birlikte ata).

### 🤔 "Bu, lineer regresyonla aynı değil mi?"

Denklemler **aynı görünür** ama algoritmalar **farklıdır**, çünkü **`f(x)` tanımı farklıdır**:

| | `f(x)` |
|---|---|
| Lineer regresyon | `w · x + b` |
| Lojistik regresyon | `sigmoid( w · x + b )` |

### Eski araçlar burada da geçerli

- **Yakınsama kontrolü:** Öğrenme eğrisi (cost - iterasyon) çizilebilir.
- **Vektörleştirme:** Daha hızlı çalışması için kullanılabilir.
- **Özellik ölçekleme:** Özellikleri benzer aralığa (örn. −1 ile +1) getirmek lojistik regresyonda da dereceli azalmayı hızlandırır.
- **scikit-learn:** Lojistik regresyonu birkaç satırda eğitmek için kullanılabilir (isteğe bağlı lab'da gösteriliyor).

---

## 7. Aşırı uyum ve yetersiz uyum

Bazen algoritma, **aşırı uyum (overfitting)** nedeniyle kötü performans gösterir. Zıt problem: **yetersiz uyum (underfitting)**.

**Genelleme (generalization):** Modelin **daha önce hiç görmediği yeni örneklerde** de iyi tahmin yapması. Amaç budur.

### Regresyon örneği: ev fiyatı

| Model | Ne olur? | Terim |
|---|---|---|
| Düz çizgi (`x`) | Veriye uymaz (fiyat büyüklükle düzleşiyor ama çizgi bunu yakalayamıyor) | **Yetersiz uyum** = **yüksek önyargı (high bias)** |
| İkinci derece (`x`, `x²`) | Veriye oldukça iyi uyar, yeni evlerde de iyi çalışması beklenir | ✅ **Tam doğru** ("Goldilocks" 🐻) |
| Dördüncü derece (`x`, `x²`, `x³`, `x⁴`) | 5 eğitim noktasının hepsinden **tam** geçer (cost = 0) ama eğri çok kıpır kıpır; bazı yerlerde daha büyük evi daha ucuz tahmin eder | **Aşırı uyum** = **yüksek varyans (high variance)** |

### Terimler

- **Yüksek önyargı:** Algoritma verinin yapısını yakalayamıyor (örn. "veri doğrusaldır" gibi çok güçlü bir varsayım).
  > Not: ML'de "bias" kelimesinin iki anlamı var: (1) burada kullanılan teknik anlamı, (2) cinsiyet/etnik köken gibi konularda **adil olmayan önyargı**. İkincisi için de modellerin kontrol edilmesi çok önemlidir ama bu videonun konusu değil.
- **Yüksek varyans:** Algoritma her eğitim örneğine uymak için aşırı uğraşıyor. Eğitim seti **biraz** değişse (bir evin fiyatı biraz farklı olsa) öğrenilen fonksiyon **tamamen** değişebilir. İki mühendis biraz farklı verilerle çok farklı modeller elde eder.
- "Aşırı uyum" ≈ "yüksek varyans", "yetersiz uyum" ≈ "yüksek önyargı" (neredeyse eş anlamlı kullanılır).

### Sınıflandırmada da aynısı

`x₁` = tümör boyutu, `x₂` = yaş:

| Model | Karar sınırı | Durum |
|---|---|---|
| Sadece `x₁`, `x₂` | Düz çizgi | Yetersiz uyum (yüksek önyargı) |
| `x₁²`, `x₂²` vb. eklenmiş | Elips benzeri | ✅ Doğru (tüm örnekleri mükemmel ayırmasa da genelleşir) |
| Çok yüksek dereceli polinom | Aşırı kıvrımlı, her noktaya uyan sınır | Aşırı uyum (yüksek varyans) |

**Hedef:** Ne yüksek önyargı ne yüksek varyansı olan bir model bulmak.

---

## 8. Aşırı uyumu azaltmanın 3 yolu

| # | Yöntem | Açıklama | Dezavantaj |
|---|---|---|---|
| 1 | **Daha fazla eğitim verisi topla** | Daha büyük veri setiyle algoritma daha az kıpır kıpır bir fonksiyon öğrenir. Yüksek dereceli polinom kullansan bile yeterli veri varsa iyi olur. **Bir numaralı araç.** | Her zaman mümkün olmaz (satılan ev sayısı sınırlı olabilir). |
| 2 | **Daha az özellik kullan (özellik seçimi)** | Örn. 100 özellik yerine en faydalı olanları seç (boyut, yatak odası sayısı, yaş). Çok özellik + az veri = aşırı uyum riski. | Seçilmeyen özelliklerdeki **bilgi çöpe gider**. (Kurs 2'de otomatik özellik seçimi algoritmaları görülecek.) |
| 3 | **Düzenlileştirme (regularization)** | Özellikleri atmak yerine **parametreleri `w₁...wₙ` küçültmeyi** teşvik eder. Tüm özellikler kalır ama hiçbirinin etkisi aşırı büyümez. | — |

### Düzenlileştirmenin fikri

- Aşırı uyumlu modelde parametreler genelde **çok büyük** olur.
- Bir parametreyi tam **0** yapmak = o özelliği tamamen atmak (sert).
- Düzenlileştirme bunu **nazikçe** yapar: parametreleri **küçültmeye** teşvik eder, sıfırlamaya zorlamaz.
- Geleneksel olarak **sadece `w₁...wₙ`** düzenlenir; **`b` genelde düzenlenmez** (düzenlense de pratikte çok az fark yaratır).
- Yazarın notu: "Ben her zaman düzenlileştirme kullanırım." Sinir ağları dahil birçok modelde çok kullanışlıdır.

---

## 9. Düzenlileştirmeli maliyet fonksiyonu

### Sezgi

Dördüncü derece polinomda `w₃` ve `w₄` küçük olursa (≈ 0) model neredeyse ikinci derece fonksiyona döner. Bunu zorlamak için maliyete ceza ekleyebiliriz, örn.:

```
J = (orijinal maliyet) + 1000·w₃² + 1000·w₄²
```

Bu maliyeti küçültmenin tek yolu `w₃`, `w₄`'ü **0'a yakın** tutmaktır. (1000 sadece "büyük bir sayı" örneği.)

**Genel durumda** (100 özellik varsa hangisinin önemli olduğunu bilemeyiz), **tüm `wⱼ`'leri** biraz cezalandırırız.

### Düzenlileştirmeli maliyet (lineer regresyon için)

```
J(w,b) = (1/2m) · Σ ( f(x⁽ⁱ⁾) − y⁽ⁱ⁾ )²   +   (λ/2m) · Σⱼ wⱼ²
          └──── kare hata (ortalama) ────┘     └── düzenlileştirme terimi ──┘
                                                  (j = 1..n)
```

- **`λ` (lambda):** **Düzenlileştirme parametresi**. `α` gibi **sen seçersin**.
- Her iki terim de `1/2m` ile ölçeklendiği için `λ`'yı seçmek kolaylaşır. Eğitim seti büyüse (`m` artsa) bile aynı `λ` çalışmaya devam etme olasılığı yüksektir.

### İki hedef arasındaki denge

| Terim | Ne yapar? |
|---|---|
| 1. terim (kare hata) | Modeli **eğitim verisine iyi uymaya** iter |
| 2. terim (düzenlileştirme) | **`wⱼ`'leri küçük** tutmaya iter → aşırı uyumu azaltır |

`λ` bu iki hedef arasındaki **dengeyi** belirler.

### `λ` değerinin etkisi

| `λ` | Ne olur? | Sonuç |
|---|---|---|
| **0** | Düzenlileştirme terimi yok olur | Kıpır kıpır, karmaşık eğri → **aşırı uyum** |
| **Çok büyük** (örn. 10¹⁰) | Tüm `wⱼ ≈ 0` olur → `f(x) ≈ b` | **Yatay düz çizgi** → **yetersiz uyum** |
| **Orta, "tam doğru"** | Denge sağlanır | Tüm özellikler korunur ama **düzgün, makul** bir eğri ✅ |

> İyi `λ` değerlerini seçme yöntemleri ilerleyen derslerde (model seçimi) gelecek.

---

## 10. Düzenlileştirmeli lineer regresyon

### Dereceli azalma

Güncelleme formülü **aynı yapıdadır**, sadece `J` değişti. `wⱼ`'nin türevine **ek bir terim** gelir; `b`'nin türevi **aynı kalır** (çünkü `b` düzenlenmiyor):

```
wⱼ = wⱼ − α · [ (1/m) · Σ ( f(x⁽ⁱ⁾) − y⁽ⁱ⁾ ) · xⱼ⁽ⁱ⁾  +  (λ/m) · wⱼ ]      (j = 1..n)

b  = b  − α · (1/m) · Σ ( f(x⁽ⁱ⁾) − y⁽ⁱ⁾ )
```

Yine **eş zamanlı** güncelle.

### Güncellemenin ilginç bir yeniden yazımı (sezgi)

`wⱼ` terimlerini topla:

```
wⱼ = wⱼ · ( 1 − α·λ/m )  −  α · (1/m) · Σ ( f(x⁽ⁱ⁾) − y⁽ⁱ⁾ ) · xⱼ⁽ⁱ⁾
      └─ küçültme çarpanı ─┘   └──── düzenlileştirmesiz olağan güncelleme ────┘
```

- İkinci kısım, hafta 2'de gördüğümüz **olağan** dereceli azalma adımıdır.
- Birinci kısım: her iterasyonda `wⱼ`, **1'den biraz küçük** bir sayıyla çarpılır → **`wⱼ` biraz küçülür.**

**Sayısal örnek:** `α = 0.01`, `λ = 1`, `m = 50`

```
α·λ/m = 0.01 · 1 / 50 = 0.0002   →   çarpan = 1 − 0.0002 = 0.9998
```

Yani her iterasyonda `wⱼ`, olağan güncellemeden önce **0.9998** ile çarpılır. Düzenlileştirme parametreleri böyle "aşağı çeker".

### Türevin hesabı (isteğe bağlı kısım)

- Videonun geri kalanı (türev hesabı) **tamamen isteğe bağlı**. Lab ve quizler için gerekmiyor.
- Kısaca: kare hata kısmının türevi, `2`'ler sadeleşerek `(1/m) Σ (f − y) xⱼ` verir. Düzenlileştirme terimi `(λ/2m) Σ wⱼ²`'nin türevi ise `(λ/m) wⱼ` olur (toplam sembolü kalkar, çünkü türev sadece ilgili `wⱼ` için alınır).

> Çok özellik + az veri durumunda düzenlileştirmeli lineer regresyon aşırı uyumu ciddi biçimde azaltır.

> 📌 **Sıradaki konu:** Aynı düzenlileştirme fikrinin **lojistik regresyona** uygulanması (bu özette yer alan transkriptlerde yok).

---

## 11. Hızlı tekrar / hatırlatma kartı

- 🔀 **Sınıflandırma:** `y` küçük bir kümeden bir değer (ikili: 0 veya 1). Lineer regresyon uygun değil.
- 📈 **Sigmoid:** `g(z) = 1/(1 + e⁻ᶻ)`, çıktı 0–1, `g(0) = 0.5`.
- 🧠 **Lojistik regresyon:** `f(x) = g(w·x + b)` = `y=1` olma **olasılığı**.
- ✂️ **Karar sınırı:** `w·x + b = 0`. `w·x + b ≥ 0 → ŷ = 1`. Ham özelliklerle düz çizgi, polinom özelliklerle eğri.
- 💥 **Kare hata + sigmoid** = konveks olmayan maliyet (yerel minimumlar). Bunun yerine **log kayıp** kullan.
- 📐 **Kayıp:** `−y·log(f) − (1−y)·log(1−f)`. Maliyet = ortalama kayıp.
- 🔽 **Dereceli azalma (lojistik):** Formül lineer regresyonla aynı görünür, **`f(x)` farklı** (sigmoid). Eş zamanlı güncelle, özellik ölçekle.
- 🎯 **Aşırı uyum (yüksek varyans):** Eğitime mükemmel uyar, yeni veride kötü. **Yetersiz uyum (yüksek önyargı):** Eğitim verisine bile uymaz.
- 🛠️ **Çözümler:** (1) Daha fazla veri, (2) daha az özellik, (3) **düzenlileştirme**.
- ⚖️ **Düzenlileştirme:** `J = kare hata + (λ/2m)·Σwⱼ²`. `b` düzenlenmez.
  - `λ = 0` → aşırı uyum
  - `λ` çok büyük → yetersiz uyum (düz çizgi)
- 🔄 **Düzenlileştirmeli lineer regresyon:** `wⱼ`'ye `+ (λ/m)·wⱼ` eklenir; her adımda `wⱼ` biraz küçülür (`× (1 − αλ/m)`).

---

## 12. Örnek Python kodu

Sigmoid, lojistik maliyet, dereceli azalma ve düzenlileştirmeli lineer regresyon gradyanı:

```python
import numpy as np

# ---------- Lojistik regresyon ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    f = sigmoid(X @ w + b)
    return -(1 / m) * np.sum(y * np.log(f) + (1 - y) * np.log(1 - f))

def compute_gradient_logistic(X, y, w, b):
    m = X.shape[0]
    f = sigmoid(X @ w + b)          # lineer regresyondan TEK fark: sigmoid
    err = f - y
    dj_dw = (1 / m) * (X.T @ err)
    dj_db = (1 / m) * np.sum(err)
    return dj_dw, dj_db

def gradient_descent_logistic(X, y, w, b, alpha, num_iters):
    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
        w = w - alpha * dj_dw       # eş zamanlı güncelleme
        b = b - alpha * dj_db
    return w, b

def predict(X, w, b, threshold=0.5):
    return (sigmoid(X @ w + b) >= threshold).astype(int)

# Küçük örnek veri
X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

w, b = gradient_descent_logistic(X_train, y_train, np.zeros(2), 0.0, alpha=0.1, num_iters=10000)
print("w =", w, " b =", b)
print("Tahminler:", predict(X_train, w, b))

# ---------- Düzenlileştirmeli lineer regresyon ----------
def compute_cost_linear_reg(X, y, w, b, lambda_):
    m = X.shape[0]
    err = X @ w + b - y
    cost = (1 / (2 * m)) * np.sum(err ** 2)
    reg = (lambda_ / (2 * m)) * np.sum(w ** 2)      # b düzenlenmez
    return cost + reg

def compute_gradient_linear_reg(X, y, w, b, lambda_):
    m = X.shape[0]
    err = X @ w + b - y
    dj_dw = (1 / m) * (X.T @ err) + (lambda_ / m) * w   # ek terim: (λ/m)·w
    dj_db = (1 / m) * np.sum(err)                        # b için değişmez
    return dj_dw, dj_db
```

> Notlar:
> - Kod, ders içeriğini örneklemek için tarafımdan yazıldı, dersten alınmadı. Veri ve hiperparametreler örnektir.
> - `compute_cost_logistic` içinde `f` tam 0 veya 1 olursa `log(0)` hatası çıkabilir; gerçek uygulamalarda buna karşı küçük bir güvenlik payı (epsilon) eklenir. Derste geçmeyen pratik bir ayrıntıdır.

---

## ✏️ Not: Transkriptteki küçük hatalar

Ders metinleri (otomatik çeviri/transkript kaynaklı) bazı yerlerde bozuktu. Özette bağlamdan düzelttim:

- "**Bir yarıyı toplamın içine koymak**" ve "**kaybın 1.5'ine eşit**" ifadeleri aslında **½·(f(x) − y)²** (kare hata kaybı, ½ çarpanı) demek.
- "**Lojistik regresyon** ... **iyi huylu olma olasılığı**" gibi geçen yerler bağlama göre **kötü huylu (y=1)** olma olasılığıdır; 0.7 → %70 kötü huylu.
- "**Doğrusal regresyon ... albümü**" aslında "algoritmayı" (zaten bildiğin lineer regresyon algoritması).
- "**Lambda m çarpı w_j**" ifadesi **(λ/m)·wⱼ** demektir.
- "**Yürütme özellikleri**" ifadesi aslında **x₃ ve x₄ özellikleri**.
- "**Kıpır kıpır**" = İngilizce "wiggly", yani çok dalgalı/eğri-büğrü.
