# Dereceli Azalmayı Daha İyi Çalıştırmak: Özellik Ölçekleme, Öğrenme Oranı, Özellik Mühendisliği ve Polinom Regresyon – Ders Özeti

> Makine Öğrenmesi – Hafta 2 (devam) notları.
> Önceki özetler: [Dereceli Azalma](gradient-descent-ozet.md) · [Çoklu Lineer Regresyon ve Vektörleştirme](coklu-lineer-regresyon-ozet.md)
> Terim karşılıkları: **özellik ölçekleme = feature scaling**, **yakınsama = convergence**, **öğrenme eğrisi = learning curve**, **özellik mühendisliği = feature engineering**, **polinom regresyon = polynomial regression**.

---

## İçindekiler

1. [Özellik ölçekleme: neden gerekli?](#1-özellik-ölçekleme-neden-gerekli)
2. [Özellik ölçekleme yöntemleri](#2-özellik-ölçekleme-yöntemleri)
3. [Dereceli azalma çalışıyor mu? (yakınsama kontrolü)](#3-dereceli-azalma-çalışıyor-mu-yakınsama-kontrolü)
4. [Doğru öğrenme oranını (α) seçmek](#4-doğru-öğrenme-oranını-α-seçmek)
5. [Özellik mühendisliği](#5-özellik-mühendisliği)
6. [Polinom regresyon (eğri uydurma)](#6-polinom-regresyon-eğri-uydurma)
7. [Scikit-learn ve haftanın sonu](#7-scikit-learn-ve-haftanın-sonu)
8. [Hızlı tekrar / hatırlatma kartı](#8-hızlı-tekrar--hatırlatma-kartı)
9. [Örnek Python kodu](#9-örnek-python-kodu)

---

## 1. Özellik ölçekleme: neden gerekli?

**Özellik ölçekleme**, farklı büyüklükteki özellikleri **birbirine yakın aralıklara** getirerek dereceli azalmanın **çok daha hızlı** çalışmasını sağlar.

### Özelliğin büyüklüğü ve parametresinin büyüklüğü ters orantılıdır

Örnek: `x₁` = evin boyutu (300–2000 feet²), `x₂` = oda sayısı (0–5). Gerçek örnek: 2000 feet², 5 oda, fiyat **500 bin $**.

| Deneme | w₁ | w₂ | b | Tahmin (bin $) | Sonuç |
|---|---|---|---|---|---|
| A | 50 | 0.1 | 50 | 50·2000 + 0.1·5 + 50 ≈ **100 milyon $** | ❌ Çok yanlış |
| B | 0.1 | 50 | 50 | 0.1·2000 + 50·5 + 50 = 200 + 250 + 50 = **500 bin $** | ✅ Doğru |

**Çıkarım:**

- Değer aralığı **büyük** olan özellik (boyut) → iyi modelde parametresi **küçük** olur (örn. 0.1).
- Değer aralığı **küçük** olan özellik (oda sayısı) → parametresi **büyük** olur (örn. 50).

### Dereceli azalmaya etkisi

- `w₁` küçük bir değişse bile (çünkü çok büyük bir sayıyla çarpılıyor) tahmin ve **cost J çok değişir**.
- `w₂`'nin cost'u fark edilir şekilde değiştirmesi için **büyük** değişiklik gerekir.
- Sonuç: cost fonksiyonunun kontür grafiği **uzun ve ince oval (elips)** olur.
- Dereceli azalma bu ince vadide **sağa sola zıplayarak** (bouncing) global minimuma uzun sürede ulaşır. 🐌

### Ölçekleme sonrası

- `x₁` ve `x₂` benzer aralıklara gelince (örn. ikisi de 0–1) kontürler **daireye yakın** olur.
- Dereceli azalma minimuma **çok daha doğrudan bir yol** bulur. 🚀

> **Özet:** Özellikler çok farklı aralıklara sahipse dereceli azalma yavaşlar. Hepsini benzer aralığa getirmek hızı ciddi şekilde artırır.

---

## 2. Özellik ölçekleme yöntemleri

Örnek: `x₁` ∈ [300, 2000], ortalama `μ₁ = 600`, standart sapma `σ₁ = 450`; `x₂` ∈ [0, 5], `μ₂ = 2.3`, `σ₂ = 1.4`.

### Yöntem 1: Maksimuma bölmek

```
x₁_ölçekli = x₁ / 2000        → aralık: 0.15 ... 1
x₂_ölçekli = x₂ / 5           → aralık: 0 ... 1
```

### Yöntem 2: Ortalama normalizasyonu (mean normalization)

Değerleri **sıfır etrafında** ortalar (hem negatif hem pozitif değerler çıkar).

```
x₁_ölçekli = (x₁ − μ₁) / (max − min)  = (x₁ − 600) / (2000 − 300)   → −0.18 ... 0.82
x₂_ölçekli = (x₂ − μ₂) / (max − min)  = (x₂ − 2.3) / (5 − 0)        → −0.46 ... 0.54
```

(`μ` = eğitim setindeki ortalama.)

### Yöntem 3: Z-skor normalizasyonu (Z-score normalization)

Ortalama `μ` ve **standart sapma `σ`** (sigma) kullanılır.

```
x₁_ölçekli = (x₁ − μ₁) / σ₁  = (x₁ − 600) / 450   → −0.67 ... 3.1
x₂_ölçekli = (x₂ − μ₂) / σ₂  = (x₂ − 2.3) / 1.4   → −1.6  ... 1.9
```

> Standart sapmayı bilmiyorsan sorun değil, bu kurs için bilmek gerekmiyor; formülü uygulaman yeterli.

### Ne kadar ölçekleme yeterli? (pratik kural)

Her özelliğin yaklaşık **−1 ile +1** arasında olmasını hedefle. Bu sınırlar **esnektir**:

| Özelliğin aralığı | Ölçekleme gerekli mi? |
|---|---|
| −3 ... +3 veya −0.3 ... +0.3 | Gerekmez, sorun yok |
| 0 ... 3 | Gerekmez (istersen yapabilirsin) |
| −2 ... +0.5 | Gerekmez (zararı da olmaz) |
| **−100 ... +100** | ✅ **Ölçekle** (çok büyük) |
| **−0.001 ... +0.001** | ✅ **Ölçekle** (çok küçük) |
| **98.6 ... 105** (örn. hasta vücut sıcaklığı °F) | ✅ **Ölçekle** (değerler ~100, diğerlerine göre büyük, dereceli azalmayı yavaşlatır) |

> 💡 **Ders hocasının tavsiyesi:** Ölçeklemenin neredeyse hiçbir zaman zararı yoktur. **Emin değilsen yap.**

---

## 3. Dereceli azalma çalışıyor mu? (yakınsama kontrolü)

### Öğrenme eğrisi (learning curve)

Her **iterasyonda** (= `w` ve `b`'nin her eş zamanlı güncellenmesinden sonra) eğitim setindeki **cost J**'yi hesapla ve çiz.

- **Yatay eksen:** İterasyon sayısı (`w` veya `b` **değil**!)
- **Dikey eksen:** Cost `J`

> Önceki grafiklerde yatay eksen `w` veya `b` idi. Bu grafik farklı: buradaki eksen **iterasyon sayısı**.

Örnek okuma: 100. iterasyondaki nokta = 100 güncelleme sonrası elde edilen `w`, `b` için hesaplanan `J`.

### Grafikten ne anlarız?

| Gözlem | Anlamı |
|---|---|
| `J` **her iterasyonda azalıyor** | ✅ Dereceli azalma düzgün çalışıyor |
| `J` **bir iterasyonda bile artıyor** | ❌ `α` çok büyük **veya kodda hata (bug)** var |
| Eğri **düzleşiyor** (örn. 300–400. iterasyonda) | ✅ Yaklaşık **yakınsadı** (converged) |

Yakınsama için gereken iterasyon sayısı uygulamaya göre çok değişir (30, 1000 veya 100.000 olabilir). Önceden tahmin etmek zordur; bu yüzden eğriye bakılır.

### Otomatik yakınsama testi (alternatif)

Küçük bir sayı `ε` (epsilon) seç, örn. `0.001`.

```
Eğer bir iterasyonda J, ε'dan daha az azalırsa → yakınsadı say.
```

> ⚠️ Doğru `ε` eşiğini seçmek **zordur**. Hoca otomatik teste güvenmek yerine **grafiğe bakmayı** tercih ediyor; çünkü grafik dereceli azalma yanlış çalışıyorsa **erkenden uyarı** verir.

---

## 4. Doğru öğrenme oranını (α) seçmek

Hatırlatma: `α` çok küçükse algoritma **çok yavaş**, çok büyükse **yakınsamayabilir**.

### Cost grafiğine göre teşhis

| Grafikte ne görüyorsun? | Olası sebep | Çözüm |
|---|---|---|
| Cost bazen **artıyor, bazen azalıyor** (zıplıyor) | `α` çok büyük (minimumu aşıyor) **veya bug** | `α`'yı küçült |
| Cost **sürekli artıyor** | `α` çok büyük **veya bug** (örn. işaret hatası) | `α`'yı küçült, kodu kontrol et |

**Aşma (overshoot) mantığı:** `α` çok büyükse her güncelleme minimumun öbür tarafına atlar, cost düşmek yerine artabilir.

### Klasik bug: yanlış işaret

```python
w1 = w1 + alpha * dj_dw   # ❌ YANLIŞ: cost'u minimumdan uzaklaştırır
w1 = w1 - alpha * dj_dw   # ✅ DOĞRU: eksi işareti
```

### Hata ayıklama (debugging) ipucu 🔧

Dereceli azalma doğru uygulanmışsa, **yeterince küçük bir `α` ile cost HER iterasyonda azalmalıdır.**

- `α`'yı **çok küçük** bir sayıya ayarla ve cost'a bak.
- Hâlâ bazen **artıyorsa** → **kodda hata vardır.**
- ⚠️ Çok küçük `α` sadece **debug içindir**. Gerçek eğitimde verimsizdir (çok iterasyon gerekir).

### İyi `α` nasıl seçilir?

1. Bir aralıktaki değerleri dene, örn: `0.001 → 0.003 → 0.01 → 0.03 → 0.1 → ...` (**her değer bir öncekinin kabaca 3 katı**; 10 katı da olur).
2. Her değer için **birkaç iterasyon** çalıştır ve **cost-iterasyon grafiğini** çiz.
3. **Çok küçük** olan bir değer bul **ve** **çok büyük** olan bir değer bul.
4. **Cost'u hızlı ama tutarlı (her adımda) düşüren**, mümkün olan **en büyük** `α`'yı (veya ona çok yakın biraz küçüğünü) seç.

---

## 5. Özellik mühendisliği

Özelliklerin seçimi/tasarımı, bir öğrenme algoritmasının performansını **çok büyük** ölçüde etkileyebilir. Pek çok uygulamada **doğru özellikleri seçmek/üretmek kritik bir adımdır.**

### Örnek: ev fiyatı tahmini (arsa verileri)

- `x₁` = arsanın **cephesi/genişliği** (frontage)
- `x₂` = arsanın **derinliği** (depth)

Basit model:

```
f(x) = w₁x₁ + w₂x₂ + b
```

**Fikir:** Arsa alanı = genişlik × derinlik. Alanın fiyatı, genişlik ve derinlikten **ayrı ayrı** daha iyi tahmin etmesi muhtemel. Yeni özellik:

```
x₃ = x₁ · x₂        (arsanın alanı)

f(x) = w₁x₁ + w₂x₂ + w₃x₃ + b
```

Model artık verinin gösterdiğine göre genişlik, derinlik veya alandan hangisinin önemli olduğuna **kendisi karar verir** (`w₁, w₂, w₃`'ü öğrenerek).

### Özellik mühendisliği nedir?

> Problem hakkındaki **bilgi veya sezgiyi** kullanarak, orijinal özellikleri **dönüştürüp veya birleştirerek yeni özellikler tasarlamak**, böylece algoritmanın doğru tahmin yapmasını kolaylaştırmak.

Elindeki ham özelliklerle yetinmek yerine yenilerini tanımlamak çoğu zaman **çok daha iyi bir model** verir. Bu tekniğin bir çeşidi, düz çizgi yerine **eğri (doğrusal olmayan fonksiyon)** uydurmayı da sağlar → sıradaki bölüm.

---

## 6. Polinom regresyon (eğri uydurma)

Şimdiye kadar hep **düz çizgi** uydurduk. Çoklu lineer regresyon + özellik mühendisliği = **polinom regresyon**: veriye **eğri** uydurur.

### Örnek: ev boyutu → fiyat

| Model | Özellikler | Sorun / Not |
|---|---|---|
| Düz çizgi | `x` | Veriye iyi uymuyor |
| Karesel (quadratic) | `x`, `x²` | Eğri **sonunda aşağı döner**; büyük evin fiyatının düşmesi mantıksız ❌ |
| Kübik (cubic) | `x`, `x²`, `x³` | Boyut büyüdükçe tekrar yükselir, daha iyi uyum ✅ |
| Karekök | `x`, `√x` | Eğim azalır ama **hiç tamamen düzleşmez ve asla aşağı inmez** ✅ |

Örnek karekök modeli:

```
f(x) = w₁·x + w₂·√x + b
```

Hepsinde **orijinal özelliği** (`x`) üslü/kökle dönüştürüp **yeni özellik** olarak ekliyoruz.

### ⚠️ Polinom özelliklerde ölçekleme çok önemli!

Evin boyutu 1–1.000 feet² ise:

| Özellik | Aralık |
|---|---|
| `x` | 1 ... 1.000 |
| `x²` | 1 ... 1.000.000 (bir milyon) |
| `x³` | 1 ... 1.000.000.000 (bir milyar) |

Aralıklar çok farklı olduğu için dereceli azalma kullanırken **özellik ölçekleme şarttır**.

### Hangi özellikler seçilmeli?

- Şimdilik: **hangi özellikleri kullanacağın konusunda seçeneğin olduğunu** bil.
- İkinci kursta, farklı özellik/model seçeneklerinin **performansını ölçmeyi** ve buna göre karar vermeyi öğreneceksin.

---

## 7. Scikit-learn ve haftanın sonu

- **Scikit-learn:** Çok yaygın kullanılan **açık kaynaklı** ML kütüphanesi. Birçok üst düzey AI/internet/ML şirketinde kullanılıyor; iş hayatında da büyük ihtimalle karşına çıkacak.
- Lineer regresyon bu kütüphaneyle **birkaç satır kodla** yapılabilir (isteğe bağlı lab'da gösterilecek).
- ⚠️ Hocanın vurgusu: Algoritmaları **kendin sıfırdan yazmayı** bil, scikit-learn fonksiyonunu sadece "kara kutu" olarak çağırma. Ama pratikte scikit-learn'ün önemli bir yeri var.
- Hafta sonunda: pratik quizler ve **pratik lab** (lineer regresyonu kendin uygularsın).
- **Gelecek hafta:** Regresyonun ötesine geçip **sınıflandırma (classification)** yani **kategori tahmini** öğrenilecek.

---

## 8. Hızlı tekrar / hatırlatma kartı

- 📏 **Özellik ölçekleme:** Özellikler çok farklı aralıklardaysa kontürler ince-uzun olur, dereceli azalma yavaşlar. Benzer aralığa getir (hedef ≈ −1...+1).
- 🧮 **3 yöntem:**
  - Maksimuma böl: `x / max`
  - Ortalama normalizasyonu: `(x − μ) / (max − min)`
  - Z-skor: `(x − μ) / σ`
- ✅ Çok büyük (±100), çok küçük (±0.001) veya ~100 civarı değerlerde **kesin ölçekle**. Emin değilsen **yine ölçekle**.
- 📉 **Öğrenme eğrisi:** Yatay = iterasyon, dikey = cost J. **Her iterasyonda azalmalı.** Düzleşiyorsa yakınsamış demektir.
- 🚨 **Cost artıyor/zıplıyor?** → `α` çok büyük **veya bug**. Çözüm: `α`'yı küçült, kodda `−` işaretini kontrol et.
- 🔧 **Debug:** Çok küçük `α` ile bile cost her adımda azalmıyorsa kodda hata vardır.
- 🎚️ **α seçimi:** `0.001, 0.003, 0.01, 0.03, 0.1, ...` (her biri ~3×). Çok küçüğü ve çok büyüğü bul, **en büyük makul** değeri (veya biraz küçüğünü) seç.
- 🏗️ **Özellik mühendisliği:** Bilgi/sezgiyle yeni özellik üret (örn. alan = genişlik × derinlik).
- 📈 **Polinom regresyon:** `x`, `x²`, `x³`, `√x` gibi özelliklerle **eğri** uydur. Bu durumda **ölçekleme şart**.
- 🧰 **Scikit-learn:** Pratikte çok kullanılır ama algoritmaları kendin yazmayı da bil.

---

## 9. Örnek Python kodu

Z-skor ölçekleme + öğrenme eğrisi için cost takibi + polinom özellikler:

```python
import numpy as np

def zscore_normalize_features(X):
    """Her özelliği (sütunu) (x - mu) / sigma ile ölçekler."""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def compute_cost(X, y, w, b):
    m = X.shape[0]
    err = X @ w + b - y
    return (1 / (2 * m)) * np.sum(err ** 2)

def compute_gradient(X, y, w, b):
    m = X.shape[0]
    err = X @ w + b - y
    dj_dw = (1 / m) * (X.T @ err)
    dj_db = (1 / m) * np.sum(err)
    return dj_dw, dj_db

def gradient_descent(X, y, w, b, alpha, num_iters):
    cost_history = []                       # öğrenme eğrisi için
    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw               # eksi işareti!
        b = b - alpha * dj_db
        cost_history.append(compute_cost(X, y, w, b))
    return w, b, cost_history

# --- Polinom regresyon örneği: y = 1 + x^2 ---
x = np.arange(0, 20, 1.0)
y = 1 + x ** 2
X = np.c_[x, x ** 2, x ** 3]                # özellikler: x, x², x³

X_norm, mu, sigma = zscore_normalize_features(X)   # ölçekleme!

w0 = np.zeros(X_norm.shape[1])
b0 = 0.0
w, b, history = gradient_descent(X_norm, y, w0, b0, alpha=0.1, num_iters=1000)

# Öğrenme eğrisi kontrolü: cost her iterasyonda azalmalı
print("Cost her adımda azalıyor mu?", all(np.diff(history) <= 0))
print("Son cost:", history[-1])

# Yeni veriyle tahmin: AYNI mu ve sigma ile ölçeklemeyi unutma
x_new = np.array([[10.0, 10.0 ** 2, 10.0 ** 3]])
print("x=10 için tahmin:", (((x_new - mu) / sigma) @ w + b)[0])

# α deneme döngüsü (her biri ~3 kat büyük):
# for alpha in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
#     _, _, h = gradient_descent(X_norm, y, w0, b0, alpha, 100)
#     print(alpha, h[0], h[-1])
```

> Notlar:
> - Kod, ders içeriğini örneklemek için tarafımdan yazıldı, dersten alınmadı. Sayılar örnektir.
> - Yeni veriyle tahmin yaparken **eğitimde hesaplanan `μ` ve `σ`** ile ölçekleme yapılır. Bu ayrıntı derste geçmedi, uygulamada önemli olduğu için ekledim.

---

## ✏️ Not: Transkriptteki küçük hatalar

Ders metninde (otomatik transkript kaynaklı) birkaç hata vardı, özette düzelttim:

- "x₁ ranges from **3**–2,000" yazılmıştı, doğrusu **300–2.000** (hesaplardaki 300 ile uyumlu: 300/2000 = 0.15).
- "great in dissent" / "grading descent" ifadeleri aslında **gradient descent** (dereceli azalma).
- Öğrenme oranı seçiminde "decrease the *learning rate* rapidly" yazılmıştı, kastedilen **cost'u (J) hızlı düşürmek**.
- "intersense / inter sense" ifadeleri muhtemelen **"gradient descent"** veya "linear regression" demek istiyor; bağlamdan çıkarıldı.
