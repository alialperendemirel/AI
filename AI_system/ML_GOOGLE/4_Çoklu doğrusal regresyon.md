# Çoklu Lineer Regresyon ve Vektörleştirme – Ders Özeti

> Makine Öğrenmesi – Hafta 2 notları. Önceki özet: [Dereceli Azalma (Gradient Descent)](gradient-descent-ozet.md)
> Terim karşılıkları: **çoklu lineer regresyon = multiple linear regression**, **vektörleştirme = vectorization**, **nokta çarpımı = dot product**, **normal denklem = normal equation**.

---

## İçindekiler

1. [Birden fazla özellik: notasyon](#1-birden-fazla-özellik-notasyon)
2. [Çoklu lineer regresyon modeli](#2-çoklu-lineer-regresyon-modeli)
3. [Vektör notasyonu ve nokta çarpımı](#3-vektör-notasyonu-ve-nokta-çarpımı)
4. [Vektörleştirme (kod tarafı)](#4-vektörleştirme-kod-tarafı)
5. [Vektörleştirme neden hızlı? (perde arkası)](#5-vektörleştirme-neden-hızlı-perde-arkası)
6. [Çoklu regresyonda dereceli azalma](#6-çoklu-regresyonda-dereceli-azalma)
7. [Normal denklem (dipnot)](#7-normal-denklem-dipnot)
8. [Hızlı tekrar / hatırlatma kartı](#8-hızlı-tekrar--hatırlatma-kartı)
9. [Örnek Python kodu](#9-örnek-python-kodu)

---

## 1. Birden fazla özellik: notasyon

Önceden sadece **bir** özellik vardı (evin boyutu `x`). Şimdi ev fiyatını tahmin etmek için **birden fazla** özellik kullanıyoruz.

### Örnek veri (ev fiyatı)

| Özellik | Anlamı |
|---|---|
| `x₁` | Evin boyutu (feet²) |
| `x₂` | Oda sayısı |
| `x₃` | Kat sayısı |
| `x₄` | Evin yaşı |

### Notasyon tablosu

| Sembol | Anlamı |
|---|---|
| `xⱼ` | **j. özellik** (j = 1, 2, ..., n) |
| `n` | **Toplam özellik sayısı** (bu örnekte n = 4) |
| `x⁽ⁱ⁾` | **i. eğitim örneğinin tüm özelliklerini** içeren **vektör** (satır vektörü) |
| `xⱼ⁽ⁱ⁾` | **i. eğitim örneğindeki j. özellik** (tek bir sayı) |

### Somut örnek

- `x⁽²⁾ = [1416, 3, 2, 40]` → ikinci evin tüm özellikleri (vektör).
- `x₃⁽²⁾ = 2` → ikinci evin **3. özelliği** (kat sayısı).

> Vektörün üstüne ok işareti (`x⃗`) çizmek **isteğe bağlı**; sadece "bu bir sayı değil, sayı listesi" demek için kullanılır.

---

## 2. Çoklu lineer regresyon modeli

**Eski model (tek özellik):**

```
f(x) = w·x + b
```

**Yeni model (n özellik):**

```
f(x) = w₁x₁ + w₂x₂ + w₃x₃ + ... + wₙxₙ + b
```

### Somut örnek (fiyat, bin dolar cinsinden)

```
f(x) = 0.1·x₁ + 4·x₂ + 10·x₃ − 2·x₄ + 80
```

### Parametreler nasıl yorumlanır?

| Parametre | Değer | Anlamı |
|---|---|---|
| `b` | 80 | Taban fiyat: 80 bin $ (tüm özellikler 0 olsaydı) |
| `w₁` | 0.1 | Her ekstra feet² için fiyat **+0.1 bin $ = +100 $** |
| `w₂` | 4 | Her ekstra yatak odası **+4 bin $** |
| `w₃` | 10 | Her ekstra kat **+10 bin $** |
| `w₄` | −2 | Evin yaşı her 1 arttığında fiyat **−2 bin $** (negatif ağırlık = fiyatı düşürür) |

---

## 3. Vektör notasyonu ve nokta çarpımı

Modeli kısa yazmak için parametreleri ve özellikleri **vektör** (sayı listesi) olarak tanımlıyoruz:

```
w = [w₁, w₂, w₃, ..., wₙ]     ← vektör
x = [x₁, x₂, x₃, ..., xₙ]     ← vektör
b                              ← tek bir SAYI (vektör değil)
```

`w` vektörü ile `b` sayısı birlikte **modelin parametrelerini** oluşturur.

### Kısa model

```
f(x) = w · x + b
```

### Nokta çarpımı (dot product) nedir?

İki vektörde **karşılıklı elemanlar çarpılır, sonra hepsi toplanır:**

```
w · x = w₁x₁ + w₂x₂ + w₃x₃ + ... + wₙxₙ
```

Sonuna `b` eklenince yukarıdaki uzun formülün aynısı çıkar. Yani nokta çarpımı sadece **daha derli toplu yazım** sağlar.

### İsimlendirme notu

- Bu modelin adı **"Çoklu Lineer (Doğrusal) Regresyon"**.
- "Multivariate regression" ismi başka bir şeye karşılık geldiği için **kullanılmıyor**.

---

## 4. Vektörleştirme (kod tarafı)

**Vektörleştirme:** Döngü yazmak yerine, tüm vektör işlemini tek seferde yapan kütüphane fonksiyonlarını kullanmak.

**Faydaları:**
1. Kod **daha kısa** ve okunaklı olur.
2. Kod **çok daha hızlı** çalışır (CPU/GPU paralel donanımını kullanır).

### Python / NumPy notları

- **NumPy:** ML ve Python'da en çok kullanılan sayısal lineer cebir kütüphanesi.
- Matematikte indeks **1'den** başlar (`w₁`), Python'da **0'dan** başlar (`w[0]`).
- Örnek: `w = np.array([1.0, 2.5, -3.3])`, `x = np.array([10, 20, 30])`

### Aynı hesabın 3 yolu (n = 3 için)

**❌ 1) Elle tek tek yazmak** (n = 100000 olunca imkansız ve verimsiz):

```python
f = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + b
```

**⚠️ 2) for döngüsü** (daha iyi ama hâlâ vektörleştirilmemiş, yavaş):

```python
f = 0
for j in range(n):          # j = 0, 1, ..., n-1
    f = f + w[j] * x[j]
f = f + b
```

**✅ 3) Vektörleştirilmiş** (tek satır, en hızlı):

```python
f = np.dot(w, x) + b
```

> `range(n)` → 0'dan başlar, **n dahil değil** (n−1'e kadar gider).

---

## 5. Vektörleştirme neden hızlı? (perde arkası)

### Tahmin hesabı (`w·x`)

| | for döngüsü | Vektörleştirilmiş (`np.dot`) |
|---|---|---|
| Nasıl çalışır | Çarpmaları **sırayla**, her zaman adımında **birer tane** yapar (t₀, t₁, ..., t₁₅) | Tüm `w` ve `x` çiftlerini **aynı anda (paralel)** çarpar |
| Toplama | Tek tek toplar | Özel donanımla **verimli şekilde** hepsini toplar |

### Dereceli azalma güncellemesi (16 parametre örneği)

`wⱼ = wⱼ − 0.1 · dⱼ` işlemi (`d` = türevleri tutan dizi, öğrenme oranı = 0.1):

**❌ Döngü ile** (16 hesap sırayla):

```python
for j in range(16):
    w[j] = w[j] - 0.1 * d[j]
```

**✅ Vektörleştirilmiş** (16 hesap aynı anda):

```python
w = w - 0.1 * d
```

### Ne kadar fark eder?

- 16 özellikte fark küçük olabilir.
- **Binlerce özellik + büyük veri setinde** fark çok büyüktür: biri **1-2 dakikada** bitirirken diğeri **saatler** sürebilir.
- Bu yüzden vektörleştirme, ML algoritmalarını büyük veriye ölçeklemenin **anahtar tekniğidir.**

---

## 6. Çoklu regresyonda dereceli azalma

Vektör notasyonuyla her şey kısalıyor:

```
Model:   f(w,b)(x) = w · x + b
Cost:    J(w, b)      ← w VEKTÖRÜ ve b sayısının fonksiyonu, çıktı tek bir sayı
```

### Tek özellik vs. çok özellik

**Tek özellik (eski):**

```
w = w − α · (1/m) · Σ ( f(x⁽ⁱ⁾) − y⁽ⁱ⁾ ) · x⁽ⁱ⁾
b = b − α · (1/m) · Σ ( f(x⁽ⁱ⁾) − y⁽ⁱ⁾ )
```

**n özellik (yeni):** Her `wⱼ` için ayrı güncelleme (j = 1, ..., n):

```
w₁ = w₁ − α · (1/m) · Σ ( f(x⁽ⁱ⁾) − y⁽ⁱ⁾ ) · x₁⁽ⁱ⁾
w₂ = w₂ − α · (1/m) · Σ ( f(x⁽ⁱ⁾) − y⁽ⁱ⁾ ) · x₂⁽ⁱ⁾
 ⋮
wₙ = wₙ − α · (1/m) · Σ ( f(x⁽ⁱ⁾) − y⁽ⁱ⁾ ) · xₙ⁽ⁱ⁾

b  = b  − α · (1/m) · Σ ( f(x⁽ⁱ⁾) − y⁽ⁱ⁾ )
```

(Toplamlar i = 1'den m'ye.)

### Fark nedir?

- Yapı **aynı**; sadece her `wⱼ` için, ilgili özellik `xⱼ⁽ⁱ⁾` ile çarpılan hata kullanılır.
- `w` ve `x` artık vektör; hata terimi `f(x⁽ⁱ⁾) − y⁽ⁱ⁾` yine "tahmin − gerçek".
- `b` için formülde sonda `x` çarpanı **yok** (tek özellikteki gibi).
- Güncellemeler yine **eş zamanlı** yapılır (tüm `w₁...wₙ` ve `b` birlikte güncellenir).

---

## 7. Normal denklem (dipnot)

Lineer regresyonda `w` ve `b`'yi bulmanın **alternatif** yolu.

| | Dereceli azalma | Normal denklem |
|---|---|---|
| Yöntem | Tekrarlı (iteratif) | `w` ve `b`'yi **tek seferde** (lineer cebirle) hesaplar |
| Uygulanabilirlik | Çok geniş (lojistik regresyon, sinir ağları vb.) | **Sadece lineer regresyon** |
| Özellik sayısı büyükse | İyi çalışır | **Yavaşlar** |

- Bu uzmanlık serisinde **neredeyse hiçbir algoritma** bunu kullanmaz.
- Bazı **eski ML kütüphaneleri** lineer regresyon için arka planda bunu kullanıyor olabilir.
- Mülakatta "normal denklem" duyarsan: bahsedilen şey budur. Nasıl çalıştığını bilmek şart değil.

---

## 8. Hızlı tekrar / hatırlatma kartı

- 🏠 **Çoklu lineer regresyon:** Birden fazla özellikle (`x₁...xₙ`) tahmin.
- 🔢 **Notasyon:** `n` = özellik sayısı, `x⁽ⁱ⁾` = i. örneğin özellik **vektörü**, `xⱼ⁽ⁱ⁾` = i. örneğin j. özelliği.
- 📐 **Model:** `f(x) = w · x + b` (nokta çarpımı = karşılıklı çarp, hepsini topla).
- 🧮 **b** tek bir sayı, **w** bir vektör.
- ⚡ **Vektörleştirme:** `for` döngüsü yerine `np.dot(w, x) + b` ve `w = w - α*d`. Daha kısa ve **çok daha hızlı** (paralel donanım).
- 🐍 **Python indeksi 0'dan** başlar; `range(n)` n'i dahil etmez.
- 🔽 **Dereceli azalma (çoklu):** Her `wⱼ` için `wⱼ = wⱼ − α · (1/m) Σ (f − y) · xⱼ⁽ⁱ⁾`, `b` için `x` çarpanı yok. **Eş zamanlı** güncelle.
- 📘 **Normal denklem:** Sadece lineer regresyona özel, tek adımlı yöntem; pratikte pek kullanılmıyor.
- ➡️ **Sıradaki konular:** Özellik ölçekleme (feature scaling), özellik seçme, doğru öğrenme oranı seçimi.

---

## 9. Örnek Python kodu

Vektörleştirilmiş çoklu lineer regresyon için dereceli azalma:

```python
import numpy as np

def predict(x, w, b):
    """Tek örnek için tahmin: f(x) = w·x + b"""
    return np.dot(w, x) + b

def compute_gradient(X, y, w, b):
    """
    X: (m, n) matris  -> m örnek, n özellik
    y: (m,)  vektör
    w: (n,)  vektör
    """
    m = X.shape[0]
    f = X @ w + b                  # tüm örnekler için tahminler (vektörleştirilmiş)
    error = f - y                  # (m,)
    dj_dw = (1 / m) * (X.T @ error)   # (n,) her özellik için türev
    dj_db = (1 / m) * np.sum(error)   # tek sayı
    return dj_dw, dj_db

def gradient_descent(X, y, w, b, alpha, num_iters):
    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        # Eş zamanlı güncelleme (vektörleştirilmiş)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b

# Kullanım örneği (küçük, yapay veri)
X_train = np.array([[2104, 5, 1, 45],
                    [1416, 3, 2, 40],
                    [852,  2, 1, 35]], dtype=float)
y_train = np.array([460, 232, 178], dtype=float)

w_init = np.zeros(X_train.shape[1])
b_init = 0.0

# Not: Özellikler çok farklı ölçeklerde olduğundan alpha küçük tutuldu.
# Bir sonraki konu (özellik ölçekleme) bunu çözecek.
w, b = gradient_descent(X_train, y_train, w_init, b_init, alpha=1e-7, num_iters=1000)
print("w =", w, " b =", b)
print("Tahmin:", predict(X_train[1], w, b))
```

> Notlar:
> - `X @ w` ve `X.T @ error` işlemleri döngüsüz, vektörleştirilmiş hesaplardır.
> - Veri ve `alpha` değeri örnek amaçlıdır, dersten alınmamıştır.

---

## ✏️ Not: Transkriptteki küçük hatalar

Ders metninde (otomatik çeviri/transkript kaynaklı) birkaç yazım hatası vardı, özette düzelttim:

- `f = f + w[j] + x[j]` yazılmış, doğrusu **`f = f + w[j] * x[j]`** (toplama değil, **çarpma**).
- "Bu ifade sadece J=1 için" kısmı, aslında **`j = 1` için** (küçük j, özellik indeksi).
- Cost fonksiyonunda "J" ve özellik indeksindeki "j" farklı şeylerdir: `J` cost, `j` indeks.
