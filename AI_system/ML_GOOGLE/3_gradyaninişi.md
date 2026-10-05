# Dereceli Azalma (Gradient Descent) – Ders Özeti

> Makine Öğrenmesi – Hafta 1 (Lineer Regresyon) notları.
> Terim karşılıkları: **dereceli azalma / alçalma = gradient descent**, **cost fonksiyonu = maliyet fonksiyonu**, **öğrenme oranı = learning rate**, **türev = derivative**.

---

## İçindekiler

1. [Dereceli azalma nedir?](#1-dereceli-azalma-nedir)
2. [Algoritma ve güncelleme formülü](#2-algoritma-ve-güncelleme-formülü)
3. [Türevin mantığı (sezgi)](#3-türevin-mantığı-sezgi)
4. [Öğrenme oranı (α) seçimi](#4-öğrenme-oranı-α-seçimi)
5. [Lineer regresyonda dereceli azalma](#5-lineer-regresyonda-dereceli-azalma)
6. [Algoritmayı çalışırken izlemek ve "toplu" kavramı](#6-algoritmayı-çalışırken-izlemek-ve-toplu-kavramı)
7. [Hızlı tekrar / hatırlatma kartı](#7-hızlı-tekrar--hatırlatma-kartı)
8. [Örnek Python kodu](#8-örnek-python-kodu)

---

## 1. Dereceli azalma nedir?

**Amaç:** `J(w, b)` cost fonksiyonunu **mümkün olan en küçük değere** getiren `w` ve `b` değerlerini sistematik olarak bulmak.

- Önceki derste cost fonksiyonunu görselleştirip `w` ve `b`'yi elle denedik. Daha sistematik bir yola ihtiyaç var → **dereceli azalma**.
- Sadece lineer regresyonda değil, **makine öğrenmesinin her alanında** kullanılır (derin öğrenme / sinir ağları dahil).
- Sadece 2 parametreyle sınırlı değil: `J(w1, w2, ..., wn, b)` için de çalışır.
- **Herhangi bir** cost fonksiyonunu küçültmek için kullanılabilir.

### Genel fikir

1. `w` ve `b` için bir başlangıç değeri seç (lineer regresyonda genelde **ikisi de 0**; başlangıç değeri çok önemli değil).
2. `w` ve `b`'yi **yavaş yavaş** değiştirerek `J`'yi azalt.
3. `J` minimuma ulaşana (veya yeterince yaklaşana) kadar tekrarla.

### Dağ/vadi benzetmesi 🏔️

`J(w, b)` yüzeyini dağlık bir park gibi düşün: yüksek yerler = tepe, alçak yerler = vadi (düşük cost).

- Bir tepenin üzerinde duruyorsun, hedefin **en kısa yoldan vadiye inmek**.
- Her adımda etrafına 360° bakıp şunu soruyorsun: *"Minik bir adım atacaksam, hangi yöne atarsam en hızlı aşağı inerim?"*
- Matematiksel olarak bu yön = **en dik alçalan yön** (eğimin tersi yönü).
- O yöne minik adım at → yeni noktada tekrar bak → tekrar adım at → ... → vadinin dibine (**yerel minimum**) ulaş.

### Yerel minimum (local minimum) özelliği

- Başlangıç noktasını değiştirirsen **farklı bir vadiye** inebilirsin.
- Birinci vadiye inmeye başladıysan algoritma seni ikinci vadiye götürmez (ve tersi).
- Bu vadilerin dipleri **yerel minimum** olarak adlandırılır.
- Bu, genel (örn. sinir ağı) cost fonksiyonları için geçerli. **Lineer regresyonda böyle bir sorun yok** (bkz. bölüm 5).

---

## 2. Algoritma ve güncelleme formülü

Her adımda **her iki parametre** şöyle güncellenir:

```
w = w − α · ∂J(w,b)/∂w
b = b − α · ∂J(w,b)/∂b
```

Bu iki güncellemeyi **yakınsayana kadar** (w ve b artık neredeyse değişmeyene kadar) tekrarla.

### Sembollerin anlamı

| Sembol | Anlamı |
|---|---|
| `=` | Burada **atama operatörü** (kodlardaki `=` gibi). Matematikteki "eşittir" iddiası değil. `a = a + 1` matematikte imkansız ama kodda "a'yı 1 artır" demek. |
| `α` (alfa) | **Öğrenme oranı.** Genelde 0 ile 1 arası pozitif küçük sayı (örn. 0.01). Her adımın **ne kadar büyük** olacağını belirler. |
| `∂J/∂w` | `J`'nin `w`'ye göre türevi. **Hangi yöne** gitmemiz gerektiğini söyler. (Kesin adı kısmi türev; ML'de pratikte "türev" diyoruz, bu ayrım önemli değil.) |

> Python'da eşitlik **testi** `==` ile yapılır (`a == c`). Tek `=` atamadır.

### ⚠️ Eş zamanlı (simultaneous) güncelleme — ÇOK ÖNEMLİ

`w` ve `b` **aynı anda** güncellenmeli. Önce ikisinin de yeni değeri **eski değerler kullanılarak** hesaplanır, sonra birlikte atanır.

**✅ Doğru:**

```
tmp_w = w − α · ∂J(w,b)/∂w
tmp_b = b − α · ∂J(w,b)/∂b
w = tmp_w
b = tmp_b
```

**❌ Yanlış:**

```
tmp_w = w − α · ∂J(w,b)/∂w
w = tmp_w                         # w ERKEN güncellendi!
tmp_b = b − α · ∂J(w,b)/∂b        # artık YENİ w ile hesaplanıyor
b = tmp_b
```

Yanlış versiyonda `b`'nin türevi **güncellenmiş w** ile hesaplanır, yani doğru algoritmadan farklı bir şey olur. Kabaca çalışabilir ama doğru yöntem değildir, farklı özellikleri olan başka bir algoritmadır. **Her zaman eş zamanlı olanı kullan.** (Koda dökerken de zaten doğal olan budur.)

---

## 3. Türevin mantığı (sezgi)

Tek parametreli basit örnek: `J(w)` (yatay eksen `w`, dikey eksen `J`).

Bir noktadaki **türev = o noktada eğriye çizilen teğet doğrunun eğimi** (yükseklik / genişlik).

### Durum A: Minimumun sağındasın → eğim **pozitif**

- Teğet sağ yukarı gider → türev > 0.
- `w = w − α · (pozitif sayı)` → **`w` azalır** → grafikte **sola** gidersin → `J` azalır ✅

### Durum B: Minimumun solundasın → eğim **negatif**

- Teğet sağ aşağı gider → türev < 0 (örn. −2).
- `w = w − α · (negatif sayı)` = negatiften çıkarmak = pozitif eklemek → **`w` artar** → grafikte **sağa** gidersin → `J` azalır ✅

### Sonuç

Türevin **işareti** otomatik olarak seni minimuma doğru yönlendirir:

| Türev | `w`'ye ne olur | Hareket |
|---|---|---|
| Pozitif | Azalır | Sola |
| Negatif | Artar | Sağa |
| 0 | Değişmez | Durur (minimumdasın) |

---

## 4. Öğrenme oranı (α) seçimi

α'nın seçimi algoritmanın verimini çok etkiler; kötü seçilirse algoritma hiç çalışmayabilir.

### α çok küçükse 🐢

- Adımlar minnacık olur.
- Dereceli azalma **çalışır ama çok yavaştır**; minimuma ulaşmak için çok sayıda adım gerekir.

### α çok büyükse 🚀

- Adımlar o kadar büyük olur ki minimumu **aşıp karşı tarafa** geçersin.
- Cost azalmak yerine **artabilir**; her adımda minimumdan daha da uzaklaşabilirsin.
- Sonuç: **yakınsayamaz (fail to converge) ve hatta "yolundan sapar" (diverge)**.

### Zaten yerel minimumdaysan?

- Minimumda teğetin eğimi **0** → türev = 0.
- `w = w − α · 0 = w` → **`w` değişmez.**
- Örn. `w = 5`, `α = 0.1` ise güncelleme sonrası yine `5`. Yani zaten minimumdaysan dereceli azalma seni orada tutar (istediğimiz de bu).

### Sabit α ile bile yakınsama

Minimuma yaklaştıkça **türev küçülür** → güncelleme adımları **otomatik olarak küçülür**.

- Başta eğim dik → büyük adımlar.
- Yaklaştıkça eğim yatıklaşır → daha küçük adımlar.
- Sonunda türev ≈ 0 → adımlar ≈ 0.

Yani α'yı hiç değiştirmesen bile algoritma kendiliğinden yavaşlayarak minimuma oturur. (İyi bir α seçme yöntemleri ilerleyen derslerde.)

---

## 5. Lineer regresyonda dereceli azalma

Hepsini birleştiriyoruz: **lineer regresyon modeli + hata kareler cost fonksiyonu + dereceli azalma**.

### Parçalar

```
Model:      f(x) = w·x + b

Cost:       J(w,b) = (1 / 2m) · Σ_{i=1..m} ( f(x⁽ⁱ⁾) − y⁽ⁱ⁾ )²
```

### Türevler (kalkülüsle türetilmiş)

```
∂J/∂w = (1/m) · Σ_{i=1..m} ( f(x⁽ⁱ⁾) − y⁽ⁱ⁾ ) · x⁽ⁱ⁾

∂J/∂b = (1/m) · Σ_{i=1..m} ( f(x⁽ⁱ⁾) − y⁽ⁱ⁾ )
```

> `∂J/∂b` formülü `∂J/∂w` ile aynı, tek fark sonunda `x⁽ⁱ⁾` çarpanı **yok**.
> Cost'taki `1/2m` içindeki **2**, türev alınırken gelen 2 ile sadeleşir, bu yüzden formüller temiz çıkar (cost'ta 2 olmasının sebebi bu).
> Türevin nasıl çıkarıldığı (isteğe bağlı kalkülüs kısmı) bilinmek zorunda **değil**; formülleri kullanmak yeterli.

### Lineer regresyon için tam algoritma

Yakınsayana kadar tekrarla (**eş zamanlı**):

```
w = w − α · (1/m) · Σ ( f(x⁽ⁱ⁾) − y⁽ⁱ⁾ ) · x⁽ⁱ⁾
b = b − α · (1/m) · Σ ( f(x⁽ⁱ⁾) − y⁽ⁱ⁾ )
```

### Neden lineer regresyonda yerel minimum sorunu yok?

- Genel cost fonksiyonlarında (örn. sinir ağları) birçok yerel minimum olabilir ve başlangıç noktasına göre farklı yerlere inersin.
- Lineer regresyondaki **hata kareler cost fonksiyonu kase (çanak) şeklindedir** → **konveks (convex / dışbükey)** fonksiyondur.
- Konveks fonksiyonun **yalnızca bir tane minimumu** vardır: **global minimum** (tüm noktalar arasındaki en küçük değer).
- Dolayısıyla öğrenme oranı uygun seçildiği sürece dereceli azalma **her zaman global minimuma yakınsar.**

| | Genel (örn. sinir ağı) cost | Lineer regresyon cost |
|---|---|---|
| Şekil | Çok tepeli/vadili | Kase (konveks) |
| Minimum sayısı | Birden fazla yerel minimum olabilir | Tek global minimum |
| Başlangıç noktası önemli mi? | Evet | Hayır |

---

## 6. Algoritmayı çalışırken izlemek ve "toplu" kavramı

Dersteki gösterim (başlangıç `w = −0.1`, `b = 900` → `f(x) = −0.1x + 900`):

- Her adımda `(w, b)` kontür grafiğinde minimuma doğru ilerler.
- Aynı anda soldaki grafikte **doğru, veriye gitgide daha iyi uyar.**
- Cost her güncellemede azalır; global minimumda doğru, veriye göre görece iyi bir modeldir.
- Bu model ile **ev fiyatı tahmini** yapılabilir (örn. 1250 feet² bir ev için ~250 bin dolar).

### Toplu dereceli azalma (Batch Gradient Descent)

- Her güncelleme adımında **eğitim setindeki tüm örnekler** (i = 1'den m'ye) kullanılır, bir alt küme değil.
- İsim biraz garip ama ML topluluğunda böyle kullanılıyor. (DeepLearning.AI'nin bülteni *The Batch* de adını buradan alır.)
- İleride her adımda **verinin sadece küçük bir kısmını** kullanan başka versiyonlar da görülecek.
- Lineer regresyon için **toplu dereceli azalma** kullanılıyor.

---

## 7. Hızlı tekrar / hatırlatma kartı

- 🎯 **Amaç:** `J(w,b)`'yi minimuma indirecek `w`, `b`'yi bulmak.
- 🔁 **Yöntem:** Başlangıç değeri seç → eğimin ters yönünde küçük adımlar at → yakınsayana kadar tekrarla.
- ✍️ **Formül:** `w = w − α · ∂J/∂w`, `b = b − α · ∂J/∂b`
- ⏱️ **Eş zamanlı güncelle:** Önce `tmp_w`, `tmp_b` hesapla, sonra ikisini birden ata.
- 🧭 **Türev:** Yönü söyler (pozitif → sola, negatif → sağa, 0 → dur).
- 📏 **α (öğrenme oranı):** Adım büyüklüğü.
  - Çok küçük → yavaş.
  - Çok büyük → minimumu aşar, yakınsamaz, uzaklaşabilir.
- 🔽 **Minimuma yaklaşınca** türev küçülür → adımlar otomatik küçülür (α sabit olsa bile).
- 🥣 **Lineer regresyon + hata kareler cost = konveks (kase)** → tek global minimum, yerel minimum tuzağı yok.
- 📦 **Toplu (batch):** Her adımda **tüm** eğitim verisi kullanılır.

---

## 8. Örnek Python kodu

Dersin formüllerinin doğrudan koda dökülmüş hali (tek özellikli lineer regresyon):

```python
import numpy as np

def compute_gradient(x, y, w, b):
    """J'nin w ve b'ye göre türevlerini hesaplar."""
    m = x.shape[0]
    f = w * x + b                     # tüm tahminler
    error = f - y                     # f(x) - y
    dj_dw = (1 / m) * np.sum(error * x)
    dj_db = (1 / m) * np.sum(error)
    return dj_dw, dj_db

def gradient_descent(x, y, w, b, alpha, num_iters):
    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # Eş zamanlı güncelleme: türevler eski w, b ile zaten hesaplandı
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b

# Kullanım örneği
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

w, b = gradient_descent(x_train, y_train, w=0.0, b=0.0, alpha=0.01, num_iters=10000)
print(f"w = {w:.2f}, b = {b:.2f}")   # yaklaşık w=200, b=100
```

> Not: Kodda `compute_gradient` iki türevi de **eski** `w`, `b` ile hesapladığı için sonra yapılan güncelleme otomatik olarak eş zamanlı olur.

---

*Sonraki konu: birden fazla özellikli (multiple features) lineer regresyon ve doğrusal olmayan eğriler.*
