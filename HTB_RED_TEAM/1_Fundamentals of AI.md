# HTB AI Red Teamer: ML / DL / GenAI Temelleri Notları

> Amaç: AI sistemlerine saldırmadan önce nasıl çalıştıklarını anlamak. Model nasıl öğreniyorsa, zafiyeti de orada aranır.

## İçindekiler

**Bölüm A: Temeller**
1. [AI / ML / DL İlişkisi](#1-ai--ml--dl-ilişkisi)
2. [Matematik Cheat Sheet](#2-matematik-cheat-sheet)

**Bölüm B: Supervised Learning**

3. [Supervised Learning Temelleri](#3-supervised-learning-temelleri)
4. [Linear Regression](#4-linear-regression)
5. [Logistic Regression](#5-logistic-regression)
6. [Decision Trees](#6-decision-trees)
7. [Naive Bayes](#7-naive-bayes)
8. [SVM](#8-support-vector-machines-svm)

**Bölüm C: Unsupervised Learning**

9. [Unsupervised Learning Temelleri](#9-unsupervised-learning-temelleri)
10. [K-Means](#10-k-means-clustering)
11. [PCA](#11-pca-principal-component-analysis)
12. [Anomaly Detection](#12-anomaly-detection)

**Bölüm D: Reinforcement Learning**

13. [RL Temelleri](#13-reinforcement-learning-temelleri)
14. [Q-Learning](#14-q-learning)
15. [SARSA](#15-sarsa)

**Bölüm E: Deep Learning**

16. [Deep Learning Girişi](#16-deep-learning-girişi)
17. [Perceptron](#17-perceptron)
18. [Neural Networks (MLP)](#18-neural-networks-mlp)
19. [CNN](#19-cnn-convolutional-neural-networks)
20. [RNN](#20-rnn-recurrent-neural-networks)

**Bölüm F: Generative AI**

21. [Generative AI Girişi](#21-generative-ai-girişi)
22. [LLM](#22-large-language-models-llm)
23. [Diffusion Models](#23-diffusion-models)

**Bölüm G: Özet**

24. [Hızlı Karşılaştırma](#24-hızlı-karşılaştırma)
25. [Saldırı Yüzeyi Haritası (kendi eklemem)](#25-saldırı-yüzeyi-haritası)

---

# BÖLÜM A: TEMELLER

## 1. AI / ML / DL İlişkisi

**İç içe üç katman:** AI ⊃ ML ⊃ DL

| Katman | Tanım | Neden var? |
|---|---|---|
| **AI** | İnsan zekası gerektiren görevleri yapan sistemler (NLP, Computer Vision, Robotics, Expert Systems) | Hedef insanın yerini almak değil, yeteneğini artırmak |
| **ML** | Açıkça programlanmadan **veriden öğrenen** AI alt alanı | Kuralları elle yazmak yerine kalıpları veriden çıkarır |
| **DL** | Çok katmanlı **neural network** kullanan ML alt alanı | Görüntü, ses, metin gibi yapısal olmayan veride feature'ı kendisi çıkarır |

**ML türleri**
- **Supervised:** Etiketli veri (spam tespiti, görüntü sınıflandırma)
- **Unsupervised:** Etiketsiz veri (clustering, anomaly detection, boyut indirgeme)
- **Reinforcement:** Ödül/ceza ile deneme-yanılma (oyun, robotik, otonom sürüş)

**DL temel fikirleri**
- **Hiyerarşik feature öğrenme:** Alt katman kenar/doku, üst katman şekil/nesne görür.
- **End-to-end:** Ham girdiden çıktıya, manuel feature engineering yok.
- **Scalable:** Veri ve hesaplama arttıkça iyileşir.

**DL mimarileri:** CNN (görüntü), RNN (sıralı veri), Transformer (NLP, LLM'lerin temeli).

---

## 2. Matematik Cheat Sheet

**Gösterimler:** `x_t` indis (t. adım), `x^2` üs, `Σ_{i=1}^{n} a_i` toplam.

**Norm (vektör uzunluğu):** Mesafe ölçme, regularization, normalizasyon için.
- L2 (Öklid): `sqrt(v1² + ... + vn²)`
- L1 (Manhattan): `|v1| + ... + |vn|`
- L∞: `max(|v1|, ..., |vn|)`

**Log / üstel:** `log2(8) = 3` (entropi), `ln(x)` doğal log, `e^x` büyüme/bozulma ve olasılık dağılımları.

**Matris / vektör** (neural network katmanlarının temeli)
- `A·v`, `A·B` çarpımlar, `A^T` transpoz, `A^-1` ters matris
- `det(A)`: 0 değilse matris tersine çevrilebilir (2x2: `ad - bc`)
- `tr(A)`: ana köşegen toplamı
- **Eigenvector/Eigenvalue:** `A·v = λ·v`. Verideki maksimum varyans yönlerini bulur → **PCA**

**Küme teorisi:** `|S|` eleman sayısı, `∪` birleşim, `∩` kesişim, `A^c` tümleyen.

**İstatistik**

| Sembol | Anlam |
|---|---|
| `P(x\|y)` | y verilmişken x olasılığı |
| `E[X]` | Beklenen değer (ortalama) |
| `Var(X) = E[(X-E[X])²]` | Ortalama etrafındaki yayılım |
| `σ = sqrt(Var)` | Standart sapma |
| `Cov(X,Y)` | İki değişkenin birlikte değişimi |
| `ρ = Cov/(σX·σY)` | Korelasyon, -1 ile 1 arası |

---

# BÖLÜM B: SUPERVISED LEARNING

## 3. Supervised Learning Temelleri

**Fikir:** Model, doğru cevapları bilinen örneklerden öğrenir, görmediği veriyi tahmin eder (çocuğa elma/portakal göstermek gibi).

**İki problem:** **Classification** (kategori: spam/değil) ve **Regression** (sayı: ev fiyatı).

| Kavram | Anlam |
|---|---|
| **Training data** | Etiketli veri. Kalitesi ve miktarı doğruluğu belirler |
| **Features / Labels** | Girdi değişkenleri / doğru cevap |
| **Model** | Feature → label ilişkisini temsil eden fonksiyon |
| **Training** | Hatayı azaltmak için parametre ayarlama |
| **Prediction** | Yeni veriden etiket üretmek |
| **Inference** | Tahminden geniş: modelin neden o sonucu verdiğini yorumlamak da dahil |
| **Generalization** | Görülmemiş veride de doğru çalışma |

**Metrikler:** **Accuracy** (doğru oranı), **Precision** (pozitif dediklerimin kaçı gerçekten pozitif), **Recall** (gerçek pozitiflerin kaçını yakaladım), **F1** (precision ve recall'un harmonik ortalaması).

**Overfitting vs Underfitting**
- **Overfitting:** Gürültüyü bile ezberler, eğitimde iyi, yeni veride kötü
- **Underfitting:** Model fazla basit, her yerde kötü
- **Cross-validation:** Veriyi fold'lara böl, farklı kombinasyonlarla eğit/doğrula, güvenilir performans tahmini
- **Regularization:** Loss'a ceza ekleyerek karmaşıklığı sınırlar. **L1** = katsayıların mutlak değeri, **L2** = karesi

---

## 4. Linear Regression

**Ne yapar:** Sürekli hedefi, feature'larla **doğrusal** ilişki kurarak tahmin eder.

- Basit: `y = mx + c`
- Çoklu: `y = b0 + b1x1 + ... + bnxn`

**OLS (Ordinary Least Squares):** (1) residual = gerçek − tahmin, (2) karesini al (negatifleri yok eder, büyük hatayı daha çok cezalandırır), (3) topla → **RSS**, (4) RSS'yi minimize edecek katsayıları bul.

**Varsayımlar** (bozulursa tahmin yanıltıcı olur): Linearity, Independence, **Homoscedasticity** (hata varyansı sabit), Normality (hatalar normal dağılımlı).

---

## 5. Logistic Regression

**Ne yapar:** İsmine rağmen **classification**. İkili sonuç için **olasılık (0-1)** üretir.

**Neden sigmoid?** Doğrusal çıktı (-∞,+∞) aralığında, olasılık 0-1. Sigmoid "S" eğrisiyle bunu 0-1'e sıkıştırır.

```
P(x) = 1 / (1 + e^-z)
z = m1x1 + ... + mnxn + c
```

**Threshold:** Varsayılan 0.5. Yükseltmek = daha temkinli pozitif (precision/recall dengesi).

**Decision boundary:** Sınıfları ayıran sınır. 2D çizgi, 3D düzlem, daha yüksekte **hyperplane** (boyutu uzaydan 1 eksik alt uzay).

**Varsayımlar:** Hedef ikili, **log-odds linearity** (`log(P/(1-P))`), az multicollinearity, yeterince büyük örneklem.

---

## 6. Decision Trees

**Fikir:** Veriyi ardışık evet/hayır sorularıyla böler. Classification + regression, yorumlaması kolay.

**Yapı:** **Root** (tüm veri) → **Internal nodes** (feature kuralları) → **Leaf** (tahmin)

| Ölçü | Formül | Not |
|---|---|---|
| **Gini impurity** | `1 - Σ(pi)²` | Düşük = saf |
| **Entropy** | `-Σ pi·log2(pi)` | Düşük = homojen |
| **Information Gain** | `Entropy(S) - Σ(\|Sv\|/\|S\|)·Entropy(Sv)` | Bölünmeyle entropi azalması, en yüksek olan feature seçilir |

**Örnek (30 A + 20 B):** Gini = 0.48, Entropy ≈ 0.971

**Durdurma:** Max depth (overfitting'i önler), minimum örnek sayısı, pure node.

**Avantaj:** Doğrusallık/normallik varsayımı yok, outlier'a dayanıklı (mesafe değil eşik bölünmesi), ağır ön işleme gerekmez.

---

## 7. Naive Bayes

**Bayes teoremi:** Yeni kanıta göre inancı günceller.
```
P(A|B) = P(B|A) · P(A) / P(B)
```
`P(A|B)` posterior, `P(B|A)` likelihood, `P(A)` prior, `P(B)` evidence.

**Neden önemli:** Hastalık yaygınlığı %1, test %95 doğru, %5 false positive → test pozitif olsa bile hasta olma ihtimali sadece **%16**. Sebep düşük **base rate**. *(Güvenlikte aynı: nadir saldırıyı arayan alarm sistemleri çok false positive üretir.)*

**Adımlar:** prior hesapla → likelihood hesapla → posterior hesapla → en yüksek posterior'lu sınıfı seç.

**"Naive" varsayım:** Sınıf verildiğinde feature'lar **koşullu bağımsız**. Gerçekte çoğu zaman yanlış ama pratikte yine iyi çalışır.

**Türler:** **Gaussian** (sürekli, çan eğrisi), **Multinomial** (kelime sıklığı, metin), **Bernoulli** (var/yok).

---

## 8. Support Vector Machines (SVM)

**Fikir:** Sınıfları **maksimum marjinle** ayıran hyperplane'i bulur. Marjin = hyperplane ile en yakın noktalar arası mesafe, bu noktalara **support vectors** denir.

**Neden geniş marjin?** Gürültüye toleranslı, daha az yanlış sınıflama, daha iyi generalization.

```
Hyperplane: w·x + b = 0
Minimize:   1/2 ||w||²
Kısıt:      yi(w·xi + b) >= 1   (yi ∈ {-1, +1})
```
`||w||` küçüldükçe marjin büyür.

**Non-linear SVM + Kernel Trick:** Veriyi **yüksek boyutlu uzaya** taşır, orada düz hyperplane bulur (masadaki karışık bilyeleri havaya kaldırmak gibi). Orijinal uzayda bu eğri sınıra denk gelir.

| Kernel | Özellik |
|---|---|
| **Polynomial** | Polinom terimleriyle eğri sınırlar |
| **RBF** | En popüler, karmaşık doğrusal olmayan kalıplar |
| **Sigmoid** | Logistic regression'daki sigmoid'e benzer |

**Avantaj:** Dağılım varsayımı yok, yüksek boyutta iyi, outlier'a nispeten dayanıklı.

---

# BÖLÜM C: UNSUPERVISED LEARNING

## 9. Unsupervised Learning Temelleri

**Fikir:** Etiket yok, "doğru cevap" yok. Algoritma verideki gizli yapıyı kendisi keşfeder (haritasız şehri keşfetmek gibi). Etiketli verinin kıt/pahalı olduğu durumlarda değerli.

**Üç ana problem**
- **Clustering:** Benzer noktaları grupla
- **Dimensionality Reduction:** Önemli bilgiyi koruyarak feature sayısını azalt
- **Anomaly Detection:** Normdan sapanı bul

**Temel kavramlar**

| Kavram | Anlam |
|---|---|
| **Similarity measures** | **Euclidean** (düz çizgi mesafesi), **Cosine** (vektörler arası açı, yüksek = benzer), **Manhattan** (mutlak farkların toplamı) |
| **Clustering tendency** | Veri doğal olarak kümelenmiyorsa clustering anlamsız sonuç verir, önce kontrol et |
| **Cluster validity** | **Cohesion** (küme içi yakınlık) ve **Separation** (kümeler arası uzaklık). Silhouette score, Davies-Bouldin index bunları ölçer |
| **Curse of dimensionality** | Boyut artınca veri seyrelir, mesafeler anlamını yitirir |
| **Intrinsic dimensionality** | Verinin gerçek bilgi boyutu, feature sayısından düşük olabilir |
| **Anomaly vs Outlier** | Anomali: beklenen kalıptan sapma (olay/hata/fraud). Outlier: daha geniş, çoğunluktan uzak herhangi bir nokta |

**Feature scaling (şart):** Mesafe tabanlı algoritmalarda büyük ölçekli feature baskın çıkar. **Min-Max** (sabit aralığa), **Standardization / Z-score** (ortalama 0, varyans 1).

---

## 10. K-Means Clustering

**Ne yapar:** Veriyi K adet örtüşmeyen kümeye böler. Hedef: küme içi varyansı minimize etmek.

**Adımlar**
1. **Init:** Rastgele K nokta merkez (centroid) seç
2. **Assign:** Her noktayı en yakın centroid'e ata (Euclidean)
3. **Update:** Her kümenin ortalamasını yeni centroid yap
4. **Iterate:** Centroid'ler sabitlenene kadar 2-3'ü tekrarla

`d(x,y) = sqrt(Σ(xi - yi)²)`

**Optimum K nasıl seçilir?** Tek doğru yöntem yok.
- **Elbow method:** K'ya karşı **WCSS** (küme içi kareler toplamı) çiz, düşüşün yavaşladığı "dirsek" noktası iyi K'dır. Dirsekten sonra artırmak gürültüyü kümelemeye başlar.
- **Silhouette analysis:** Skor -1 ile 1 arası. **1'e yakın** = noktası kendi kümesine iyi uyuyor, **0** = sınırda, **-1** = muhtemelen yanlış kümede. En yüksek ortalama skoru veren K seçilir.
- **Domain bilgisi:** Örn. yönetebileceğin kampanya sayısı K'yı belirleyebilir. Hesaplama maliyeti ve yorumlanabilirlik de hesaba katılır.

**Varsayımlar / zayıflıklar:** Kümeler küresel ve benzer boyutta olmalı, **ölçeğe hassas** (önce standardize et), **outlier'a hassas** (centroid'i kaydırır).

---

## 11. PCA (Principal Component Analysis)

**Ne yapar:** Yüksek boyutlu veriyi, mümkün olduğunca bilgiyi koruyarak düşük boyuta indirir. Orijinal feature'ların doğrusal kombinasyonları olan **principal component**'leri bulur (verideki maksimum varyans yönleri).

**Kullanım:** Feature extraction, görselleştirme, gürültü azaltma.

**Üç kavram:** **Variance** (yayılım, maksimize edilir), **Covariance** (feature'lar arası ilişki), **Eigenvector/Eigenvalue** (eigenvector = bileşenin yönü, eigenvalue = o bileşenin açıkladığı varyans miktarı).

**Adımlar**
1. Veriyi standardize et
2. Kovaryans matrisini `C` hesapla
3. Eigenvector ve eigenvalue'ları bul (`C·v = λ·v`), eigenvalue'ya göre büyükten küçüğe sırala
4. En üstteki **k** eigenvector'ü seç
5. Veriyi projekte et: `Y = X·V`

**Çözüm yöntemleri:** Eigenvalue decomposition veya **SVD** (daha sayısal kararlı).

**Kaç bileşen?** Açıklanan varyans oranı grafiğine bak, toplam varyansın yeterli bir kısmını (örn. %95) yakalayan sayıyı seç.

**Varsayımlar:** Doğrusal ilişki, feature'lar arası anlamlı korelasyon varsa en iyi çalışır, **ölçeğe hassas** (standardize şart).

---

## 12. Anomaly Detection

**Fikir:** Normal davranışı öğren, sapmayı işaretle (gece binaya girmeye çalışan kişi alarmı tetikler).

**Üç tür**
- **Point:** Tek nokta farklı (ani trafik artışı, aşırı yüksek işlem tutarı)
- **Contextual:** Bağlamda anormal (30°C yazın normal, kışın anomali)
- **Collective:** Tek tek normal ama grup olarak anormal (birçok IP'den ani login dalgası, koordineli saldırı)

**Teknik aileleri**
- **İstatistiksel:** Z-score, modified z-score, boxplot (normal verinin bir dağılımı izlediğini varsayar)
- **Clustering tabanlı:** Hiçbir kümeye ait olmayan veya küçük/seyrek kümedeki noktalar (K-means, density-based)
- **ML tabanlı:** One-Class SVM, Isolation Forest, LOF

**One-Class SVM:** Normal verinin etrafına sınır çizer (koyun ağılına çit gibi), dışına düşen anomali. Kernel ile doğrusal olmayan sınır öğrenebilir.

**Isolation Forest:** Rastgele feature ve rastgele bölme değeriyle veriyi böler. Anomaliler "az ve farklı" olduğu için **daha az bölmede (kısa yolda) izole olur** ("20 soru" oyunu: az soruda bulunuyorsa anomali).
```
score(x) = 2^( -E(h(x)) / c(n) )
```
`E(h(x))` ortalama yol uzunluğu, `c(n)` normalizasyon. **1'e yakın = anomali, 0.5 civarı = normal.**

**LOF (Local Outlier Factor):** Noktanın **yerel yoğunluğunu** komşularınınkiyle kıyaslar (yoğun mahallede seyrek bölgedeki ev). Yoğunluğun bölgeye göre değiştiği veride etkili.
```
LOF(p) = (Σ lrd(o) / k) / lrd(p)
lrd(p) = 1 / (Σ reach_dist(p,o) / k)
```
**Yüksek LOF = outlier olma ihtimali yüksek.** `reach_dist` = gerçek mesafe ile `o`'nun k-mesafesinin maksimumu.

**Dikkat:** Bazı yöntemler Gauss dağılımı varsayar, feature seçimi performansı çok etkiler, bazı ML yöntemleri etiketli veri ister.

---

# BÖLÜM D: REINFORCEMENT LEARNING

## 13. Reinforcement Learning Temelleri

**Fikir:** **Agent** bir **environment** ile etkileşir, **reward/ceza** ile deneme-yanılma yoluyla öğrenir (köpek eğitimi gibi). Dinamik ortamda ardışık karar verme için uygun.

**Kavramlar**

| Kavram | Anlam |
|---|---|
| **Agent** | Öğrenen ve karar veren |
| **Environment** | Agent'ın dışındaki her şey, geri bildirimi verir |
| **State** | Ortamın anlık durumu |
| **Action** | Agent'ın hamlesi |
| **Reward** | Eylemin ne kadar iyi olduğunu gösteren skaler geri bildirim |
| **Policy** | State → action stratejisi (deterministic veya stochastic). Hedef: kümülatif ödülü maksimize eden policy |
| **Value function** | Bir state/action'ın uzun vadeli beklenen kümülatif ödülü (state-value ve action-value) |
| **Discount factor γ** | 0-1. 0 = sadece anlık ödül, 1'e yakın = uzun vadeli ödüle ağırlık |
| **Episodic vs Continuous** | Episodic: terminal state ile biter (labirent). Continuous: sonu yok (robot kol kontrolü) |

**Model-Based vs Model-Free**
- **Model-Based:** Ortamın modelini öğrenir, plan yapar (labirentin haritası elinde)
- **Model-Free:** Modelsiz, doğrudan deneyimle öğrenir (haritasız labirent)

---

## 14. Q-Learning

**Ne yapar:** Model-free RL. Her (state, action) çiftinin **Q-value**'sunu (beklenen kümülatif ödül) öğrenir. **Off-policy**.

**Q-Table:** Satırlar state, sütunlar action, hücreler Q-value (arama tablosu).

**Güncelleme kuralı (Bellman tabanlı)**
```
Q(s,a) = Q(s,a) + α · [ r + γ · max Q(s',a') - Q(s,a) ]
```
`α` öğrenme oranı, `r` ödül, `γ` discount, `max Q(s',a')` sonraki state'in **en iyi** action değeri.

**Örnek:** `Q(S1,Right)=0.2`, `r=0.5`, `α=0.1`, `γ=0.9`, `max Q(S2)=1.0` → `0.2 + 0.1·(0.5 + 0.9 - 0.2) = 0.32`

**Algoritma:** Q-table'ı sıfırla → action seç (explore/exploit) → uygula ve gözlemle → Q güncelle → state güncelle → yakınsayana kadar tekrarla.

**Exploration vs Exploitation:** Yeni action dene mi, bilinen en iyiyi mi kullan (restoran seçimi gibi).

**Epsilon-greedy:** ε olasılıkla rastgele action (keşif), 1-ε olasılıkla en yüksek Q'lu action (sömürü). Başta yüksek ε (çok keşif), zamanla düşürülür.

**Varsayımlar:** **Markov property** (sonraki state sadece mevcut state+action'a bağlı), **stationary environment** (dinamikler zamanla değişmez).

---

## 15. SARSA

**Ne yapar:** Q-Learning'e benzer ama **on-policy**: güncellemede sonraki state'te **gerçekten seçilen** action'ı (`a'`) kullanır, `max` kullanmaz.

```
Q(s,a) <- Q(s,a) + α · ( r + γ · Q(s',a') - Q(s,a) )
```

| | **Q-Learning** | **SARSA** |
|---|---|---|
| Tür | **Off-policy** | **On-policy** |
| Sonraki değer | `max Q(s',a')` (en iyi action) | `Q(s',a')` (gerçekte seçilen action) |
| Ne öğrenir | Optimal policy'nin değerini | Takip ettiği (keşif dahil) policy'nin değerini |
| Karakter | Daha keşifçi, riskli | **Temkinli**, güvenli |

**Neden SARSA?** Keşif hamlelerinin riskini de hesaba kattığı için tehlikeli action'lardan kaçınır. Güvenlik/istikrarın kritik olduğu yerlerde (engelli odada robot) avantajlı. "Görev başında öğrenir", bazen optimalite kaybı göze alınır.

**Adımlar:** Q init → `a` seç → uygula, `s'` ve `r` gözle → `a'` seç (aynı policy ile) → Q güncelle → `s=s', a=a'` → tekrarla.

**Keşif stratejileri:** Epsilon-greedy (SARSA'da keşif sonuçlarını hesaba kattığı için daha temkinli), **Softmax** (Q-value'ya göre olasılık verir, orta değerli action'lar da seçilebilir).

**Parametreler:** Yüksek `α` hızlı ama kararsız, düşük `α` kararlı ama yavaş. Yüksek `γ` uzun vadeli, düşük `γ` anlık odaklı. Grid search/cross-validation ile ayarlanır. Optimal policy'ye yakınsama garantisi: `α` yeterince küçük ve tüm state-action çiftleri sonsuz kez ziyaret edilirse.

**Varsayımlar:** Q-Learning ile aynı (Markov property, stationary environment).

---

# BÖLÜM E: DEEP LEARNING

## 16. Deep Learning Girişi

**Fikir:** Çok katmanlı **ANN** (beyinden esinlenmiş) ile karmaşık kalıpları öğrenir. Klasik ML'de **manuel feature engineering** gerekir, DL ham veriden feature'ı **otomatik** öğrenir (hiyerarşik temsil). Bu yüzden daha karmaşık problemleri çözer.

**Motivasyon:** (1) Klasik yöntemlerin zorlandığı karmaşık problemleri çözmek (görüntü, konuşma, dil), (2) insan beyninin hiyerarşik bilgi işlemesini taklit etmek.

**Temel bileşenler**

| Bileşen | Anlam |
|---|---|
| **ANN** | Katmanlı nöronlar. Her bağlantının bir **ağırlığı** var, öğrenme = ağırlıkları ayarlamak |
| **Katmanlar** | **Input** (veriyi alır), **Hidden** (hesaplar, feature çıkarır, birden fazlaysa "derin"), **Output** (nihai tahmin) |
| **Activation function** | Ağa **non-linearity** katar. **Sigmoid** (0-1), **ReLU** (negatif=0, pozitif=aynen), **Tanh** (-1 ile 1) |
| **Backpropagation** | Loss'un ağırlıklara göre gradyanını hesaplayıp ağırlıkları loss'u azaltacak yönde günceller |
| **Loss function** | Tahmin-gerçek farkı. Regression: **MSE**, Classification: **cross-entropy** |
| **Optimizer** | Ağırlık güncelleme yöntemi: **SGD, Adam, RMSprop** |
| **Hyperparameters** | Eğitimden önce ayarlanır: learning rate, katman sayısı, nöron sayısı |

---

## 17. Perceptron

**Ne:** Biyolojik nöronun en basit modeli, sinir ağlarının yapı taşı.

**Yapı ve akış:** Girdiler `x` × ağırlıklar `w` → toplama `Σ(wi·xi)` → **bias** `b` ekle → **activation (step)** → çıktı (0 veya 1).

```
çıktı = f(Σ wi·xi + b)      f(x) = 1 if x > 0 else 0
```

**Örnek (tenis oynanır mı?):** `w = [0.3, 0.2, -0.4, -0.2]`, `b = 0.1`, girdi `[0,1,0,0]` → `0.2 + 0.1 = 0.3 > 0` → **1 (oyna)**

```python
def step_activation(x):
    return 1 if x > 0 else 0

weighted_sum = sum(w * x for w, x in zip(weights, inputs))
output = step_activation(weighted_sum + bias)
```

**Sınırlama:** Tek katmanlı perceptron sadece **doğrusal ayrılabilir** problemleri çözer (tek düz çizgi/hyperplane). **XOR** klasik örnek: tek çizgiyle ayrılamaz. Çözüm: çok katman (→ MLP).

---

## 18. Neural Networks (MLP)

**MLP (Multi-Layer Perceptron):** Input layer + bir veya daha fazla hidden layer + output layer.

**Nöron:** Girdi × ağırlık + bias → activation. Perceptron'dan farkı: step yerine sigmoid/ReLU/tanh kullanabilir → doğrusal olmayan ilişkiler ve sürekli çıktı.

**Hidden layer'daki nöron:** Önceki katmanın tüm nöronlarından girdi alır → ağırlıklı toplam → bias → activation → sonraki katmana. İlk katmanlar basit, sonrakiler karmaşık feature öğrenir.

**Output layer:** Binary classification = 1 nöron, multi-class = sınıf başına 1 nöron.

**Neden çok katman?** Non-linear activation'lı çoklu hidden layer, doğrusal olmayan karar sınırlarını öğrenir (XOR çözülür). **Activation olmasa ağ özünde doğrusal bir model olurdu.**

**Activation türleri**
- **Sigmoid:** 0-1, vanishing gradient yüzünden artık az kullanılıyor
- **ReLU:** Basit, hızlı eğitim, genelde en iyi performans
- **Tanh:** -1 ile 1, sigmoid gibi ama 0 merkezli
- **Softmax:** Output layer'da, ham skorları sınıflar üzerinde **olasılık dağılımına** çevirir

**Eğitim = Backpropagation + Gradient Descent**

*Backpropagation:*
1. **Forward pass:** Girdi ağdan geçer, çıktı hesaplanır
2. **Loss hesapla:** Tahmin-gerçek farkı
3. **Backward pass:** Hata çıkıştan girişe katman katman yayılır, **chain rule** ile her ağırlığın gradyanı bulunur
4. **Güncelle:** Ağırlıklar ve bias'lar hatayı azaltacak şekilde ayarlanır

*Gradient Descent:* Loss'un **negatif gradyanı** yönünde adım atar, adım boyutunu **learning rate** belirler. Rastgele başla → gradyanı hesapla (backprop) → ağırlıkları güncelle → yakınsayana kadar tekrarla. **Backprop gradyanı hesaplar, gradient descent onu kullanıp parametreleri günceller.**

---

## 19. CNN (Convolutional Neural Networks)

**Ne için:** Izgara benzeri veri (görüntü, video). Uzamsal feature hiyerarşisini yakalar. Görüntü tanıma, nesne tespiti, segmentasyon.

**Üç katman türü**
- **Convolutional layer:** Öğrenilebilir **filtre**ler girdi üzerinde kayar, her konumda nokta çarpımı alır → **feature map** (kenar, köşe, doku gibi özelliklerin nerede ve ne kadar güçlü olduğunu gösterir). Her katmanda birden çok filtre.
- **Pooling layer:** Feature map'i küçültür (**max pooling** / **average pooling**). Hesaplama maliyetini ve overfitting'i azaltır, küçük kaymalara dayanıklılık verir.
- **Fully connected layer:** Sonda, çıkarılan feature'lardan üst düzey karar verir. Pooling çıktısı önce **flatten** edilir.

**Akış:** `Görüntü → [Conv → Pool] x N → Flatten → FC → Tahmin`

**Hiyerarşik feature öğrenme:** İlk katmanlar kenar/leke, ara katmanlar köşe/eğri, derin katmanlar şekil/nesne parçası (tekerlek, pencere, araba). Filtre ağırlıkları eğitimde öğrenilir.

**Veri varsayımları (neden bu yapı çalışıyor)**

| Varsayım | Anlam |
|---|---|
| **Grid yapısı** | Görüntü 2D grid (yükseklik, genişlik, kanal), video 3D (+zaman) |
| **Spatial hierarchy** | Basit yerel feature'lar üstüne karmaşık feature'lar kurulur |
| **Feature locality** | İlgili ilişkiler yerel komşulukta, filtreler küçük **receptive field**'lara bakar |
| **Feature stationarity** | Dikey kenar solda da sağda da aynı feature. **Weight sharing** ile aynı filtre her konuma uygulanır |
| **Yeterli veri + normalizasyon** | Büyük etiketli veri (yoksa overfitting), girdi 0-1 veya -1 ile 1'e ölçeklenmeli |

---

## 20. RNN (Recurrent Neural Networks)

**Ne için:** **Sıralı veri** (metin, konuşma, zaman serisi), sıranın önemli olduğu yerlerde. Geleneksel ağlardan farkı: geçmiş girdilerin "hafızasını" tutar.

**Nasıl çalışır:** Her zaman adımında modül iki girdi alır: **mevcut girdi** (örn. kelime) + önceki adımdan **hidden state** (geçmişin özeti). İki çıktı üretir: o adımın çıktısı + güncellenmiş hidden state (sonrakine aktarılır). Cümle işlenirken bağlam hidden state'te birikir, "mat" kelimesine gelindiğinde tüm cümlenin bilgisi içindedir.

**Vanishing Gradient problemi:** Eğitimde **BPTT** (Backpropagation Through Time) kullanılır. Gradyanlar zaman adımları boyunca tekrar tekrar çarpılır, 1'den küçükse **üstel olarak küçülip sıfıra yaklaşır**. Sonuç: erken girdilerin etkisi kaybolur, **uzun vadeli bağımlılıklar öğrenilemez.**

**Çözüm: kapılı mimariler**
- **LSTM:** Bellek hücresi + 3 kapı. **Input gate** (ne eklenecek), **Forget gate** (ne unutulacak), **Output gate** (ne çıkacak). Seçici hatırlama/unutma vanishing gradient'ı hafifletir.
- **GRU:** Daha basit, 2 kapı: **Update gate** (önceki durumun ne kadarı korunur), **Reset gate** (önceki durum girdiyle ne kadar birleşir). Daha verimli, birçok görevde LSTM ile benzer performans.

**Bidirectional RNN:** Diziyi hem ileri hem geri işler (iki RNN), çıktılar birleştirilir. Tüm dizi baştan mevcutsa (NLP) hem geçmiş hem gelecek bağlamı yakalanır.

---

# BÖLÜM F: GENERATIVE AI

## 21. Generative AI Girişi

**Fikir:** Tahmin/sınıflandırma yerine **yeni içerik üretir** (metin, görüntü, müzik, kod). Verinin istatistiksel dağılımını öğrenir, benzer yeni örnekler üretir.

**Süreç:** **Training** (veri kalıplarını öğren) → **Generation** (öğrenilen dağılımdan örnekleme, seed/girdiden başlayıp yinelemeli iyileştirme) → **Evaluation** (kalite, özgünlük, insana benzerlik; öznel veya metrikle).

**Model türleri**

| Model | Mantık |
|---|---|
| **GAN** | **Generator** sahte üretir, **Discriminator** gerçek/sahteyi ayırt eder. Rekabet ikisini de geliştirir |
| **VAE** | Sıkıştırılmış temsil (latent) öğrenir, oradan örnekleyerek üretir. Kontrollü ve çeşitli |
| **Autoregressive** | İçeriği sırayla, her öğeyi öncekilere göre üretir (metin/LLM) |
| **Diffusion** | Veriye kademeli gürültü ekler, sonra bu süreci **tersine çevirmeyi** öğrenir |

**Kavramlar**
- **Latent space:** Verinin sıkıştırılmış gizli temsili. Benzer noktalar yakın kümelenir, oradan örnekleyerek yeni içerik üretilir
- **Sampling:** Öğrenilen dağılımdan çekim yaparak üretim
- **Mode collapse:** Generator, veri çeşitliliği geniş olsa da sadece dar bir çıktı çeşitliliği üretir (bir "mod"da sıkışır)
- **Overfitting:** Eğitim örneklerini ezberler, yaratıcılık/özgünlük düşer

**Metrikler**
- **IS (Inception Score):** Üretilen görüntünün netliği ve sınıf çeşitliliği
- **FID (Fréchet Inception Distance):** Üretilen vs gerçek dağılım. **Düşük = daha iyi**
- **BLEU:** Üretilen metnin referans metne benzerliği

---

## 22. Large Language Models (LLM)

**Ne:** Devasa metin verisiyle eğitilmiş, insan benzeri metni anlayıp üreten modeller (çeviri, özet, soru-cevap, yaratıcı yazı). **Transformer** mimarisine dayanır. Eğitim pahalıdır, GPU/TPU gerektirir.

**Üç özellik:** **Massive scale** (milyarlarca/trilyonlarca parametre), **Few-shot learning** (birkaç örnekle yeni görev), **Contextual understanding** (bağlamı anlayıp tutarlı yanıt).

**Bileşenler (metnin modelden geçişi)**

| Adım | Açıklama |
|---|---|
| **Tokenization** | Metin token'lara (kelime/alt-kelime/karakter) ayrılır: `"I love AI" → ["I","love","AI"]` |
| **Embeddings** | Her token anlamı yakalayan **vektöre** çevrilir. Benzer anlam = yakın vektör (king ile queen, king ile table'dan yakındır) |
| **Transformer** | Cümleyi **paralel** işler (RNN sıralıydı), bu yüzden çok daha hızlı |
| **Self-attention** | Her kelime çifti arasında **attention skoru** hesaplar, kelimenin diğerlerine ne kadar "dikkat etmesi" gerektiğini belirler. Uzak bağımlılıkları yakalar ("which" kelimesinin "mat"e atıf yaptığını anlar) |
| **Encoder / Decoder** | Encoder girdiyi anlar, decoder onun çıktısına dayanıp metin üretir |

**Eğitim:** Devasa metin üzerinde, tahmin ile gerçek metin farkını azaltmak için **gradient descent** varyantlarıyla parametre ayarlanır. Kaynak "unsupervised" der; pratikte buna **self-supervised** denir (etiket metnin kendisinden, sonraki kelimeden gelir).

**Üretim:** Prompt verilir (`"Once upon a time, there was a cat named Whiskers."`), model bağlamı ve dil bilgisini kullanarak devamını **kelime kelime** üretir.

---

## 23. Diffusion Models

**Fikir:** GAN/VAE'den farklı olarak veri dağılımını **gürültü ekleme ve çıkarma** adımlarıyla öğrenir. Yüksek kaliteli görüntü, ses vb. üretir.

**İleri süreç (forward, gürültüleme):** Veriye adım adım gürültü eklenir, sonunda saf gürültü olur.
```
x_0 (orijinal) → x_1 → ... → x_T (saf gürültü)      x_t ~ q(x_t | x_{t-1})
```

**Ters süreç (reverse, denoising):** Bir sinir ağı her adımda **eklenen gürültüyü tahmin etmeyi** öğrenir ve gürültüyü geri çıkarır.
```
x_{t-1} ~ p_θ(x_{t-1} | x_t)
Loss: L = E[ ||ε - ε_pred||² ]     (MSE: gerçek gürültü vs tahmin)
```

**Noise schedule:** Her adımda ne kadar gürültü eklendiğini belirler. Yaygın: doğrusal, `β_t = β_min + (t/T)·(β_max - β_min)`. İyi tasarım performansı ciddi etkiler.

**Denoising network:** Genelde derin **CNN** veya **transformer**. Girdi: gürültülü `x_t`, çıktı: tahmin edilen gürültü.

**Eğitim:** Ağı başlat → orijinal veriye gürültü ekle → ağı gürültüyü tahmin edecek şekilde eğit → loss hesapla → gradient descent ile güncelle → yakınsayana kadar epoch'larla tekrarla.

**Örnekleme (üretim):** Saf gürültü `x_T` ile başla → T'den 1'e her adımda gürültüyü tahmin edip çıkar → sonunda `x_0` üretilen görüntüdür.

**Text-to-image (örn. "şapkalı bir kedi"):**
1. **Text encoding:** Prompt, önceden eğitilmiş text encoder (Transformer/CLIP) ile anlamı yakalayan vektöre çevrilir
2. **Conditioning:** Bu vektör denoising ağını **koşullandırır**, görüntünün prompt ile uyumlu olmasını sağlar
3. **Sampling:** Her ters adımda ağ hem gürültülü görüntüyü hem **text embedding**'i kullanır
4. **Sonuç:** Yeterli adımdan sonra prompt'a uyan görüntü

**Varsayımlar:** **Markov property** (her adım sadece bir önceki adıma bağlı), **statik veri dağılımı**, **smoothness** (katı gereklilik değil, küçük girdi değişimi küçük çıktı değişimi).

---

# BÖLÜM G: ÖZET

## 24. Hızlı Karşılaştırma

**Supervised**

| Algoritma | Tür | Çıktı | Ana fikir | Güçlü yanı |
|---|---|---|---|---|
| **Linear Regression** | Regression | Sayı | En iyi doğru (OLS) | Basit, yorumlanabilir |
| **Logistic Regression** | Classification | Olasılık 0-1 | Sigmoid + threshold | Hızlı baseline |
| **Decision Tree** | İkisi de | Sınıf/sayı | Entropy/Gini ile bölme | Yorumlanabilir |
| **Naive Bayes** | Classification | Olasılık | Bayes + bağımsızlık | Hızlı, metin için iyi |
| **SVM** | İkisi de | Sınıf/sayı | Maks. marjin + kernel | Yüksek boyut, non-linear |

**Unsupervised**

| Algoritma | Amaç | Ana fikir | Dikkat |
|---|---|---|---|
| **K-Means** | Clustering | Centroid'e ata, ortalamayla güncelle | K seçimi (elbow/silhouette), outlier ve ölçek hassasiyeti |
| **PCA** | Boyut indirgeme | Max varyans yönleri (eigenvector) | Standardize et, doğrusal |
| **One-Class SVM** | Anomaly | Normal verinin etrafına sınır | Kernel gerekir |
| **Isolation Forest** | Anomaly | Anomali kısa yolda izole olur | Skor 1'e yakın = anomali |
| **LOF** | Anomaly | Yerel yoğunluk karşılaştırması | Yüksek LOF = outlier |

**Reinforcement**

| Algoritma | Policy | Güncelleme | Karakter |
|---|---|---|---|
| **Q-Learning** | Off-policy | `max Q(s',a')` | Keşifçi, optimal'i öğrenir |
| **SARSA** | On-policy | `Q(s',a')` (seçilen) | Temkinli, güvenli |

**Deep Learning**

| Mimari | Veri | Kilit fikir | Ana sorun/çözüm |
|---|---|---|---|
| **Perceptron** | Basit | Ağırlıklı toplam + step | XOR'u çözemez |
| **MLP** | Genel | Hidden layer + non-linear activation | Backprop + gradient descent ile eğitilir |
| **CNN** | Görüntü/video | Conv + pooling, weight sharing | Çok veri ve normalizasyon ister |
| **RNN** | Sıralı | Hidden state hafızası | Vanishing gradient → LSTM/GRU |
| **Transformer/LLM** | Metin | Self-attention, paralel işleme | Pahalı eğitim |

**Generative**

| Model | Mantık | Tipik sorun |
|---|---|---|
| **GAN** | Generator vs Discriminator | **Mode collapse** |
| **VAE** | Latent space'ten örnekleme | Detay/keskinlik |
| **Autoregressive** | Sırayla sonraki öğe | Yavaş, sıralı üretim |
| **Diffusion** | Gürültüyü tersine çevir | Hesaplama maliyeti |

---

## 25. Saldırı Yüzeyi Haritası

> Bu bölüm kaynak metinlerde yok, kendi eklemem. Amaç: öğrenilen kavramı, ileride HTB AI Red Teamer'da göreceğin saldırı alanına bağlamak. Detayları ilgili modül gelince doğrulayıp güncelle.

| Öğrenilen kavram | Neden saldırı yüzeyi? |
|---|---|
| **Training data kalitesi** (Bölüm 3) | Model veriden öğrenir, veri zehirlenirse (data poisoning) model yanlış kalıp öğrenir |
| **Overfitting / Generalization** | Ezberleyen model eğitim verisini sızdırabilir, görülmemiş girdide beklenmedik davranır |
| **Decision boundary** (Bölüm 5, 8) | Sınıra çok yakın girdi küçük değişiklikle yanlış sınıflandırılabilir (adversarial örnekler) |
| **Threshold / Base rate** (Bölüm 5, 7) | Anomaly/spam sistemlerinde eşik ve false positive dengesi kaçınma ve alarm yorgunluğu doğurur |
| **CNN, feature hiyerarşisi** (Bölüm 19) | Görüntüye insan gözünün fark etmediği pertürbasyonlar eklenip model yanıltılabilir |
| **Anomaly detection** (Bölüm 12) | Saldırgan, "normal" davranışı yavaş yavaş kaydırarak dedektörü kandırmaya çalışabilir |
| **LLM, prompt, tokenization** (Bölüm 22) | Model girdiyi talimat ve veri olarak ayırt etmekte zorlanır, prompt injection bu yüzden mümkün |
| **Text conditioning** (Bölüm 23) | Prompt, üretimi doğrudan yönlendirir, prompt manipülasyonu çıktıyı kontrol eder |
| **Q-table / reward** (Bölüm 13-15) | Ödül fonksiyonu yanlış tasarlanırsa agent ödülü istismar eden istenmeyen davranış öğrenir |
