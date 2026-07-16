# Tarımsal Sulama İhtiyacının Tahminlenmesi

**Kaggle Playground Series — Season 6, Episode 4 (S6E4)** yarışması için geliştirilen makine öğrenimi çözümü. Toprak, iklim ve tarım pratiği verilerinden yola çıkarak sulama ihtiyacını üç sınıfta (**Low / Medium / High**) tahmin eder.

> **ISU AI 4 Social Good Kulübü** bünyesinde 3 kişilik takım projesi olarak geliştirilmiştir.
> Kaggle sonucu: **4.316 takım arasında 852. sıra (üst %19,7)**.

## Problem

Veri setinde her satır bir tarım parseline ait gözlemi temsil eder: toprak nemi, sıcaklık, yağış, nem, rüzgar hızı, toprak pH'ı gibi sayısal değişkenler ile toprak tipi, ürün tipi, büyüme evresi, mevsim, sulama tipi, su kaynağı, malçlama ve bölge gibi kategorik değişkenler. Hedef değişken `Irrigation_Need` ciddi biçimde dengesizdir: **Low %58,7 — Medium %38,0 — High %3,3**. Tarımsal açıdan en kritik sınıf olan "High" aynı zamanda en az örneğe sahip sınıftır; bu da problemi standart bir çok sınıflı sınıflandırmadan daha zorlu hale getirir.

## Yaklaşım

### 1. Keşifsel Veri Analizi (EDA)

- Sayısal değişkenlerde IQR tabanlı aykırı değer taraması ve eksik değer raporu
- Kategorik değişkenlerin hedefle ilişkisi için **Cramér's V** analizi
- Sınıf bazlı boxplot, korelasyon ısı haritaları ve pairplot ile dağılım incelemesi
- "High" sınıfının içindeki anomalilerin ayrıştırılması (kuru toprak kaynaklı "normal High" ile düşük sıcaklık + yüksek nem koşullarındaki "biyolojik anomali" ayrımı; büyüme evresi bazlı lift analizi)

### 2. Veri Hazırlama

- Yarışma verisi (630 bin satır) ile orijinal kaynak veri seti (10 bin satır) birleştirilerek **640 bin satır × 21 kolonluk** eğitim seti oluşturuldu
- 8 kategorik kolon için `OrdinalEncoder`, **veri sızıntısını (data leakage) önlemek amacıyla** scikit-learn `Pipeline` içinde her çapraz doğrulama katmanında ayrı ayrı fit edildi
- Görülmemiş kategorilere karşı `handle_unknown='use_encoded_value', unknown_value=-1` ile dayanıklılık sağlandı

### 3. Özellik Mühendisliği

Alan bilgisine dayalı 8 yeni özellik üretildi:

| Özellik | Tanım |
|---|---|
| `Is_Active_Growth` | Aktif büyüme evresi bayrağı (Flowering / Vegetative) |
| `Mulching_x_ActiveGrowth` | Malçsız + aktif büyüme etkileşimi |
| `Evap_Stress` | Buharlaşma baskısı proxy'si (sıcaklık − nem − yağış/10) |
| `Water_Deficit` | Düşük önceki sulama + düşük toprak nemi bayrağı |
| `Heat_Load` | Sıcaklık × güneşlenme saati |
| `Soil_Fertility` | Toprak nemi × organik karbon |
| `Wind_Evap_Index` | Rüzgar hızı × (100 − nem) / 100 |
| `Rainfall_Cat` | Yağış miktarının kategorik binlenmesi |

### 4. Modelleme ve Değerlendirme

- **5-katlı StratifiedKFold** çapraz doğrulama, out-of-fold (OOF) tahminlerle değerlendirme
- Modeller: **LightGBM**, **XGBoost** (early stopping ile), **CatBoost** (yerel kategorik destek) ve **Optuna** ile hiperparametre araması yapılmış LightGBM (50 trial)
- LightGBM ve XGBoost olasılıkları OOF doğruluğuna göre ağırlıklı **ensemble (blend)** ile birleştirildi
- Sınıf bazlı classification report, confusion matrix ve özellik önem grafikleriyle analiz; verinin yalnızca %3,3'ünü oluşturan kritik **"High" sınıfında %92 recall** elde edildi

## Depo İçeriği

```
.
└── ai4sg-playground-series-s6e4.ipynb   # EDA + veri hazırlama + modelleme + submission (tek notebook)
```

## Çalıştırma

Notebook, Kaggle ortamında çalışacak şekilde yazılmıştır (veri yolları `/kaggle/input/...` altındadır).

### Kaggle üzerinde (önerilen)

1. Notebook'u Kaggle'a yükleyin veya yeni bir Kaggle Notebook'a kopyalayın.
2. Sağ panelden **Add Input** ile şu iki veri kaynağını ekleyin:
   - `playground-series-s6e4` yarışma verisi (train.csv, test.csv, sample_submission.csv)
   - `cdeotte/s6e4-the-original-dataset` (orijinal kaynak veri seti)
3. Tüm hücreleri çalıştırın; sonunda `submission.csv` üretilir.

### Yerel ortamda

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna matplotlib seaborn scipy jupyter
```

Verileri [yarışma sayfasından](https://www.kaggle.com/competitions/playground-series-s6e4) indirdikten sonra notebook'un başındaki `/kaggle/input/...` yollarını kendi dizininize göre güncelleyip Jupyter ile çalıştırın.

> Not: 640 bin satırlık eğitim seti üzerinde 5-katlı CV ile üç ayrı gradient boosting modeli eğitildiği için tam koşum, donanıma bağlı olarak uzun sürebilir.

## Kullanılan Teknolojiler

Python, Pandas, NumPy, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna, Matplotlib, Seaborn, SciPy
