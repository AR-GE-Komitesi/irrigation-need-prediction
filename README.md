# Tarımsal Sulama İhtiyacının Tahminlenmesi

**Kaggle Playground Series — Season 6, Episode 4 (S6E4)** yarışması için geliştirilen makine öğrenimi çözümü. Toprak, iklim ve tarım pratiği verilerinden yola çıkarak sulama ihtiyacını üç sınıfta (**Low / Medium / High**) tahmin eder.

> **ISU AI 4 Social Good Kulübü** bünyesinde 3 kişilik takım projesi olarak geliştirilmiştir.
> Kaggle sonucu: **4.316 takım arasında 852. sıra (üst %19,7)**.

Yukarıdaki sıralama, yarışma sürecinde kullanılan ilk çözümle (LightGBM + XGBoost ensemble, "High" sınıfında %92 recall) elde edilmiştir. Bu depodaki notebook, yarışma sonrasında geliştirilen **v2** sürümüdür: ensemble'a CatBoost eklenmiş, değerlendirme balanced accuracy'ye taşınmış ve "High" sınıfı için ağırlıklandırma + olasılık eşiği optimizasyonu güçlendirilmiştir.

## Problem

Veri setinde her satır bir tarım parseline ait gözlemi temsil eder: toprak nemi, sıcaklık, yağış, nem, rüzgar hızı, toprak pH'ı gibi sayısal değişkenler ile toprak tipi, ürün tipi, büyüme evresi, mevsim, sulama tipi, su kaynağı, malçlama ve bölge gibi kategorik değişkenler. Hedef değişken `Irrigation_Need` ciddi biçimde dengesizdir: **Low %58,7 — Medium %38,0 — High %3,3**. Tarımsal açıdan en kritik sınıf olan "High" aynı zamanda en az örneğe sahip sınıftır; bu nedenle çözüm, dengesiz sınıf problemine odaklanan **balanced accuracy** metriği etrafında kurgulanmıştır.

## Yaklaşım

### 1. Veri Hazırlama

- Yarışma verisi ile orijinal kaynak veri seti (`cdeotte/s6e4-the-original-dataset`) birleştirilerek genişletilmiş bir eğitim seti oluşturuldu
- 8 ham kategorik kolon + üretilen etkileşim kolonları için `OrdinalEncoder` kullanıldı; görülmemiş kategorilere karşı `handle_unknown='use_encoded_value', unknown_value=-1` ile dayanıklılık sağlandı
- CatBoost için kategorik kolonlar **encode edilmeden ham haliyle** ayrı bir kopyada tutuldu ve modelin yerel kategorik desteğinden yararlanıldı
- Hedef değişken `LabelEncoder` ile sayısallaştırıldı

### 2. Dengesiz Sınıf Stratejisi

- `compute_class_weight('balanced')` ile sınıf ağırlıkları hesaplandı ve kritik **"High" sınıfının ağırlığı ek olarak 2 katına çıkarıldı**
- Bu ağırlıklar her üç modele de örnek ağırlığı (sample weight) olarak verildi
- Tahmin aşamasında "High" sınıfı için **olasılık eşiği optimizasyonu** yapıldı: OOF tahminler üzerinde 0.05–0.50 aralığı taranarak balanced accuracy'yi maksimize eden eşik bulundu; test tahminlerinde bu eşiği aşan örnekler doğrudan "High" olarak etiketlendi

### 3. Özellik Mühendisliği

Alan bilgisine dayalı 17 yeni özellik üretildi:

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
| `is_dry_soil` | Kuru toprak bayrağı (toprak nemi < 25) |
| `high_risk_flag` | Aktif büyüme + kuru toprak birleşik risk bayrağı |
| `no_mulching` | Malçlama yapılmamış parsel bayrağı |
| `rain_deficit` | Yağış açığı (2500 − yağış, alt sınır 0) |
| `moisture_stress` | Yağış açığı / (toprak nemi + 1) oranı |
| `evap_proxy` | Sıcaklık × güneşlenme / (nem + 1) buharlaşma proxy'si |
| `season_growth` | Mevsim × büyüme evresi kategorik etkileşimi |
| `source_type` | Su kaynağı × sulama tipi kategorik etkileşimi |
| `region_soil` | Bölge × toprak tipi kategorik etkileşimi |

### 4. Modelleme ve Değerlendirme

- **5-katlı StratifiedKFold** çapraz doğrulama, out-of-fold (OOF) tahminlerle **balanced accuracy** üzerinden değerlendirme
- Üç gradient boosting modeli, üçü de **GPU üzerinde** ve early stopping ile eğitildi:
  - **LightGBM** — Optuna ile bulunmuş hiperparametreler
  - **XGBoost** — `hist` + CUDA
  - **CatBoost** — yerel kategorik özellik desteği ve sınıf ağırlıkları ile
- Üç modelin olasılıkları, OOF balanced accuracy'yi maksimize eden ağırlıklar **grid search ile taranarak** ensemble (blend) edildi
- Ensemble çıktısına "High" eşik optimizasyonu uygulandıktan sonra sınıf bazlı classification report ile nihai OOF performansı raporlandı

## Depo İçeriği

```
.
├── README.md
└── prediction irrigation need v2.ipynb   # Veri hazırlama + özellik müh. + modelleme + submission (tek notebook)
```

## Çalıştırma

Notebook, Kaggle ortamında **GPU açık** olarak çalışacak şekilde yazılmıştır (veri yolları `/kaggle/input/...` altındadır, modeller `device='gpu'` / `device='cuda'` / `task_type='GPU'` ile eğitilir).

### Kaggle üzerinde (önerilen)

1. Notebook'u Kaggle'a yükleyin veya yeni bir Kaggle Notebook'a kopyalayın.
2. Sağ panelden **Accelerator** olarak GPU seçin.
3. **Add Input** ile şu iki veri kaynağını ekleyin:
   - `playground-series-s6e4` yarışma verisi (train.csv, test.csv, sample_submission.csv)
   - `cdeotte/s6e4-the-original-dataset` (orijinal kaynak veri seti)
4. Tüm hücreleri çalıştırın; sonunda `submission.csv` üretilir.

### Yerel ortamda

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost
```

Verileri [yarışma sayfasından](https://www.kaggle.com/competitions/playground-series-s6e4) indirdikten sonra notebook'un başındaki `/kaggle/input/...` yollarını kendi dizininize göre güncelleyip Jupyter ile çalıştırın. GPU yoksa parametrelerdeki `device` / `task_type` ayarlarını CPU'ya çevirmeniz gerekir (`device='cpu'`, `tree_method='hist'`, `task_type='CPU'`).

> Not: Genişletilmiş eğitim seti üzerinde 5-katlı CV ile üç ayrı gradient boosting modeli eğitildiği için tam koşum, donanıma bağlı olarak uzun sürebilir.

## Kullanılan Teknolojiler

Python, Pandas, NumPy, scikit-learn, LightGBM, XGBoost, CatBoost
