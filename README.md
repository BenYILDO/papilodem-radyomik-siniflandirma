# Radyomik Ozellikler ile Papilodem Siniflandirmasi

Yapay Zeka Dersi - Final Odevi (Secenek A). Yuksek boyutlu radyomik ozellikler
kullanilarak "Normal" ve "Papilodem" siniflarinin ayrilmasi icin veri-sizintisiz
(data-leakage-free) bir makine ogrenmesi pipeline'i.

**Ogrenci:** Muhammed Yildiray SURMEN  |  **No:** 254329041
**Uskudar Universitesi - Yapay Zeka Muhendisligi (Tezsiz Yuksek Lisans)**

## Klasor Yapisi (bu zip icinde, kendi kendine yeterli)
```
Surmen_254329041_YapayZeka_FinalOdevi/
├── data/                       # VERI SETLERI (zip icine dahildir)
│   ├── normal_radiomics.csv
│   ├── papilodem_radiomics.csv
│   └── radiomics_classification_workflow.png   # Ek Sekil 1
├── src/
│   ├── pipeline.py             # Tek calistirmada uctan uca pipeline
│   └── run_stage.py            # Asamali calistirici (opsiyonel)
├── notebook/
│   └── papilodem_pipeline.ipynb
├── figures/                    # ROC, PR, confusion matrix, feature importance, calibration, karsilastirma
├── results/                    # Metrik tablolari, secilen ozellikler, en iyi parametreler, istatistik testleri
├── report/
│   ├── Final_Rapor.pdf
│   └── Final_Rapor.docx
├── requirements.txt
└── README.md
```

## Kurulum
```bash
pip install -r requirements.txt
```

## Calistirma
Veri yollari klasor yapisina gore otomatik cozumlenir; ekstra ayar gerekmez.

**1) Tek dosyada tam pipeline:**
```bash
cd src
python pipeline.py          # data/ klasorunu otomatik bulur
```
Cikti: ust klasordeki `figures/` ve `results/` guncellenir.

**2) Jupyter Notebook:**
`notebook/papilodem_pipeline.ipynb` dosyasini acip hucreleri sirayla calistirin
(veri yolu `../data` olarak ayarlidir).

**3) Asamali calistirma (opsiyonel, sinirli sureli ortamlar icin):**
```bash
cd src
python run_stage.py prep
for m in LR SVM RF ET GB KNN; do python run_stage.py model:$m; done
python run_stage.py finalize
```

> Not: Veri yolunu degistirmek isterseniz `DATA_DIR` ortam degiskenini verebilirsiniz:
> `DATA_DIR=/baska/yol python pipeline.py`

## Pipeline Adimlari
1. Veri birlestirme + etiketleme (Normal=0, Papilodem=1)
2. Patient-level (hasta seviyesinde) train/test bolme (ayni hasta tek tarafta)
3. On isleme (YALNIZCA egitimde fit): median impute -> low-variance filtre ->
   korelasyon eleme (Pearson>0.95) -> RobustScaler
4. MRMR ozellik secimi (relevance=Mutual Information, redundancy=Pearson)
5. Optuna HPO (TPE, 50 trial, macro-F1) + StratifiedGroupKFold ic dogrulama
6. 6 model (LR, SVM, RF, ET, GB, KNN) + sigmoid kalibrasyon
7. Soft-voting ensemble (RF+ET+GB)
8. Test metrikleri + istatistiksel testler (Friedman, Wilcoxon, Bonferroni)
9. Grafikler

## Veri Sizintisini Onleme
- Tum on isleme ve olcekleme yalnizca egitim verisinde `fit` edilir.
- Ozellik secimi cross-validation icinde, her fold'un kendi egitim kismindan yapilir.
- Ayni hastanin goruntuleri asla farkli subsetlere dusmez (grup-bazli bolme).
- Test verisi yalnizca son degerlendirmede kullanilir.
