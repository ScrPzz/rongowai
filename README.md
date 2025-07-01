Classificatore binario acqua/terra.

#### 1. Data preprocessing e model prototyping: [eda, model_prototype]clf_binary_land_water.ipynb
#### 2. XGboost fine tuning:[model_training]xgboost_finetune.ipynb
#### 3. Catboost fine tuning: [model_training]catoost_finetune.ipynb
#### 4. Voting classifier XGBost + Catboost: [model_training]voting_classifier.ipynb

### 1. Preprocessamento dati e prototipazione modelli con Pycaret.
Questo notebook contiene il codice per preprocessare i file netcdf e creare quello che diventerà il dataset di addestramento per i modelli.

I file netcdf vengono processati con la classe **NetCDFPreprocessor**: 
Implementate tre modalità:
- **Filtered**: applica filtri su guadagno copolare/cross-polare, SNR e distanza satellite-superficie
- **With_lat_lons**: include coordinate geografiche dei punti speculari
- **Unfiltered**: rimuove solo il filtro SNR mantenendo gli altri

Sono inclusi:
- Controllo integrità file netCDF
- Mascheratura e filtraggio dei dati
- Estrazione di labels binarie basate sul tipo di superficie
- Processamento batch con campionamento stratificato

Le features vengono create dalla classe **DDMFeatureExtractor** con questi punti :

- **Statistiche base**: media, deviazione standard, skewness, kurtosis, entropia, coefficiente di Gini
- **Features posizionali**: indice del picco, centro di massa, momenti di inerzia
- **Segmentazione spaziale**: analisi per quadranti e regione centrale della matrice DDM
- **Analisi temporale**: derivate prime, autocorrelazioni, trasformata di Fourier
- **Features comparative**: differenze statistiche tra quadranti e centro

Usando le features così ottenute, usando Pycaret, si ricercano i modelli più promettenti.


### 2 Pipeline CatBoost per Classificazione Binaria

Pipeline completa per classificazione binaria con CatBoost e tracciamento esperimenti tramite MLflow.


### Logica generale

#### Fase Sviluppo
1. Caricamento campione bilanciato per tuning (25K campioni per classe)
2. Esecuzione pipeline completa: preparazione, tuning, training, valutazione
3. Salvataggio modello di sviluppo e generazione artifact

#### Fase Produzione
1. Caricamento dataset bilanciato esteso (250K campioni per classe)
2. Finalizzazione modello usando il modello di sviluppo come base
3. Riaddestramento scaler opzionale ed early stopping
4. Confronto feature importance tra modello sviluppo e produzione
5. Generazione artifact per deployment


### 3  XGBoost Binary Classification Pipeline
Si applica quanto detto per Catboost.


### 4. Voting Classifier (al momento i risultati non giustificano l'aggiunta di complessità)

## Architettura del Sistema


### Gestione Dati

**DataLoader**: Gestisce il caricamento efficiente di dati da file Parquet con:
- Caricamento completo o campionato
- Bilanciamento automatico delle classi
- Caching opzionale per ottimizzare la memoria
- Validazione dell'integrità dei dati

### MaxProbVotingClassifier
Classificatore ensemble che combina predizioni di modelli multipli (CatBoost, XGBoost):
-**Seleziona la classe con probabilità massima tra i modelli base, In caso di pareggio, privilegia CatBoost**
- Supporta scaling differenziato per ogni modello
- Implementa interfaccia scikit-learn standard

## Pipeline di Valutazione

Il sistema implementa una pipeline completa di valutazione che include:

1. **Ottimizzazione soglia**: ricerca della soglia ottimale per massimizzare F1-score
2. **Validazione robusta**: test su dataset bilanciato (50K samples) oltre al test set standard
3. **Analisi comparativa**: confronto performance tra diversi modelli e configurazioni
4. **Metriche estese**: accuracy, precision, recall, F1, AUC-ROC, specificity, NPV
5. **Visualizzazioni**: ROC curves, precision-recall curves, confusion matrices, curve di calibrazione

## Output

- Modelli ensemble ottimizzati salvati in formato joblib/CatBoost
- Configurazioni di soglie ottimali per massimizzare specifiche metriche
- Report dettagliati di performance con visualizzazioni comparative
- Dataset preprocessati in formato Parquet per riutilizzo futuro