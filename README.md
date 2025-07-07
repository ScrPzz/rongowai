Binary water/land classifier Based on Delay Doppler Maps from Rongowai NASA dataset.

Notebooks: 
#### 1. Data preprocessing and model prototyping: [eda, model_prototype]clf_binary_land_water.ipynb
#### 2. XGBoost fine tuning:[model_training]xgboost_finetune.ipynb
#### 3. Catboost fine tuning: [model_training]catboost_finetune.ipynb
#### 4. Voting classifier XGBoost + Catboost: [model_training]voting_classifier.ipynb
#### 5. Model performance testing: [model_test]all_models_deep_test.ipynb

### 1. Data Preprocessing and Model Prototyping with Pycaret
This notebook contains the code for preprocessing netCDF files and creating what will become the training dataset for the models.

The netCDF files are processed with the **NetCDFPreprocessor** class: 
Three modes are implemented:
- **Filtered**: applies filters on co-polar/cross-polar gain, SNR and satellite-surface distance
- **With_lat_lons**: includes geographic coordinates of specular points
- **Unfiltered**: removes only the SNR filter while maintaining the others

Included features:
- NetCDF file integrity checking
- Data masking and filtering
- Binary label extraction based on surface type
- Batch processing with stratified sampling

Features are created by the **DDMFeatureExtractor** class with these aspects:

- **Basic statistics**: mean, standard deviation, skewness, kurtosis, entropy, Gini coefficient
- **Positional features**: peak index, center of mass, moments of inertia
- **Spatial segmentation**: analysis by quadrants and central region of the DDM matrix
- **Temporal analysis**: first derivatives, autocorrelations, Fourier transform
- **Comparative features**: statistical differences between quadrants and center

Using the features obtained this way, the most promising models are researched using Pycaret.

### 2. CatBoost Pipeline for Binary Classification

Complete pipeline for binary classification with CatBoost and experiment tracking via MLflow.

#### General Logic

#### Development Phase
1. Loading balanced sample for tuning (25K samples per class)
2. Executing complete pipeline: preparation, tuning, training, evaluation
3. Saving development model and generating artifacts

#### Production Phase
1. Loading extended balanced dataset (250K samples per class)
2. Model finalization using the development model as base
3. Optional scaler retraining and early stopping
4. Feature importance comparison between development and production models
5. Artifact generation for deployment

### 3. XGBoost Binary Classification Pipeline
The same approach described for CatBoost is applied.

### 4. Voting Classifier (currently results don't justify the added complexity)
In this notebook, two pre-trained models (CatBoost and XGBoost) are loaded and a Voting classifier is created.

#### MaxProbVotingClassifier
Ensemble classifier that combines predictions from multiple models (CatBoost, XGBoost):
- **Selects the class with maximum probability among base models. In case of tie, CatBoost is favored**
- Supports differentiated scaling for each model
- Implements standard scikit-learn interface

#### Evaluation Pipeline

The system implements a complete evaluation pipeline that includes:

1. **Threshold optimization**: search for optimal threshold to maximize F1-score
2. **Robust validation**: testing on balanced dataset (50K samples) in addition to standard test set
3. **Comparative analysis**: performance comparison between different models and configurations
4. **Extended metrics**: accuracy, precision, recall, F1, AUC-ROC, specificity, NPV
5. **Visualizations**: ROC curves, precision-recall curves, confusion matrices, calibration curves

#### Output

- Optimized ensemble models saved in joblib/CatBoost format
- Optimal threshold configurations to maximize specific metrics
- Detailed performance reports with comparative visualizations
- Preprocessed datasets in Parquet format for future reuse

### 5. Model Performance Testing

In this notebook, n different test sets of chosen dimension are created and the model performance metrics are calculated on each one and then averaged.
