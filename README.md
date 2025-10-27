# aiap21-Ling-Kheng-Aik-Jony-149A
Contains the deliverables for the AIAP Batch 21 Technical Assessment, addressing both Task 1 (Exploratory Data Analysis) and Task 2 (End-to-end Machine Learning Pipeline). 

# AIAP_Batch21_TA

AIAP Batch 21 Technical Assessment Submission

Candidate Information
Full Name: Ling Kheng Aik Jony
Email: jonyling@hotmail.com

Overview
This repository contains the deliverables for the AIAP Batch 21 Technical Assessment, addressing both Task 1 (Exploratory Data Analysis) and Task 2 (End-to-end Machine Learning Pipeline). The goal is to predict elderly residents' activity levels using environmental sensor data to support ElderGuard Analytics' non-invasive early warning system.

Repository Structure

aiap21-Ling-Kheng-Aik-Jony-149A
├── .github/                    # GitHub Actions scripts (provided in template)
├── src/                        # Python modules for ML pipeline
│   ├── data_preparation.py     # Data loading and preprocessing
│   ├── model_training.py       # Model training and evaluation
│   └── config.yaml             # Configuration file for pipeline parameters
├── data/                       # Data folder (not uploaded, contains gas_monitoring.db)
├── models/                     # Trained model files (generated during execution)
├── eda.ipynb                   # Jupyter notebook for Task 1 (EDA)
├── requirements.txt            # Python dependencies
├── run.sh                      # Bash script to execute the pipeline
└── README.md                   # This file

## Table of Contents
- [Pipeline Execution Instructions](##PipelineExecutionInstructions)
- [PipelineLogicalFlow](#PipelineLogicalFlow)
- [Key_EDA_Findings_and_Pipeline_Choices](#Key_EDA_Findings_and_Pipeline_Choices)
- [Feature_Processing](#Feature_Processing)
- [Model_Selection_and_Justification](#Model_Selection_and_Justification)
- [Model_Evaluation](#Model_Evaluation)
- [Deployment_Considerations](#Deployment_Considerations)
- [Configuration](#configuration)
- [MLmodel_results](#MLmodel_results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)


## PipelineExecutionInstructions
Setup:
- Ensure Python 3.11+ is installed.
- Place gas_monitoring.db in the data/ folder (relative path: data/gas_monitoring.db).
- Install dependencies: pip install -r requirements.txt.
- Run the Pipeline: github-actions.yaml. 
- Execute the bash script: bash run.sh: This runs data_preparation.py followed by model_training.py.
- Modify Parameters: Edit src/config.yaml to adjust model hyperparameters, feature selections, or preprocessing steps. Example: Change max_depth for DecisionTree or n_estimators for RandomForest.

## PipelineLogicalFlow
- Data Loading: Load data from data/gas_monitoring.db using SQLite.
- Data Preprocessing:
    - Drop duplicates. 
    - Handle missing values (imputation for numerical, mode for categorical).
    - Remove outliers using IQR method.
- Feature engineering: 
    - Create Temperature_x_Humidity, CO2_sum, MetalOxideSensor_sum, and Session_ID_Bands.
    - Encode categorical variables (one-hot for nominal, ordinal for ordered categories).
- Split Data:
    - Split data with 80/20 stratified proportion for training and test sets.
- Build Pipelines with Resampling
    - Numerical features = {'Temperature_x_Humidity', 'CO2_sum', 'MetalOxideSensor_sum'}
    - Ordinal features = {'Time of Day', 'Session_ID_Bands', 'CO_GasSensor'}
    - Nomimal feature = 'HVAC Operation Mode'
    - Resampling:
        - SMOTE for 'High Activity'
        - Undersample for 'Low Activity'
- Model Training: 
    - Train four models: DecisionTree, RandomForest, LogisticRegression, and XGBoost.
- Evaluation:
    - Evaluate models using accuracy, macro F1, weighted F1, and confusion matrix.
    - Perform hyperparameter tuning via RandomizedSearchCV.
    - Save trained models to models/ directory.
- Feature Importance:
    - Analyze feature importance for tree-based models and coefficients for LogisticRegression.
- Visualization: Below is a simplified flowchart of the pipeline:
    graph TD
        A[Load Data from SQLite] --> B[Preprocess Data]
        B --> C[Feature Engineering]
        C --> D[Split Data]
        D --> E[Build Pipelines with Resampling]
        E --> F[Train Models]
        F --> G[Evaluate Models]
        G --> H[Analyze Feature Importance]
        H --> I[Save Models]

## Key_EDA_Findings_and_Pipeline_Choices
- Data Quality: Missing values in CO2_ElectroChemicalSensor (706), MetalOxideSensor_Unit3 (2566), CO_GasSensor (1369), and Ambient Light Level (2532) were handled via imputation. 
- Feature Engineering:
    - Temperature_x_Humidity: Product of temperature and humidity to capture interaction effects.
    - CO2_sum: Sum of CO2_InfraredSensor and CO2_ElectroChemicalSensor for a combined CO2 metric.
    - MetalOxideSensor_sum: Sum of all MetalOxideSensor units to aggregate sensor readings.
    - Session_ID_Bands: Binned Session ID to capture temporal patterns.
- Key Features: Temperature_x_Humidity and MetalOxideSensor_sum showed strong correlations with Activity Level (Spearman correlation > 0.3).
    - Implications: Features capturing environmental conditions (Temperature_x_Humidity) and total volatile organic carbons (MetalOxideSensor_sum) are critical for predicting activity levels, influencing model selection and preprocessing.

## Feature_Processing
| Feature | Processing Method | Description|
|----------|----------|----------|
|Time of Day | Ordered into {morning, afternoon, evening, night}.  | Time of Day ordered into logical flow. Spearman_corr indicates there is some correlation, thus this feature is used. |
|Temperature | Impossible temperatures exceeding 288 degC assummed to be in Kelvin, converted to Celsius by deducting 273.15 | Environmental temperature (°C) |
|Humidity | Impossible humidities that are negative, abs() less than 40% and/or more than 100% are suitably converted to appropriate values (abs(), +30, /10) | Relative humidity (%) |
|Temperature_x_Humidity | New feature is done calculated by Temperature x Humidity | Derivate feature to capture both environmental factors (Temperature & Humidity). Spearman_corr indicates there is strong correlation.|
|CO2_InfraredSensor | By comparing with CO2_ElectroChemicalSensor, and by general knowledge, the readings are off by 450 ppm -> Shifted by +450 | CO2 in the environmental air by Infrared |
|CO2_ElectroChemicalSensor | Impute (median) | CO2 in the environmental air by ElectroChemicalSensor |
|CO2_sum | New feature is done calculated by CO2_InfraredSensor + CO2_ElectroChemicalSensor | CO2 in the environmental air by both methods. Spearman_corr indicates there is some correlation, thus this feature is used. |
|MetalOxideSensor_Unit[1-2] | None, histology looks fine | Total Volatile Organic Carbon in the environmental air by both units |
|MetalOxideSensor_Unit3 | I have KNN-imputed the missing 2566 datapoints by the other 3 MetalOxideSensor_Units because they are highly correlated | Total Volatile Organic Carbon in the environmental air by MetalOxideSensor_Unit3 |
|MetalOxideSensor_Unit4 | None, histology looks fine | Total Volatile Organic Carbon in the environmental air by the 4th unit. |
|MetalOxideSensor_sum | None, histology looks fine | Total Volatile Organic Carbon in the environmental air by summing all 4 units.  Spearman_corr indicates there is strong correlation.|
|CO_GasSensor | Order into {extremely low, low, moderate, high, extremely high} | Carbon monoxide in the environmental air. Spearman_corr indicates there is some correlation, thus this feature is used.|
|Session ID | Order into {1000-2000, 2000-3000, 3000-4000, 4000-5000, 5000-6000, 6000-7000, 7000-8000, 8000-9000, 9000-10000} | Supposed to be just an identifier, but Spearman says that there is some correlation with Activity Level|
|HVAC Operation Mode| The entries were input haphazardly; sorted out. |Heating, Ventilation, Air Conditioning machinery affecting the indoor environment. Spearman_corr indicates there is some correlation|
| Ambient Light Level | There were 2519 missing datapoints, they were filled up witht the 5 levels randomly.  | Light intensity; Spearman correlation shows almost no correlation with Activity Level, dropped. |

## Model_Selection_and_Justification
- DecisionTree: Simple, interpretable, handles non-linear relationships. Used as a baseline.
- RandomForest: Ensemble method, robust to overfitting, effective for imbalanced data with class weights.
- LogisticRegression: Linear model, interpretable coefficients, suitable for multi-class problems with multinomial setting.
- XGBoost: Gradient boosting, high performance on tabular data, handles imbalanced classes well.
- Rationale: These models balance interpretability, performance, and robustness to class imbalance, as observed in EDA (Low Activity: 5231, Moderate Activity: 3092, High Activity: 1677).
- Why not KNN? It is not chosen due to the high dimensionality of the data and the computationally expensive nature of the method (would take too long to compute). 
- Why not AUC-PR? It is primarily designed for binary classification problems and is most effective when evaluating the performance of a positive class (e.g., detecting "High Activity" as a health emergency) against all others. It also focuses on the trade-off between precision and recall for the (one) positive class, potentially overlooking the overall model performance across all classes (Low, Moderate, High activity).

## Model_Evaluation
- Metrics:
    - Accuracy: Overall correctness of predictions.
    - Macro F1: Unweighted average F1 score across classes, suitable for imbalanced data.
    - Weighted F1: Weighted average F1 score, accounts for class distribution.
    - Confusion Matrix: Detailed breakdown of classification performance.
- Justification: 
    - Macro F1 prioritizes performance across all activity levels equally, critical for detecting rare High Activity cases. 
    - Weighted F1 reflects real-world class distribution.
- Results: RandomForest and XGBoost typically outperform others due to ensemble nature, with macro F1 scores around 0.75–0.85 (based on eda.ipynb results). 

## Deployment_Considerations
- Scalability: Pipeline uses modular Python scripts, allowing easy integration into production systems.
- Real-time Processing: SQLite queries can be optimized for streaming data if needed.
- Model Updating: Retrain models periodically using saved pipelines in models/ directory.
- Monitoring: Log model predictions and feature distributions to detect data drift.
- Interpretability: Feature importance and coefficient analysis aid in explaining predictions to stakeholders.

## Configuration
The project uses a config.yaml file for configuration. Key parameters include:
    db_path: "data/gas_monitoring.db"
    numerical_features: Temperature_x_Humidity, CO2_sum, MetalOxideSensor_sum
    nominal_features: HVAC Operation Mode
    ordinal_features: Time of Day, Session_ID_Bands, CO_GasSensor
    param_grid: Hyperparameter grid for classification algorithm using RandomizedSearchCV.
Edit config.yaml to customize the pipeline for your dataset.

## MLmodel_results
- Decision Tree: 
    - Initial model:
        - Performs best on "Low" (F1 0.69) due to higher support, with moderate recall (0.63) but some false positives (176 to "Moderate," 110 to "High").
        "Moderate" has balanced but low performance (F1 0.46), with 117 false positives to "Low" and 97 to "High."
        - "High" has the lowest precision (0.22) and F1 (0.28), with 62 false positives to "Moderate," reflecting difficulty in distinguishing this minority class.
        - Accuracy (0.55) is decent but skewed by the majority class ("Low"). Macro F1 (0.48) shows poor average performance across classes, while weighted F1 (0.57) reflects the imbalance.
    - Tuned model: 
        - Tuning slightly reduces accuracy (0.5277) and F1 scores across classes, with "Low" F1 dropping to 0.65 due to lower recall (0.58) and more misclassifications (201 to "Moderate," 121 to "High"). 
        - "Moderate" remains stable (F1 0.46), with 115 false positives to "Low." 
        - "High" F1 drops to 0.26, with recall decreasing to 0.34, indicating tuning didn’t improve minority class performance. 
        The model may be overfitting with max_depth=None, reducing generalization. Macro F1 (0.46) and weighted F1 (0.55) suggest tuning didn’t enhance overall balance.
    - Conclusion: Decision Tree is not a very good model to use here, due to the imbalanced classes, and tuning does not improve the performance much. 
- Random Forest:
    - Initial model:
        - Outperforms DecisionTree with "Low" F1 at 0.75 (recall 0.70), though 159 misclassifications to "Moderate." 
        - "Moderate" improves to F1 0.54, with 92 false positives to "Low," showing better separation.
        - "High" remains weak (F1 0.31), with 69 false positives to "Moderate," reflecting imbalance challenges. 
        - Accuracy (0.62) and macro F1 (0.53) indicate solid performance, with weighted F1 (0.64) reflecting "Low" dominance.
    - Tuned model: 
        - Tuning improves accuracy (0.6304) and F1 scores: "Low" F1 rises to 0.76, "Moderate" to 0.56, and "High" holds at 0.31. 
        - "Moderate" recall increases to 0.62, with fewer false positives to "Low" (77), showing better class separation. 
        - "High" remains the weakest, with 78 false positives to "Moderate." 
        - Macro F1 (0.54) and weighted F1 (0.64) improve slightly, suggesting tuning enhances ensemble performance, likely due to more trees (200).
    - Conclusion: Random Forest  is a stronger model to use compared to Decision Tree, with a solid performance in the initial model and improved performance for the tuned model. 
- Logistic Regression
    - Initial model:
        - "Low" performs well (F1 0.73), with 135 misclassifications to "Moderate" and 109 to "High." 
        - "Moderate" is weak (F1 0.46), with 99 false positives to "Low" and 128 to "High," indicating poor separation. 
        - "High" has low precision (0.20) but decent recall (0.38), with 59 false positives to "Moderate." 
        - Accuracy (0.57) and macro F1 (0.49) are moderate, with weighted F1 (0.60) reflecting imbalance.
    - Tuned model:
        - Tuning slightly improves "Low" F1 to 0.74 (recall 0.69), with similar misclassifications. 
        - "Moderate" F1 stays at 0.47, with 104 false positives to "Low," showing minimal improvement. 
        - "High" remains weak (F1 0.26), with 60 false positives to "Moderate." 
        - Accuracy (0.58) and weighted F1 (0.60) improve marginally, but macro F1 (0.49) suggests tuning with a lower C (0.01) didn’t significantly enhance balance.
    - Conclusion:
        - A slightly stronger model to use relative to Decision Tree, but still not much performance improved even after tuning. 
- XGBoost:
    - Initial Model: 
        - "Low" F1 is 0.74, with 169 misclassifications to "Moderate" and 75 to "High."
        - "Moderate" F1 is 0.54, with 92 false positives to "Low," showing reasonable separation.
        - "High" F1 is 0.29, with 66 false positives to "Moderate," indicating difficulty with the minority class.
        - Accuracy (0.61) and macro F1 (0.52) are strong, with weighted F1 (0.62) reflecting imbalance.
    - Tuned model:
        - Tuning improves "Low" recall to 0.70 (F1 0.74), with 168 misclassifications to "Moderate."
        - "Moderate" F1 rises to 0.52, with 108 false positives to "Low," showing slight improvement. 
        - "High" F1 increases to 0.33, with 63 false positives to "Moderate," indicating better minority class handling.
        - Accuracy (0.61) and macro F1 (0.53) improve, with weighted F1 (0.62) reflecting balanced gains, likely due to more trees (200) and controlled depth (5).
    - Conclusion: A rather solid model in terms of performace, it is comparable to Random Forest. 
- Overal Conclusion:
    - Best Overall Model: Random Forest (tuned) with accuracy 0.6304 and macro F1 0.5436 outperforms others, showing robust ensemble performance across classes. Second best is XGBoost. 
    - Minority Class ("High") Challenge: All models struggle with "High" (F1 0.26–0.33), due to low support (157) and imbalance. Tuning helps slightly (e.g., XGBoost F1 0.33), but more aggressive resampling (e.g., adjusting SMOTE/undersampling ratios in config.yaml) could improve this.
    - Trade-offs: DecisionTree and LogisticRegression show stability but limited gains from tuning, while RandomForest and XGBoost benefit from ensemble adjustments (more trees, controlled depth).
- Feature Importance:
    - Random Forest: It relies heavily on environmental sensor data (MetalOxideSensor_sum, Temperature_x_Humidity, CO2_sum), with temporal and operational features playing supporting roles. The tuned model (200 trees, no max depth) likely amplifies these contributions across its ensemble.
    - XGBoost: It balances environmental sensors (MetalOxideSensor_sum, CO_GasSensor) with temporal (Time of Day) and HVAC operational features, reflecting its ability to model complex interactions (tuned with 200 estimators, max_depth=5, subsample=0.8).
    - Comparison: 
        - Both models prioritize MetalOxideSensor_sum, Temperature_x_Humidity, and CO2_sum, confirming these environmental factors are fundamental to activity level prediction.
        - RandomForest emphasizes MetalOxideSensor_sum (0.278535) more, while XGBoost spreads importance across CO_GasSensor (0.102012) and HVAC modes, reflecting its gradient boosting approach.
        - RandomForest’s higher accuracy (0.6304) aligns with its focus on MetalOxideSensor_sum, which may better capture the majority class ("Low"). XGBoost’s balanced feature use (macro F1 0.5294) supports its slight edge in minority class performance ("High" F1 0.33). 
        - Feature Engineering: Enhancing MetalOxideSensor_sum and CO_GasSensor data (e.g., adding derived features like trends) could boost both models. HVAC mode interactions might benefit XGBoost more.
        - Model Selection: RandomForest’s feature importance suggests it’s robust for the current feature set, while XGBoost’s broader focus could be leveraged with additional HVAC or temporal data.

## Contributing
We welcome contributions! To contribute:
    Fork the repository.
    Create a new branch (git checkout -b feature/your-feature-name).
    Commit your changes (git commit -m 'Add your feature').
    Push to the branch (git push origin feature/your-feature-name).
    Open a Pull Request.
Please follow our Code of Conduct (CODE_OF_CONDUCT.md) and ensure code adheres to PEP 8 style guidelines.

## License
This project is licensed under the MIT License (LICENSE).

## Contact 
For questions or feedback, reach out to:
    Email: jonyling@hotmail.com
    GitHub: jonyling
    X: @JonyLing1

This README is concise yet informative, tailored to your project's structure. Adjust the repository URL, contact details, and dataset path as needed. Let me know if you'd like to expand any section!
