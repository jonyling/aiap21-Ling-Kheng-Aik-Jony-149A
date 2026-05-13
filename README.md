# AIAP Batch 21 Technical Assessment

Candidate: Ling Kheng Aik Jony  
Task: Predict elderly residents' activity level from gas-monitoring and indoor environmental sensor data.

## Repository Structure

```text
.
├── data/
│   └── gas_monitoring.db
├── src/
│   ├── config.yaml
│   ├── data_preparation.py
│   └── model_training.py
├── eda.ipynb
├── main.py
├── requirements.txt
├── run.sh
└── README.md
```

`eda.ipynb` contains the exploratory analysis and model experiments. The Python scripts are the reproducible pipeline version of the final notebook workflow.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the SQLite database at:

```text
data/gas_monitoring.db
```

## Run

```bash
python main.py
```

or:

```bash
bash run.sh
```

The run creates:

- `data/processed/preprocessed_data.csv`
- trained model pipelines in `models/*.pkl`
- model comparison tables and feature-importance/confusion-matrix artifacts in `models/`

## Pipeline Summary

The final pipeline follows the notebook's post-EDA modeling design:

1. Load `gas_monitoring` from SQLite.
2. Drop duplicate rows.
3. Clean inconsistent labels in `Activity Level`, `HVAC Operation Mode`, `Time of Day`, and `CO_GasSensor`.
4. Correct sensor anomalies:
   - Convert Kelvin-like temperature readings to Celsius.
   - Repair impossible humidity values.
   - Add a 450 ppm offset to `CO2_InfraredSensor`.
   - Median-impute `CO2_ElectroChemicalSensor`.
   - KNN-impute `MetalOxideSensor_Unit3` using the other metal-oxide sensors.
5. Engineer EDA-supported features:
   - `Temperature_x_Humidity`
   - `co2_ratio`
   - `avg_metal_oxide`
   - `Temperature_Band`
   - `is_daytime`
   - session-grouped lag and rolling features
   - previous HVAC mode and time since last HVAC mode change within a session
6. Drop rows that cannot support the selected lag/rolling features.
7. Train SMOTE-balanced classification pipelines.
8. Save fitted pipelines and evaluation artifacts.

## Feature Policy

The final model deliberately excludes raw `Session ID` from the main row-split model because the notebook found it can behave like an identifier/proxy. The script keeps a group-based generalization check by `Session ID` to estimate performance on unseen sessions more honestly.

The main selected features are:

- Numeric: `Temperature`, `Temperature_x_Humidity`, `co2_ratio`, `avg_metal_oxide`, selected session lag/rolling features, and `time_since_last_change_session`
- Ordinal: `Time of Day`, `CO_GasSensor`, `is_daytime`, `Temperature_Band`
- Nominal: `HVAC_prev_session`

`Ambient Light Level` is not used in the final model because the notebook did not find meaningful target signal.

## Models

The training script compares the models used in the final notebook baseline:

- Random Forest
- Decision Tree
- Logistic Regression
- XGBoost
- LightGBM
- CatBoost

Each model is wrapped in an imbalanced-learn pipeline:

```text
preprocessing -> SMOTE -> model
```

This keeps resampling inside the training fold only and leaves the test distribution untouched.

## Evaluation

The primary metric is macro F1, because the activity-level classes are imbalanced and minority-class performance matters. The scripts also report accuracy, weighted F1, classification reports, confusion matrices, and feature importance where supported.

For generalization risk, the pipeline also runs:

- `GroupShuffleSplit` by `Session ID`
- `GroupKFold` by `Session ID`

These are saved as `models/group_split_result.csv` and `models/group_cv_results.csv`.

## Configuration

Edit `src/config.yaml` to adjust:

- input/output paths
- train/test split and random seed
- SMOTE settings
- feature prefixes used for session time-series features
- model hyperparameters
- whether to run the group generalization check
