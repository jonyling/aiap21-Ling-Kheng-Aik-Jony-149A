"""Model training pipeline aligned with eda.ipynb."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "src" / "config.yaml"
ACTIVITY_LEVEL_ORDER = ["Low Activity", "Moderate Activity", "High Activity"]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_processed_data(config: dict[str, Any]) -> pd.DataFrame:
    path = PROJECT_ROOT / config.get("processed_data_path", "data/processed/preprocessed_data.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run `python src/data_preparation.py` first."
        )
    return pd.read_csv(path)


def select_features(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[str], list[str]]:
    current_numeric_features = config.get(
        "current_numeric_features",
        ["Temperature", "Temperature_x_Humidity", "co2_ratio", "avg_metal_oxide"],
    )
    important_ts_prefixes = tuple(
        config.get(
            "important_ts_prefixes",
            [
                "Temperature_",
                "CO2_InfraredSensor_",
                "MetalOxideSensor_Unit1_",
                "MetalOxideSensor_Unit2_",
                "avg_metal_oxide_",
                "co2_ratio_",
            ],
        )
    )
    session_ts_numeric_features = [
        col
        for col in df.columns
        if ("_session_lag" in col or "_session_roll_mean_" in col or "_session_roll_std_" in col)
        and col.startswith(important_ts_prefixes)
    ]
    selected_numeric_features = current_numeric_features + session_ts_numeric_features + [
        "time_since_last_change_session"
    ]
    selected_ordinal_features = ["Time of Day", "CO_GasSensor", "is_daytime", "Temperature_Band"]
    selected_nominal_features = ["HVAC_prev_session"]

    numerical_features = [col for col in selected_numeric_features if col in df.columns]
    ordinal_features = [col for col in selected_ordinal_features if col in df.columns]
    nominal_features = [col for col in selected_nominal_features if col in df.columns]
    selected_features = numerical_features + ordinal_features + nominal_features

    X = df[selected_features].copy()
    y = df["Activity Level"].astype(str)
    groups = df["Session ID"]

    return X, y, groups, numerical_features, ordinal_features, nominal_features


def build_preprocessor(
    numerical_features: list[str],
    ordinal_features: list[str],
    nominal_features: list[str],
) -> ColumnTransformer:
    ordinal_categories_lookup = {
        "Time of Day": ["morning", "afternoon", "evening", "night"],
        "Temperature_Band": ["14-16", "16-18", "18-20", "20-22", "22-24", "24-26"],
        "CO_GasSensor": ["extremely low", "low", "medium", "high", "extremely high"],
        "is_daytime": [0, 1],
    }
    ordinal_categories = [ordinal_categories_lookup[col] for col in ordinal_features]

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    nominal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("ord", ordinal_transformer, ordinal_features),
            ("cat", nominal_transformer, nominal_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(config: dict[str, Any]) -> dict[str, Any]:
    model_params = config.get("models", {})
    return {
        "RandomForest": RandomForestClassifier(
            **({"n_estimators": 300, "random_state": 42, "n_jobs": -1} | model_params.get("RandomForest", {}))
        ),
        "DecisionTree": DecisionTreeClassifier(
            **({"max_depth": 8, "min_samples_leaf": 25, "random_state": 42} | model_params.get("DecisionTree", {}))
        ),
        "LogisticRegression": LogisticRegression(
            **({"solver": "lbfgs", "max_iter": 2000, "random_state": 42} | model_params.get("LogisticRegression", {}))
        ),
        "XGBoost": XGBClassifier(
            **(
                {
                    "objective": "multi:softprob",
                    "num_class": len(ACTIVITY_LEVEL_ORDER),
                    "n_estimators": 300,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "eval_metric": "mlogloss",
                    "random_state": 42,
                    "n_jobs": -1,
                }
                | model_params.get("XGBoost", {})
            )
        ),
        "LightGBM": LGBMClassifier(
            **(
                {
                    "objective": "multiclass",
                    "num_class": len(ACTIVITY_LEVEL_ORDER),
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbose": -1,
                }
                | model_params.get("LightGBM", {})
            )
        ),
        "CatBoost": CatBoostClassifier(
            **(
                {
                    "iterations": 300,
                    "depth": 4,
                    "learning_rate": 0.05,
                    "loss_function": "MultiClass",
                    "random_seed": 42,
                    "verbose": False,
                }
                | model_params.get("CatBoost", {})
            )
        ),
    }


def build_pipelines(preprocessor: ColumnTransformer, config: dict[str, Any]) -> dict[str, ImbPipeline]:
    smote = SMOTE(
        sampling_strategy=config.get("smote_sampling_strategy", "not majority"),
        random_state=config.get("random_state", 42),
        k_neighbors=config.get("smote_k_neighbors", 5),
    )
    return {
        name: ImbPipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("smote", clone(smote)),
                ("model", model),
            ]
        )
        for name, model in build_models(config).items()
    }


def predict_labels(model_name: str, pipeline: ImbPipeline, X_test: pd.DataFrame, label_inverse: dict[int, str]) -> np.ndarray:
    predictions = pipeline.predict(X_test)
    predictions = np.ravel(predictions)
    if model_name == "XGBoost":
        return pd.Series(predictions).map(label_inverse).to_numpy()
    return predictions


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, model_name: str, output_dir: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=ACTIVITY_LEVEL_ORDER)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=ACTIVITY_LEVEL_ORDER,
        yticklabels=ACTIVITY_LEVEL_ORDER,
    )
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_confusion_matrix.png")
    plt.close()


def save_feature_importance(pipeline: ImbPipeline, model_name: str, output_dir: Path, top_n: int = 20) -> None:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "get_feature_importance"):
        importances = model.get_feature_importance()
    elif hasattr(model, "coef_"):
        importances = np.mean(np.abs(model.coef_), axis=0)
    else:
        return

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    metric_name = "mean_abs_coefficient" if hasattr(model, "coef_") else "importance"
    importance_df = (
        pd.DataFrame({"feature": feature_names, metric_name: importances})
        .sort_values(metric_name, ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(output_dir / f"{model_name}_feature_importance.csv", index=False)

    plt.figure(figsize=(12, max(6, top_n * 0.35)))
    sns.barplot(data=importance_df.head(top_n), x=metric_name, y="feature")
    plt.title(f"Top {top_n} Features: {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_feature_importance.png")
    plt.close()


def evaluate_pipelines(
    pipelines: dict[str, ImbPipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
) -> pd.DataFrame:
    activity_label_map = {label: idx for idx, label in enumerate(ACTIVITY_LEVEL_ORDER)}
    activity_label_inverse = {idx: label for label, idx in activity_label_map.items()}
    y_train_encoded = y_train.map(activity_label_map)

    results: list[dict[str, Any]] = []
    for model_name, pipeline in pipelines.items():
        print(f"\nTraining {model_name} + SMOTE")
        fit_target = y_train_encoded if model_name == "XGBoost" else y_train
        pipeline.fit(X_train, fit_target)
        y_pred = predict_labels(model_name, pipeline, X_test, activity_label_inverse)

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(classification_report(y_test, y_pred, labels=ACTIVITY_LEVEL_ORDER, zero_division=0))

        joblib.dump(pipeline, output_dir / f"{model_name}.pkl")
        save_confusion_matrix(y_test, y_pred, model_name, output_dir)
        save_feature_importance(pipeline, model_name, output_dir)
        results.append(
            {
                "model": f"{model_name} + SMOTE",
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            }
        )

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    results_df.to_csv(output_dir / "model_results.csv", index=False)
    print("\nModel comparison sorted by Macro F1:")
    print(results_df.to_string(index=False))
    return results_df


def group_generalization_check(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    preprocessor: ColumnTransformer,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Run the notebook's honest Session ID generalization check."""

    if groups.nunique() < 5:
        return

    group_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.get("random_state", 42))
    train_idx, test_idx = next(group_splitter.split(X, y, groups=groups))
    group_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "smote",
                SMOTE(
                    sampling_strategy=config.get("smote_sampling_strategy", "not majority"),
                    random_state=config.get("random_state", 42),
                    k_neighbors=config.get("smote_k_neighbors", 5),
                ),
            ),
            (
                "model",
                CatBoostClassifier(
                    iterations=500,
                    depth=5,
                    learning_rate=0.04,
                    loss_function="MultiClass",
                    random_seed=42,
                    verbose=False,
                ),
            ),
        ]
    )
    group_pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = np.ravel(group_pipeline.predict(X.iloc[test_idx]))
    group_result = {
        "split": "GroupShuffleSplit by Session ID",
        "accuracy": accuracy_score(y.iloc[test_idx], y_pred),
        "macro_f1": f1_score(y.iloc[test_idx], y_pred, average="macro"),
        "train_sessions": groups.iloc[train_idx].nunique(),
        "test_sessions": groups.iloc[test_idx].nunique(),
    }

    cv_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "smote",
                SMOTE(
                    sampling_strategy=config.get("smote_sampling_strategy", "not majority"),
                    random_state=config.get("random_state", 42),
                    k_neighbors=config.get("smote_k_neighbors", 5),
                ),
            ),
            (
                "model",
                LGBMClassifier(
                    objective="multiclass",
                    num_class=len(ACTIVITY_LEVEL_ORDER),
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )
    cv_scores = cross_validate(
        cv_pipeline,
        X,
        y,
        groups=groups,
        cv=GroupKFold(n_splits=5),
        scoring=["f1_macro", "accuracy"],
        n_jobs=-1,
    )
    group_cv_summary = pd.DataFrame(
        {
            "fold": range(1, 6),
            "macro_f1": cv_scores["test_f1_macro"],
            "accuracy": cv_scores["test_accuracy"],
        }
    )
    group_cv_summary.to_csv(output_dir / "group_cv_results.csv", index=False)
    pd.DataFrame([group_result]).to_csv(output_dir / "group_split_result.csv", index=False)

    print("\nGroup split by Session ID:")
    print(pd.DataFrame([group_result]).to_string(index=False))
    print("\nGroupKFold by Session ID:")
    print(group_cv_summary.agg({"macro_f1": ["mean", "std"], "accuracy": ["mean", "std"]}).to_string())


def main() -> None:
    config = load_config()
    output_dir = PROJECT_ROOT / config.get("model_output_dir", "models")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_processed_data(config)
    X, y, groups, numerical_features, ordinal_features, nominal_features = select_features(df, config)
    preprocessor = build_preprocessor(numerical_features, ordinal_features, nominal_features)
    pipelines = build_pipelines(preprocessor, config)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42),
        stratify=y,
    )
    print(f"Training rows: {len(X_train):,}; test rows: {len(X_test):,}; features: {X.shape[1]}")

    evaluate_pipelines(pipelines, X_train, X_test, y_train, y_test, output_dir)

    if config.get("run_group_generalization_check", True):
        group_generalization_check(X, y, groups, preprocessor, config, output_dir)

    print("\nModel training and evaluation completed.")


if __name__ == "__main__":
    main()
