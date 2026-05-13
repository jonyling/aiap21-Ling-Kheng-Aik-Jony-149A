"""Data preparation for the gas monitoring activity-level pipeline.

The cleaning and feature engineering steps here are the production version of
the decisions made in eda.ipynb.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import KNNImputer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "src" / "config.yaml"

ACTIVITY_LEVEL_ORDER = ["Low Activity", "Moderate Activity", "High Activity"]
TIME_OF_DAY_ORDER = ["morning", "afternoon", "evening", "night"]
CO_GAS_ORDER = ["extremely low", "low", "medium", "high", "extremely high"]
TEMPERATURE_BAND_LABELS = ["14-16", "16-18", "18-20", "20-22", "22-24", "24-26"]
SESSION_ID_BAND_LABELS = [
    "1000-2000",
    "2000-3000",
    "3000-4000",
    "4000-5000",
    "5000-6000",
    "6000-7000",
    "7000-8000",
    "8000-9000",
    "9000-10000",
]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_data(config: dict[str, Any]) -> pd.DataFrame:
    """Load raw data from the SQLite table used in the assessment."""

    db_path = PROJECT_ROOT / config.get("db_path", "data/gas_monitoring.db")
    table_name = config.get("table_name", "gas_monitoring")

    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def clean_activity_level(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Low Activity": "Low Activity",
        "Low_Activity": "Low Activity",
        "LowActivity": "Low Activity",
        "Moderate Activity": "Moderate Activity",
        "ModerateActivity": "Moderate Activity",
        "High Activity": "High Activity",
    }
    df["Activity Level"] = df["Activity Level"].map(mapping).fillna("Low Activity")
    df["Activity Level"] = pd.Categorical(
        df["Activity Level"],
        categories=ACTIVITY_LEVEL_ORDER,
        ordered=True,
    )
    return df


def clean_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Time of Day"] = df["Time of Day"].replace("", np.nan)
    df["Time of Day"] = pd.Categorical(
        df["Time of Day"],
        categories=TIME_OF_DAY_ORDER,
        ordered=True,
    )

    hvac_mapping = {
        "off": "off",
        "Off": "off",
        "OFF": "off",
        "ventilation_only": "ventilation_only",
        "Ventilation_only": "ventilation_only",
        "Ventilation_Only": "ventilation_only",
        "VENTILATION_ONLY": "ventilation_only",
        "eco_mode": "eco_mode",
        "Eco_mode": "eco_mode",
        "Eco_Mode": "eco_mode",
        "ECO_MODE": "eco_mode",
        "heating_active": "heating_active",
        "Heating_active": "heating_active",
        "Heating_Active": "heating_active",
        "HEATING_ACTIVE": "heating_active",
        "cooling_active": "cooling_active",
        "Cooling_active": "cooling_active",
        "Cooling_Active": "cooling_active",
        "COOLING_ACTIVE": "cooling_active",
        "maintenance_mode": "maintenance_mode",
        "Maintenance_mode": "maintenance_mode",
        "Maintenance_Mode": "maintenance_mode",
        "MAINTENANCE_MODE": "maintenance_mode",
    }
    df["HVAC Operation Mode"] = (
        df["HVAC Operation Mode"].map(hvac_mapping).fillna("unknown").replace("", "unknown")
    )

    mode_value = df["CO_GasSensor"].mode(dropna=True)
    df["CO_GasSensor"] = df["CO_GasSensor"].fillna(mode_value.iloc[0] if not mode_value.empty else "medium")
    df["CO_GasSensor"] = pd.Categorical(df["CO_GasSensor"], categories=CO_GAS_ORDER, ordered=True)

    return df


def clean_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df["Temperature"] >= 288, "Temperature"] -= 273.15
    df["Temperature_Band"] = pd.cut(
        df["Temperature"],
        bins=[14, 16, 18, 20, 22, 24, 26],
        labels=TEMPERATURE_BAND_LABELS,
        right=True,
    )

    df.loc[df["Humidity"] < 0, "Humidity"] = df.loc[df["Humidity"] < 0, "Humidity"].abs()
    df.loc[df["Humidity"] >= 100, "Humidity"] = df.loc[df["Humidity"] >= 100, "Humidity"] / 100
    df.loc[df["Humidity"] < 40, "Humidity"] = df.loc[df["Humidity"] < 40, "Humidity"] + 30
    df.loc[df["Humidity"] > 100, "Humidity"] = 100
    df["Temperature_x_Humidity"] = df["Temperature"] * df["Humidity"]

    df["CO2_InfraredSensor"] = df["CO2_InfraredSensor"] + 450
    df["CO2_ElectroChemicalSensor"] = df["CO2_ElectroChemicalSensor"].fillna(
        df["CO2_ElectroChemicalSensor"].median()
    )
    df["CO2_sum"] = df["CO2_InfraredSensor"] + df["CO2_ElectroChemicalSensor"]
    df["co2_ratio"] = df["CO2_InfraredSensor"] / (df["CO2_ElectroChemicalSensor"] + 1e-6)

    mos_cols = [f"MetalOxideSensor_Unit{i}" for i in range(1, 5)]
    df[mos_cols] = KNNImputer(n_neighbors=5).fit_transform(df[mos_cols])
    df["MetalOxideSensor_sum"] = df[mos_cols].sum(axis=1)
    df["avg_metal_oxide"] = df[mos_cols].mean(axis=1)
    df["mos_variance"] = df[mos_cols].std(axis=1)

    return df


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    df["time_index"] = np.arange(len(df))
    df["is_daytime"] = df["Time of Day"].isin(["morning", "afternoon"]).astype(int)
    df["hvac_active"] = (df["HVAC Operation Mode"] != "off").astype(int)
    df["Session_ID_Bands"] = pd.cut(
        df["Session ID"],
        bins=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
        labels=SESSION_ID_BAND_LABELS,
        right=True,
    )

    df = df.sort_values(["Session ID", "time_index"]).reset_index(drop=True)
    ts_base_cols = [
        "Temperature",
        "Humidity",
        "CO2_InfraredSensor",
        "MetalOxideSensor_Unit1",
        "MetalOxideSensor_Unit2",
        "avg_metal_oxide",
        "co2_ratio",
    ]

    for col in ts_base_cols:
        session_group = df.groupby("Session ID")[col]
        for lag in (1, 2, 3):
            df[f"{col}_session_lag{lag}"] = session_group.shift(lag)

        shifted = session_group.shift(1)
        for window in (3, 5, 10):
            df[f"{col}_session_roll_mean_{window}"] = (
                shifted.groupby(df["Session ID"])
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            df[f"{col}_session_roll_std_{window}"] = (
                shifted.groupby(df["Session ID"])
                .rolling(window=window, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
            )

    df["HVAC_prev_session"] = df.groupby("Session ID")["HVAC Operation Mode"].shift(1)
    df["HVAC_changed_session"] = (df["HVAC Operation Mode"] != df["HVAC_prev_session"]).astype(int)
    df.loc[df["HVAC_prev_session"].isna(), "HVAC_changed_session"] = 0

    change_block = df.groupby("Session ID")["HVAC_changed_session"].cumsum()
    df["time_since_last_change_session"] = df.groupby(["Session ID", change_block]).cumcount()

    return df


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "Activity Level",
        "Session ID",
        "Time of Day",
        "CO_GasSensor",
        "Temperature_Band",
        "is_daytime",
        "Temperature",
        "Temperature_x_Humidity",
        "co2_ratio",
        "avg_metal_oxide",
        "time_since_last_change_session",
        "HVAC_prev_session",
    ]
    session_features = [
        col
        for col in df.columns
        if "_session_lag" in col or "_session_roll_mean_" in col or "_session_roll_std_" in col
    ]
    required_columns.extend(session_features)
    required_columns = [col for col in required_columns if col in df.columns]

    return df.dropna(subset=required_columns).copy()


def preprocess_data(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    df = clean_activity_level(df)
    df = clean_categorical_features(df)
    df = clean_sensor_features(df)
    df = add_session_features(df)
    df = prepare_model_frame(df)

    output_path = PROJECT_ROOT / config.get("processed_data_path", "data/processed/preprocessed_data.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved preprocessed model frame to {output_path.relative_to(PROJECT_ROOT)} ({len(df):,} rows).")
    return df


def main() -> None:
    config = load_config()
    raw_df = load_data(config)
    preprocess_data(raw_df, config)


if __name__ == "__main__":
    main()
