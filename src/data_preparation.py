from sklearn.impute import KNNImputer
import sqlite3
import pandas as pd
import numpy as np
import yaml
import os

# Load data function
def load_data(config):
    """
    Loads data from a SQLite database into a pandas DataFrame.
    """
    db_file = 'data/gas_monitoring.db'
    # NOTE: The provided code assumes a database file at 'data/gas_monitoring.db'.
    # If you are running this in a different environment, you may need to
    # modify this path.
    if not os.path.exists(db_file):
        print(f"Error: Database file '{db_file}' not found.")
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(db_file)
        df = pd.read_sql_query("SELECT * FROM gas_monitoring", conn)
        conn.close()
        return df
    except sqlite3.OperationalError as e:
        print(f"An error occurred while reading from the database: {e}")
        return pd.DataFrame()

# EDA Function
def preprocess_data(df, config):

    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # --- Data Cleaning and Correction ---
    
    numerical_cols = ['Temperature', 'Humidity', 'CO2_InfraredSensor', 'CO2_ElectroChemicalSensor',
                     'MetalOxideSensor_Unit1', 'MetalOxideSensor_Unit2', 'MetalOxideSensor_Unit3',
                     'MetalOxideSensor_Unit4']
    categorical_cols = ['CO_GasSensor', 'HVAC Operation Mode', 'Time of Day']

    # Remove outliers using IQR
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]

# Feature engineering

    # Put "Activity Level" to the proper expressions.
    mode_Activity_Level_mapping = {
        'Low Activity': 'Low Activity',
        'High Activity': 'High Activity',
        'Moderate Activity': 'Moderate Activity',
        'ModerateActivity': 'Moderate Activity',
        'Low_Activity': 'Low Activity',
        'LowActivity': 'Low Activity'
    }
    df['Activity Level'] = df['Activity Level'].map(mode_Activity_Level_mapping)

    # Temperature analysis and correction (convert Kelvin to Celsius)
    anomaly_mask = df['Temperature'] >= 288
    df.loc[anomaly_mask, 'Temperature'] = df.loc[anomaly_mask, 'Temperature'] - 273.15
    
    # Humidity analysis and correction
    humidity_strange = df[df['Humidity'] < 40].shape[0] + df[df['Humidity'] < 0].shape[0] + df[df['Humidity'] >= 100].shape[0]
    df.loc[df['Humidity'] < 0, 'Humidity'] = df.loc[df['Humidity'] < 0, 'Humidity'].abs()
    df.loc[df['Humidity'] >= 100, 'Humidity'] = df.loc[df['Humidity'] >= 100, 'Humidity'] / 100
    df.loc[df['Humidity'] < 40, 'Humidity'] = df.loc[df['Humidity'] < 40, 'Humidity'] + 30
    
    # --- Feature Engineering: Create the new interaction feature ---
    df['Temperature_x_Humidity'] = df['Temperature'] * df['Humidity']
   
    # CO2_InfraredSensor correction
    # Typical CO2 in Indoor environments are in the range of 350~1000 ppm. Also, the readings in CO2_ElectroChemicalSensor have their median at 563 ppm; this suggests the readings here are off by 450 ppm -> Apply 450 ppm offset.
    df['CO2_InfraredSensor'] = df['CO2_InfraredSensor'] + 450
        
    # CO2_ElectroChemicalSensor imputation (Median)
    median_value = df['CO2_ElectroChemicalSensor'].median()
    df['CO2_ElectroChemicalSensor'] = df['CO2_ElectroChemicalSensor'].fillna(median_value)
    
    # --- Feature Engineering: Create the new interaction feature ---
    df['CO2_sum'] = df['CO2_InfraredSensor'] + df['CO2_ElectroChemicalSensor']
    
    # MetalOxideSensor_Unit3
    # There are 2566 datapoints in MetalOxideSensor_Unit3 that are NaN. I would KNN impute the NaN values with the other metal oxide sensor units, because from the shapes of their graphs it is likely they are highly correlated.
    # MetalOxideSensor_Unit3 imputation (KNN)
    impute_cols = ['MetalOxideSensor_Unit1', 'MetalOxideSensor_Unit2', 'MetalOxideSensor_Unit3', 'MetalOxideSensor_Unit4']
    imputer = KNNImputer(n_neighbors=5)
    df[impute_cols] = imputer.fit_transform(df[impute_cols])
    
    # --- Feature Engineering: Create the new interaction feature ---
    df['MetalOxideSensor_sum'] = df['MetalOxideSensor_Unit1'] + df['MetalOxideSensor_Unit2'] + df['MetalOxideSensor_Unit3'] + df['MetalOxideSensor_Unit4']
    
    # CO_GasSensor
    # There are 1369 datapoints in CO_GasSensor that are NaN. Impute with mode.
    mode_value = df['CO_GasSensor'].mode()[0]  # Get the most frequent value
    df['CO_GasSensor'] = df['CO_GasSensor'].fillna(mode_value)
    
    # Session ID
    
    # Define Session_ID bins and labels for analysis
    Session_ID_bins = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    Session_ID_labels = ['1000-2000', '2000-3000', '3000-4000', '4000-5000', '5000-6000', '6000-7000', '7000-8000', '8000-9000', '9000-10000']
    # Create a new categorical series for Session_ID_Bands
    df['Session_ID_Bands'] = pd.cut(df['Session ID'], bins=Session_ID_bins, labels=Session_ID_labels, right=True)
    
    # HVAC Operation Mode
    
    # --- Categorical Feature Cleaning ---

    # HVAC Operation Mode mapping
    mode_HVAC_mapping = {
        'off': 'off', 'ventilation_only': 'ventilation_only', 'eco_mode': 'eco_mode',
        'Heating_active': 'heating_active', 'heating_active': 'heating_active',
        'maintenance_mode': 'maintenance_mode', 'cooling_active': 'cooling_active',
        'Eco_Mode': 'eco_mode', 'Cooling_Active': 'cooling_active',
        'MAINTENANCE_MODE': 'maintenance_mode', 'Eco_mode': 'eco_mode',
        'HEATING_ACTIVE': 'heating_active', 'COOLING_ACTIVE': 'cooling_active',
        'VENTILATION_ONLY': 'ventilation_only', 'Ventilation_Only': 'ventilation_only',
        'Heating_Active': 'heating_active', 'Off': 'off', 'ECO_MODE': 'eco_mode',
        'Ventilation_only': 'ventilation_only', 'OFF': 'off',
        'Maintenance_mode': 'maintenance_mode', 'Maintenance_Mode': 'maintenance_mode',
        'Cooling_active': 'cooling_active'
    }
    df['HVAC Operation Mode'] = df['HVAC Operation Mode'].map(mode_HVAC_mapping).fillna('unknown').replace('', 'unknown')
    
    # Drop unused columns
    df = df.drop(['Ambient Light Level'], axis=1)

    # Save preprocessed data
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/preprocessed_data.csv', index=False)
    print("Data preprocessing completed.")

    return df

def main():
    """
    Main function to load and preprocess data.
    """
    with open('src/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    df = load_data(config)
    if df.empty:
        return
    
    df = preprocess_data(df, config)

if __name__ == "__main__":
    main()
