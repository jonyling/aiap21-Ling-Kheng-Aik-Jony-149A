#!/bin/bash
# Script to run the end-to-end machine learning pipeline

# Run the data preparation script
python src/data_preparation.py

# Run the model training script
python src/model_training.py

echo "Pipeline execution completed."