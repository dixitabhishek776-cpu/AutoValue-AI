# Submission report notes

## Project title
AutoValue AI: Used Car Price Prediction Using Machine Learning

## Abstract
This project demonstrates a machine-learning workflow for estimating the resale price of a used
vehicle from structured attributes such as brand, age, distance driven, fuel type, transmission,
ownership count, mileage, engine capacity and seating capacity. A Streamlit application provides
an interactive prediction interface and descriptive market analytics. A scikit-learn pipeline
performs missing-value imputation, categorical encoding, feature scaling and Random Forest
regression. The training script evaluates the model on a held-out test split using MAE, RMSE and R².

## Methodology
1. Acquire a legally usable used-car listing dataset.
2. Validate schema, units, target and data quality.
3. Separate predictors and selling-price target.
4. Impute missing values and one-hot encode categorical variables.
5. Split data into training and test subsets.
6. Train Random Forest regression.
7. Evaluate with MAE, RMSE and R².
8. Save model artifact and evaluation metadata.
9. Load the artifact in Streamlit for interactive inference.

## Honest experimental reporting
The repository contains a synthetic demonstration CSV so the software can run immediately.
Any metrics generated from this file are synthetic-data pipeline checks only. Replace the CSV
with a real, licensed dataset and rerun training before reporting model accuracy as a result.

## Limitations and future work
Use a larger time-stamped, geographically diverse dataset; perform time-based validation;
compare CatBoost/XGBoost; calibrate uncertainty intervals; add vehicle model/variant and city;
monitor drift; version datasets and models; and deploy behind authentication where needed.
