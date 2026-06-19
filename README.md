# Injection Molding Defect Prediction

A tool that predicts whether a plastic injection molding cycle produced a defective part from its process parameters, and explains which parameter caused the risk. Built as my advanced machine learning capstone.

## Status

Proposal stage. Waiting on technical mentor approval before development starts. See `PROPOSAL.md`.

## Problem

Plastics plants catch defective parts after production, during inspection, by which point material and machine time are wasted. This tool flags a likely defective cycle in real time from the machine's own process parameters (temperatures, pressures, cycle time, and more) and points to the parameter that caused it.

## Planned Approach

- Supervised classification: Logistic Regression baseline, then Random Forest and Gradient Boosting, with class imbalance handling
- Unsupervised anomaly detection: Isolation Forest and an autoencoder for catching abnormal cycles without labels
- SHAP for explaining each prediction
- A deployed Streamlit web app for the demo

## Data

- Primary: public injection molding quality dataset (airtlab / iGuzzini), 1,451 real cycles, 13 process parameters, quality label
- Secondary: SECOM (UCI), real semiconductor quality data with strong class imbalance

## Tools

Python, pandas, numpy, scikit-learn, imbalanced-learn, TensorFlow, SHAP, Streamlit.

## Author

James Mahinda
