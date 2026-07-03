# Injection Molding Defect Prediction

A tool that predicts whether a plastic injection molding cycle produced a good or a scrap part from the machine's process parameters, and explains which settings drove the risk. Built as my advanced machine learning capstone.

## Status

In progress. Proposal approved. The exploratory analysis and the modeling are done. Still to come: a short drift forecasting notebook and a Streamlit demo app.

## Problem

Plastics plants catch defective parts after production, during inspection, by which point material and machine time are wasted. This tool flags a likely scrap cycle from the machine's own process parameters (temperatures, pressures, cycle time, and more) and points to the settings that caused it.

## Approach

- Exploratory analysis of the process parameters, including whether they separate good from scrap parts (`notebooks/eda.ipynb`)
- A linear separability check (PCA plus a straight line vs curved SVM) to justify the model choice
- Supervised classification: a Logistic Regression baseline, then a curved SVM and XGBoost, with class weighting for the rarer scrap class
- SHAP to explain each prediction and name the settings behind it
- Isolation Forest for catching unusual cycles without labels
- Modeling in `notebooks/modeling.ipynb`

## Data

Public injection molding quality dataset (airtlab / iGuzzini): 1,451 real production cycles, 13 process parameters, and a quality label. The raw file is not committed (see `.gitignore`); the notebooks load it from `data/raw/`.

## Tools

Python, pandas, numpy, scikit-learn, XGBoost, SHAP, matplotlib, seaborn, and Streamlit for the planned demo app.

## Author

James Mahinda
