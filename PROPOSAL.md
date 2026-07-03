# Capstone Proposal: Early Defect Prediction for Plastic Injection Molding

**Student:** James Mahinda

## Description

Plastics manufacturers run injection molding machines that produce one part per cycle. Each cycle has a set of process parameters such as melt temperature, mold temperature, injection pressure, cycle time, clamping force, screw position, and shot volume. When these parameters drift out of their good range, the part comes out defective.

The problem is that defects are usually caught after production, during quality inspection or once a customer complains. By then the material and machine time are already wasted. I want to build a tool that predicts whether a molding cycle produced a good or scrap part from its process parameters, and points to the settings that caused it so an operator can correct the machine before more scrap is made.

This is a real problem I see in the plastics manufacturing space, which is why I picked it.

## Core Functionalities

1. Predict whether a molding cycle is good or scrap from its process parameters
2. Give a defect risk score for each cycle rather than just a yes or no
3. Explain the prediction by showing which settings pushed the risk up, so the output is actionable on the shop floor
4. Detect anomalous cycles that do not match normal good production, even without a label, to catch new types of defects
5. A deployed web app where a user enters cycle parameters and gets the risk score and the top contributing settings back

## Machine Learning Approach

The main task is supervised classification. Before choosing models I test whether good and scrap parts can be split by a simple straight line (linear separability), using PCA to visualize the data in 2D and a straight line versus a curved Support Vector Machine to confirm it. Because a straight line is not enough, I move to stronger models.

I start with Logistic Regression as a baseline, then move to a curved SVM and XGBoost. Because scrap is rarer than good parts, I handle the class imbalance with class weighting, and I evaluate on precision, recall, and F1 for the scrap class rather than plain accuracy, since accuracy is misleading on imbalanced data.

For explainability I use SHAP to show which settings drove each prediction, which turns the model from a black box into something an operator can act on.

I also add unsupervised anomaly detection with Isolation Forest, for the case where defect labels are not available and the goal is just to flag cycles that look abnormal.

As an extra angle, since the cycles happen in sequence over time, I use a time series model (ARIMA/SARIMA) to forecast where a key parameter is heading and flag when the machine is drifting toward scrap conditions, a simple predictive maintenance view.

## Data

A public injection molding quality dataset from airtlab (iGuzzini Illuminazione), with 1,451 real production cycles, 13 process parameters, and a quality label. It is real industrial data, free, and published in a peer reviewed paper.

Using a public dataset keeps the project shareable, since the real plant data I work with is confidential.

## Tools and Technologies

Python, pandas, and numpy for data work, matplotlib and seaborn for visualization, scikit-learn for the models, PCA, and Isolation Forest, XGBoost for the main classifier, SHAP for explainability, statsmodels for the time series forecasting, and Streamlit for the deployed web app. Code and documentation on GitHub.

## Deployment

The deliverable is a deployed Streamlit web app. A user enters or loads a cycle's process parameters and gets back a defect risk score, a good or scrap flag, and the settings that contributed most to the risk.

In a real factory this model would sit inside the manufacturing execution system, scoring each cycle as its parameters land in the database and showing the result to operators and managers. That full integration is the real world roadmap beyond this capstone, which focuses on the model and a working demo app.

## Expected Outcomes

A working, deployed tool that flags likely scrap cycles better than a naive baseline, explains its predictions, and is honest about its limits. The model is evaluated on metrics that suit imbalanced data, and I show the supervised, the anomaly detection, and the forecasting angles.

## Limitations and Future Work

The model learns from the parameters in a public dataset, so it reflects those machines and parts rather than any specific plant. To use it in a real factory it would need to be retrained on that factory's own labeled cycles, and it would improve over time as more quality results are fed back in. Future work also includes connecting it directly to a live machine feed and building the operator and manager views inside a manufacturing execution system.
