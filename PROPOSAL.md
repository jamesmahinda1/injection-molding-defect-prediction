# Capstone Proposal: Early Defect Prediction for Plastic Injection Molding

**Student:** James Mahinda

## Description

Plastics manufacturers run injection molding machines that produce one part per cycle. Each cycle has a set of process parameters such as melt temperature, mold temperature, injection pressure, holding pressure, cycle time, clamping force, screw position, and shot volume. When these parameters drift out of their good range, the part comes out defective.

The problem is that defects are usually caught after production, during quality inspection or once a customer complains. By then the material and machine time are already wasted. I want to build a tool that predicts whether a molding cycle produced a defective part from its process parameters, in real time, and points to the parameter that caused it so an operator can correct the machine before more scrap is made.

This is a real problem I see in the plastics manufacturing space, which is why I picked it.

## Core Functionalities

1. Predict whether a molding cycle is good or defective from its process parameters
2. Give a defect risk score for each cycle rather than just a yes or no
3. Explain the prediction by showing which parameters pushed the risk up, so the output is actionable on the shop floor
4. Detect anomalous cycles that do not match normal good production, even without a label, to catch new types of defects
5. A deployed web app where a user enters cycle parameters and gets the risk score and the top contributing parameters back

## Machine Learning Approach

The main task is supervised classification. I start with Logistic Regression as a baseline, then move to Random Forest and Gradient Boosting. Because defects are rarer than good parts, I handle class imbalance with techniques like SMOTE and class weighting, and I evaluate with precision, recall, F1, and PR-AUC rather than plain accuracy, since accuracy is misleading on imbalanced data.

I also add unsupervised anomaly detection with Isolation Forest and an autoencoder, for the case where defect labels are not available and the goal is just to flag cycles that look abnormal.

For explainability I use SHAP to show which parameter drove each prediction, which turns the model from a black box into something an operator can act on.

## Data

Primary dataset: a public injection molding quality dataset from airtlab (iGuzzini Illuminazione), with 1,451 real production cycles, 13 process parameters, and a quality label. It is real industrial data, free, and published in a peer reviewed paper.

Secondary dataset: SECOM from the UCI repository, a real semiconductor manufacturing quality dataset with a strong class imbalance (about 14 to 1), used to demonstrate the imbalance handling techniques on a harder case.

Using public datasets keeps the project shareable, since the real plant data I work with is confidential.

## Tools and Technologies

Python, pandas, and numpy for data work, matplotlib and seaborn for visualization, scikit-learn for the classification models and Isolation Forest, imbalanced-learn for SMOTE, TensorFlow for the autoencoder, SHAP for explainability, and Streamlit for the deployed web app. Code and documentation on GitHub.

## Deployment

The deliverable is a deployed Streamlit web app. A user enters or loads a cycle's process parameters and gets back a defect risk score, a good or scrap flag, and the parameters that contributed most to the risk.

In a real factory this model would sit inside the manufacturing execution system, scoring each cycle as its parameters land in the database and showing the result to operators and managers. That full integration is the real world roadmap beyond this capstone, which focuses on the model and a working demo app.

## Expected Outcomes

A working, deployed tool that flags likely defective cycles better than a naive baseline, explains its predictions, and is honest about its limits. The model is evaluated on metrics that suit imbalanced data, and I show both the supervised and the anomaly detection approaches.

## Limitations and Future Work

The model learns from the parameters in the public datasets, so it reflects those machines and parts rather than any specific plant. To use it in a real factory it would need to be retrained on that factory's own labeled cycles, and it would improve over time as more quality results are fed back in. Future work also includes connecting it directly to a live machine feed and building the operator and manager views inside a manufacturing execution system.
