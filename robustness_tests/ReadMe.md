# Robustness Tests for Fairness Auditing Framework

This part contains a series of robustness tests conducted on our fairness auditing framework. The purpose of these tests is to evaluate the stability and robustness of the framework under different configurations.

## Contents

The tests cover the following aspects:

1. **Fitting Model Changes**:
   - **Random Forest (RF)**
   - **XGBoost**
   
2. **Prediction Model Changes**:
   - **Random Forest (RF)**
   - **XGBoost**

3. **Peer Identification Threshold Adjustments**:
   - Thresholds of **$0.2 \times Std_{IC}$**, **$0.4 \times Std_{IC}$**, and **$0.5 \times Std_{IC}$** were tested.

## Research Findings

The results of these tests are compiled and can be found in the `robustness results` file within this folder.


---
## Summary of Results

The results demonstrate that our fairness auditing framework remains stable and robust regardless of the changes in model configurations or peer identification thresholds. This consistency underscores the reliability of our framework across various settings.
