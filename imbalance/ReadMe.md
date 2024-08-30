# Imbalance Level Adjustment

In this section, we explore the effectiveness of our fairness auditing framework by systematically altering the imbalance levels within the dataset. By resampling from the original dataset, which has an initial imbalance level of **41.33%**, we adjust the imbalance levels to **11.33%**, **16.33%**, **21.33%**, **26.33%**, **31.33%**, and **36.33%**. 


## Peer Identification

- **Threshold:** $0.3 \times \text{Std}_{IC}$

## Model Specifications

- **Fitting Model:** Logistic 
- **Prediction Model:** Logistic


## Research Findings

- **High Consistency**: Over **95%** of micro-firms continue to experience either discrimination or privilege, regardless of changes in imbalance levels.
- **Stable Auditing Results**: **95%** of micro-firms showed consistent auditing outcomes across different imbalance levels.
- **Robust Framework**: The framework remains effective even when using a pair-wise IOR calculation method.

---

## Summary of Results
The results of this analysis demonstrate the resilience of our fairness auditing framework, proving its effectiveness even under challenging conditions such as data scarcity and imbalance. This makes it a reliable tool, regardless of the underlying data distribution.

