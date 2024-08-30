# Auditing Implementation and Results

This section presents the auditing implementation and the results, with a focus on treating **micro-firms** as the protected group.

## Sensitive Attribute: Firm Size

We define firms based on their size using the following criteria:

- **Micro-firms**:
  - **Workers:** Less than 10 employees
  - **Annual Turnover:** Less than 2 million pounds

- **Non-Micro-firms**:
  - **Workers:** More than 10 employees
  - **Annual Turnover:** More than 2 million pounds

## Peer Identification

- **Threshold:** $0.3 \times \text{Std}_{IC}$

## Model Specifications

- **Fitting Model:** Logistic 
- **Prediction Model:** Logistic 

## Research Findings

- **Fair Treatment**: Only **2.48%** of micro-firms are treated fairly.
- **Discrimination vs. Privilege**: 
  - **41.51%** of micro-firms are discriminated against.
  - **56.40%** of micro-firms are privileged.
  - **26.71%** of micro-firms are extremely discriminated against.
  - **32.17%** of micro-firms are extremely privileged.
- **Rejection Rates**:
  - Nearly half of the micro-firms facing extreme discrimination are rejected.
  - Discriminated micro-firms have a significantly higher rejection rate compared to their peers (**52.42%** vs **9.97%**).
- **Framework Stability**:
  - **95%** of micro-firms maintained consistent auditing results despite changing imbalance levels, showcasing the robustness of the framework.


---
# Summary of Results
The results highlight significant algorithmic bias against micro-firms, reflecting challenges in the banking sector. Our "peer-induced fairness" framework effectively identifies these disparities and visually represents individual-level discrepancies across the entire dataset.

---
> **Note:** This auditing framework is adaptable and can be applied to other sectors and sensitive attributes beyond firm size.
