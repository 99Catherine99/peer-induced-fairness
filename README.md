# Peer-Induced Fairness Framework for Algorithmic Bias Auditing

## Aim

This project aims to establish a robust and scientifically grounded fairness auditing framework that adheres to the regulatory requirements of the EU AI Act. The proposed framework, known as *peer-induced fairness*, utilizes the concept of peer comparison by quantitatively identifying similar peers through $IC$ and comparing their algorithmic treatments.

Our framework is a credible, stable, and universal tool that effectively addresses data scarcity and imbalance issues. It also provides transparent explanations for individuals who, despite being treated fairly by the algorithm, are still rejected.

Given its effectiveness and strength, this framework can serve as both an external auditing tool for regulators and third-party auditors, as well as an internal auditing tool for stakeholders.


> **For a detailed discussion of the theoretical foundation and methodology, please refer to our related publication: [*Peer-induced Fairness: A Causal Approach to Reveal Algorithmic Unfairness in Credit Approval*](https://arxiv.org/abs/2408.02558) (Fang, Chen, & Ansell, 2024).**


## Data

Our framework is validated using data on loan application behaviors of SMEs when interacting with banks. The data, sourced from the UK Data Archive and the SME Finance Monitor, spans the period from 2012 to 2020 and includes 4,500 labeled records.

- **Target Variable (Y):** The final outcomes of accessing finance serve as the target label for constructing our predictive model.
- **Features (X):** We have selected 19 variables that are historically significant in the context of SME loans and available in our dataset. After filtering out features with a high missing ratio, 15 features were retained for prediction.
- **Sensitive Attribute (S):** Firm size, categorized into micro firms and non-micro firms.

[UK Archive and the SME Finance Monitor](https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=6888)

## Package Overview

This project comprises several key packages:

- **datasets**: Contains the datasets required for analysis, along with the selected features.
- **preprocessing**: Includes preprocessing steps such as imputation, encoding, and other data cleaning processes.
- **analysis_tools**: Provides tools for data analysis and visualization used throughout the project.
- **metrics**: Encompasses a comprehensive collection of fairness and predictive performance metrics for evaluation.
- **models**: Contains the predictive models employed in the study.
- **peer_identification**: Implements the peer identification methods and processes.
- **example**: Includes the main experiment demonstrating the framework.
- **imbalance**: Contains experiments related to varying imbalance levels in the data.
- **robustness_tests**: Features additional experiments for robustness testing, along with a summary of the robustness results.

These packages are integral to the project's framework, enabling a rigorous examination of the dataset, model predictions, and fairness considerations.

## Robustness

To ensure the robustness of our findings, we varied the peer identification threshold, as well as the fitting and prediction models. This comprehensive approach ensures that our conclusions are methodologically sound and applicable across diverse research settings.

---
## Conclusion

This study reveals significant findings in the context of SMEs in the UK. Notably, it highlights that start-ups and micro firms, especially those led by older individuals or those from non-white ethnic backgrounds, often face discrimination in bank decisions and unfairness in algorithmic predictions when applying for loans.

---
## Tools and Version

This project is built on Python 3.10.9.
