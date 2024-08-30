# Metrics Package

This package contains a collection of metrics for assessing model performance, fairness, and distributional distances. These metrics are designed to aid in the evaluation of machine learning models, especially in the context of fairness-aware machine learning.

## Metrics for Model Performance

1. **Accuracy Calculator**
   - Measures the accuracy of model predictions.
   - $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$

4. **Precision Calculator**
   - Evaluates the precision of model predictions.
   - $Precision = \frac{TP}{TP + FP}$

5. **Recall Calculator**
   - Calculates the recall (true positive rate) of the model.
   - $Recall = \frac{TP}{TP + FN}$
 
3. **F1 Calculator**
   - Computes the F1 score, which balances precision and recall.
   - $F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$

6. **ROC Calculator**
   - Computes the Receiver Operating Characteristic (ROC) curve.

7. **Gini Calculator**
   - Measures the Gini coefficient as a measure of predictive power.
   - $Gini=2*AUC-1$

## Metrics for Fairness

1. **Balanced Accuracy Calculator (BACC)**
   - Evaluates balanced accuracy, considering sensitivity and specificity.
   - $BACC = \frac{1}{2}(Sensitivity + Specificity)$

2. **CV Calculator**
   - Computes coefficient of variation as a fairness metric.
   - $Pr(\hat Y=1|S=1)-Pr(\hat Y=1|S=0)$

3. **Discrimination Calculator**
   - Measures discrimination between protected groups.
   - $Pr(\hat Y=1|S=1)/Pr(\hat Y=1|S=0)$

4. **FNR Calculator**
   - Calculates the False Negative Rate.
   - $FNR = \frac{FN}{FN + TP}$

5. **FPR Calculator**
   - Computes the False Positive Rate.
   - $FPR = \frac{FP}{FP + TN}$

6. **TNR Calculator**
   - Measures the True Negative Rate.
   - $TNR = \frac{TN}{TN + FP}$

7. **TPR Calculator**
   - Calculates the True Positive Rate.
   - $TPR = \frac{TP}{TP + FN}$

8. **NPV Calculator**
   - Computes the Negative Predictive Value.
   - $NPV = \frac{TN}{TN + FN}$

9. **PPV Calculator**
   - Measures the Positive Predictive Value.
   - $PPV = \frac{TP}{TP + FP}$

10. **Proportion Calculator**
    - Calculates the $Pr(Y=1)$ among groups.

11. **SP Calculator**
    - Evaluates $Pr(\hat Y=1)$ among groups

12. **Mutual Information Calculator**
    - Measures mutual information between variables.
    - $I(X;Y) = \sum_x \sum_y p(x, y) \log \left(\frac{p(x, y)}{p(x)p(y)}\right)$

13. **Information Gain Calculator**
    - Calculates information gain.
    - $IG(D, A) = H(D) - H(D|A)$

## Metrics for Distributional Distances

1. **KL Calculator**
   - Computes the Kullback-Leibler (KL) divergence.
   - $KL(P||Q) = \sum_x P(x) \log \left(\frac{P(x)}{Q(x)}\right)$

2. **KS Calculator**
   - Measures the Kolmogorov-Smirnov (KS) statistic.
   - $KS = \max |F_1(x) - F_2(x)|$

3. **Wasserstein Calculator**
   - Evaluates the Wasserstein distance between distributions.
   - $W(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \int_{X \times Y} c(x, y) d\gamma(x, y)$