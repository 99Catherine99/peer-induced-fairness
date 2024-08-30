import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import sem
from sklearn.model_selection import cross_val_score


class ROC:
    def __init__(self, model, X, y, cv=5):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv

    def cross_validate(self):
        cv_auc_scores = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring='roc_auc')

        print(f"{self.cv}-fold Cross Validation AUC:", cv_auc_scores)
        print("Mean AUC:", cv_auc_scores.mean())
        print("Standard Deviation of the Mean AUC:", cv_auc_scores.std())
        print("Standard Error of the Mean AUC:", sem(cv_auc_scores))

        return cv_auc_scores

