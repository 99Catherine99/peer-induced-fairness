from sklearn.model_selection import cross_val_score
from scipy.stats import sem


class precision:
    def __init__(self, model, X, y, cv=5):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv

    def cross_validate(self, scoring='precision'):
        cv_precision_scores = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring=scoring)

        print(f"{self.cv}-fold Cross Validation Precision:", cv_precision_scores)
        print("Mean Precision:", cv_precision_scores.mean())
        print("Standard Deviation of the Mean Precision:", cv_precision_scores.std())
        print("Standard Error of the Mean Precision:", sem(cv_precision_scores))

        return cv_precision_scores