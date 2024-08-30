from sklearn.model_selection import cross_val_score
from scipy.stats import sem

class f1:
    def __init__(self, model, X, y, cv=5):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv

    def cross_validate(self, scoring='f1'):
        cv_f1_scores = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring=scoring)

        print(f"{self.cv}-fold Cross Validation F1:", cv_f1_scores)
        print("Mean F1:", cv_f1_scores.mean())
        print("Standard Deviation of the Mean F1:", cv_f1_scores.std())
        print("Standard Error of the Mean F1:", sem(cv_f1_scores))

        return cv_f1_scores