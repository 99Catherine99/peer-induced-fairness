from sklearn.model_selection import cross_val_score
from scipy.stats import sem

class recall:
    def __init__(self, model, X, y, cv=5):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv

    def cross_validate(self, scoring='recall'):
        cv_recall_scores = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring=scoring)

        print(f"{self.cv}-fold Cross Validation Recall:", cv_recall_scores)
        print("Mean Recall:", cv_recall_scores.mean())
        print("Standard Deviation of the Mean Recall:", cv_recall_scores.std())
        print("Standard Error of the Mean Recall:", sem(cv_recall_scores))
        return cv_recall_scores