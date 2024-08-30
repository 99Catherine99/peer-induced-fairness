from sklearn.model_selection import cross_val_score
from scipy.stats import sem

class accuracy:
    def __init__(self, model, X, y, cv=5):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv

    def cross_validate(self, scoring='accuracy'):
        cv_accuracy_scores = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring=scoring)

        print(f"{self.cv}-fold Cross Validation Accuracy:", cv_accuracy_scores)
        print("Mean Accuracy:", cv_accuracy_scores.mean())
        print("Standard Deviation of the Mean Accuracy:", cv_accuracy_scores.std())
        print("Standard Error of the Mean Accuracy:", sem(cv_accuracy_scores))

        return cv_accuracy_scores