from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from metrics.accuracy_calculator import accuracy
from metrics.precision_calculator import precision
from metrics.recall_calculator import recall
from metrics.f1_calculator import f1
from metrics.roc_calculator import ROC
from sklearn.metrics import classification_report



class LogisticRegressionModel:
    """
    Initialize the Logistic Regression Model.

    Parameters:
    - X (array-like): Features for training.
    - y (array-like): Target variable for training.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.lr = LogisticRegression(max_iter=1000, random_state=42)
        self.param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2']
        }
        self.grid_search = GridSearchCV(estimator=self.lr, param_grid=self.param_grid, cv=5, scoring='roc_auc')

    def train_model(self):
        """
        Train the logistic regression model using grid search.
        """
        self.grid_search.fit(self.X_train, self.y_train)

    def evaluate_performance(self):
        """
        Evaluate the performance of the logistic regression model using various metrics.
        """
        AUC_evaluator = ROC(self.grid_search, self.X, self.y, cv=5)
        accuracy_evaluator = accuracy(self.grid_search, self.X, self.y, cv=5)
        precision_evaluator = precision(self.grid_search, self.X, self.y, cv=5)
        recall_evaluator = recall(self.grid_search, self.X, self.y, cv=5)
        f1_evaluator = f1(self.grid_search, self.X, self.y, cv=5)

        # cv_auc_scores = AUC_evaluator.cross_validate(scoring='roc_auc')

        # accuracy_evaluator.cross_validate(scoring='roc_auc')
        AUC_evaluator.cross_validate()
        accuracy_evaluator.cross_validate(scoring='accuracy')
        precision_evaluator.cross_validate(scoring='precision')
        recall_evaluator.cross_validate(scoring='recall')
        f1_evaluator.cross_validate(scoring='f1')

    def generate_test_report(self):
        """
        Generate classification report on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        print(report)

    def calculate_auc(self):
        """
        Calculate AUC (Area Under the ROC Curve) on the test set.
        """
        y_prob = self.grid_search.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, y_prob)
        print('AUC:', auc)

    def calculate_acc(self):
        """
        Calculate Accuracy on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print('Accuracy:', acc)

    def calculate_f1(self):
        """
        Calculate F1 Score on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred)
        print('F1 Score:', f1)

    def calculate_precision(self):
        """
        Calculate Precision on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        precision = precision_score(self.y_test, y_pred)
        print('Precision:', precision)

    def calculate_recall(self):
        """
        Calculate Recall on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        recall = recall_score(self.y_test, y_pred)
        print('Recall:', recall)

    def predict(self, X_pred):
        """
        Predict the target variable for a given set of features.

        Parameters:
        - X_pred (array-like): Features for prediction.

        Returns:
        - array-like: Predicted target variable.
        """
        return self.grid_search.predict(X_pred)

    def predict_proba(self, X_pred):
        """
        Predict class probabilities for a given set of features.

        Parameters:
        - X_pred (array-like): Features for prediction.

        Returns:
        - array-like: Predicted class probabilities.
        """
        return self.grid_search.predict_proba(X_pred)




