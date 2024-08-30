from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV



class ModelTrainer:
    """
    A class for training different classifiers using grid search for hyperparameter tuning.

    Methods:
    grid_search(self, model): Perform grid search for hyperparameter tuning and return the best model.

    Attributes:
    classifier: The classifier instance to be trained, which we could choose.
    param_grid: The hyperparameter grid to be used in grid search, different classifier has different hyperparameters.
    """

    def __init__(self):
        """
        Initialize the ModelTrainer class.
        """

        self.classifier = None
        self.param_grid = None

    def grid_search(self, model):
        """
        Perform grid search for hyperparameter tuning and return the best model.

        Args:
        model (str): The type of classifier to train. Supported values: 'LogisticRegression', 'RandomForest', 'SVM', we could add more then.

        Returns:
        grid_search (GridSearchCV): A GridSearchCV object configured with the specified classifier and parameter grid.
        """

        if model == "LogisticRegression":
            self.classifier = LogisticRegression(random_state=42)
            self.param_grid = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2']
            }
        elif model == "RandomForest":
            self.classifier = RandomForestClassifier(random_state=42)
            self.param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model == "SVM":
            self.classifier = SVC(probability=True, random_state=42)
            self.param_grid = {
                'C': [1.0],
                'kernel': ['linear'],
                'gamma': ['auto']
            }
        elif model == "XGBClassifier":
            self.classifier = XGBClassifier(random_state=42, use_label_encoder=False)
            self.param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.1],
                'subsample': [1.0]
            }
        elif model == "DecisionTree":
            self.classifier = DecisionTreeClassifier(random_state=42)
            self.param_grid = {
                'max_depth': [10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'criterion': ['gini']
            }

        else:
            raise ValueError("Invalid model type. Supported models: 'LogisticRegression', 'RandomForest', 'SVM'")

        grid_search = GridSearchCV(estimator=self.classifier, param_grid=self.param_grid, cv=5)

        return grid_search