from sklearn.ensemble import RandomForestRegressor

# To use the experimental IterativeImputer, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import numpy as np

rng = np.random.RandomState(42)
regressor = RandomForestRegressor(random_state=0)
N_SPLITS = 4


def get_scores_for_imputer(imputer, X, Y):
    estimator = make_pipeline(imputer, regressor)
    impute_scores = cross_val_score(
        estimator, X, Y, scoring="neg_mean_squared_error", cv=N_SPLITS
    )
    return impute_scores


def impute_zero_score(X, Y):
    imputer = SimpleImputer(
        missing_values=np.nan, add_indicator=True, strategy="constant", fill_value=0
    )
    return imputer.fit_transform(X, Y)


def get_impute_zero_score(X, Y):
    imputer = SimpleImputer(
        missing_values=np.nan, add_indicator=True, strategy="constant", fill_value=0
    )
    zero_impute_scores = get_scores_for_imputer(imputer, X, Y)
    return zero_impute_scores.mean(), zero_impute_scores.std()

def impute_knn_score(X, Y):
    imputer = KNNImputer(missing_values=np.nan, add_indicator=True)
    return imputer.fit_transform(X, Y)


def get_impute_knn_score(X, Y):
    imputer = KNNImputer(missing_values=np.nan, add_indicator=True)
    knn_impute_scores = get_scores_for_imputer(imputer, X, Y)
    return knn_impute_scores.mean(), knn_impute_scores.std()

def impute_mean(X, Y):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean", add_indicator=True)
    return imputer.fit_transform(X, Y)


def get_impute_mean(X, Y):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean", add_indicator=True)
    mean_impute_scores = get_scores_for_imputer(imputer, X, Y)
    return mean_impute_scores.mean(), mean_impute_scores.std()

def impute_iterative(X, Y):
    imputer = IterativeImputer(
        missing_values=np.nan,
        add_indicator=True,
        random_state=0,
        n_nearest_features=10,
        max_iter=1,
        sample_posterior=True,
    )
    return imputer.fit_transform(X, Y)


def get_impute_iterative(X, Y):
    imputer = IterativeImputer(
        missing_values=np.nan,
        add_indicator=True,
        random_state=0,
        n_nearest_features=3,
        max_iter=1,
        sample_posterior=True,
    )
    iterative_impute_scores = get_scores_for_imputer(imputer, X, Y)
    return iterative_impute_scores.mean(), iterative_impute_scores.std()
