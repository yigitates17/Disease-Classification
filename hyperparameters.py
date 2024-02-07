from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import numpy as np

data_loc = "../Training.csv"
data = pd.read_csv(data_loc)
X = data.drop("prognosis", axis=1)
y = data["prognosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=354792)

# Define the grid of hyperparameters to search
rf_param_grid = {
    'n_estimators': range(50, 200, 25),
    'max_depth': [3, 5, 7],
    'max_features': np.arange(0.1, 0.5, 0.1),
    'min_samples_split': [2, 3]
}

lr_param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001],
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    'class_weight': ['balanced']
}

gbm_param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.1, 0.01],
    'subsample': [0.8],
    'min_samples_split': [2, 3, 4]
}

svm_param_grid = {
    'C': np.arange(0.1, 1.0, 0.1),
    'gamma': [0.1, 0.01, 0.001],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

# Create grid search objects
rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=rf_param_grid, cv=5, scoring="accuracy")
lr_grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=lr_param_grid, cv=5, n_jobs=-1,
                              scoring="accuracy")
gbm_grid_search = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=gbm_param_grid, cv=5,
                               scoring="accuracy")
svm_grid_search = GridSearchCV(estimator=SVC(), param_grid=svm_param_grid, cv=5, scoring="accuracy", refit=True)

# Fit the grid search object
rf_grid_search.fit(X_train, y_train)
lr_grid_search.fit(X_train, y_train)
gbm_grid_search.fit(X_train, y_train)
svm_grid_search.fit(X_train, y_train)

# Get the best hyperparameters
rf_best_params = rf_grid_search.best_params_
lr_best_params = lr_grid_search.best_params_
gbm_best_params = gbm_grid_search.best_params_
svm_best_params = svm_grid_search.best_params_
