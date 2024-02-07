import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data_loc = "../Training.csv"
data = pd.read_csv(data_loc)
X = data.drop("prognosis", axis=1)
y = data["prognosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=354792)


# Random Forest
def rf_prediction():
    rf_best_params = {'max_depth': 7, 'max_features': 0.1, 'min_samples_split': 2, 'n_estimators': 125}
    rf_model = RandomForestClassifier(**rf_best_params, random_state=67)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    rf_accuracy = accuracy_score(y_test, rf_preds)
    rf_precision = precision_score(y_test, rf_preds, average='macro', zero_division=1)
    rf_recall = recall_score(y_test, rf_preds, average='macro')
    rf_f1 = f1_score(y_test, rf_preds, average='macro')
    rf_sensitivity = recall_score(y_test, rf_preds, average='macro')

    print("RF Accuracy on test set: {:.3f}".format(rf_accuracy))
    print("RF Precision on test set: {:.3f}".format(rf_precision))
    print("RF Recall on test set: {:.3f}".format(rf_recall))
    print("RF F1 score on test set: {:.3f}".format(rf_f1))
    print("RF Sensitivity on test set: {:.3f}".format(rf_sensitivity))

    return rf_preds


# Logistic Regression
def lr_prediction():
    lr_best_params = {'C': 0.01, 'class_weight': 'balanced', 'penalty': 'l2', 'solver': 'saga'}
    lr_model = LogisticRegression(**lr_best_params)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    lr_accuracy = accuracy_score(y_test, lr_preds)
    lr_precision = precision_score(y_test, lr_preds, average="macro", zero_division=1)
    lr_recall = recall_score(y_test, lr_preds, average="macro")
    lr_f1 = f1_score(y_test, lr_preds, average="macro")
    lr_sensitivity = recall_score(y_test, lr_preds, average="macro")

    print("\nLR Accuracy on test set: {:.3f}".format(lr_accuracy))
    print("LR Precision on test set: {:.3f}".format(lr_precision))
    print("LR Recall on test set: {:.3f}".format(lr_recall))
    print("LR F1 score on test set: {:.3f}".format(lr_f1))
    print("LR Sensitivity on test set: {:.3f}".format(lr_sensitivity))

    return lr_preds


# Gradient Boosted Machines
def gbm_prediction():
    gbm_best_params = {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 3, 'n_estimators': 150, 'subsample': 0.8}
    gbm_model = GradientBoostingClassifier(**gbm_best_params, random_state=50)
    gbm_model.fit(X_train, y_train)
    gbm_preds = gbm_model.predict(X_test)

    gbm_accuracy = accuracy_score(y_test, gbm_preds)
    gbm_precision = precision_score(y_test, gbm_preds, average="macro", zero_division=1)
    gbm_recall = recall_score(y_test, gbm_preds, average="macro")
    gbm_f1 = f1_score(y_test, gbm_preds, average="macro")
    gbm_sensitivity = recall_score(y_test, gbm_preds, average="macro")

    print("\nGBM Accuracy on test set: {:.3f}".format(gbm_accuracy))
    print("GBM Precision on test set: {:.3f}".format(gbm_precision))
    print("GBM Recall on test set: {:.3f}".format(gbm_recall))
    print("GBM F1 score on test set: {:.3f}".format(gbm_f1))
    print("GBM Sensitivity on test set: {:.3f}".format(gbm_sensitivity))

    return gbm_preds


# SVM
def svm_prediction():
    svm_best_params = {'C': 0.3, 'gamma': 0.1, 'kernel': 'linear'}
    svm_model = SVC(**svm_best_params)
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)

    svm_accuracy = accuracy_score(y_test, svm_preds)
    svm_precision = precision_score(y_test, svm_preds, average="macro", zero_division=1)
    svm_recall = recall_score(y_test, svm_preds, average="macro")
    svm_f1 = f1_score(y_test, svm_preds, average="macro")
    svm_sensitivity = recall_score(y_test, svm_preds, average="macro")

    print("\nSVM Accuracy on test set: {:.3f}".format(svm_accuracy))
    print("SVM Precision on test set: {:.3f}".format(svm_precision))
    print("SVM Recall on test set: {:.3f}".format(svm_recall))
    print("SVM F1 score on test set: {:.3f}".format(svm_f1))
    print("SVM Sensitivity on test set: {:.3f}".format(svm_sensitivity))

    return svm_preds


rf_prediction()
lr_prediction()
gbm_prediction()
svm_prediction()



