import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from main import rf_prediction, lr_prediction, gbm_prediction, svm_prediction, y_test, cm_labels

data_loc = "../Training.csv"
data = pd.read_csv(data_loc)

X = data.drop("prognosis", axis=1)
y = data["prognosis"]


def heat_map():
    plt.figure(figsize=(40, 30))
    sns.heatmap(data=X.corr(), cmap='coolwarm')
    plt.title('Correlations of Symptoms')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def rf_confusion():
    cf_matrix = confusion_matrix(y_test, rf_prediction(), labels=cm_labels)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cf_matrix, annot=True, xticklabels=cm_labels, yticklabels=cm_labels, cbar=False)
    plt.yticks(rotation=360)
    plt.show()


def lr_confusion():
    cf_matrix = confusion_matrix(y_test, lr_prediction())
    plt.figure(figsize=(20, 20))
    sns.heatmap(cf_matrix, annot=True, xticklabels=cm_labels, yticklabels=cm_labels, cbar=False)
    plt.yticks(rotation=360)
    plt.show()


def gbm_confusion():
    cf_matrix = confusion_matrix(y_test, gbm_prediction())
    plt.figure(figsize=(20, 20))
    sns.heatmap(cf_matrix, annot=True, xticklabels=cm_labels, yticklabels=cm_labels, cbar=False)
    plt.yticks(rotation=360)
    plt.show()


def svm_confusion():
    cf_matrix = confusion_matrix(y_test, svm_prediction())
    plt.figure(figsize=(20, 20))
    sns.heatmap(cf_matrix, annot=True, xticklabels=cm_labels, yticklabels=cm_labels, cbar=False)
    plt.yticks(rotation=360)
    plt.show()


def count():
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='prognosis')
    plt.xticks(rotation=90)
    plt.xlabel('Diseases')
    plt.ylabel('Frequency of Diseases')
    plt.tight_layout()
    plt.show()


rf_confusion()
lr_confusion()
gbm_confusion()
svm_confusion()
count()
heat_map()

