import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix 
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import threading
import time
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

loan_dataset= pd.read_csv('train.csv')

print(type(loan_dataset))
print(loan_dataset.head())
print(loan_dataset.shape)
print(loan_dataset.describe())
print(loan_dataset.isnull().sum())
# loan_dataset= loan_dataset.dropna()
# print(loan_dataset.isnull().sum())


loan_dataset['Loan_Status']= loan_dataset['Loan_Status'].map({'Rejected':0,'Approved':1})
print(loan_dataset)
print(loan_dataset.shape)

# dependents_count= loan_dataset['Dependents'].value_counts()
# print(dependents_count)

# sns.countplot(x='Married', hue='Loan_Status',data=loan_dataset)
# plt.show()

loan_dataset['Education']= loan_dataset['Education'].map({'Not Graduate':0,'Graduate':1})
loan_dataset['Employment_Status']= loan_dataset['Employment_Status'].map({'Unemployed':0,'Self-Employed':1, 'Salaried':2})
loan_dataset['Property_Area']= loan_dataset['Property_Area'].map({'Rural':0,'Semiurban':1,'Urban':2})
print(loan_dataset)

x= loan_dataset.drop(columns= ['Loan_ID', 'Loan_Status', "Married", "Gender"])
y= loan_dataset['Loan_Status']
print(x)
print(y)
# print(y.value_counts())
# print(y.value_counts(normalize=True) * 100)

def train_with_live_timer(classifier, x_train, y_train):
    finished = threading.Event()

    def show_timer():
        start = time.monotonic()
        with tqdm(
            total=None,
            desc="SVM Training",
            bar_format="{desc} | Elapsed: {elapsed} | {postfix}",
        ) as pbar:
            while not finished.wait(0.5):
                pbar.set_postfix_str("Training in progress")
                pbar.update(0)
                pbar.refresh()
            pbar.set_postfix_str(
                f"Finished in {time.monotonic() - start:.1f}s"
            )
            pbar.refresh()

    timer_thread = threading.Thread(target=show_timer)
    timer_thread.start()

    try:
        classifier.fit(x_train, y_train)
    finally:
        finished.set()
        timer_thread.join()

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.1,stratify=y,random_state=2)
print(x.shape, x_train.shape, x_test.shape)

x_train = x_train.copy()
x_test = x_test.copy()


categorical_features = ['Education', 'Employment_Status', 'Property_Area']
numerical_features = ['Applicant_Income', 'Loan_Amount']

categorical_imputer = SimpleImputer(strategy='most_frequent')
numerical_imputer = SimpleImputer(strategy='median')

x_train[categorical_features] = categorical_imputer.fit_transform(x_train[categorical_features])
x_train[numerical_features] = numerical_imputer.fit_transform(x_train[numerical_features])
x_test[categorical_features] = categorical_imputer.transform(x_test[categorical_features])
x_test[numerical_features] = numerical_imputer.transform(x_test[numerical_features])

print("\nMissing values in training data:")
print(x_train.isnull().sum())

print("\nMissing values in testing data:")
print(x_test.isnull().sum())


# classifier= svm.SVC(kernel='linear')

knn_model = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski', p=2)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
knn_model.fit(x_train_scaled, y_train)

# baseline_prediction = x_test['Credit_History']

# print("\nCredit History Only Baseline")
# print("Accuracy:", accuracy_score(y_test, baseline_prediction))
# print(classification_report(y_test, baseline_prediction, zero_division=0))
train_with_live_timer(knn_model, x_train_scaled, y_train)

x_train_prediction= knn_model.predict(x_train_scaled)
x_train_accuracy= accuracy_score(y_train, x_train_prediction)
print('Accuracy score of training data:', x_train_accuracy)
print('Recall score of training data:', recall_score(y_train, x_train_prediction))
print('Precision score of training data:', precision_score(y_train, x_train_prediction))
print('F1 score of training data:', f1_score(y_train, x_train_prediction))
print('Classification report of training data:')
print(classification_report(y_train, x_train_prediction))
print('Confusion matrix of training data:')
print(confusion_matrix(y_train, x_train_prediction))

x_test_prediction= knn_model.predict(x_test_scaled)
x_test_accuracy= accuracy_score(y_test, x_test_prediction)
print('Accuracy score of test data:', x_test_accuracy)

print('Recall score of test data:', recall_score(y_test, x_test_prediction))
print('Precision score of test data:', precision_score(y_test, x_test_prediction))
print('F1 score of test data:', f1_score(y_test, x_test_prediction))
print('Classification report of test data:')
print(classification_report(y_test, x_test_prediction))
print('Confusion matrix of test data:')
print(confusion_matrix(y_test, x_test_prediction))