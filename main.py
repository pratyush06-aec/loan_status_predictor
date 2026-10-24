import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import threading
import time

loan_dataset= pd.read_csv('train.csv')

print(type(loan_dataset))

print(loan_dataset.head())

print(loan_dataset.shape)

print(loan_dataset.describe())

print(loan_dataset.isnull().sum())

loan_dataset= loan_dataset.dropna()
print(loan_dataset.isnull().sum())

## Lable encoding 
loan_dataset['Loan_Status']= loan_dataset['Loan_Status'].map({'Rejected':0,'Approved':1})
print(loan_dataset)

print(loan_dataset.shape)

dependents_count= loan_dataset['Dependents'].value_counts()
print(dependents_count)

sns.countplot(x='Married', hue='Loan_Status',data=loan_dataset)
plt.show()

loan_dataset.replace({"Married":{'No':0,'Yes':1}},inplace=True)
loan_dataset.replace({"Gender":{'Female':0,'Male':1}},inplace=True)
loan_dataset.replace({"Education":{'Not Graduate':0,'Graduate':1}},inplace=True)
loan_dataset.replace({"Employment_Status":{'Unemployed':0,'Self-Employed':1, 'Salaried':2}},inplace=True)
loan_dataset.replace({"Property_Area":{'Rural':0,'Semiurban':1,'Urban':2}},inplace=True)
print(loan_dataset)

x= loan_dataset.drop(columns= ['Loan_ID', 'Loan_Status'])
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


classifier= svm.SVC(kernel='linear')
train_with_live_timer(classifier, x_train, y_train)

x_train_prediction= classifier.predict(x_train)
x_train_accuracy= accuracy_score(x_train_prediction, y_train)
print('Accuracy score of training data:', x_train_accuracy)

x_test_prediction= classifier.predict(x_test)
x_test_accuracy= accuracy_score(x_test_prediction, y_test)
print('Accuracy score of test data:', x_test_accuracy)