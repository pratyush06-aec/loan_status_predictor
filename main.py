import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

loan_dataset= pd.read_csv('train.csv')
print(type(loan_dataset))
print(loan_dataset.head())
print(loan_dataset.shape)
print(loan_dataset.describe())
print(loan_dataset.isnull().sum())

# --- Step 1: Optional but important — flag missingness before filling it ---
# If missingness itself correlates with Loan_Status (e.g. people who don't
# report income get rejected more), these flags become useful features.
for col in ['Gender', 'Married', 'Education', 'Employment_Status',
            'Applicant_Income', 'Loan_Amount']:
    loan_dataset[f'{col}_was_missing'] = loan_dataset[col].isna().astype(int)

# --- Step 2: Categorical — impute with mode ---
cat_cols = ['Gender', 'Married', 'Education', 'Employment_Status']
for col in cat_cols:
    loan_dataset[col] = loan_dataset[col].fillna(loan_dataset[col].mode()[0])

# --- Step 3: Numeric — impute with median (robust default) ---
num_cols = ['Applicant_Income', 'Loan_Amount']
for col in num_cols:
    loan_dataset[col] = loan_dataset[col].fillna(loan_dataset[col].median())

print(loan_dataset.isnull().sum())

updated_loan_status_column= loan_dataset.replace({"Loan_Status": {"Approved": 1, "Rejected": 0}}, inplace=True)
print(loan_dataset["Loan_Status"].head())