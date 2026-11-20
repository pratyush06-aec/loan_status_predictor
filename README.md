# Loan Status Predictor

## 1. Project Overview

This project is a binary loan-approval prediction experiment written in Python. It reads a tabular loan application dataset, converts selected categorical values into numeric form, handles missing values, scales numerical features, and trains a K-Nearest Neighbors (KNN) classifier.

The model learns to predict whether a loan application should be **Approved** or **Rejected**.

- **Input data:** `train.csv`
- **Main program:** `main.py`
- **Dependencies:** `requirements.txt`
- **Target variable:** `Loan_Status`
- **Approved label:** `1`
- **Rejected label:** `0`
- **Evaluation split:** 90% training, 10% testing

This repository is primarily an educational machine-learning workflow rather than a deployable loan decision system. It is not intended to replace regulated, audited, and human-reviewed lending decisions.

---

## 2. Dataset Details

The project uses `train.csv`, which contains 3,192 records and 14 columns.

| Column | Description |
|---|---|
| `Loan_ID` | Unique application identifier |
| `Gender` | Applicant gender |
| `Married` | Applicant marital status |
| `Dependents` | Number of dependents |
| `Education` | Graduate or Not Graduate |
| `Employment_Status` | Unemployed, Self-Employed, or Salaried |
| `Applicant_Income` | Applicant income |
| `Coapplicant_Income` | Co-applicant income |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Loan term, in months |
| `Credit_History` | Credit-history indicator |
| `Property_Area` | Rural, Semiurban, or Urban |
| `Age` | Applicant age |
| `Loan_Status` | Approved or Rejected |

The dataset is perfectly balanced:

- Approved: 1,596 records (50%)
- Rejected: 1,596 records (50%)

### Missing Values

The original dataset contains missing values in the following columns:

| Column | Missing values |
|---|---:|
| `Gender` | 77 |
| `Married` | 66 |
| `Education` | 66 |
| `Employment_Status` | 56 |
| `Applicant_Income` | 35 |
| `Loan_Amount` | 41 |

All other columns in the dataset have no missing values. `Coapplicant_Income`, `Loan_Term`, `Credit_History`, `Age`, `Property_Area`, `Loan_ID`, `Dependents`, and `Loan_Status` have zero missing values.

The script removes `Loan_ID`, `Gender`, `Married`, and `Loan_Status` from the feature set. It therefore does not require imputation for `Gender` or `Married`; they are excluded before training.

---

## 3. Machine-Learning Pipeline

### 3.1 Load and Inspect the Dataset

`pandas.read_csv()` loads `train.csv` as a DataFrame. The script prints:

- the Python and pandas object type
- the first five records
- the full row and column shape
- descriptive statistics
- missing-value counts

### 3.2 Encode the Target

`Loan_Status` is converted from text to a binary label:

```text
Rejected  -> 0
Approved  -> 1
```

### 3.3 Encode Categorical Features

The following columns are mapped to numeric values:

| Column | Mapping |
|---|---|
| `Education` | Not Graduate: 0, Graduate: 1 |
| `Employment_Status` | Unemployed: 0, Self-Employed: 1, Salaried: 2 |
| `Property_Area` | Rural: 0, Semiurban: 1, Urban: 2 |

The script no longer encodes `Gender` or `Married` because both columns are excluded from the feature matrix.

### 3.4 Select Features and Target

The feature matrix contains 10 columns:

- `Dependents`
- `Education`
- `Employment_Status`
- `Applicant_Income`
- `Coapplicant_Income`
- `Loan_Amount`
- `Loan_Term`
- `Credit_History`
- `Property_Area`
- `Age`

The target vector is `Loan_Status`.

`Loan_ID`, `Loan_Status`, `Married`, and `Gender` are excluded. `Loan_ID` is an identifier rather than a predictive feature, while `Gender` and `Married` are explicitly excluded from this experiment.

### 3.5 Split the Data

The dataset is split with:

- test size: 10%
- stratification by `Loan_Status`
- random state: `2`

Stratification preserves the class balance in both partitions:

- training records: 2,872
- testing records: 320
- feature columns: 10

### 3.6 Handle Missing Values

Imputation is fitted only on training data and then applied to test data:

- categorical features: most-frequent (`mode`) imputation
- numerical features: median imputation

The categorical imputation set is:

- `Education`
- `Employment_Status`
- `Property_Area`

The numerical imputation set is:

- `Applicant_Income`
- `Loan_Amount`

This design prevents test-data information from influencing the training imputers. It is preferable to simply removing every row containing a missing value because only some rows need imputation and more data remains available for learning.

### 3.7 Scale Features

`StandardScaler` standardizes the feature matrix using z-score scaling:

```text
scaled value = (value - training mean) / training standard deviation
```

The scaler is fitted on `x_train` and then used to transform `x_test`. This is necessary for KNN because KNN calculates distances, and unscaled features such as income or loan amount could otherwise dominate smaller-range features.

### 3.8 Train the KNN Model

The active classifier is:

```python
KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    metric='minkowski',
    p=2
)
```

With `metric='minkowski'` and `p=2`, the effective distance metric is Euclidean distance.

The script also contains a live elapsed-time indicator while the model is fitting. It creates a background thread and updates a `tqdm` progress display every 0.5 seconds. The event is set in a `finally` block so the timer thread terminates even if model fitting raises an exception.

The timer prints the fitting description as “SVM Training” even though the active model is KNN. This is a stale label left from the earlier SVM implementation.

### 3.9 Evaluate the Model

The script calculates:

- accuracy
- recall
- precision
- F1 score
- classification report
- confusion matrix

Metrics are printed for both training and test data.

---

## 4. Measured Results

The following results were obtained by running `main.py` in the project's `.venv`.

### Training Data

| Metric | Result |
|---|---:|
| Accuracy | 98.8162% |
| Recall | 99.7911% |
| Precision | 97.8825% |
| F1 score | 98.8276% |

Training confusion matrix:

```text
                Predicted Rejected  Predicted Approved
Actual Rejected               1405                 31
Actual Approved                  3               1433
```

- true negatives: 1,405
- false positives: 31
- false negatives: 3
- true positives: 1,433

### Test Data

| Metric | Result |
|---|---:|
| Accuracy | 98.7500% |
| Recall | 99.3750% |
| Precision | 98.1481% |
| F1 score | 98.7578% |

Test confusion matrix:

```text
                Predicted Rejected  Predicted Approved
Actual Rejected                157                  3
Actual Approved                  1                159
```

- true negatives: 157
- false positives: 3
- false negatives: 1
- true positives: 159

The test set has 160 Approved and 160 Rejected records. The model misclassified four test records in total.

### Result Interpretation

The reported test accuracy is high. However, accuracy should not be considered sufficient evidence of real-world reliability:

- the dataset is synthetic or experimental-looking in structure and should be validated before production use
- the split is a random in-sample holdout, not time-based, cross-validation, or external validation
- the 320-record test set is relatively small
- the 50/50 class balance may not represent real lending application distributions
- no cross-validation or hyperparameter search is performed
- no model artifact or deployment pipeline is included
- the model is not trained to detect bias or fairness risks

The small gap between training accuracy (98.82%) and test accuracy (98.75%) does not show substantial overfitting in this specific split, but it is not enough to prove generalization.

---

## 5. Environment Setup

Run all commands from the repository root.

### Windows PowerShell

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

### Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

The virtual environment exists as `.venv` in this workspace. Use the interpreter inside the environment instead of a global Python interpreter:

```powershell
.\.venv\Scripts\python.exe main.py
```

---

## 6. Dependencies

The main runtime dependencies are:

- NumPy
- pandas
- seaborn
- scikit-learn
- matplotlib
- tqdm
- colorama
- SciPy

Supporting packages, such as joblib, cycler, Pillow, and date-handling packages, are also pinned in `requirements.txt`.

Package versions can be listed with:

```powershell
python -m pip freeze
```

Recreate the lock set with:

```powershell
python -m pip install -r requirements.txt
```

Pinning exact versions improves reproducibility, but it can make installation or upgrades more difficult if the pinned versions are unavailable for a user's Python version or operating system.

---

## 7. Project History and Evolution

Git history shows that the project evolved through several stages:

1. Dataset loading and dependency setup.
2. Handling missing values.
3. Encoding `Loan_Status`.
4. Initial model training and testing.
5. Expansion of metrics and model evaluation.
6. Replacement of the initial linear SVM approach with KNN.
7. Addition of imputers and feature scaling.

The old linear SVM code remains only as a commented example:

```python
# classifier = svm.SVC(kernel='linear')
```

The KNN implementation is the active model. The `SVM Training` timer label is therefore a legacy label and should not be interpreted as evidence that SVM is currently executing.

---

## 8. Drawbacks and Challenges Faced

### 8.1 NumPy Was Missing

**Observed error:**

```text
Traceback (most recent call last):
  File "D:\loan_status_predictor\main.py", line 1, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

**Cause:** the Python interpreter used to execute the script did not have NumPy installed or was not associated with the project virtual environment.

**Troubleshooting:**

1. Confirm the active interpreter:

   ```powershell
   python --version
   python -m pip --version
   ```

2. Activate the project environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install requirements:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Execute with the environment's interpreter:

   ```powershell
   .\.venv\Scripts\python.exe main.py
   ```

The project environment's interpreter successfully loaded NumPy and all runtime dependencies.

### 8.2 Package Installation Mistakes

The saved terminal history shows several misspelled installation commands:

```text
pip install pandas matplolib sckit-learn numpy seaborn imblearn
pip install pandas matplotlib sckit-learn numpy seaborn imblearn
pip install requirements.ttx
pip install requirements.txt
```

Problems included:

- misspelled package names
- `matplolib` instead of `matplotlib`
- `sckit-learn` instead of `scikit-learn`
- attempting to install `requirements.txt` as a package name
- incorrect filename `requirements.ttx`

**Correct resolution:**

```powershell
python -m pip install matplotlib scikit-learn numpy pandas seaborn imbalanced-learn
python -m pip install -r requirements.txt
```

The `-r` option is required when installing dependencies from a requirements file.

### 8.3 Confusion Between Python Interpreters

This project has both a system/global Python environment and a `.venv` environment. A package can appear installed in the global environment and still be unavailable in the virtual environment, or vice versa.

**Troubleshooting:**

```powershell
python -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
python -m pip show numpy
.\.venv\Scripts\python.exe -m pip show numpy
```

Always use `python -m pip` rather than bare `pip` so the package installer is paired with the interpreter currently executing the script.

### 8.4 Typographical Command Errors

Earlier terminal history includes commands such as:

- `python main.oy` instead of `python main.py`
- `pip install requirements.ttx` instead of `pip install -r requirements.txt`
- `git push origin mian` instead of `git push origin main`

These were command-entry mistakes rather than project-code failures.

**Resolution:** verify filenames and remotes exactly, use shell completion where available, and re-run the correctly spelled command.

### 8.5 Missing Values

The dataset initially contains missing values in `Gender`, `Married`, `Education`, `Employment_Status`, `Applicant_Income`, and `Loan_Amount`.

An earlier implementation dropped all rows containing missing values. This can discard useful records and was replaced by split-safe imputation.

**Resolution:**

- keep and encode the encoded categorical values
- use the most frequent value for categorical features
- use the median for numerical features
- fit imputers only on training data
- transform training and testing data consistently

### 8.6 Replacing All DataFrame Values Incorrectly

The original code used DataFrame-wide `replace()` calls to encode several columns. The current code uses named-column `.map()` calls and removes the target/identifier fields explicitly.

Potential drawbacks of broad replacement include accidentally encoding the wrong value in a future column or changing unexpected text values. Named mappings make the intended transformations easier to audit.

### 8.7 Feature Scaling and Data Leakage

KNN is distance-based, so unscaled features can bias distance calculations.

**Resolution:**

- fit `StandardScaler` only on training features
- transform test features using the fitted training scaler
- do not fit the scaler on the complete dataset before splitting

The current script follows this order.

### 8.8 Duplicate Imputation and Scaling Risk

`train_test_split` creates separate copies of training and test features. The script copies those DataFrames before imputation and then separately transforms both sets.

This is important because:

- imputers must not mutate training data through a shared reference unexpectedly
- test data must use training-derived imputation values
- test data must use training-derived scaling statistics

### 8.9 SVM-to-KNN Transition

The project originally used a linear SVM:

```python
svm.SVC(kernel='linear')
```

It was later replaced with KNN. The associated timer function still uses the description `SVM Training`, so users may misread the output.

**Resolution:** the README clearly documents that KNN is currently active. In a future update, the timer description should be made configurable or renamed to `Model Training`.

### 8.10 Progress Timer Naming

The elapsed-time display can look like a SVM-specific progress report even though KNN is currently used.

**No training failure occurred:** the timer finished immediately during the successful run because KNN training is fast for this dataset.

### 8.11 Terminal Output Is Very Verbose

The script prints the complete DataFrame, descriptive statistics, missing-value tables, and full classification reports.

This is useful for learning and debugging, but it can make repeat runs difficult to inspect.

**Recommended future troubleshooting approach:**

- print summaries instead of full DataFrames
- use a logging library
- separate data exploration scripts from training scripts
- write metrics to a CSV or JSON file
- provide command-line options for optional verbose output

### 8.12 No Model Persistence

The script trains a model but does not save it. Re-running the script retrains it from scratch.

Missing artifacts include:

- saved model file
- saved scaler
- saved imputer objects
- label mappings
- feature schema
- training/evaluation report

This also prevents the current model from being used in a separate prediction application.

### 8.13 No Prediction Interface

The script currently performs training and evaluation only. It has no:

- command-line prediction command
- web endpoint
- interactive form
- batch prediction input format

A future implementation would need to load the scaler, imputers, and KNN model and apply the same transformations to new records.

### 8.14 No Cross-Validation or Hyperparameter Tuning

The chosen KNN setting is `k=5`, but it was not established by validation or cross-validation.

Better practice would include:

- repeated or stratified k-fold cross-validation
- testing multiple values of `k`
- testing different distance metrics
- comparing KNN with SVM, logistic regression, random forest, or gradient boosting
- reporting mean and standard deviation across folds
- using a separate untouched test set

### 8.15 Small Test Set and Holdout-Only Evaluation

The test set contains 320 records. Although balanced, a 320-row holdout can have noticeable sampling variance.

Cross-validation is more robust and should be added before making conclusions about the model's performance on future applications.

### 8.16 Accuracy Reporting in a Financial Context

Loan approval is a consequential decision. High accuracy alone does not establish:

- fairness across protected or sensitive groups
- acceptable false-positive and false-negative costs
- compliance with local lending rules
- protection from discrimination
- explanation quality
- reliability under data drift

False approvals and false rejections can both cause harm. Approval decisions should be independently governed and audited.

### 8.17 Restricted Feature Handling

`Gender` and `Married` are removed from the active feature set. That is safer for the limited scope of this exercise, but their removal may reduce information available to the model and should be documented as an explicit design choice rather than an accidental omission.

Similarly, no feature-importance or error analysis is included.

### 8.18 Potential pandas and NumPy Version Sensitivity

`requirements.txt` includes exact package versions. Reusing the same environment improves consistency, but upgrades to pandas, NumPy, or scikit-learn can introduce behavior or API differences.

Before changing package versions:

1. create a backup or branch
2. create a clean virtual environment
3. install the new versions
4. run `main.py`
5. compare class distributions, metrics, and output
6. update this README only after confirming behavior

---

## 9. Troubleshooting Guide

### Script Cannot Import NumPy or pandas

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

### PowerShell Blocks the Virtual Environment Activation Script

Allow activation only for the current PowerShell process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

For a persistent alternative, execute the virtual environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe main.py
```

### `train.csv` Is Not Found

Run the script from the repository directory:

```powershell
cd D:\loan_status_predictor
.\.venv\Scripts\python.exe main.py
```

The script currently uses the relative path `train.csv`, so changing the working directory to another folder may cause `FileNotFoundError`.

### Missing or Incorrect Packages

Check key imports:

```powershell
.\.venv\Scripts\python.exe -c "import numpy, pandas, sklearn, matplotlib, seaborn, tqdm; print('Dependencies available')"
```

If one import fails, reinstall dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade -r requirements.txt
```

### Model Metrics Differ

Possible reasons include:

- different data
- altered split
- changed `random_state`
- changed imputation strategy
- changed feature set
- changed scaling order
- changed package versions
- changed KNN parameters

Compare the code and data against this README and regenerate the split before diagnosing a meaningful model change.

---

## 10. Repository Structure

```text
loan_status_predictor/
|-- main.py
|-- train.csv
|-- requirements.txt
|-- README.md
`-- .venv/
```

### `main.py`

Contains the complete data-loading, encoding, splitting, imputation, scaling, training, and evaluation workflow.

### `train.csv`

Contains the 3,192 loan application records.

### `requirements.txt`

Contains exact package versions for the project environment.

### `.venv`

Contains the local virtual Python environment.

---

## 11. Run Summary

From a correctly configured environment, the expected workflow is:

1. Load `train.csv`.
2. Print dataset structure and missing values.
3. Encode `Loan_Status`.
4. Encode `Education`, `Employment_Status`, and `Property_Area`.
5. Remove `Loan_ID`, `Loan_Status`, `Married`, and `Gender` from the feature matrix.
6. Split 90% for training and 10% for testing.
7. Fit imputation and scaling objects on training data.
8. Fit a five-neighbor KNN classifier.
9. Print training and test metrics.

The final test result was 98.75% accuracy, with four misclassified records out of 320.

---

## 12. Known Limitations

- No hyperparameter optimization
- No cross-validation
- No model serialization
- No unseen-data prediction interface
- No automatic test suite
- No logging configuration
- No configurable paths or command-line arguments
- No separate data-validation module
- No fairness or bias evaluation
- No feature importance or model explainability
- No production preprocessing pipeline
- No train/validation/test split
- No deployment configuration
- The timer label incorrectly says “SVM Training”
- The script prints large portions of the dataset, making logs long
- The current imputation and scaler objects are not saved for reuse
- The KNN algorithm can become slow or memory-intensive on much larger datasets

---

## 13. Recommended Improvements

1. Replace the custom timer with a configurable training timer or remove it.
2. Rename the timer description to `Model Training`.
3. Create a scikit-learn `Pipeline` with imputation and scaling.
4. Use a single `ColumnTransformer` for mixed data types.
5. Save the full preprocessing and model pipeline with `joblib`.
6. Add a command-line prediction mode.
7. Add unit tests for mappings, missing-value handling, and split lengths.
8. Add stratified k-fold cross-validation.
9. Compare KNN with several baseline classifiers.
10. Tune `n_neighbors`, weighting, and distance metrics.
11. Add error analysis for false positives and false negatives.
12. Add fairness and subgroup analysis.
13. Separate logging and exploratory printing.
14. Replace relative paths with paths derived from `__file__`.
15. Add an explicit `pandas.read_csv` validation step.
16. Add an external test dataset.
17. Add model and scaler version information.
18. Document whether the dataset is synthetic or real and how it was collected.
