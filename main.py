import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from skelearn.metrics import accuracy_score

dataset= pd.read_csv('train.csv')

