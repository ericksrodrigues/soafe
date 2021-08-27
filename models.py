import pandas as pd

from sklearn.tree import  DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

df_diabetes = pd.read_csv('datasets/diabetes.csv')
df_diabetes.head()