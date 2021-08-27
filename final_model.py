
import pandas as pd

from sklearn.tree import  DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import SimpSOM as sps
from sklearn.preprocessing import normalize
from sklearn.preprocessing import normalize
import random
import json

data_path = "./datasets/megawatt1.csv"
accuracy_array_normal = []
for x in range(30):
  df = pd.read_csv(data_path)
  df.head()
  g = df.groupby('class')
  df=g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
  X_train, X_test, y_train, y_test = train_test_split(df.drop('class',axis=1),df['class'],test_size=0.3,stratify=df['class'])
  X_train = normalize(X_train)
  X_test = normalize(X_test)
  X_train.shape,X_test.shape

  df_describe = pd.DataFrame(y_test)
  print(df_describe.describe())


  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import GridSearchCV
  from sklearn.model_selection import KFold
  clf = RandomForestClassifier()
  clf = GridSearchCV(estimator=clf, cv=KFold(10), param_grid={}, scoring='f1')
  clf.fit(X_train, y_train)
  results = clf.predict(X_test)

  print("F1 score", metrics.f1_score(y_test,results))
  print("Accuracy Score", metrics.accuracy_score(y_test,results))
  accuracy_array_normal.append(metrics.f1_score(y_test,results))

with open('results_normal.txt', 'w') as outfile:
  json.dump(accuracy_array_normal, outfile)


