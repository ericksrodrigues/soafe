

from sklearn.ensemble import RandomForestClassifier
import timeit
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel

data_path = "./datasets/ionosphere.csv"

time = []
results_l = []
for i in range(30):
  start = timeit.default_timer()
  df = pd.read_csv(data_path)
  X_train, X_test, y_train, y_test = train_test_split(df.drop('class',axis=1),df['class'],test_size=0.3,stratify=df['class'])
  fs = SelectFromModel(RandomForestClassifier().fit(X_train, y_train), prefit=True)
  X_train = fs.transform(X_train)
  X_test = fs.transform(X_test)
  clf = RandomForestClassifier()
  clf.fit(X_train, y_train)
  results = clf.predict(X_test)
  print(metrics.f1_score(results,y_test))
  results_l.append(metrics.f1_score(results,y_test))
  stop = timeit.default_timer()
  time.append(stop-start)

print(results_l)
  