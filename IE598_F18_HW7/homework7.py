import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                       header=None)
df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']

#Part 1
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

SEED = 1

X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.1,random_state=SEED,stratify=y)

feat_labels = df_wine.columns[1:]

params=[1,2,3,4,5,6,7,8,9,10]

accurary_score = []


for n in params:
    start = time.time()
    forest = RandomForestClassifier(criterion = 'gini', n_estimators = n, random_state = 1, n_jobs = 2)
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    accurary_score.append(metrics.accuracy_score(y_train, y_train_pred))
    timeout = (time.time() - start)
    print(timeout)

    
plt.plot(params, accurary_score)
plt.ylabel("Accurarcy")
plt.xlabel("N_list")
plt.title("Random Forest")
plt.show()
#Part 2
feat_labels = df_wine.columns[1:]
rf=RandomForestClassifier(criterion = 'gini', n_estimators = 5, random_state = 1, n_jobs = 2)
rf.fit(X_train,y_train)
importances = rf.feature_importances_

indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

print("My name is Mengming Guo")
print("My NetID is: mg28")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
