from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score



iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
in_scores = []
out_scores = []

for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i, stratify=y)
    tree = DecisionTreeClassifier(criterion='gini',
                                  max_depth=4,
                                  random_state=1)
    tree.fit(X_train,y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)

    score1 = metrics.accuracy_score(y_train, y_train_pred)
    score2 = metrics.accuracy_score(y_test, y_test_pred)
    
    in_scores.append(metrics.accuracy_score(y_train, y_train_pred))  
    out_scores.append(metrics.accuracy_score(y_test, y_test_pred))
    
    print('random state: %d, in sample: %.3f, out of sample: %.3f' % (i,score1,score2))

print("                 ")
print("In sample mean:",np.mean(in_scores),"        std dev:",np.std(in_scores))
print("Out of sample mean:",np.mean(out_scores),"    std dev:",np.std(out_scores))

#Part2

kfold = KFold(n_splits=10,random_state=1).split(X_train,y_train)
tree.fit(X_train,y_train)
cv_score = cross_val_score(estimator=tree,X=X_train,y=y_train,cv=10,n_jobs=1)
y_pred_test=tree.predict(X_test)
out_score=metrics.accuracy_score(y_test,y_pred_test)
print("Accuracy scores:", cv_score)
print("mean:",np.mean(cv_score),", std:", np.std(cv_score),)
print("Out of sample score:", out_score)

print("My name is Mengming guo")
print("My NetID is: mg28")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")