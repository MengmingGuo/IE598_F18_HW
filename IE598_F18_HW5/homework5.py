import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/' 'machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 
                   'Malic acid', 'Ash', 
                   'Alcalinity of ash', 
                   'Magnesium',
                   'Total phenols',
                   'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins', 
                   'Color intensity',
                   'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']
X = df_wine.iloc[:,1:].values
y = df_wine.iloc[:,0].values
print('Class labels', np.unique(df_wine['Class label']))
df_wine.describe()


#head map
cm = np.corrcoef(df_wine[df_wine.columns].values.T)
sns.set(font_scale = 1)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size':6},
                 yticklabels=df_wine.columns,
                 xticklabels=df_wine.columns,
                 )
plt.show()

#pair plot
sns.pairplot(df_wine[df_wine.columns], size=2.5)
plt.tight_layout()
plt.show()

#split train, test sub
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=42)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))


#Part 2  
#LR
lr = LogisticRegression(C=100.0,random_state =1)
lr.fit(X_train_std, y_train)
y_train_pred = lr.predict(X_train_std)
y_test_pred = lr.predict(X_test_std)
print('Train Accuracy :' + str(accuracy_score(y_train, y_train_pred)*100) + '%')
print('Test Accuracy :' + str(accuracy_score(y_test, y_test_pred)*100) + '%')


#SVM
svm = SVC(kernel='rbf', random_state=1, C=1.0)
svm.fit(X_train_std, y_train)
y_train_pred = svm.predict(X_train_std)
y_test_pred = svm.predict(X_test_std)
print('Train Accuracy :' + str(accuracy_score(y_train, y_train_pred)*100) + '%')
print('Test Accuracy :' + str(accuracy_score(y_test, y_test_pred)*100) + '%')


#PCA 
pca = PCA(n_components=2)


X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


#LR via PCA
lr.fit(X_train_pca, y_train)
y_train_pred = lr.predict(X_train_pca)
y_test_pred = lr.predict(X_test_pca)
print('Train Accuracy :' + str(accuracy_score(y_train, y_train_pred)*100) + '%')
print('Test Accuracy :' + str(accuracy_score(y_test, y_test_pred)*100) + '%')

#svm via pca
svm.fit(X_train_pca, y_train)
y_train_pred = svm.predict(X_train_pca)
y_test_pred = svm.predict(X_test_pca)
print('Train Accuracy :' + str(accuracy_score(y_train, y_train_pred)*100) + '%')
print('Test Accuracy :' + str(accuracy_score(y_test, y_test_pred)*100) + '%')

#LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

#LR via LDA
lr.fit(X_train_lda, y_train)
y_train_pred = lr.predict(X_train_lda)
y_test_pred = lr.predict(X_test_lda)
print('Train Accuracy :' + str(accuracy_score(y_train, y_train_pred)*100) + '%')
print('Test Accuracy :' + str(accuracy_score(y_test, y_test_pred)*100) + '%')

#SVM via LDA
svm.fit(X_train_lda, y_train)
y_train_pred = svm.predict(X_train_lda)
y_test_pred = svm.predict(X_test_lda)
print('Train Accuracy :' + str(accuracy_score(y_train, y_train_pred)*100) + '%')
print('Test Accuracy :' + str(accuracy_score(y_test, y_test_pred)*100) + '%')

#KPCA
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_train_skernpca = scikit_kpca.fit_transform(X_train_std)
X_test_skernpca = scikit_kpca.transform(X_test_std)


#lr via kpca
lr.fit(X_train_skernpca, y_train)
y_train_pred = lr.predict(X_train_skernpca)
y_test_pred = lr.predict(X_test_skernpca)
print('Accuracy :' + str(accuracy_score(y_train, y_train_pred)*100) + '%')
print('Accuracy :' + str(accuracy_score(y_test, y_test_pred)*100) + '%')

#svm via kpca
svm.fit(X_train_skernpca, y_train)
y_train_pred = svm.predict(X_train_skernpca)
y_test_pred = svm.predict(X_test_skernpca)
print('Accuracy :' + str(accuracy_score(y_train, y_train_pred)*100) + '%')
print('Accuracy :' + str(accuracy_score(y_test, y_test_pred)*100) + '%')

#test gamma for kpca in lr 
range = [0.01, 0.1, 1, 2.5, 5, 15]
scores = []

for g in range:
    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=g)
    X_train_skernpca = scikit_kpca.fit_transform(X_train_std)
    X_test_skernpca = scikit_kpca.transform(X_test_std)
    lr.fit(X_train_skernpca, y_train)
    y_train_pred = lr.predict(X_train_skernpca)
    y_test_pred = lr.predict(X_test_skernpca)
    scores.append(metrics.accuracy_score(y_train, y_train_pred))
    
i = 0
while i < len(range):
    print("gamma =", range[i], "", "Accuracy score is", scores[i])
    i += 1

#test gamma for kpca in svm
range = [0.01, 0.1, 1, 2.5, 5, 15]
scores = []
for g in range:
    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=g)
    X_train_skernpca = scikit_kpca.fit_transform(X_train_std)
    X_test_skernpca = scikit_kpca.transform(X_test_std)
    svm.fit(X_train_skernpca, y_train)
    y_train_pred = svm.predict(X_train_skernpca)
    y_test_pred = svm.predict(X_test_skernpca)
    scores.append(metrics.accuracy_score(y_train, y_train_pred))
    
i = 0
while i < len(range):
    print("gamma =", range[i], "", "Accuracy score is", scores[i])
    i += 1

print("My name is Mengming guo")
print("My NetID is: mg28")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")