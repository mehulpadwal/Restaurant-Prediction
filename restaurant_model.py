import sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import os

from sklearn import svm
df = pd.read_csv("final.csv", encoding='latin-1')



# print(df.head())
# print(df.shape)

X_train = df.ix[ : , 0: 148].values
y = df.ix[: , 148].values

X_scaled = StandardScaler().fit_transform(X_train)




clf = svm.SVC(gamma=0.1 , C = 100)

clf.fit(X_scaled, y)


# print('Prediction:', clf.predict(X_scaled[5]))

with open('example.csv', 'w') as f:
    for i in range(len(X_scaled)):
        for j in X_scaled[i]:
            f.write("%s," % j)
        f.write("%s\n" % y[i])



