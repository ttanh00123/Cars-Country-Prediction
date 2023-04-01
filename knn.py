import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
df = pd.read_csv('cars.csv')
df.dropna()
k=4
x = df[['mpg','cylinders','cubicinches','hp','weightlbs','time-to-60','year']].values
y = df['brand']
x_trainset, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)
x_train = [[float(ele) for ele in sub] for sub in x_trainset]
knn = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
pred = knn.predict(x_test)
print("KNN's Accuracy: ", metrics.accuracy_score(y_test, pred))
#0.658, low accuracy