import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
df = pd.read_csv('cars.csv')
df.dropna()
x = df[['mpg','cylinders','cubicinches','hp','weightlbs','time-to-60','year']].values
y = df['brand']
x_trainset, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)
x_train = [[float(ele) for ele in sub] for sub in x_trainset]
sapling = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
sapling.fit(x_train,y_train)
predTree = sapling.predict(x_test)
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))
#0.772, better than KNN