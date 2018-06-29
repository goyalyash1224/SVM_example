# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)

# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)

# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)


## Get the data

#*Use seaborn to get the iris data by using: iris = sns.load_dataset('iris') **

import seaborn as sns
iris = sns.load_dataset('iris')

## Exploratory Data Analysis

#*Import some libraries you think you'll need.**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
iris.head(2)

#analysis on data
sns.pairplot(data=iris,hue='species',palette='Set1')

setosa = iris[iris['species']=='setosa']
sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],cmap='plasma',shade=True,shade_lowest=False)

# Train Test Split

from sklearn.model_selection import train_test_split

X= iris.drop('species',axis=1)
y= iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

# Train a Model



from sklearn.svm import SVC

model =SVC()

model.fit(X_train,y_train)

predictions=model.predict(X_test)

## Model Evaluation

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(predictions,y_test))

print(confusion_matrix(predictions,y_test))

print(classification_report(predictions,y_test))


## Gridsearch for best parameter to obtain good results


from sklearn.grid_search import GridSearchCV


param_grid= {'C':[0.1,1,10,100],'gamma':[0.1,0.01,0.001,0.0001]}

#Create a GridSearchCV object and fit it to the training data.

grid = GridSearchCV(SVC(),param_grid,verbose=2)
grid.fit(X_train,y_train)

grid_predictions =grid.predict(X_test)

print(confusion_matrix(grid_predictions,y_test))

print(classification_report(grid_predictions,y_test))


##Thanku

