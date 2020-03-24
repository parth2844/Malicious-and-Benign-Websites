import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

dataset = pd.read_csv('dataset.csv')
for i in range(0,1781):
    dataset['SERVER'][i] = re.sub('[^a-zA-Z0-9]', ' ',dataset['SERVER'][i])
    dataset['SERVER'][i] = dataset['SERVER'][i].lower()
X = dataset.iloc[:, [1,2,3,4,5,10,11,12,13,14,15,16,17,18,19]].values
y = dataset.iloc[:, 20].values

#Missing data
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer=imputer.fit(X[:,4:19])
X[:,4:19]=imputer.transform(X[:,4:19])

#categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X= LabelEncoder();
X[:,2]=labelencoder_X.fit_transform(X[:,2])
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder= OneHotEncoder(categorical_features=[2,3])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y= LabelEncoder();
y=labelencoder_y.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest classifier 
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators= 10,random_state=0,criterion='entropy' )
classifier.fit(X_train,y_train)

# Fitting Kernel SVM classifier 
from sklearn.svm import SVC
classifier= SVC(kernel='rbf', random_state= 0 )
classifier.fit(X_train, y_train)

# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier= SVC(kernel= 'sigmoid',random_state= 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Claculating the model's performance
accuracy= (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
#error=(cm[0][1]+cm[1][0])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
precision=(cm[1][1])/(cm[0][1]+cm[1][1])
recall=(cm[1][1])/(cm[1][0]+cm[1][1])
f1score= 2*(precision*recall)/(precision+recall)

#For Random Forest 5 estimators
# acc=0.95, prec=0.85, recall= 0.74, f1=0.79

#For Random Forest 10 estimators
# acc=0.97, prec=0.97, recall= 0.74, f1=0.84

#For rbf kernel svm
#acc= 0.90,prec=0.875,recall=0.14,f1=0.25

#For linear svm
#acc= 0.94,prec=0.76,recall=0.68,f1=0.71

#For sigmoid svm
#acc= 0.90,prec=0.85,recall=0.12,f1=0.22








