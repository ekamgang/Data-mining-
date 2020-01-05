import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import random
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import preprocessing

data = pd.read_csv("clean_data.csv")

#Logistic regression
X=data[[ 'Trip Seconds','Pickup Community Area']]     #Features
y=data['Tip']             #target variable
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
#predicting
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
#scoring
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#with standardized
print("after standardizing")
standardized_X = preprocessing.scale(X)
print(standardized_X)
print(X)
X_train,X_test,y_train,y_test=train_test_split(standardized_X,y,test_size=0.20,random_state=0)
#predicting
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
#scoring
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# #Try increasing performance by using more independent variables

X=data[[  'Fare', 'Trip Miles','Trip Seconds','Pickup Community Area','Trip Total']]     #Features
y=data['Tip']             #target variable
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
#predicting
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
#scoring
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#with standardized
print("after standardizing")
standardized_X = preprocessing.scale(X)
print(standardized_X)
print(X)
X_train,X_test,y_train,y_test=train_test_split(standardized_X,y,test_size=0.20,random_state=0)
#predicting
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
#scoring
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

















