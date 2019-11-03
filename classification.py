# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:47:31 2019

@author: rmuh
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

#---------------- Min-Max normalization function ------------------------------
def minmax_norm(data) :
    return (data - data.min()) / (data.max() - data.min())

df = pd.read_csv("runidades_kmeans3_ageb.csv")

x = minmax_norm(df.iloc[:,5:]) #Normalize the data

#--------------- Creating the train and test sets -----------------------------
df_train, df_test, c_train, c_test = train_test_split(x,df['cluster'],
                                                      test_size=0.30, 
                                                      random_state=2411)

#--------------- Support Vector Machine ---------------------------------------

SVC_model = SVC() #Create the model
SVC_model.fit(df_train, c_train) #Training

SVC_prediction = SVC_model.predict(df_test) #Testing

print("Accuracy of SVM: "accuracy_score(SVC_prediction, c_test))
print("Confusion Matrix: ", confusion_matrix(SVC_prediction, c_test))

#----------------------- Decision Tree ----------------------------------------

dtc_model = DecisionTreeClassifier()
dtc_model.fit(df_train, c_train)

dtc_prediction = dtc_model.predict(df_test)

print("Accuracy of Decision Tree: ", accuracy_score(dtc_prediction, c_test))
print("Confusion Matrix: ", confusion_matrix(dtc_prediction, c_test))

#------------------------Random Forest-----------------------------------------

rfc = RandomForestClassifier(n_estimators=700)
rfc.fit(df_train,c_train)
rfc_predict = rfc.predict(df_test)

print("Accuracy of RF: ", accuracy_score(rfc_predict, c_test))
print("Confusion Matrix: ", confusion_matrix(rfc_predict, c_test))

#---------------------- Neural Network ----------------------------------------

mlp = MLPClassifier(hidden_layer_sizes=(21),max_iter=800)
mlp.fit(df_train,c_train)

mlp_prediction = mlp.predict(df_test)

print("Accuracy of MLP: ", accuracy_score(mlp_prediction, c_test))
print("Confusion Matrix: ", confusion_matrix(mlp_prediction, c_test))

#---------------------- Gradient Boosting -------------------------------------

gbm = GradientBoostingClassifier(n_estimators=1000)
gbm.fit(df_train,c_train)
gbm_predict = rfc.predict(df_test)

print("Accuracy of Gradient Boosting: ", accuracy_score(gbm_predict, c_test))
print("Confusion Matrix: ", confusion_matrix(gbm_predict, c_test))


