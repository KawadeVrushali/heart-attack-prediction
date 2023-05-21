#!/usr/bin/env python
# coding: utf-8
# # problem_Statement to predict the risk of heart attack by using the different features
# # import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import zscore,skewimage.png
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,f1_score,plot_roc_curve
import warnings
warnings.filterwarnings('ignore')
import pickle
df=pd.read_csv('heart.csv')
x=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=20)
x_train
# # model_trainning
heart_reg=LogisticRegression()
heart_reg.fit(x_train,y_train)
y_predict=heart_reg.predict(x_test)
# # Testing Data Evalution
cnf=confusion_matrix(y_test,y_predict)
acc=accuracy_score(y_test,y_predict)
clf_report=classification_report(y_test,y_predict)
y_predict_train=heart_reg.predict(x_train)
cnf=confusion_matrix(y_train,y_predict_train)
acc=accuracy_score(y_train,y_predict_train)
clf_report=classification_report(y_train,y_predict_train)
# # Feature Scalling
# # Normalization
x_df=df.drop('target',axis=1)
norm_scalar=MinMaxScaler()
array=norm_scalar.fit_transform(x_df)
x_normal_df=pd.DataFrame(array,columns=x_df.columns)
x=x_normal_df.copy()
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=24,stratify=y)
heart_reg=LogisticRegression()
heart_reg.fit(x_train,y_train)
#Tesing after normalisation
y_predict=heart_reg.predict(x_test)
cnf=confusion_matrix(y_test,y_predict)
acc=accuracy_score(y_test,y_predict)
clf_report=classification_report(y_test,y_predict)
#Trainning after normalisation
y_predict_train=heart_reg.predict(x_train)
cnf=confusion_matrix(y_train,y_predict_train)
acc=accuracy_score(y_train,y_predict_train)
clf_report=classification_report(y_train,y_predict_train)
# # Standardization
x_df=df.drop('target',axis=1)
Std_Scalar=StandardScaler()
array=Std_Scalar.fit_transform(x_df)
x_std_df=pd.DataFrame(array,columns=x_df.columns)
x=x_std_df.copy()
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=24,stratify=y)
#Tesing after Standardization
y_predict_train=heart_reg.predict(x_train)
cnf=confusion_matrix(y_train,y_predict_train)
acc=accuracy_score(y_train,y_predict_train)
clf_report=classification_report(y_train,y_predict_train)
#Trainning after Standardization
y_predict_train=heart_reg.predict(x_train)
cnf=confusion_matrix(y_train,y_predict_train)
acc=accuracy_score(y_train,y_predict_train)
clf_report=classification_report(y_train,y_predict_train)
with open('heart_attack_model.pkl', 'wb') as file:
    pickle.dump(heart_reg, file)

