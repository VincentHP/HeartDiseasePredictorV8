import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

#Dataset
df = pd.read_csv('heart.csv')
df_mean_data = df.groupby(by = ['age', 'sex']).mean().reset_index()

#Removing Outliers
q1 = df.chol.quantile(0.25)
q3 = df.chol.quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 1.5 * IQR
upper_limit = q3 + 1.5 * IQR
df=df[df["chol"] < upper_limit]
q1 = df.trestbps.quantile(0.25)
q3 = df.trestbps.quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 1.5 * IQR
upper_limit = q3 + 1.5 * IQR
df=df[df["trestbps"] < upper_limit]
q1 = df.thalach.quantile(0.25)
q3 = df.thalach.quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 1.5 * IQR
df=df[df["thalach"] > lower_limit]
oldpeak_q1 = df.oldpeak.quantile(0.25)
oldpeak_q3 = df.oldpeak.quantile(0.75)
oldpeak_IQR = oldpeak_q3 - oldpeak_q1
oldpeak_lower_limit = oldpeak_q1 - 1.5 * oldpeak_IQR
oldpeak_upper_limit = oldpeak_q3 + 1.5 * oldpeak_IQR
df=df[df["oldpeak"] < oldpeak_upper_limit]

#Splitting and scaling the data
y = df["target"]
x = df.drop("target", axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#XGB
xgb_model = XGBClassifier(booster='gbtree', max_depth=6, sampling_method='uniform')
xgb_model.fit(x_train, y_train)

#Function

class Person:
    def __init__(self, sex, age):
        if sex.lower() == 'male' or sex.lower() == 'm':
            s = 1 
        elif sex.lower() == 'female' or sex.lower() == 'f':
            s = 0
        self.sex = s
        age = int(age)
        if age < 29 or age > 77:
            print("Sorry, we have no data for that age")
            exit()
        self.age = age
        df = pd.read_csv('heart.csv')
        self.data = df.groupby(by = ['age', 'sex']).mean().reset_index()

    def predict(self):
        features = self.data[(self.data['sex'] == self.sex) & (self.data['age'] == self.age)]
        features.drop(["target"], axis = 1, inplace = True)
        pred = xgb_model.predict(features)
        new_list = list(pred)
        if (new_list == [0]):
            return False
        else:
            return True

    
    
    
    
