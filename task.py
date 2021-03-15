# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def pre(x,theta):
    p=(x * theta.T)
    return p
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))



def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost

#=============================================================


data2=pd.read_csv("C:\\Users\\Mazen\\Downloads\\Bio-ML-Assignment-1\\Bodyfat-Percentage.csv")

# # rescaling data
data2 = (data2 - data2.mean()) / data2.std()


data2.insert(0, 'Ones', 1)

X2 = data2[['Ones','Density', 'Age','Chest','Weight','Height','Neck','Abdomen','Hip','Thigh','Knee','Ankle','Biceps','Forearm','Wrist']]
y2 = data2.iloc[:,2:3]

x_train, x_test, y_train, y_test = train_test_split(X2, y2, test_size=0.1, random_state=4)



# for col in x_train.columns: 
#     print(col)
x_train = np.matrix(x_train.values)
y_train = np.matrix(y_train.values)
theta2 = np.matrix(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
print(x_train.shape[1])



alpha = 0.1
iters = 100

# perform linear regression on the data set
g2, cost2 = gradientDescent(x_train, y_train, theta2, alpha, iters)

thiscost = computeCost(x_train, y_train, g2)
x_test.to_csv("C:\\Users\\Mazen\\.spyder-py3\\BodyfatPercentage-Test-Data.csv")
temp_data=pd.read_csv("C:\\Users\\Mazen\\.spyder-py3\\BodyfatPercentage-Test-Data.csv")
print(temp_data.head())
#print(temp_data['Unnamed: 0'])

temp_data.drop(['Unnamed: 0'], axis=1,inplace=True)
print(temp_data.head())
test_data = np.matrix(temp_data.values)
#print(test_data)
y_pre=pre(test_data,g2)
print(pre(test_data,g2))
print(y_test)
y_test = np.matrix(y_test.values)
score = r2_score(y_test,y_pre)*100

print(score)