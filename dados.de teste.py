import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.datasets import data
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv("abalone.data",   sep=";")
evaluation_data=dataset[1044:]
data_X=evaluation_data.iloc[:,7:8]
data_Y=evaluation_data.iloc[:,8:9]
print(type(evaluation_data))
print(type(data_X))
loaded_model = p1.load(open('abalone.data', 'rb'))
print("Coefficients: \n", loaded_model.coef_)
y_pred=loaded_model.predict(data_X)
print(type(data_X))
z_pred=y_pred-data_Y.values

right   =    0; wrong   =    0; total   =    0;
for x in z_pred:
    z=int(x)
    total=total+1
    if z==0:
        right   +=  1

    else:
        wrong   +=  1
print("accuraccy1= ",right/total,"accuraccy2= ",wrong/total)