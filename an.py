import pandas as pd
import numpy as np
from sklearn import datasets, svm 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA


df=pd.read_csv(r"C:\Users\kamal\Desktop\DA\train.csv")
print("readed")
X=df.drop(['Congestion_Type'],axis=1)
X=df.drop(['cell_name'],axis=1)




my = pd.crosstab(index=nc_val['Congestion_Type'],columns="count")  





labels = '3G_BACKHAUL_CONGESTION', '4G_BACKHAUL_CONGESTION', '4G_RAN_CONGESTION', 'NC'
sizes = [19449, 19650, 19768, 19693]
explode = (0, 0, 0.1, 0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  
plt.show()



day1=df[df['par_day']==1]
time_min=nc_val['par_min'].value_counts()

