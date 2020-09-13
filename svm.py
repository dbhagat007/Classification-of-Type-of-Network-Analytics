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
from sklearn.exceptions import DataConversionWarning



df=pd.read_csv(r"C:\Users\kamal\Desktop\DA\train_upd.csv")
print("data readed")
#df=df.iloc[1:50000,:]
def enc(k):

   if k=='4G_RAN_CONGESTION':
      return 1

   elif k=='4G_BACKHAUL_CONGESTION':
      return 2

   elif k=='3G_BACKHAUL_CONGESTION':
      return 3

   else :
      return 0
df['Congestion_Type'] = df['Congestion_Type'].apply(enc)

y=df['Congestion_Type']

#df.drop(['Congestion_Type'],axis=1)
X=df.drop(['cell_name','Congestion_Type','par_year','par_month'],axis=1)
#X=df.drop([],axis=1)
print(type(X))
dummies=pd.get_dummies(X['ran_vendor'],prefix='ran_vendor')
X=pd.concat([X,dummies],axis=1)
X.drop(['ran_vendor'],axis=1,inplace=True)

print(X.head())
#print(y.head())



scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

pca=PCA(n_components=22)
pca.fit(X_train)
X_train=pca.transform(X_train)
X_test=pca.transform(X_test)
variance=pca.explained_variance_ratio_
print("retained data = "+str(sum(variance*100)))
def mat(model):
   y_pred = model.predict(X_test)
   test=pd.Series.tolist(y_test)
   pred=np.ndarray.tolist(y_pred)
   #print(test)
   p=matthews_corrcoef(test,pred)
   return p



   
kernels = ('linear','poly','rbf')

for index, kernel in enumerate(kernels):
   print("*"*60)
   print("kernel type : "+str(kernel))
   model = svm.SVC(kernel=kernel,C=0.1)
   model.fit(X_train, y_train)
   #scores = cross_val_score(model, X, y, cv=10)
   #print(scores)
   y_pred=model.predict(X_test)
   y_pred1=model.predict(X_train)
   #print(np.ndarray.tolist(y_pred))
   mcc=mat(model)
   print("matthews correlation coefficient = ",mcc)
   s=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
   s1=accuracy_score(y_train, y_pred1, normalize=True, sample_weight=None)
   print("test Accuracy = "+str(s*100)+"%")
   print("train Accuracy = "+str(s1*100)+"%")
   
