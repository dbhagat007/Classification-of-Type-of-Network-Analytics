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



df=pd.read_csv(r"C:\Users\kamal\Desktop\DA\train_upd.csv")
print("readed")
df=df.iloc[1:1000,:]
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


dummies=pd.get_dummies(X['ran_vendor'],prefix='ran_vendor')
X=pd.concat([X,dummies],axis=1)
X.drop(['ran_vendor'],axis=1,inplace=True)

#print(X.head())
#print(y.head())
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

pca=PCA(n_components=2)
pca.fit(X_train)
X_train=pca.transform(X_train)
X_test=pca.transform(X_test)
variance=pca.explained_variance_ratio_
print("retainced data = "+str(sum(variance*100)))
def mat(model):
   y_pred = model.predict(X_test)
   test=pd.Series.tolist(y_test)
   pred=np.ndarray.tolist(y_pred)
   #print(test)
   p=matthews_corrcoef(test,pred)
   return p




from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train,y_train)





#scores = cross_val_score(classifier, X, y, cv=10)
print(scores)
y_pred=classifier.predict(X_test)
y_pred1=classifier.predict(X_train)
#print(np.ndarray.tolist(y_pred))
mcc=mat(classifier)
print("matthews correlation coefficient = ",mcc)
s=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
s1=accuracy_score(y_train, y_pred1, normalize=True, sample_weight=None)
print("test Accuracy = "+str(s*100)+"%")
print("train Accuracy = "+str(s1*100)+"%")











