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
X=df.drop(['cell_name','par_year','par_month'],axis=1)

dummies=pd.get_dummies(X['ran_vendor'],prefix='ran_vendor')
X=pd.concat([X,dummies],axis=1)
X.drop(['ran_vendor'],axis=1,inplace=True)




import seaborn as sns

correlation = X.corr()
#plt.figure(figsize=(15,15))
#sns.heatmap(correlation, square=True, cmap='viridis',linecolor='white',center=0)
#plt.title('correlation between different features')
#plt.show()
