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
import itertools


df=pd.read_csv(r"C:\Users\kamal\Desktop\DA\train_upd.csv")
print("data readed")
#df=df.iloc[1:50000,:]
class_names=['4G_RAN_CONG.','4G_BACKHAUL_CONG.','3G_BACKHAUL_CONG.','NO CONG.']
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
print("22222222222222222222222222")
print(X_train.shape)
pca=PCA(n_components=22)
pca.fit(X_train)
X_train=pca.transform(X_train)
X_test=pca.transform(X_test)
variance=pca.explained_variance_ratio_
print(variance)
print("retained data = "+str(sum(variance*100)))
def mat(model):
   y_pred = model.predict(X_test)
   test=pd.Series.tolist(y_test)
   pred=np.ndarray.tolist(y_pred)
   #print(test)
   p=matthews_corrcoef(test,pred)
   return p

kernels = ('linear')


print("*"*60)
print("kernel type : "+str(kernels))
model = svm.SVC(kernel=kernels,C=0.1)
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
   







##########################################################

#df_pred=pd.read_csv(r"C:\Users\kamal\Desktop\DA\test_upd.csv")
#print("readed for prediction")

#X_pred=df_pred.drop(['cell_name','par_year','par_month'],axis=1)

#dummies=pd.get_dummies(X_pred['ran_vendor'],prefix='ran_vendor')
#X_pred=pd.concat([X_pred,dummies],axis=1)
#X_pred.drop(['ran_vendor'],axis=1,inplace=True)
#scaler = StandardScaler()
#scaler.fit(X_pred)
#X_pred=scaler.transform(X_pred)
#X_pred=pca.transform(X_pred)
#final=model.predict(X_pred)

#########################################################










def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = (cm.astype('float')) / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        cm=cm*100
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion Matrix')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion Matrix')

plt.show()

























   
