from math import *
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
y_pred=[1,0,1,1]
y_true=[1,1,1,0]
p=matthews_corrcoef(y_true,y_pred)
print(p)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(tn,tp,fp,fn)
d=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
mcc=((tp*tn)-(fp*fn))/(sqrt(d))
print(mcc)
