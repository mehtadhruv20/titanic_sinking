import numpy as np
import pandas as pd
train=pd.read_csv('Train.csv')
#test=pd.read_csv('Test.csv')
train.head(10)
train.drop(['PassengerId','Name','Cabin','Ticket'],inplace=True,axis=1)
y=train.Survived
train.drop('Survived',axis=1,inplace=True)
dum1=pd.get_dummies(train['Pclass'])
dum2=pd.get_dummies(train['Embarked'])
dum3=pd.get_dummies(train['Sex'])
dum1.drop(1,inplace=True,axis=1)
dum2.drop('C',inplace=True,axis=1)
dum3.drop('female',inplace=True,axis=1)
train=pd.concat((train,dum1,dum2,dum3),axis=1)
train['Family']=train['SibSp']+train['Parch']
train.drop(['Pclass','Embarked','Sex','SibSp','Parch'],inplace=True,axis=1)


'''#for test set
hold=test.PassengerId.values
test.drop(['PassengerId','Name','Cabin','Ticket'],inplace=True,axis=1)

dum4=pd.get_dummies(test['Pclass'])
dum5=pd.get_dummies(test['Embarked'])
dum6=pd.get_dummies(test['Sex'])
dum4.drop(1,inplace=True,axis=1)
dum5.drop('C',inplace=True,axis=1)
dum6.drop('female',inplace=True,axis=1)
test=pd.concat((test,dum4,dum5,dum6),axis=1)
test['Family']=test['SibSp']+test['Parch']
test.drop(['Pclass','Embarked','Sex','SibSp','Parch'],inplace=True,axis=1)
#test set ends

#deleting dummy variables'''
del(dum1,dum2,dum3)

'''X_train=train.values
y_train=y.values
X_test=test.values

from sklearn.preprocessing import Imputer
imp=Imputer()
X_train=imp.fit_transform(X_train,y_train)
X_test=imp.transform(X_test)



from sklearn.preprocessing import StandardScaler
s=StandardScaler()
X_train=s.fit_transform(X_train,y)
X_test=s.transform(X_test)

from sklearn.linear_model import LogisticRegression
regressor=LogisticRegression()
regressor.fit(X_train,y)
y_pred=regressor.predict(X_test)
from pandas import DataFrame
hold=pd.DataFrame(hold)
y_pred=pd.DataFrame(y_pred)
hold=pd.concat((hold,y_pred),axis=1)
hold.to_csv('out.csv',index=False)
'''

X=train.values
y=y.values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.667)

from sklearn.preprocessing import Imputer
imp=Imputer()
X_train=imp.fit_transform(X_train,y_train)
X_test=imp.transform(X_test)

from sklearn.preprocessing import StandardScaler
s=StandardScaler()
X_train=s.fit_transform(X_train,y)
X_test=s.transform(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

regressor=KNeighborsClassifier(n_neighbors=10)
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

#con=confusion_matrix(y_test,y_pred)
#print(con)

from sklearn.model_selection import cross_val_score
c=cross_val_score(regressor,X_test,y_test,cv=10).mean()
print(c)
