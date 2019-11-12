
# coding: utf-8

# # Pattern Project

# ## Import Necessary Library

# In[371]:


import pandas as pd
import numpy as np
import glob
import os
from os import listdir
from os.path import isfile, join
import csv
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

#Gaussian Naive bayes
from sklearn.naive_bayes import GaussianNB

#svm with rbf karnel
import sklearn
from sklearn import svm, preprocessing
#SVM
from sklearn import datasets,svm
from sklearn.cross_validation import train_test_split
from sklearn import metrics
#CN Matrix
from sklearn.metrics import accuracy_score, confusion_matrix

#Algo
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection


# ## Import DataSet

# In[162]:


dataFrame0 = pd.read_csv("F:/python/4.2/project/main/dataset/DATASET_01.csv")
dataFrame1 = pd.read_csv("F:/python/4.2/project/main/dataset/DATASET_02.csv")
dataFrame2 = pd.read_csv("F:/python/4.2/project/main/dataset/DATASET_03.csv")
dataFrame3 = pd.read_csv("F:/python/4.2/project/main/dataset/DATASET_04.csv")
dataFrame4 = pd.read_csv("F:/python/4.2/project/main/dataset/DATASET_05.csv")
dataFrame5 = pd.read_csv("F:/python/4.2/project/main/dataset/DATASET_06.csv")

dataFrame0.head()
#dataFrame1.head()
#dataFrame2.head()
#dataFrame3.head()
#dataFrame4.head()
#dataFrame5.head()


# ## Dataset Concatenate

# In[165]:


#path = "F:/python/4.2/project/main/dataset/"
#filenames = [path+'DATASET_01.csv', 'dataset/DATASET_02.csv', 'dataset/DATASET_03.csv', 'dataset/DATASET_04.csv',
 #           'dataset/DATASET_05.csv', 'dataset/DATASET_06.csv']

#dataFrames = []
#for f in filenames:
#    dataFrames.append(pd.read_csv(f))
#outputFile = "dataset/finalData/finalData.csv"
#data_dir= glob.glob(os.path.join(path, '*.csv'))
#print(os.listdir(data_dir))


# In[62]:


#filenames = glob('dataset/DATASET*.csv')
#dataFrames = [pd.read_csv(f) for f in filenames]


# In[201]:


def concatenate (indir = "F:/python/4.2/project/main/dataset/", outfile = "F:/python/4.2/project/main/dataset/Total/TotalfinalData.csv"):
    #os.chdir(indir)
    #fileList = glob.glob(indir+"*.csv")
    fileList = [f for f in listdir(indir) if isfile(join(indir, f))]
    dfList = []
    #colnames=["ID"]
    for filename in fileList:
        print(filename)
        df = pd.read_csv(filename, header= None)
        dfList.append(df)
    concateDf= pd.concat(dfList, axis=0)
    concateDf.to_csv(outfile, index = None)
concatenate()


# ### Column Add from other Dataset

# In[212]:


# Loading data and preprocessing
filePath1 = "F:/python/4.2/project/main/dataset/Total/TotalfinalData_1.csv"
filePath2 = "F:/python/4.2/project/main/extra/EXTENDED_1.csv"
mergedFile = "F:/python/4.2/project/main/dataset/Total/TotalfinalData_3.csv"

df1 = pd.read_csv(filePath1)
df2 = pd.read_csv(filePath2)
columnsNeed = ["ID", "AcademicResult", "Co_curri", "Skill","Achi","Physical","Higher"]
for column in df2.columns:
    dropChecker = True
    for colName in columnsNeed:
        if column == colName:
            dropChecker = False
            break

    if dropChecker == True:
        df2.drop(column, axis=1, inplace=True)

merged = df1.merge(df2, on="ID")
# for column in df1
merged.to_csv(mergedFile, index=False)

data_frame = pd.read_csv("F:/python/4.2/project/main/dataset/Total/TotalfinalData_final.csv", usecols=[1,2,3,8,9,11,12,13,14])


# ## Dataset Processed

# In[206]:


cols = ["ID", "Name", "OLat", "OLong","PI", "PILat", "PIlong", "Food", "Leisure", "FP","LR","AcademicResult", "Co_curri", "Skill", "Achi", "Physical", "Higher"]
df = pd.read_csv("F:/python/4.2/project/main/dataset/Total/TotalfinalData_final.csv",)
df.columns = [''] * len(df.columns)
df = df[1:]
df.reset_index(drop=True)
df.columns = cols

df.drop(['Name'], axis = 1, inplace = True)
df.drop(['ID'], axis = 1, inplace = True)
df.head()


# ## Label the Dataset 

# In[238]:


df.dropna(axis = 0, inplace = True)
lb_make = LabelEncoder()
df["PI"] = lb_make.fit_transform(df["PI"])
df["Food"] = lb_make.fit_transform(df["Food"])
df["Leisure"] = lb_make.fit_transform(df["Leisure"])
df["FP"] = lb_make.fit_transform(df["FP"])
df["LR"] = lb_make.fit_transform(df["LR"])
df["Co_curri"] = lb_make.fit_transform(df["Co_curri"])
df["Skill"] = lb_make.fit_transform(df["Skill"])
df["Physical"] = lb_make.fit_transform(df["Physical"])
df["Higher"] = lb_make.fit_transform(df["Higher"])
df.head()


# ## Classifier - Problem 1

# In[239]:


df = sklearn.utils.shuffle(df)
x = df.drop("Higher",axis = 1).values
y = df["Higher"].values


# ### Decision Tree - Accuracy Test

# In[253]:


test_size = int(0.2*len(x))
train_x = x[:-test_size]
train_y = y[:-test_size]
test_x = x[-test_size:]
test_y = y[-test_size:]

clf = DecisionTreeClassifier()
clf.fit(train_x, train_y)
clf.score(test_x, test_y)

#print(clf.predict([train_x[7]]))


# ### Test single Data - Decision Tree

# In[225]:


print(clf.predict([train_x[6]]))
#1-Yes, 0 -No


# ### Gaussian Naive Bayes - Accuracy Test

# In[231]:


#Gaussian Naive bayes
clf = GaussianNB()
clf.fit(train_x, train_y)
clf.score(test_x, test_y)


# ### Bernouli Naive Bayes

# In[332]:


from sklearn.naive_bayes import BernoulliNB
model1=BernoulliNB()
model1.fit(X_train,y_train)
predict1=model1.predict(X_test)
x=y_test.iloc[:].values

length1=len(predict1)
correct1=0
for i in range(length1):
    if(x[i]==predict1[i]):
        correct1=correct1+1
print(correct1)

accuracy1=correct1*100/len(x)
print(accuracy1)


# In[333]:


print(metrics.confusion_matrix(y_test,predict1,labels=[0,1,2]))
cm=metrics.confusion_matrix(y_test,predict1,labels=[0,1,2])
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confussion Matrix for Bernoulli Naive Bayes')
plt.xlabel('Prediction')
plt.ylabel('True')
plt.show()


# ### svm with rbf kernel

# In[261]:


#svm with rbf karnel
df = sklearn.utils.shuffle(df)
x = df.drop("Higher",axis = 1).values
y = df["Higher"].values

x = preprocessing.scale(x)
test_size = int(0.2*len(x))
train_x = x[:-test_size]
train_y = y[:-test_size]
test_x = x[-test_size:]
test_y = y[-test_size:]

clf = svm.SVR(kernel = "rbf")
clf.fit(train_x, train_y)
abs(clf.score(test_x, test_y))


# ### SVM

# In[325]:


X=df.drop('Higher',axis=1)
y=df['Higher']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.12)

model=svm.SVC(kernel='linear')
model.fit(X_train,y_train)


# In[330]:


predict=model.predict(X_test)
x=y_test.iloc[:].values
length=len(predict)

correct=0
for i in range(length):
    if(predict[i]==x[i]):
        correct+=1
print(correct)

accuracy=correct*100/len(x)
print(accuracy)


# In[331]:


#print("{0}".format(metrics.confusion_matrix(y_test,predict,labels=[0,1,2])))
print(metrics.confusion_matrix(y_test,predict,labels=[0,1,2]))
cm=metrics.confusion_matrix(y_test,predict,labels=[0,1,2])
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confussion Matrix for SVM')
plt.xlabel('Prediction')
plt.ylabel('True')
plt.show() 


# ### Random Forest Classifier

# In[356]:


model=RandomForestClassifier()
mode2=DecisionTreeRegressor()

model.fit(X_train,y_train)
mode2.fit(X_train,y_train)

predict3=model3.predict(X_test)
predict4=model4.predict(X_test)

length3=len(predict3)
correct3=0
for i in range(length3):
    if(x[i]==predict3[i]):
        correct3=correct3+1
print(correct3)
accuracy3=correct3*100/len(x)
print(accuracy3)


# In[335]:


print(metrics.confusion_matrix(y_test,predict3,labels=[0,1,2]))
cm=metrics.confusion_matrix(y_test,predict3,labels=[0,1,2])
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confussion Matrix for Random Forest')
plt.xlabel('Prediction')
plt.ylabel('True')
plt.show()


# ### Box Plot for Algorithm Comparison

# In[345]:


models = []
models.append(('DT', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('GB', GaussianNB()))
models.append(('NB', BernoulliNB()))
models.append(('SVC', SVC()))
models.append(('LSVC', LinearSVC()))
models.append(('RFC', RandomForestClassifier()))

seed = 7
results = []
names = []
X = X_train
Y = y_train

for name, model in models:
    kfold = model_selection.KFold(
        n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(
        model, X, Y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f " % (name, cv_results.mean()*100)
    print(msg)


# In[346]:


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# ## Classifier - Problem 2

# In[263]:


df = sklearn.utils.shuffle(df)
x = df.drop("Physical",axis = 1).values
y = df["Physical"].values


# In[283]:


#Decision Tree
test_size = int(0.2*len(x))
train_x = x[:-test_size]
train_y = y[:-test_size]
test_x = x[-test_size:]
test_y = y[-test_size:]

clf = DecisionTreeClassifier()
clf.fit(train_x, train_y)
#clf.score(test_x, test_y)
print('Accuracy:', clf.score(test_x, test_y))

#print(clf.predict([train_x[7]]))


# In[295]:


#Gaussian Naive bayes
clf = GaussianNB()
clf.fit(train_x, train_y)
#clf.score(test_x, test_y)
print('Accuracy:', clf.score(test_x, test_y))


# In[297]:


#svm with rbf karnel
df = sklearn.utils.shuffle(df)
x = df.drop("Higher",axis = 1).values
y = df["Higher"].values

x = preprocessing.scale(x)
test_size = int(0.2*len(x))
train_x = x[:-test_size]
train_y = y[:-test_size]
test_x = x[-test_size:]
test_y = y[-test_size:]

clf = svm.SVR(kernel = "rbf")
clf.fit(train_x, train_y)
#abs(clf.score(test_x, test_y))
print('Accuracy:', abs(clf.score(test_x, test_y)))


# ### SVM

# In[363]:


X=df.drop('LR',axis=1)
y=df['LR']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.12)

model=svm.SVC(kernel='linear')
model.fit(X_train,y_train)  


# In[364]:


predict=model.predict(X_test)
x=y_test.iloc[:].values
length=len(predict)

correct=0
for i in range(length):
    if(predict[i]==x[i]):
        correct+=1
print(correct)

accuracy=correct*100/len(x)
print(accuracy)


# In[366]:


#print("{0}".format(metrics.confusion_matrix(y_test,predict,labels=[0,1,2])))
print(metrics.confusion_matrix(y_test,predict,labels=[0,1,2]))
cm=metrics.confusion_matrix(y_test,predict,labels=[0,1,2])
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confussion Matrix for SVM')
plt.xlabel('Prediction')
plt.ylabel('True')
plt.show() 


# ### Random Forest Classifier 

# In[367]:


model=RandomForestClassifier()
mode2=DecisionTreeRegressor()

model.fit(X_train,y_train)
mode2.fit(X_train,y_train)

predict3=model3.predict(X_test)
predict4=model4.predict(X_test)

length3=len(predict3)
correct3=0
for i in range(length3):
    if(x[i]==predict3[i]):
        correct3=correct3+1
print(correct3)
accuracy3=correct3*100/len(x)
print(accuracy3)


# In[368]:


print(metrics.confusion_matrix(y_test,predict3,labels=[0,1,2]))
cm=metrics.confusion_matrix(y_test,predict3,labels=[0,1,2])
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confussion Matrix for Random Forest')
plt.xlabel('Prediction')
plt.ylabel('True')
plt.show()


# ### Boxplot for algorithm comparison 

# In[369]:


models = []
models.append(('DT', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('GB', GaussianNB()))
models.append(('NB', BernoulliNB()))
models.append(('SVC', SVC()))
models.append(('LSVC', LinearSVC()))
models.append(('RFC', RandomForestClassifier()))

seed = 7
results = []
names = []
X = X_train
Y = y_train

for name, model in models:
    kfold = model_selection.KFold(
        n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(
        model, X, Y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f " % (name, cv_results.mean()*100)
    print(msg)


# In[370]:


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# ### Error Rate Calculation

# In[351]:


#error rate = (1 - (test_y / train_y)) * 100


# In[ ]:




