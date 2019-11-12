
# coding: utf-8

# k-Nearest Neighbour Classifier using sklearn
# =================================
# 

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[10]:


#Loading data and preprocessing
url='http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df=pd.read_csv(url)
df.columns=['sepal_length','sepal_width','petal_length','petal_width','flower_type']
df['flower_type'] = df['flower_type'].astype('category')
df.flower_type = df.flower_type.cat.rename_categories([0,1,2])
D=df.values


# Get the labelled set
c1=D[:20,:]; c2=D[50:70,:];  c3=D[100:120,:]
trainSet = np.concatenate((c1,c2,c3),axis=0)

# Get the testing set
c1 = D[21:50,:]; c2=D[71:100,:];  c3=D[121:,:]
testSet = np.concatenate((c1,c2,c3),axis=0)

xTrain=trainSet[:,:-1]; yTrain=trainSet[:,-1]
xTest=testSet[:,:-1]; yTest=testSet[:,-1]


# In[11]:


# create a knn classifier with K=3
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(xTrain, yTrain.astype(int))


# In[12]:


# Make predictions

#accuracy_score function nije banate hbe
yPred=clf.predict(xTest)
acc=accuracy_score(yTest.astype(int), yPred.astype(int))
print('Accuracy with 3 neighbours: ',acc)


# In[13]:


#confusion matrix function nije banate hbe
def plot_conf_mat(lTrue, lPred, title):
    """ A function for plotting the confusion matrix given true and predicted labels."""
    cm = confusion_matrix(lTrue.astype(int), lPred.astype(int))
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()    


# In[14]:


plot_conf_mat(yTest, yPred, 'K=3')


# In[2]:


#f = open("SRR6055422", "r")
f = open("data.csv", encoding="utf8")
text = f.readlines()
print(text)


# In[3]:


import pandas as pd
#import sys
#reload(sys)
#sys.setdefaultencoding("ISO-8859-1")

#importing the dataset
#dataset = pd.read_csv('SRR6055422.csv')
f    = open("data.csv","rb")
text = f.read().decode(errors='replace')
print(text[10])


# In[6]:


#import pandas as pd
df=pd.read_csv('data.csv', encoding="ISO-8859-1")
#print(text[:1000])
df.head()


# In[1]:


#print(text[:100000])
#c03de09c8e7a70a37fd398b725415bd6
text2 = text.split()
print(len(text2))
'''
for i in range(len(text2)):
    if text2[i] == "*idx1":
        print(text2[i-1])
'''


# In[6]:


print(text[:100000])


# In[15]:


'''
import gzip
import shutil
with gzip.open('SRR6055422.fastq.gz', 'rb') as f_in:
    with open('SRR6055422.fastq', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
'''
symbol=['@','+']
"""Stores both the sequence and the quality values for the sequence"""
f = open('SRR6055422.csv','rU')
for lines in f:
    #if symbol not in lines.startwith()
    data = f.readlines()
    print(data)


# In[1]:


import pandas as pd
df = pd.read_csv("data.csv")

