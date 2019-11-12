
# coding: utf-8

# Preamble
# ========
# 
# This assignment is meant to provide you with an opportunity to create your first machine learning algorithm, and begin exploring the relationship between the answers the algorithm provides and the structure of the data.
# 
# Attached to this assignment is a dataset containing 4 columns, x (a feature), y (another feature) TL (the true label of the class) and L (the known label of the class). Typically with machine learning, you would not have access to TL for all datapoint or you would not need to build a classifier, but it is provided here as a learning exercise. You should not use the TL column during training, but you can use it after training to see how the structure of the data interacts with the algorithm to produce answers that are correct or incorrect. Most of the data in the L column is empty (NaN in scipy). Those cells which have a label are your labels training data. Those cells which do not have a label are the data that you need to classify.
# 
# You will be building, from first principles a K- nearest neighbour classifier in Python. You cannot use open-source code which implements this algorithm, that is the second step of your assignment. You are encouraged to use the lower-level components of SciPy, such as Pandas data frames to speed up your code, and save you the work of implementing low-level data management tasks.
# 
# Assignment
# ==========





import pandas as pd
import numpy as np
import math
import operator
from collections import Counter
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score  
from sklearn.metrics import confusion_matrix  

# In[ ]:

# Loading and preprocessing the dataset

#dataFrame = pd.read_csv('knnDataSetCopy.csv')
dataFrame = pd.read_csv('knnDataSet.csv')
dataFrame.columns = ['no','x','y','TL','L']
Set1 = dataFrame.drop('no', axis=1)
global outputCheck
# Preparing Test Data
f = Set1[~Set1.L.notnull()]
g = f.values
removeNon = g[:, :-1]  # With true label
testSet11 = removeNon
testData = removeNon[:, :-1]  # without true label
#print(testData)  # test set

# Preparing Training Data Set
E = Set1.dropna(axis=0, how='any', thresh=None, subset=None)  # STEP 3
k = E.values
trainData = k[:, :-1]
#print(trainData)  # training set


# 2) Implement the k-nearest neighbour algorithm in Python. Just use simple data structures, the small datasets we are employing will not require kD-tree optimization. Your algorithm should take a labeled dataset (3 column table), and unlabeled dataset (2 column table), the parameter k, and a boolean value feedback_classification. The parameter k is the number of neighbours to consider when classifying. The boolean feedback_classification parameter is used to set the behaviour of the classifier. When it is true, previously unlabeled data that has been classified becomes part of the training set. When it is false, only data in the training set is used for nearest neighbour determination for all unlabeled data. (20 marks)

# In[ ]:

#def knn(trainSet, xTest, K, feedback_classification):
    #pass

def similarSet (trainingSet, testInstance, k):
    distance = []
    for x in range(len(trainingSet)):
        teuclid = sum(pow(trainingSet[x, 0:2] - testInstance,2))
        dist = np.sqrt(teuclid)
        distance.append((trainingSet[x].tolist(), dist))
    distance.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distance[x][0])
    return neighbors


def response(neighbors):


def accuracyFunction(testSet, prediction):
    

def knn(trainData, testData, k, feedback_classification):
    predictions = []

    for i in range(len(testData)):
        neighbours = similarSet(trainData,testData[i],k)
        result = response(neighbours)
        #print(result)
        predictions.append(result)
        newV = ([[testData[i, 0], testData[i, 1], result]])
        if feedback_classification == True:
            trainData = np.append(trainData, newV, axis=0)
        #print(trainData)

    #print(neighbors)
    #print(result)
    set111=testSet11[:, 2]
    #print(predictions)
  
    cMatrix = # NEED TO BE FIXED 
    print(cMatrix)

    accuracy = accuracyFunction(testSet11, predictions)
    print(accuracy)
    #USE AFTER CLASSIFICATION
    plt.scatter(trainData[:, 0], trainData[:, 1], c=trainData[:, 2], label='TrainDataSet', s=70, alpha=0.7, marker='.')
    plt.scatter(removeNon[:, 0], removeNon[:, 1], c=predictions, label='TestingDataSet', s=50, alpha=0.5, marker='x')
    # ON WHEN RANDOM NEEDED
    #plt.scatter(testSet11[:, 0], testSet11[:, 1], c=predictions, label='TestingDataSet', s=50, alpha=0.5, marker='x')
    plt.xlabel('Feature value of x',fontsize=12)
    plt.ylabel('Feature value of y',fontsize=12)
    plt.title('After classification', fontsize=15)
    plt.legend()
    plt.show()


def main():
    print("main function")
    global trainData
# random check
    #print(testData)3
    #print(trainData)
    #print(testSet11)
    #print(len(testData))
    #np.random.shuffle(testSet11)# ON WHEN RANDOM NEEDED
    #print(df2)
    #randTestData = testSet11[:, :-1]# ON WHEN RANDOM NEEDED
    #print(testSet11)
    #print(randTestData)
# end of random check
    """
    #for randomization for randomization for randomization for randomization
    testSet11Copy=testSet11
    np.random.shuffle(testSet11Copy)
    print(testSet11Copy)
    testDataRand = testSet11Copy[:, :-1]
    print(testDataRand)
    """

    k = 3
    feedback_classification = False
    predictions = []


    print(feedback_classification)
    # Main KNN
    knn(trainData, testData, k, feedback_classification)

    
main()

