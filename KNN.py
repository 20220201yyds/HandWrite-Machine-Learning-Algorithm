from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from heapq import *

class KNN:
    def __init__(self,k):
        self.k=k

    def loaddata(self,train_data):
        self.train_data=train_data

    def predict(self,test_data):
        amount=0
        correct=0
        returnlist=[]
        for i in range(test_data.shape[0]):
            # need to much resource
            '''closestlist=[]
            for j in range(self.train_data.shape[0]):
                distance=sum((train_data.iloc[j,:]-test_data.iloc[i,:])**2)
                closestlist.append((distance,j))
            heapify(closestlist)
            closestlist=nsmallest(self.k,closestlist)
            print(closestlist)'''

            closestlist = []
            for j in range(self.train_data.shape[0]):
                distance = sum((train_data.iloc[j, :] - test_data.iloc[i, :]) ** 2)
                if j<self.k:
                    closestlist.append((-distance, j))
                    if j==self.k-1:
                        heapify(closestlist)
                else:
                    heappushpop(closestlist,(-distance,j))
            resultdic={}
            for point in closestlist:
                result=train_data.iloc[point[1],:].target
                resultdic[result]=resultdic.get(result,0)+1
            maxkey=max(resultdic,key=resultdic.get)

            amount+=1
            if maxkey==test_data.iloc[i,:].target:
                correct+=1
            returnlist.append(maxkey)

        return returnlist,correct/amount

def createDataSet():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target

    train_data=data.sample(n=100)
    test_data=data.drop(train_data.index)

    return train_data,test_data

train_data,test_data=createDataSet()
knn=KNN(5)
knn.loaddata(train_data)
predict_data,accuracy=knn.predict(test_data)
print(predict_data)
print(accuracy)