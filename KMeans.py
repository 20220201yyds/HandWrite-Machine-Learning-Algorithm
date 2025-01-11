from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

class KMeans:
    def __init__(self,k=3):
        self.k=k
        self.maxiter=100

    def classify(self,data):
        self.data = data
        col=self.data.columns[:-1]
        self.data['cluster']=0
        lastSeries=self.data['cluster']
        init=data.sample(3)[col]
        kcenter_list = [list(init.iloc[0]), list(init.iloc[1]), list(init.iloc[2])]
        i=0

        while i<self.maxiter and  (i==0 or sum(lastSeries == self.data['cluster'])!=data.shape[0]):
            lastSeries=self.data['cluster'].copy()
            #update cluster
            for j in range(data.shape[0]):
                tempdistancelist=[]
                for center in kcenter_list:
                    tempdistance=0
                    for p in range(len(col)):
                        tempdistance+=(center[p]-data.iloc[j,p])**2
                    tempdistancelist.append(np.sqrt(tempdistance))

                minvalue=min(tempdistancelist)
                minindex=tempdistancelist.index(minvalue)
                self.data.loc[j,'cluster']=minindex

            #update kcenter_list
            for j in range(self.k):
                kcenter_list[j]=list(data.loc[data.cluster==j][col].mean())
            print(kcenter_list)
            i+=1
            print(data)
        return data




def createDataSet():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['true_target'] = iris.target
    return data

data = createDataSet()
kmeans=KMeans()
kmeans.classify(data)

