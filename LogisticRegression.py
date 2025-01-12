import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# the The mathematical derivation of the derivative of the loss function:
# https://blog.csdn.net/weixin_60737527/article/details/124141293

class LogisticRegression:
    def __init__(self,alpha=0.001,batchsize=20,circle=1000):
        self.alpha=alpha
        self.batchsize=batchsize
        self.circle=circle

    def data_process(self):
        self.data=self.data.sample(n=self.data.shape[0])
        batchdata=[self.data[i:i+self.batchsize] for i in range(0,self.data.shape[0],self.batchsize)]
        return batchdata

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train1(self,data):
        #Maximum Likelihood Estimation (MLE) loss function
        self.data = data
        self.w = np.random.normal(size=(self.data.shape[1], 1))
        for i in range(self.circle):

            batches=self.data_process()

            for batch in batches:
                d_w=np.zeros(shape=(self.data.shape[1],1))
                for j in range(0,batch.shape[0]):
                    x0=np.r_[batch.iloc[j,:][:-1],1] #add 1 to present the intercept
                    x=np.mat(x0).T
                    y=batch.iloc[j,:][-1]
                    dw=(self.sigmoid(self.w.T*x)-y)[0,0]*x
                    d_w+=dw
                self.w-=self.alpha*d_w/self.batchsize

    def train2(self,data):
        # MSE loss function
        self.data = data
        self.w = np.random.normal(size=(self.data.shape[1], 1))
        for i in range(self.circle):

            batches=self.data_process()

            for batch in batches:
                d_w=np.zeros(shape=(self.data.shape[1],1))
                for j in range(0,batch.shape[0]):
                    x0=np.r_[batch.iloc[j,:][:-1],1] #add 1 to present the intercept
                    x=np.mat(x0).T
                    y=batch.iloc[j,:][-1]
                    dw= ((self.sigmoid(self.w.T * x)-y) * self.sigmoid(self.w.T*x)*(1-self.sigmoid(self.w.T*x)))[0,0]*x
                    d_w+=dw
                self.w-=self.alpha*d_w/self.batchsize

    def predict(self, x):
        x=np.append(x,1)
        x=np.mat(x)
        print(x * self.w)
        s = self.sigmoid(x * self.w)[0, 0]
        print('s',s)
        if s >= 0.5:
            return 1
        elif s < 0.5:
            return 0


def createDataSet():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['true_target'] = iris.target
    return data

data = createDataSet()
data=data.loc[data['true_target']!=2]#2 classes
lr=LogisticRegression()
lr.train1(data)
lr.predict(np.array([5,3.6,1.5,0.1]))#data close to class 0
lr.predict(np.array([5.7,2.9,4,0.8]))#data close to class 1
lr.train2(data)
lr.predict(np.array([5,3.6,1.5,0.1]))
lr.predict(np.array([5.7,2.9,4,0.8]))