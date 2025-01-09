import numpy as np
import pandas as pd


def createDataSet():
    dataSet = [['青绿','蜷缩','浊响','清晰','凹陷','硬滑',0.697,0.46,'yes'],
               ['乌黑','蜷缩','沉闷','清晰','凹陷','硬滑',0.774,0.376,'yes'],
               ['乌黑','蜷缩','浊响','清晰','凹陷','硬滑',0.634,0.264,'yes'],
               ['青绿','蜷缩','沉闷','清晰','凹陷','硬滑',0.608,0.318,'yes'],
               ['浅白','蜷缩','浊响','清晰','凹陷','硬滑',0.556,0.215,'yes'],
               ['青绿','稍蜷','浊响','清晰','稍凹','软粘',0.403,0.237,'yes'],
               ['乌黑','稍蜷','浊响','稍糊','稍凹','软粘',0.481,0.149,'yes'],
               ['乌黑','稍蜷','浊响','清晰','稍凹','硬滑',0.437,0.211,'yes'],
               ['乌黑','稍蜷','沉闷','稍糊','稍凹','硬滑',0.666,0.091,'no'],
               ['青绿','硬挺','清脆','清晰','平坦','软粘',0.243,0.267,'no'],
               ['浅白','硬挺','清脆','模糊','平坦','硬滑',0.245,0.057,'no'],
               ['浅白','蜷缩','浊响','模糊','平坦','软粘',0.343,0.099,'no'],
               ['青绿','稍蜷','浊响','稍糊','凹陷','硬滑',0.639,0.161,'no'],
               ['浅白','稍蜷','沉闷','稍糊','凹陷','硬滑',0.657,0.198,'no'],
               ['乌黑','稍蜷','浊响','清晰','稍凹','软粘',0.360,0.37,'no'],
               ['浅白','蜷缩','浊响','模糊','平坦','硬滑',0.593,0.042,'no'],
               ['青绿','蜷缩','沉闷','稍糊','稍凹','硬滑',0.719,0.103,'no']]

    labels=['色泽','根蒂','敲声','纹理','脐部','触感','密度','含糖率','是否好瓜']
    dataSet = pd.DataFrame(dataSet, columns=labels)
    return dataSet
# Press the green button in the gutter to run the script.

class Treenode:
    def __init__(self):
        self.value=None
        self.children=[]

    def show(self):
        print(self.value)

        for child in self.children:
            child.show()

    def print_tree(self, level=0):
        print("  " * level + str(self.value))  # 打印当前节点，添加缩进表示层级
        for child in self.children:  # 遍历子节点
            child.print_tree(level + 1)

class DecisionTree:
    def __init__(self):
        self.kind='ID3'
        self.selected_feature=[]
        self.entropy = 0
        self.tree=Treenode()
        self.max_depth = 2
        self.current_depth=0

    def select_feature(self,data):
        data_columns=data.columns
        bestentropy=100
        bestfeature=data_columns[0]
        bestsplitpoint=False
        for feature in data_columns[:-1]:

            # different method to handle the continuous/discrete feature
            # calculate conditional entropy
            c_entropy = 0
            if data.dtypes[feature]=='float64' :
                possible_value_list = data[feature].unique()
                possible_value_list=sorted(possible_value_list)
                possible_split_list=[round((possible_value_list[i+1]+possible_value_list[i])/2,5) for i in range(len(possible_value_list)-1)]
                #print(possible_split_list)

                for splitpoint in possible_split_list:
                    c_entropy = 0
                    tempdf1=data.loc[data[feature]>=splitpoint]
                    tempamount1 = tempdf1.shape[0]
                    tempdf2=data.loc[data[feature]<splitpoint]
                    tempamount2 = tempdf2.shape[0]
                    mydic = {}
                    for result_value in tempdf1.iloc[:, -1].values:
                        mydic[result_value] = mydic.get(result_value, 0) + 1
                    for key in mydic.keys():
                        p = mydic[key] / tempamount1
                        c_entropy -= p * np.log(p) * tempamount1/(tempamount1+tempamount2)
                        #print(splitpoint, p)
                    mydic = {}
                    for result_value in tempdf2.iloc[:, -1].values:
                        mydic[result_value] = mydic.get(result_value, 0) + 1
                    for key in mydic.keys():
                        p = mydic[key] / tempamount2
                        c_entropy -= p * np.log(p) * tempamount2/(tempamount1+tempamount2)
                    #print(feature, c_entropy, splitpoint)
                    if c_entropy < bestentropy:
                        bestentropy = c_entropy
                        bestfeature = feature
                        bestsplitpoint = splitpoint

            else:
                possible_value_list=data[feature].unique()
                amount=data.shape[0]
                for value in possible_value_list:
                    tempamount=data.loc[data[feature]==value].shape[0]
                    tempdf=data.loc[data[feature]==value]
                    mydic = {}
                    for result_value in tempdf.iloc[:, -1].values:
                        mydic[result_value] = mydic.get(result_value, 0) + 1
                    for key in mydic.keys():
                        p = mydic[key] / tempamount
                        c_entropy -= p * np.log(p)* tempamount/amount
                #print(feature,c_entropy)
                if c_entropy<bestentropy:
                    bestentropy=c_entropy
                    bestfeature=feature
                    bestsplitpoint=False
        #print(bestentropy,bestfeature,bestsplitpoint)
        return bestfeature,bestsplitpoint

    # calculating entropy:H(D)=∑-plog(p)
    # default that the result col is the last col of the df
    def cal_whole_entropy(self,data):
        amount=len(data.iloc[:,-1])
        mydic={}
        for value in data.iloc[:, -1].values:
            mydic[value]=mydic.get(value,0)+1
        for key in mydic.keys():
            p=mydic[key]/amount
            self.entropy-=p*np.log(p)


    def build_tree(self,data,treenode):
        print(data,self.current_depth)
        if self.current_depth>=self.max_depth:
            return 0
        if len(data.iloc[:,-1].unique())==1:
            return 0
        select_column, splitpoint = self.select_feature(data)
        self.current_depth += 1
        treenode.value=[select_column, splitpoint]
        if splitpoint:
            data1 = data.loc[data[select_column] >= splitpoint].drop(columns=[select_column])
            data2 = data.loc[data[select_column] < splitpoint].drop(columns=[select_column])
            newnode1 = Treenode()
            newnode2 = Treenode()

            self.build_tree(data1,newnode1)
            self.build_tree(data2, newnode2)

            treenode.children.append(newnode1)
            treenode.children.append(newnode2)

        else:
            values=data[select_column].unique()
            for value in values:
                tpdata=data.loc[data[select_column]==value].drop(columns=[select_column])
                newnode=Treenode()
                self.build_tree(tpdata,newnode)
                treenode.children.append(newnode)
        self.current_depth -= 1
        return



    def train(self,data):
        self.origin_data=data
        self.max_possible_lenth=len(data.columns)-1
        self.cal_whole_entropy(data)

        self.build_tree(data,self.tree)

        self.tree.print_tree()
