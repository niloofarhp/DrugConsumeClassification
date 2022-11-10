import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import  RandomUnderSampler
from collections import Counter
class DataProcess:
    def __init__(self, dataSetName):
        self.dataSetName = dataSetName
        self.usedatasetHeart = False
        self.usedataLabor = True
        pass
    def dataPreparation(self,target = 13):
        if self.usedatasetHeart:
            d = pd.read_csv(self.dataSetName)
        elif self.usedataLabor:
            d = pd.read_csv(self.dataSetName, header=None)   
        else:    
            d = pd.read_csv(self.dataSetName,header=None)
        print(len(d.columns))
        df = pd.DataFrame(d.iloc[:,:])
        dd = df.head()
        scaler = MinMaxScaler()
        if(self.usedatasetHeart):
            targetC = d['condition']
            train_test_data = df.drop(['condition'],axis=1)
        else:        
            train_test_data = df.loc[:,1:12]
            targetC = df.loc[:,int(target)] 
        scaler.fit(train_test_data)
        scaled = scaler.fit_transform(train_test_data)
        train_test_data = pd.DataFrame(scaled)
        
        X_train, X_test, y_train, y_test = train_test_split(train_test_data, targetC, train_size=0.67, random_state = 0)
        return X_train, X_test, y_train, y_test

    def calcLabelsInDataSet(self,y_dataSet):
        # print((['CL0'] == y_dataSet).sum()) # 11
        # print((['CL1'] == y_dataSet).sum()) # 9
        # print((['CL2'] == y_dataSet).sum()) # 21
        # print((['CL3'] == y_dataSet).sum()) # 58
        # print((['CL4'] == y_dataSet).sum()) # 102
        # print((['CL5'] == y_dataSet).sum()) # 255
        # print((['CL6'] == y_dataSet).sum()) # 167
        # print((['NonUser'] == y_dataSet).sum()) # 167
        # print((['User'] == y_dataSet).sum()) # 167
        return
        
    def convertMultiClassLabelsToBinaryLabels(self,y_dataSet, biThr):
        labelCls = ['CL0','CL1','CL2','CL3','CL4','CL5','CL6']
        for i in range(len(y_dataSet)):
            label = y_dataSet[i]
            if labelCls.index(label) < biThr:
                y_dataSet[i] = 'NonUser'
            else:
                y_dataSet[i] = 'User' 
                
    def ConvertMultiToBinary(self,y_train, y_test, biThr):
            y_testArray = np.array(y_test)
            y_trainArray = np.array(y_train)
            #print("before binarization label numbers:")
            self.calcLabelsInDataSet(y_trainArray)
            self.convertMultiClassLabelsToBinaryLabels(y_trainArray, biThr)
            self.convertMultiClassLabelsToBinaryLabels(y_testArray, biThr)
            #print("after binarization label numbers:")
            self.calcLabelsInDataSet(y_trainArray)
            return y_trainArray, y_testArray

    def dataSetOverSampling(self,X,y):
        oversample = RandomOverSampler(sampling_strategy='minority')
        X_over, y_over = oversample.fit_resample(X, y)
        print(Counter(y))
        print(Counter(y_over))
        return  X_over, y_over      
    def dataSetUnderSampling(self,X,y):
        undersample = RandomUnderSampler(sampling_strategy='majority')
        X_under, y_under = undersample.fit_resample(X, y)
        print(Counter(y))
        print(Counter(y_under))
        return  X_under, y_under
