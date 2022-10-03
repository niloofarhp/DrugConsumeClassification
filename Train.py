from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn import svm, tree
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel
from enum import Enum

class FeatureEnums(Enum):
    SelectFromModel = 1 
    KBest = 2

class MLModelEnums(Enum):
    DecisionTree = 1 
    RandomForest = 2
    KNearestNeighbor = 3
    SVM = 4

class featureSelection():
    def __init__(self, model, datasetTrain, datasetTes):
        self.featureSelectType = FeatureEnums #Enum('SelectFromModel', 'KBest')
        self.trainData = datasetTrain
        self.testData = datasetTes
        self.model = model
        pass
    def runFeatureSelection(self, featrueSelectName):
        if self.featureSelectType.SelectFromModel == featrueSelectName:
            X_new, X_newT = self.selectFeatureFromModel()
        return X_new, X_newT
    def selectFeatureFromModel(self):
        modelf = SelectFromModel(self.model, prefit=True)
        print(self.model.feature_importances_)
        X_new = modelf.transform(self.trainData)
        X_newT = modelf.transform(self.testData)
        return X_new, X_newT


class ML_Models():
    def __init__(self,trainX, trainY, testX, testY):
        self.modelNames = MLModelEnums #Enum('DecisionTree', 'RandomForest', 'KNearestNeighbor', 'SVM')
        self.model = tree.DecisionTreeClassifier() # default 
        self.trainX = trainX  
        self.trainY = trainY
        self.testX = testX  
        self.testY = testY 
        pass
    def runModel(self, modelName):
        if(self.modelNames.DecisionTree == modelName):
            acc, accT = self.DecisionTreeClassifier()
        return acc, accT
    def DecisionTreeClassifier(self):
        self.model = tree.DecisionTreeClassifier()
        self.model = self.model.fit(self.trainX,self.trainY)
        #tree.plot_tree(self.model)
        Y_PredTest = self.model.predict(self.testX)
        Y_PredTest_Prob = self.model.predict_proba(self.testX)
        accuracyTest = accuracy_score(self.testY, Y_PredTest) 
        accuracyTrain = accuracy_score(self.trainY, self.model.predict(self.trainX)) 
        return accuracyTrain, accuracyTest
    def SVMClassifier():
        return
    def RandomForestClassifier():
        return 
    def KNearestNeighborClassifier():
        return       


def calcLabelsInDataSet(y_dataSet):
    print((['CL0'] == y_dataSet).sum()) # 11
    print((['CL1'] == y_dataSet).sum()) # 9
    print((['CL2'] == y_dataSet).sum()) # 21
    print((['CL3'] == y_dataSet).sum()) # 58
    print((['CL4'] == y_dataSet).sum()) # 102
    print((['CL5'] == y_dataSet).sum()) # 255
    print((['CL6'] == y_dataSet).sum()) # 167
    print((['NonUser'] == y_dataSet).sum()) # 167
    print((['User'] == y_dataSet).sum()) # 167
    

def convertMultiClassLabelsToBinary(y_dataSet):
    #convert multiclasses to binary classes
    for i in range(len(y_dataSet)):
        label = y_dataSet[i]
        if label  == 'CL0' or label  == 'CL1' :
            y_dataSet[i] = 'NonUser'
        if label  == 'CL6' or label  == 'CL2' or label  == 'CL3' or label  == 'CL4' or label == 'CL5':
            y_dataSet[i] = 'User'    

def main():
    featureSelectName = [FeatureEnums.SelectFromModel,FeatureEnums.KBest]
    modelNames = ['DecisionTree', 'RandomForest', 'KNearestNeighbor', 'SVM']
    d = pd.read_csv("drug_consumption2.data",header=None)
    df = pd.DataFrame(d.iloc[:,:], columns = d.columns[:])
    # print(df.columns)
    train_test_data = df.iloc[:,:12]
    target_data_Alcohol = df.iloc[:,31]  
    X_train, X_test, y_train, y_test = train_test_split(train_test_data, target_data_Alcohol, train_size=0.67, random_state = 0)
    y_testArray = np.array(y_test)
    y_trainArray = np.array(y_train)
    calcLabelsInDataSet(y_trainArray)
    calcLabelsInDataSet(y_testArray)
    convertMultiClassLabelsToBinary(y_trainArray)
    convertMultiClassLabelsToBinary(y_testArray)
    calcLabelsInDataSet(y_testArray)

    models = ML_Models(X_train, y_trainArray, X_test, y_testArray)
    fSName = featureSelectName[0]
    mName = modelNames[0]
    if(fSName == FeatureEnums.SelectFromModel):
        accTrain, accTest = models.runModel(MLModelEnums.DecisionTree)
    featureSelect = featureSelection(models.model, X_train, X_test)
    X_trainN, X_testN = featureSelect.runFeatureSelection(FeatureEnums.SelectFromModel)
    accTrainN, accTestN = models.runModel(MLModelEnums.DecisionTree)

    return
if __name__ == '__main__':
    main() 




# linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
# linear_pred = linear.predict(X_test)
# cm_lin = confusion_matrix(y_test, linear_pred)
# rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
# poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
# sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)


# #Using Pearson Correlation
# plt.figure()#figsize=(12,10))
# cor = df.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()
# cor_target = abs(cor[12])
# relevant_features = cor_target[cor_target>0.5]
# relevant_features    