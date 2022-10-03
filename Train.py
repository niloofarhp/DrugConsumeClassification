import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, tree
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel
from enum import Enum
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
class FeatureEnums(Enum):
    SelectFromModel = 1 
    SelectFromVarianceThr = 2
    Non = 3

class MLModelEnums(Enum):
    DecisionTree = 1 
    RandomForest = 2
    KNearestNeighbor = 3
    SVM = 4

class featureSelection():
    def __init__(self, model, datasetTrain, datasetTes, data):
        self.featureSelectType = FeatureEnums
        self.trainData = datasetTrain
        self.testData = datasetTes
        self.model = model
        self.data = data
        pass
    def runFeatureSelection(self, featrueSelectName):
        X_new = self.trainData
        X_newT = self.testData
        if self.featureSelectType.SelectFromModel == featrueSelectName:
            X_new, X_newT = self.selectFeatureFromModel()
        if self.featureSelectType.SelectFromVarianceThr == featrueSelectName:
            X_new, X_newT = self.selectVarianceThreshold() 

        return X_new, X_newT
    def selectFeatureFromModel(self):
        modelf = SelectFromModel(self.model, prefit=True)
        print(self.model.feature_importances_)
        X_new = modelf.transform(self.data)
        return X_new[:len(self.trainData)], X_new[len(self.trainData):]

    def selectVarianceThreshold(self):
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        X_new = sel.fit_transform(self.data)
        return X_new[:len(self.trainData)], X_new[len(self.trainData):]
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
            self.DecisionTreeClassifier()
        if(self.modelNames.KNearestNeighbor == modelName):
            self.KNearestNeighborClassifier()
        if(self.modelNames.RandomForest == modelName):
            self.RandomForestClassifier()
        if(self.modelNames.SVM == modelName):
            self.SVMClassifier() 
        return self.calPerformance()

    def DecisionTreeClassifier(self):
        self.model = tree.DecisionTreeClassifier()
        self.model = self.model.fit(self.trainX,self.trainY)
        #tree.plot_tree(self.model)
        return 

    def SVMClassifier(self):
        self.model = svm.SVC()
        self.model = self.model.fit(self.trainX,self.trainY)
        return

    def RandomForestClassifier(self):
        self.model = RandomForestClassifier(max_depth=2, random_state=0)
        self.model = self.model.fit(self.trainX,self.trainY)
        return 

    def KNearestNeighborClassifier(self):
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model = self.model.fit(self.trainX,self.trainY)
        return 

    def calPerformance(self):
        Y_PredTest = self.model.predict(self.testX)
        accuracyTest = accuracy_score(self.testY, Y_PredTest) 
        accuracyTrain = accuracy_score(self.trainY, self.model.predict(self.trainX))
        return accuracyTrain, accuracyTest

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
    

def convertMultiClassLabelsToBinaryLabels(y_dataSet):
    #convert multiclasses to binary classes
    for i in range(len(y_dataSet)):
        label = y_dataSet[i]
        if label  == 'CL0' or label  == 'CL1' :
            y_dataSet[i] = 'NonUser'
        if label  == 'CL6' or label  == 'CL2' or label  == 'CL3' or label  == 'CL4' or label == 'CL5':
            y_dataSet[i] = 'User'    

def ConvertMultiToBinary(y_train, y_test):
        y_testArray = np.array(y_test)
        y_trainArray = np.array(y_train)
        calcLabelsInDataSet(y_trainArray)
        calcLabelsInDataSet(y_testArray)
        convertMultiClassLabelsToBinaryLabels(y_trainArray)
        convertMultiClassLabelsToBinaryLabels(y_testArray)
        calcLabelsInDataSet(y_testArray)
        return y_trainArray, y_testArray

def main():
    featureSelectName = [FeatureEnums.SelectFromModel,FeatureEnums.SelectFromVarianceThr, FeatureEnums.Non]
    modelNames = [MLModelEnums.DecisionTree, MLModelEnums.RandomForest, MLModelEnums.KNearestNeighbor, 
    MLModelEnums.SVM]
    d = pd.read_csv("drug_consumption2.data",header=None)
    df = pd.DataFrame(d.iloc[:,:], columns = d.columns[:])
    # print(df.columns)
    train_test_data = df.iloc[:,:12]
    target_data_Alcohol = df.iloc[:,31]  
    X_train, X_test, y_train, y_test = train_test_split(train_test_data, target_data_Alcohol, train_size=0.67, random_state = 0)
    y_trainArray, y_testArray = ConvertMultiToBinary(y_train, y_test)
    models = ML_Models(X_train, y_trainArray, X_test, y_testArray)
    fSName = featureSelectName[0]
    mName = modelNames[1]
    if(fSName == FeatureEnums.SelectFromModel and (mName == MLModelEnums.DecisionTree or mName == MLModelEnums.RandomForest)):
        accTrain, accTest = models.runModel(mName)
    featureSelect = featureSelection(models.model, X_train, X_test, train_test_data)
    X_trainN, X_testN = featureSelect.runFeatureSelection(fSName)
    accTrainN, accTestN = models.runModel(mName)

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