import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, tree, metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif 
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectFromModel
from enum import Enum
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
class FeatureEnums(Enum):
    SelectFromModel = 1 
    SelectFromVarianceThr = 2
    UnivariateFeature = 3
    Non = 4
class MLModelEnums(Enum):
    DecisionTree = 1 
    RandomForest = 2
    KNearestNeighbor = 3
    SVM = 4
class featureSelection():
    def __init__(self, model, datasetTrain, dataTrainY, datasetTes):
        self.featureSelectType = FeatureEnums
        self.trainData = datasetTrain
        self.testData = datasetTes
        self.model = model
        self.trainDataY = dataTrainY

    def runFeatureSelection(self, featrueSelectName):
        X_new = self.trainData
        X_newT = self.testData
        if self.featureSelectType.SelectFromModel == featrueSelectName:
            X_new, X_newT = self.selectFeatureFromModel()
        if self.featureSelectType.SelectFromVarianceThr == featrueSelectName:
            X_new, X_newT = self.selectVarianceThreshold() 
        if self.featureSelectType.UnivariateFeature == featrueSelectName:
           X_new, X_newT = self.selectUnivariateFeature()    
        return X_new, X_newT

    def selectFeatureFromModel(self):
        modelf = SelectFromModel(self.model, prefit=True)
        # print(self.model.feature_importances_)
        X_new = modelf.transform(self.trainData)
        X_newT = modelf.transform(self.testData)
        return X_new, X_newT

    def selectVarianceThreshold(self):
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        X_new = sel.fit_transform(self.trainData)
        X_newT = sel.transform(self.testData)
        return X_new, X_newT
        
    def selectUnivariateFeature(self):
        selector = SelectKBest(f_classif, k=4) 
        selector.fit(self.trainData, self.trainDataY) 
        scores = -np.log10(selector.pvalues_)
        scores /= scores.max()
        X_new = selector.fit_transform(self.trainData, self.trainDataY)
        X_newT = selector.transform(self.testData)
        return X_new, X_newT  
class ML_Models():
    def __init__(self,trainX, trainY, testX, testY):
        self.modelNames = MLModelEnums #Enum('DecisionTree', 'RandomForest', 'KNearestNeighbor', 'SVM')
        self.model = tree.DecisionTreeClassifier() # default 
        self.trainX = trainX  
        self.trainY = trainY
        self.testX = testX  
        self.testY = testY 

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
    def predictModel(self):
        predY = self.model.predict(self.testX)
        return predY

def dataPreparation(target = 13):
    d = pd.read_csv("drug_consumption2.data",header=None)
    df = pd.DataFrame(d.iloc[:,:])
    dd = df.head()
    scaler = MinMaxScaler()
    train_test_data = df.loc[:,1:12]
    scaler.fit(train_test_data)
    scaled = scaler.fit_transform(train_test_data)
    train_test_data = pd.DataFrame(scaled)
    targetC = df.loc[:,int(target)] 
    X_train, X_test, y_train, y_test = train_test_split(train_test_data, targetC, train_size=0.67, random_state = 0)
    return X_train, X_test, y_train, y_test

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
    
def convertMultiClassLabelsToBinaryLabels(y_dataSet, biThr):
    labelCls = ['CL0','CL1','CL2','CL3','CL4','CL5','CL6']
    for i in range(len(y_dataSet)):
        label = y_dataSet[i]
        if labelCls.index(label) < biThr:
            y_dataSet[i] = 'NonUser'
        else:
            y_dataSet[i] = 'User' 
               
def ConvertMultiToBinary(y_train, y_test, biThr):
        y_testArray = np.array(y_test)
        y_trainArray = np.array(y_train)
        print("before binarization label numbers:")
        calcLabelsInDataSet(y_trainArray)
        convertMultiClassLabelsToBinaryLabels(y_trainArray, biThr)
        convertMultiClassLabelsToBinaryLabels(y_testArray, biThr)
        print("after binarization label numbers:")
        calcLabelsInDataSet(y_trainArray)
        return y_trainArray, y_testArray

def main():
    featureSelectName = [FeatureEnums.SelectFromModel,FeatureEnums.SelectFromVarianceThr, FeatureEnums.UnivariateFeature, FeatureEnums.Non]
    modelNames = [MLModelEnums.DecisionTree, MLModelEnums.RandomForest, MLModelEnums.KNearestNeighbor, MLModelEnums.SVM]
    drug_names = ['alcohol 5', 'Amphetamines 1', 'Crack 1' , 'Ecstasy 1', 'Nicotine 1', 'VSA 2']
    binarizationThr = [5,1,1,1,1,2]
    targetDrugs = ['13'    ,'14'         ,'21'   ,'22'    ,'30'     ,'31']
    for i in range(len(targetDrugs)):
        X_train, X_test, y_train, y_test = dataPreparation(targetDrugs[i])
        y_trainArray, y_testArray = ConvertMultiToBinary(y_train, y_test, binarizationThr[i])
        print("the target drug is :", drug_names[i] ,"*********************************************")
        for mName in modelNames:
            bestTestAc = 0
            selectedF = ""
            print("the model is :")
            print(mName)
            for fsName in featureSelectName:
                if (mName == MLModelEnums.KNearestNeighbor or mName == MLModelEnums.SVM) and fsName == FeatureEnums.SelectFromModel:
                    continue
                # print("feature selection name :")
                # print(fsName)
                models = ML_Models(X_train, y_trainArray, X_test, y_testArray)
                accTrain, accTest = models.runModel(mName)
                # print("model performane before feature selection ->")
                # print("train data :", accTrain, "test data :", accTest)
                featureSelect = featureSelection(models.model, X_train, y_trainArray, X_test)
                X_trainN, X_testN = featureSelect.runFeatureSelection(fsName)
                modelsF = ML_Models(X_trainN, y_trainArray, X_testN, y_testArray)
                accTrainN, accTestN = modelsF.runModel(mName)
                # print("model performane after feature selection ->")
                # print("model performane -> train data :", accTrainN, "test data :", accTestN)
                if(bestTestAc < accTestN):
                    bestSelectFM = featureSelect
                    selectedF = fsName
                    selectedW = modelsF
                    bestTestAc = accTestN     
            print("confusion matrix for the model selected using:", selectedF , "testACC : ", bestTestAc)
            cfM = confusion_matrix(y_testArray, modelsF.predictModel())
            print(cfM)
            ConfusionMatrixDisplay(cfM, display_labels=modelsF.model.classes_).plot()
            precision = cfM[1][1] / (cfM[1][1] + cfM[0][1])
            recall = cfM[1][1] / (cfM[1][0] + cfM[1][1])
            sensitivity = recall
            specificity = cfM[0][0] / (cfM[0][0] + cfM[0][1])
            print("presicion: ",precision)
            print("recal: ",recall)
            print("sensivity: ",sensitivity)
            print("specificity: ",specificity)
            # fpr, tpr, thresholds = metrics.roc_curve(y_testArray,selectedW.model.predict(bestSelectFM.runFeatureSelection(selectedF)[1]))
            # roc_auc = metrics.auc(fpr, tpr)
            metrics.RocCurveDisplay.from_estimator(selectedW.model, bestSelectFM.runFeatureSelection(selectedF)[1], y_testArray) 
            plt.plot()
            # metrics.plot_roc_curve(selectedW.model, bestSelectFM.runFeatureSelection(selectedF)[1], y_testArray) 
            print("the end of model with 3 different feature selection......................")    
        print()
    return

if __name__ == '__main__':
    main() 