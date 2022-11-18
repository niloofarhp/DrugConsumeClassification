from sklearn.feature_selection import SelectFromModel
from enum import Enum
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif 
import numpy as np
from sklearn import svm, tree, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import export_text
import graphviz
from dtreeviz.trees import *
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
    MLP = 5
    GB = 6
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
        selector = SelectKBest(f_classif, k=6) 
        XX = selector.fit(self.trainData, self.trainDataY) 
        scores = -np.log10(selector.pvalues_)
        scores /= scores.max()
        indexes =[]
        for i in range(6):
            indexes.append(self.trainData.columns[scores.argmax()])
            scores[scores.argmax()] = 0
        X_new = pd.DataFrame(selector.fit_transform(self.trainData, self.trainDataY), columns = indexes)
        X_newT = pd.DataFrame(selector.transform(self.testData), columns = indexes)
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
        if(self.modelNames.MLP == modelName):
            self.MLPClassifier() 
        if(self.modelNames.GB == modelName):
            self.GBClassifier() 
        return self.calPerformance()

    def DecisionTreeClassifier(self):
        self.model = tree.DecisionTreeClassifier(criterion = "entropy",max_depth=10)
        self.model = self.model.fit(self.trainX,self.trainY)
        r = export_text(self.model)
        Ydata=np.zeros_like(self.trainY)
        for i in range(len(self.trainY)):
            if self.trainY[i] == "NonUser":
                Ydata[i] = 0
            else:
                Ydata[i] = 1  
        tree.export_graphviz(self.model,
                            feature_names = self.trainX.columns,
                            out_file="tree2.dot",
                            filled = True)        
        # by using this command we can have the png file of the tree 
        # dot -Tpng tree2.dot -o tree_11.png
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
        self.model = KNeighborsClassifier(n_neighbors=3,)
        self.model = self.model.fit(self.trainX,self.trainY)
        return

    def MLPClassifier(self):
        self.model = MLPClassifier(random_state=1, max_iter=700).fit(self.trainX, self.trainY)
        return

    def GBClassifier(self):
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1 ,max_depth=1, random_state=0).fit(self.trainX, self.trainY)
        return

    def calPerformance(self):
        Y_PredTest = self.model.predict(self.testX)
        accuracyTest = accuracy_score(self.testY, Y_PredTest) 
        accuracyTrain = accuracy_score(self.trainY, self.model.predict(self.trainX))
        return accuracyTrain, accuracyTest

    def predictModel(self):
        predY = self.model.predict(self.testX)
        return predY

