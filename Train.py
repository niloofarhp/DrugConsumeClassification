import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import seaborn as sns
from enum import Enum
from DataUtils import DataProcess
import MLUtills 
from sklearn import metrics
from sklearn.model_selection import KFold
from collections import Counter
def main():
    featureSelectName = [MLUtills.FeatureEnums.UnivariateFeature]#[MLUtills.FeatureEnums.SelectFromVarianceThr,MLUtills.FeatureEnums.SelectFromModel,MLUtills.FeatureEnums.UnivariateFeature, MLUtills.FeatureEnums.Non]
    modelNames = [MLUtills.MLModelEnums.DecisionTree]
    #, MLUtills.MLModelEnums.RandomForest,MLUtills.MLModelEnums.KNearestNeighbor, MLUtills.MLModelEnums.SVM, MLUtills.MLModelEnums.MLP, MLUtills.MLModelEnums.GB]

    drug_names = ['alcohol 5', 'Amphetamines 1', 'Crack 1' , 'Ecstasy 1', 'Nicotine 1', 'VSA 2']
    binarizationThr = [5,1,1,1,1,2]
    targetDrugs = ['13'    ,'14'         ,'21'   ,'22'    ,'30'     ,'31']
    dataProcess = DataProcess("/Users/niloofar/Documents/Projects/DrugConsumeClassification/DataSets/drug_consumption2.data")#heart_cleveland_upload.csv") #heart_cleveland_upload.csv")
    useOverSample = True
    useUnderSample = False
    useKfold = False
    useHeartLabor = False
    useLaborNegotiate = False 
    # Ecstasy is chosen from the last HW
    for i in range(len(drug_names)):
        i = 3
        X_train, X_test, y_train, y_test = dataProcess.dataPreparation(targetDrugs[i])
        y_testArray = np.array(y_test)
        y_trainArray = np.array(y_train)
        if not useHeartLabor and not useLaborNegotiate:
            y_trainArray, y_testArray = dataProcess.ConvertMultiToBinary(y_train, y_test, binarizationThr[i])
        if useOverSample:
            X_train, y_trainArray = dataProcess.dataSetOverSampling(X_train,y_trainArray)
        elif useUnderSample:
            X_train, y_trainArray = dataProcess.dataSetUnderSampling(X_train,y_trainArray)
        print("the target drug is :", drug_names[i] ,"*********************************************")
        #feature_names=X_train['feature_names']
        for mName in modelNames:
            X_trainK, y_trainK, X_testK, y_testK = X_train, y_trainArray, X_test, y_testArray
            if useKfold:
                X_trainK = np.concatenate((X_trainK,X_test),axis=0)
                y_trainArrayK = np.concatenate((y_trainArray,y_testArray),axis=0)
                kf = KFold(n_splits=10)
                foldnum = 0
                meanACC =0
                for train_index, test_index in kf.split(np.array(X_trainK)):
                    X_train, X_test = X_trainK.take(list(train_index),axis=0), X_trainK.take(list(test_index),axis=0)
                    y_train, y_test = y_trainArrayK.take(list(train_index),axis=0), y_trainArrayK.take(list(test_index),axis=0)
                    bestTestAc = 0
                    selectedF = ""
                    for fsName in featureSelectName:
                        if (mName == MLUtills.MLModelEnums.KNearestNeighbor or mName == MLUtills.MLModelEnums.SVM or mName == MLUtills.MLModelEnums.MLP) and fsName == MLUtills.FeatureEnums.SelectFromModel:
                            continue
                        models = MLUtills.ML_Models(X_train, y_train, X_test, y_test)
                        ##accTrain, accTest = models.runModel(mName)
                        featureSelect = MLUtills.featureSelection(models.model, X_train, y_train, X_test)
                        X_trainN, X_testN = featureSelect.runFeatureSelection(fsName)
                        modelsF = MLUtills.ML_Models(X_trainN, y_train, X_testN, y_test)
                        accTrainN, accTestN = modelsF.runModel(mName)
                        if(bestTestAc < accTestN):
                            bestSelectFM = featureSelect
                            selectedF = fsName
                            selectedW = modelsF
                            bestTestAc = accTestN 
                    foldnum += 1
                    meanACC+=bestTestAc
                print("the model is :", mName, "the average ACC over 10fold is:", meanACC/10 )    
            else:
                bestTestAc = 0
                selectedF = ""
                print("the model is :")
                print(mName)
                for fsName in featureSelectName:
                    if (mName == MLUtills.MLModelEnums.KNearestNeighbor or mName == MLUtills.MLModelEnums.SVM or mName == MLUtills.MLModelEnums.MLP) and fsName == MLUtills.FeatureEnums.SelectFromModel:
                        continue
                    models = MLUtills.ML_Models(X_trainK, y_trainK, X_testK, y_testK)
                    accTrain, accTest = models.runModel(mName)
                    featureSelect = MLUtills.featureSelection(models.model, X_trainK, y_trainK, X_testK)
                    X_trainN, X_testN = featureSelect.runFeatureSelection(fsName)
                    modelsF = MLUtills.ML_Models(X_trainN, y_trainK, X_testN, y_testK)
                    accTrainN, accTestN = modelsF.runModel(mName)
                    if(bestTestAc < accTestN):
                        bestSelectFM = featureSelect
                        selectedF = fsName
                        selectedW = modelsF
                        bestTestAc = accTestN     
                print("confusion matrix for the model selected using:", selectedF , "testACC : ", bestTestAc)
                cfM = confusion_matrix(y_testK, modelsF.predictModel())
                #print(cfM)
                ConfusionMatrixDisplay(cfM, display_labels=modelsF.model.classes_).plot()
                precision = cfM[1][1] / (cfM[1][1] + cfM[0][1])
                recall = cfM[1][1] / (cfM[1][0] + cfM[1][1])
                sensitivity = recall
                specificity = cfM[0][0] / (cfM[0][0] + cfM[0][1])
                # print("presicion: ",precision)
                # print("recal: ",recall)
                # print("sensivity: ",sensitivity)
                # print("specificity: ",specificity)
                # fpr, tpr, thresholds = metrics.roc_curve(y_testK,selectedW.model.predict(bestSelectFM.runFeatureSelection(selectedF)[1]))
                # roc_auc = metrics.auc(fpr, tpr)
                #metrics.RocCurveDisplay.from_estimator(selectedW.model, bestSelectFM.runFeatureSelection(selectedF)[1], y_testK) 
                #plt.plot()
                # metrics.plot_roc_curve(selectedW.model, bestSelectFM.runFeatureSelection(selectedF)[1], y_testK) 
                print("the end of model with 3 different feature selection......................")   
            print("***************************")
        break
    return

if __name__ == '__main__':
    main() 