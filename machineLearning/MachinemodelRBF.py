from sklearn import svm,datasets,metrics
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold,train_test_split,cross_val_score,cross_val_predict,GridSearchCV
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc , plot_roc_curve,confusion_matrix,f1_score
from sklearn.multiclass import OneVsRestClassifier


# import dataset
dataAll = pd.read_csv("eyedata.csv")
dataGO = dataAll.copy()
dataNO = dataAll.copy()
dataOO = dataAll.copy()

eye = dataAll.eye.values
dataGO['eye'] = dataGO['eye'].replace(2,1)
dataNO['eye'] = dataNO['eye'].replace(2,0)
dataOO['eye'] = dataOO['eye'].replace(1,0)


# prepare data for Glucoma vs Non-Glaucoma
cdr01 = np.reshape(dataGO.CDr.values,(-1,1))
cdd01 = np.reshape(dataGO.CDd.values,(-1,1))
data01 = np.hstack([cdr01,cdd01])
eye01 = dataGO.drop(['CDr','CDd'],axis=1).eye.values

trainData01,testData01,trainType01,testType01 = train_test_split(data01, eye01, test_size=0.2 , random_state=1)
skf = StratifiedKFold(n_splits=5)

plt.scatter(data01[:,0],data01[:,1], c=eye01)
plt.xlabel('CDR')
plt.ylabel('y')


# RBF model for Glucoma vs Non-Glaucoma
rbf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
i = 1
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

for train_index, test_index in skf.split(trainData01, trainType01):
    x_train, x_test = trainData01[train_index], trainData01[test_index]
    y_train, y_test = trainType01[train_index], trainType01[test_index]
    rbf.fit(x_train,y_train)
    filename = 'savemodel/GORBF_model'+str(i)+'.sav'
    joblib.dump(rbf,filename)

    viz = plot_roc_curve(rbf, x_test, y_test,name='RBF SVM model {}'.format(i), ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    i+=1

loaded_model = joblib.load('savemodel/GORBF_model3.sav')

conf_matrix = confusion_matrix(testType01,loaded_model.predict(testData01))
FP = conf_matrix[0][1]
FN = conf_matrix[1][0]
TP = conf_matrix[1][1]
TN = conf_matrix[0][0]
Accuracy = (TP+TN)/(TP+FP+FN+TN)
Specificity = TN/(TN+FP)
sensitivity = TP / (TP + FN)
Precision = TP / (TP + FP)
print('Accuracy:',Accuracy)
print('Specificity',Specificity)
print('sensitivity:',sensitivity)
print('Precision:',Precision)
print('F1 score',f1_score(testType01,loaded_model.predict(testData01), average='micro'),'\n')
print('FP',FP)
print('FN',FN)
print('TP',TP)
print('TN',TN,'\n')

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],title="Glaucoma vs Non-Gluacoma")
ax.legend(loc="lower right")
plt.show()


# prepare data for Normal vs Non-Normal
cdr02 = np.reshape(dataNO.CDr.values,(-1,1))
cdd02 = np.reshape(dataNO.CDd.values,(-1,1))
data02 = np.hstack([cdr02,cdd02])
eye02 = dataNO.drop(['CDr','CDd'],axis=1).eye.values

trainData02,testData02,trainType02,testType02 = train_test_split(data02, eye02, test_size=0.2 , random_state=1)

plt.scatter(data02[:,0],data02[:,1], c=eye02)
plt.xlabel('CDR')
plt.ylabel('CDD')


# RBF model for Normal vs Non-Normal
rbf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
i = 1
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

for train_index, test_index in skf.split(trainData02,trainType02):
    x_train, x_test = trainData02[train_index], trainData02[test_index]
    y_train, y_test = trainType02[train_index], trainType02[test_index]
    rbf.fit(x_train,y_train)
    filename = 'savemodel/NORBF_model'+str(i)+'.sav'
    joblib.dump(rbf,filename)

    viz = plot_roc_curve(rbf,x_test,y_test,name='RBF SVM model {}'.format(i), ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    i+=1

loaded_model = joblib.load('savemodel/NORBF_model1.sav')

conf_matrix = confusion_matrix(testType02,loaded_model.predict(testData02))
FP = conf_matrix[0][1]
FN = conf_matrix[1][0]
TP = conf_matrix[1][1]
TN = conf_matrix[0][0]
Accuracy = (TP+TN)/(TP+FP+FN+TN)
Specificity = TN/(TN+FP)
sensitivity = TP / (TP + FN)
Precision = TP / (TP + FP)
print('Accuracy:',Accuracy)
print('Specificity',Specificity)
print('sensitivity:',sensitivity)
print('Precision:',Precision)
print('F1 score',f1_score(testType02,loaded_model.predict(testData02), average='micro'),'\n')
print('FP',FP)
print('FN',FN)
print('TP',TP)
print('TN',TN,'\n')

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],title="Normal vs Non-Normal")
ax.legend(loc="lower right")
plt.show()


# prepare data for Other vs Non-Other
cdr12 = np.reshape(dataOO.CDr.values,(-1,1))
cdd12 = np.reshape(dataOO.CDd.values,(-1,1))
data12 = np.hstack([cdr12,cdd12])
eye12 = dataOO.drop(['CDr','CDd'],axis=1).eye.values

trainData12,testData12,trainType12,testType12 = train_test_split(data12, eye12, test_size=0.2 , random_state=0)

plt.scatter(data12[:,0],data12[:,1], c=eye12)
plt.xlabel('CDR')
plt.ylabel('CDD')


# RBF model for Other vs Non-Other
rbf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
i = 1
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

for train_index, test_index in skf.split(trainData12, trainType12):
    x_train, x_test = trainData12[train_index], trainData12[test_index]
    y_train, y_test = trainType12[train_index], trainType12[test_index]
    rbf.fit(x_train,y_train)
    filename = 'savemodel/OORBF_model'+str(i)+'.sav'
    joblib.dump(rbf,filename)

    viz = plot_roc_curve(rbf,x_test,y_test,name='RBF SVM model {}'.format(i), ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    i+=1

loaded_model = joblib.load('savemodel/OORBF_model5.sav')

conf_matrix = confusion_matrix(testType12,loaded_model.predict(testData12))
FP = conf_matrix[0][1]
FN = conf_matrix[1][0]
TP = conf_matrix[1][1]
TN = conf_matrix[0][0]
Accuracy = (TP+TN)/(TP+FP+FN+TN)
Specificity = TN/(TN+FP)
sensitivity = TP / (TP + FN)
Precision = TP / (TP + FP)
print('Accuracy:',Accuracy)
print('Specificity',Specificity)
print('sensitivity:',sensitivity)
print('Precision:',Precision)
print('F1 score',f1_score(testType12,loaded_model.predict(testData12), average='micro'),'\n')
print('FP',FP)
print('FN',FN)
print('TP',TP)
print('TN',TN,'\n')

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],title="Other vs Non-Other")
ax.legend(loc="lower right")
plt.show()


# prepare data for Glaucoma vs Normal vs Other
cdr = np.reshape(dataAll.CDr.values,(-1,1))
cdd = np.reshape(dataAll.CDd.values,(-1,1))
data = np.hstack([cdr,cdd])
eye = dataAll.drop(['CDr','CDd'],axis=1).eye.values

trainData,testData,trainType,testType = train_test_split(data, eye, test_size=0.2 , random_state=1)
skf = StratifiedKFold(n_splits=5)

plt.scatter(data[:,0],data[:,1], c=eye)
plt.xlabel('CDR')
plt.ylabel('y')
print(data)


# RBF model for Glaucoma vs Normal vs Other
rbf = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True))
i = 1

for train_index, test_index in skf.split(trainData01, trainType01):
    x_train, x_test = trainData[train_index], trainData[test_index]
    y_train, y_test = trainType[train_index], trainType[test_index]
    rbf.fit(x_train,y_train)
    filename = 'savemodel/RBF_model'+str(i)+'.sav'
    joblib.dump(rbf,filename)

    i+=1

loaded_model = joblib.load('savemodel/RBF_model2.sav')

conf_matrix = confusion_matrix(testType,loaded_model.predict(testData))
FP = conf_matrix[0][1]
FN = conf_matrix[1][0]
TP = conf_matrix[1][1]
TN = conf_matrix[0][0]
Accuracy = (TP+TN)/(TP+FP+FN+TN)
Specificity = TN/(TN+FP)
sensitivity = TP / (TP + FN)
Precision = TP / (TP + FP)
print('Accuracy:',Accuracy)
print('Specificity',Specificity)
print('sensitivity:',sensitivity)
print('Precision:',Precision)
print('F1 score',f1_score(testType,loaded_model.predict(testData), average='micro'),'\n')
print('FP',FP)
print('FN',FN)
print('TP',TP)
print('TN',TN,'\n')



