import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix

path_test = "dataset/test"
model = load_model('deeplearningVGG16.h5')


# import test data
img_test = ImageDataGenerator().flow_from_directory(
    path_test,
    target_size = (224,224),
    shuffle = False
)

result = img_test.classes

predict = model.predict(img_test)

print(result)
print(predict)


#  plot roc graph
fprG , tprG , thres = metrics.roc_curve(result,predict[:,0],pos_label=0)
fprN , tprN , thres = metrics.roc_curve(result,predict[:,1],pos_label=1)
fprO , tprO , thres = metrics.roc_curve(result,predict[:,2],pos_label=2)

roc_aucG = auc(fprG, tprG)
roc_aucN = auc(fprN, tprN)
roc_aucO = auc(fprO, tprO)

lw = 2
plt.plot(fprG,tprG,lw=2,label='ROC curve Glaucoma (area = %.2f)'%roc_aucG)
plt.plot(fprN,tprN,lw=lw,label='ROC curve Normal (area = %.2f)'%roc_aucN)
plt.plot(fprO,tprO,lw=2,label='ROC curve Other (area = %.2f)'%roc_aucO)
plt.plot([0,1],[0,1],color ='navy',lw=lw,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()


# show accuracy/f1-score/callback/
predictResult = predict.argmax(axis = 1)

print(metrics.classification_report(result,predictResult,digits=5))


# confusion matrix
cf_matrix = confusion_matrix(result, predictResult)
print(cf_matrix)


# seaborn confusion matrix
conf_mat = confusion_matrix(result, predictResult)
print(np.sum(cf_matrix))
conf_mat_normalized = np.array([conf_mat[0] / (np.sum(cf_matrix[0])),conf_mat[1] / (np.sum(cf_matrix[1])),conf_mat[2] / (np.sum(cf_matrix[2]))])

print(conf_mat_normalized)
sns.heatmap(conf_mat_normalized,annot=True,fmt='.2%',cmap='Blues')
plt.ylabel('True label')
plt.xlabel('Predicted label')


