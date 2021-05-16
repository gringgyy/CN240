import numpy as np # linear algebra
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, roc_curve, auc, plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import warnings
import os
import shutil

from PIL import ImageFile
warnings.simplefilter('error', Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# declare path & class label
datasetFolderName='dataset'
MODEL_FILENAME="model_dl.h5"
sourceFiles=[]
classLabels=['glaucoma', 'normal', 'other']

def transferBetweenFolders(source, dest, splitRate):   
    global sourceFiles
    sourceFiles=os.listdir(source)
    if(len(sourceFiles)!=0):
        transferFileNumbers=int(len(sourceFiles)*splitRate)
        transferIndex=random.sample(range(0, len(sourceFiles)), transferFileNumbers)
        for eachIndex in transferIndex:
            shutil.move(source+str(sourceFiles[eachIndex]), dest+str(sourceFiles[eachIndex]))
    else:
        print("No file moved. Source empty!")


def transferAllClassBetweenFolders(source, dest, splitRate):
    for label in classLabels:
        transferBetweenFolders(datasetFolderName+'/'+source+'/'+label+'/', 
                               datasetFolderName+'/'+dest+'/'+label+'/', 
                               splitRate)


# First, check if test folder is empty or not, if not transfer all existing files to train
transferAllClassBetweenFolders('test', 'train', 1.0)
# Now, split some part of train data into the test folders.
transferAllClassBetweenFolders('train', 'test', 0.20)


X=[]
Y=[]
def prepareNameWithLabels(folderName):
    sourceFiles=os.listdir(datasetFolderName+'/train/'+folderName+'/')
    for val in sourceFiles:
        X.append(val)
        if(folderName==classLabels[0]):
            Y.append(0)
        elif(folderName==classLabels[1]):
            Y.append(1)
        else:
            Y.append(2)


prepareNameWithLabels(classLabels[0])
prepareNameWithLabels(classLabels[1])
prepareNameWithLabels(classLabels[2])

X=np.asarray(X)
Y=np.asarray(Y)


#model
batch_size = 5
epoch=5
activationFunction='relu'
def Model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation=activationFunction, input_shape=(img_rows, img_cols, 3)))
    model.add(Conv2D(64, (3, 3), activation=activationFunction))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(32, (3, 3), padding='same', activation=activationFunction))
    model.add(Conv2D(32, (3, 3), activation=activationFunction))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(16, (3, 3), padding='same', activation=activationFunction))
    model.add(Conv2D(16, (3, 3), activation=activationFunction))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(64, activation=activationFunction)) # we can drop 
    model.add(Dropout(0.1))                  # this layers
    model.add(Dense(32, activation=activationFunction))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation=activationFunction))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax')) 
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def my_metrics(y_true, y_pred):
    accuracy=accuracy_score(y_true, y_pred)
    precision=precision_score(y_true, y_pred,average='weighted')
    f1Score=f1_score(y_true, y_pred, average='weighted')
    cm=confusion_matrix(y_true, y_pred)
    sensGlacoma = cm[0][0]/(cm[0][0]+cm[0][1]+cm[0][2])
    specGlaucoma = (cm[1][1]+cm[1][2]+cm[2][1]+cm[2][2])/(cm[1][1]+cm[1][2]+cm[2][1]+cm[2][2]+cm[1][0]+cm[2][0])
    sensNormal = cm[1][1]/(cm[1][0]+cm[1][1]+cm[1][2])
    specNormal = (cm[0][0]+cm[0][2]+cm[2][0]+cm[2][2])/(cm[0][1]+cm[2][1]+cm[0][0]+cm[0][2]+cm[2][0]+cm[2][2])
    sensOther = cm[2][2]/(cm[0][2]+cm[1][2]+cm[2][2])
    specOther = (cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])/(cm[0][2]+cm[1][2]+cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    
    print("Accuracy  : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("f1Score : {}".format(f1Score))
    print("sensGlacoma : {}".format(sensGlacoma))
    print("specGlaucoma : {}".format(specGlaucoma))
    print("sensNormal : {}".format(sensNormal))
    print("specNormal : {}".format(specNormal))
    print("sensOther : {}".format(sensOther))
    print("specOther : {}".format(specOther))
    
    print(cm)
    return accuracy, precision, f1Score, sensGlacoma, specGlaucoma, sensNormal, specNormal, sensOther, specOther

# declare img_size & path & model
img_rows, img_cols =  224, 224
train_path=datasetFolderName+'/train/'
validation_path=datasetFolderName+'/validation/'
test_path=datasetFolderName+'/test/'
model=Model()


skf = StratifiedKFold(n_splits=5, shuffle=True)
skf.get_n_splits(X, Y)
foldNumber=0
for train_index, val_index in skf.split(X, Y):
    transferAllClassBetweenFolders('validation', 'train', 1.0)
    foldNumber+=1
    print("Results for fold",foldNumber)
    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]
    for eachIndex in range(len(X_val)):
        classLabel=''
        if(Y_val[eachIndex]==0):
            classLabel=classLabels[0]
        elif(Y_val[eachIndex]==1):
            classLabel=classLabels[1]
        else:
            classLabel=classLabels[2]

        #Then, copy the validation images to the validation folder
        shutil.move(datasetFolderName+'/train/'+classLabel+'/'+X_val[eachIndex], 
                    datasetFolderName+'/validation/'+classLabel+'/'+X_val[eachIndex])
    train_datagen = ImageDataGenerator(
                rescale=1./255,
        		zoom_range=0.20,
            	fill_mode="nearest"
                )
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    #Start ImageClassification Model
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

    validation_generator = validation_datagen.flow_from_directory(
            validation_path,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode=None,  # only data, no labels
            shuffle=False)   
    
    # fit model
    history=model.fit(train_generator, 
                        epochs=epoch)
    
    predictions = model.predict(validation_generator, verbose=1)
    yPredictions = np.argmax(predictions, axis=1)
    true_classes = validation_generator.classes
    
    # evaluate validation performance
    print("***Performance on Validation data***")    
    valAcc, valPrec, valFScore, valsensGlaucoma, valspecGlaucoma, valsensNormal, valspecNormal, valsensOther, valspecOther = my_metrics(true_classes, yPredictions)


print("==============TEST RESULTS============")
test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False) 
predictions = model.predict(test_generator, verbose=1)
yPredictions = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

testAcc,testPrec, testFScore, valsensGlaucoma, valspecGlaucoma, valsensNormal, valspecNormal, valsensOther, valspecOther = my_metrics(true_classes, yPredictions)
model.save(MODEL_FILENAME)




