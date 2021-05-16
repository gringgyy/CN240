import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, roc_curve, auc, plot_roc_curve

# declare path
path_train = "dataset/train"

# import train data
img_train = tf.keras.preprocessing.image_dataset_from_directory(
    path_train,
    validation_split=0.2,
    subset = "training",
    seed = 125,
    image_size = (224,224),
    batch_size = 32
)

# import validation data
img_validation = tf.keras.preprocessing.image_dataset_from_directory(
    path_train,
    validation_split=0.2,
    subset = "validation",
    seed = 125,
    image_size = (224,224),
    batch_size = 32
)


# load model VGG16
VGG16_MODEL=tf.keras.applications.VGG16(input_shape=(224,224,3),
                                               include_top=False,
                                               weights='imagenet')
VGG16_MODEL.trainable=False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(3,activation='softmax')


#train model
model = tf.keras.Sequential([
  VGG16_MODEL,
  global_average_layer,
  prediction_layer
])

model.compile(optimizer="adam", 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

model.summary()

history = model.fit(img_train,
                    epochs=18, 
                    steps_per_epoch=30,
                    validation_steps=2,
                    validation_data=img_validation)

#save model
model.save("deeplearningVGG16.h5")


#import Test img
path_test = "dataset/test"

img_test = tf.keras.preprocessing.image_dataset_from_directory(
    path_test,
    image_size = (224,224)
)


label = ["glaucoma","normal","other"]

loadmodel = tf.keras.models.load_model("deeplearningVGG16.h5")

predict = loadmodel.predict(img_test)
prediction = np.argmax(predict)


# use this code for test number of image predict in each class
# i = 0
# glau = 0
# norm = 0
# oth = 0
# for i in predict:
#     index = np.argmax(i)
#     if index == 0:
#         glau+=1
#     elif index == 1:
#         norm+=1
#     elif index==2 :
#         oth+=1
#     
# print(glau)
# print(norm)
# print(oth)


# use this code to change file type to jpeg if decoded error

#import imghdr
#import cv2
#import os
#import glob
#
#
#for file in glob.glob('dataset/test/other/*.jpg'):
#    image = cv2.imread(file)
#    file_type = imghdr.what(file)
#    if file_type != 'jpeg':
#        print(file + " - invalid - " + str(file_type))
#        cv2.imwrite(file, image)
#    print("finish jpg",image)



