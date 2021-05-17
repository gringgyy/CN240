from fastapi import FastAPI, UploadFile, Form, File
import numpy as np
import cv2 as cv
import joblib
import tensorflow as tf
from tensorflow import keras

app = FastAPI()

def deeplearn_model(img):
    #reshape
    dlimg = cv.resize(img,(224,224))
    dlimg = dlimg.reshape(1,224,224,3)

    #import model
    model = tf.keras.models.load_model("deeplearn/deeplearningVGG16.h5")

    #predict
    predict = model.predict(dlimg)
    predictclass = np.argmax(predict)

    return predict,predictclass

@app.get('/')
def root():
    return {'msg' : 'Hello World'}

@app.post('/api/fundus')
async def upload_image(nonce: str=Form(None, title="Query Text"), image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    conf,classes = deeplearn_model(img)

    class_names = ["glaucoma", "normal", "other"]
    class_out = class_names[classes]

    class_conf = conf[0][classes]

    return {
        "nonce": nonce,
        "classification": class_out,
        "confidence_score": np.float(class_conf),
    }