from fastapi import FastAPI, UploadFile, Form, File
import numpy as np
import cv2 as cv
import joblib
from machine.MachineLearning import extract

app = FastAPI()

def machine_model(img):
    #extract
    extract_feature = extract(img)
 
    #import model
    model = joblib.load("machine/MachineModel.sav")

    #predict
    result = np.reshape(extract_feature,(1,-1))
    predict = model.predict_proba(result)
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

    conf,classes = machine_model(img)

    class_names = ["glaucoma", "normal", "other"]
    class_out = class_names[classes]

    class_conf = conf[0][classes]

    return {
        "nonce": nonce,
        "classification": class_out,
        "confidence_score": np.float(class_conf),
    }
