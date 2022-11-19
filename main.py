from fastapi import FastAPI, File, UploadFile
import numpy as np
import random
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import cv2
import tensorflow as tf
from painting import Painting


middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
]

app = FastAPI(middleware=middleware)


@app.get("/")
async def main():
    return {"Connected!"}


@app.post("/process")
async def root(file: UploadFile = File(...)):
    contents = await file.read()
    properties_list = [0] * 6
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    painting = Painting(img)
    painting.preprocessing()
    painting.calculateProperties()
    return painting.properties_list

@app.post("/predictNN")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    prediction = predNN(img)
    prediction = np.array(prediction)
    return str(prediction)

# @app.post("/predictIP")
# async def predictIP(file: UploadFile = File(...)):
#     contents = await file.read()
#     nparr = np.fromstring(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     painting = Painting(img)
#     painting.preprocessing()
#     painting.calculateProperties()
#     prediction = predIP(painting.properties_list)
#     return str(prediction)


modelNN = tf.keras.models.load_model('model_03_a_0.2411_l_1.9252_va_0.2131_vl_1.9761.h5')
def predNN(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (120, 120),interpolation=cv2.INTER_CUBIC)
    img = np.float32(img)
    img_array = tf.expand_dims(img, 0)
    predictions = modelNN.predict(img_array)
    return predictions[0]

# modelIP = tf.keras.models.load_model('MODEL NAME HERE')
# def predIP(ipScores):
#     predictions = modelIP.predict(ipScores)
#     return predictions[0]