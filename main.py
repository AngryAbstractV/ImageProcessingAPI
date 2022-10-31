from fastapi import FastAPI, File, UploadFile
import random
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import cv2
import numpy as np


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
    return {"Scores":
            [random.randint(10, 100) / 100, random.randint(10, 100) / 100,
             random.randint(10, 100) / 100, random.randint(10, 100) / 100,
             random.randint(10, 100) / 100, random.randint(10, 100) / 100,
             random.randint(10, 100) / 100, random.randint(10, 100) / 100]}


@app.post("/upload")
async def root(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # img is in BRG format
    return str(img[0][0])
