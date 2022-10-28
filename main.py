from fastapi import FastAPI, File, UploadFile
import random
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import cv2
import io


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
    return {"Scores": [random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100]}



@app.post("/upload")
async def root(file: UploadFile = File(...)):
        # request_object_content = await file.read()
        # img = cv2.imread(io.BytesIO(request_object_content))
        # firstPixel = img[0][0]
        return 'Poggers'
