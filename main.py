from fastapi import FastAPI, UploadFile, File
import random
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
from pydantic import BaseModel
from starlette.responses import HTMLResponse
from typing import List

origins = ["*"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi import UploadFile, File


@app.get("/")
async def main():
    return {"Scores": [random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100]}


@app.post("/upload")
async def upload_image(user_id: int, file: UploadFile = File(...)):
    return {"Scores": [random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100]}
