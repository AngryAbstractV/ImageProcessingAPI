from fastapi import FastAPI, File, UploadFile
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
import cv2
import random
import numpy as np
import math
from scipy.signal import argrelextrema

app = FastAPI()

origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def main():
    return {"Scores": [random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100]}