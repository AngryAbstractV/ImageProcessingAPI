from fastapi import FastAPI, UploadFile, File
import random
import cv2
import numpy as np
from pydantic import BaseModel
from starlette.responses import HTMLResponse
from typing import List

app = FastAPI()

from fastapi import UploadFile, File


@app.post("/upload_image/{user_id}")
async def upload_image(user_id: int, file: UploadFile = File(...)):
    return {"Scores": [random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100]}
