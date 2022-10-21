from fastapi import FastAPI
import random


app = FastAPI()


@app.get("/")
def hello():

    return {"Scores": [random.randint(10, 100) / 100,random.randint(10, 100) / 100,
                        random.randint(10, 100) / 100,random.randint(10, 100) / 100,
                        random.randint(10, 100) / 100,random.randint(10, 100) / 100,
                        random.randint(10, 100) / 100,random.randint(10, 100) / 100]}
