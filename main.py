from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def hello():
    return {"message": "My Name is Daniel"}
