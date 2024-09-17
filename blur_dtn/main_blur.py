# main.py
from fastapi import FastAPI
from blurapi import router
import uvicorn

app = FastAPI()

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1111)