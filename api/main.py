from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Body
import uvicorn
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from fastapi.middleware.cors import CORSMiddleware

MODEL  = tf.keras.models.load_model("./models/1")

CLASS_NAMES = ["Early Blight", "Late Blight","Healthy"]

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

endpoint = "http://localhost:8601/v1/models/pot_model:predict"


@app.post("/predict")

async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }  
    


if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)
