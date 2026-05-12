from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
import pandas as pd
from pydantic import BaseModel
import logging

# Initialize FastAPI and Logging
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Load the trained model
try:
    model = tf.keras.models.load_model('models/sales_model.keras')
except Exception as e:
    logging.error(f"Could not load model: {e}")

class PredictionInput(BaseModel):
    last_7_days_sales: list[float]

@app.get("/")
def home():
    return {"message": "Supermarket Sales Forecasting API is Live!"}

@app.post("/predict")
def predict_sales(input_data: PredictionInput):
    try:
        # 1. Convert input to numpy array
        data = np.array(input_data.last_7_days_sales).reshape(1, 7, 1)
        
        # 2. Make prediction
        prediction = model.predict(data)
        
        # 3. Return result
        return {"predicted_next_day_sales": float(prediction[0][0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
