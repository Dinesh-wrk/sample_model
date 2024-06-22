from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from io import BytesIO
import pandas as pd
import joblib
import os
import uuid

app = FastAPI()

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

# Azure Blob Storage configuration
connection_string = "your_connection_string"
container_name = "your_container_name"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # Generate a unique file name
        file_name = str(uuid.uuid4()) + "_" + file.filename
        blob_client = container_client.get_blob_client(file_name)

        # Upload the file to Azure Blob Storage
        await blob_client.upload_blob(file.file.read(), overwrite=True)

        # Return the file path
        file_path = f"{container_name}/{file_name}"
        return {"file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(file_path: str):
    try:
        # Get the blob client for the file
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_path.file_path)

        # Download the file from Azure Blob Storage
        blob_data = blob_client.download_blob()
        file_data = blob_data.readall()
        df = pd.read_csv(BytesIO(file_data))
        
        # Check if the necessary columns are present
        if not {'age', 'income'}.issubset(df.columns):
            raise HTTPException(status_code=400, detail="CSV file must contain 'age' and 'income' columns")
        
        # Perform prediction
        predictions = model.predict(df[['age', 'income']])
        
        # Prepare the response
        df['predicted_loan_amount'] = predictions
        response = df.to_dict(orient='records')
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define the PredictionInput class and predict_single endpoint as before

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
