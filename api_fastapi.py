from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

# Directory to store uploaded files
UPLOAD_DIRECTORY = "uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Directory to store result files
RESULT_DIRECTORY = "result_files"
os.makedirs(RESULT_DIRECTORY, exist_ok=True)

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FilePath(BaseModel):
    file_path: str

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to the specified directory
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Read the CSV file to include in the response
        df = pd.read_csv(file_path)
        csv_content = df.to_csv(index=False)
        
        # Return the file path and CSV content
        return {"file_path": file_path, "csv_content": csv_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(file_path: FilePath):
    try:
        # Read the CSV file from the provided file path
        df = pd.read_csv(file_path.file_path)
        
        # Check if the necessary columns are present
        if not {'age', 'income'}.issubset(df.columns):
            raise HTTPException(status_code=400, detail="CSV file must contain 'age' and 'income' columns")
        
        # Perform prediction
        predictions = model.predict(df[['age', 'income']])
        
        # Prepare the response
        df['predicted_loan_amount'] = predictions
        response = df.to_dict(orient='records')

        # Save the response as a CSV file
        result_file_name = f"result_{os.path.basename(file_path.file_path)}"
        result_file_path = os.path.join(RESULT_DIRECTORY, result_file_name)
        df.to_csv(result_file_path, index=False)

        # Return the path of the result file including the RESULT_DIRECTORY prefix
        return {"result_file_path": result_file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PredictionInput(BaseModel):
    age: float
    income: float

@app.post("/predict_single")
async def predict_single(input_data: PredictionInput):
    try:
        # Prepare the data for prediction
        data = pd.DataFrame([[input_data.age, input_data.income]], columns=['age', 'income'])
        
        # Perform prediction
        prediction = model.predict(data)

        # Round the prediction to the nearest hundred
        rounded_prediction = round(prediction[0], -1)

        # Prepare the response.
        response = {
            "predicted_loan_amount": int(rounded_prediction)
        }

        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{file_name}")
async def download_result(file_name: str):
    try:
        result_file_path = os.path.join(RESULT_DIRECTORY, file_name)
        return FileResponse(result_file_path, media_type='text/csv', filename=file_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail="File not found")
    

@app.get("/health_check")
def serverhealth():
    return JSONResponse(content="OK", status_code=200)


@app.get("/testme")
def serverhealth():
    return JSONResponse(content="Testme", status_code=200)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
