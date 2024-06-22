import requests

url = "http://localhost:8000/predict"
file_path = "test_data.csv"

with open(file_path, "rb") as f:
    response = requests.post(url, files={"file": f})

print(response.json())
