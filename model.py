import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load sample data
data = pd.read_csv('loan_data.csv')

# Prepare the features and target
X = data[['age', 'income']]
y = data['loan_amount']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model to a file
joblib.dump(model, 'linear_regression_model.pkl')
