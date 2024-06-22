import pandas as pd
from faker import Faker
import random

fake = Faker()

# Function to generate realistic data
def generate_loan_data(num_records):
    data = []
    for _ in range(num_records):
        age = random.randint(18, 70)
        income = round(random.uniform(20000, 150000), 2)
        loan_amount = round(random.uniform(5000, 50000), 2)
        data.append([age, income, loan_amount])
    return data

# Generate 1,000,000 records of loan data
num_records = 1000000
loan_data = generate_loan_data(num_records)

# Create a DataFrame
df = pd.DataFrame(loan_data, columns=['Age', 'Income', 'Loan_Amount'])

# Save to an Excel file
file_path = "./loan_data.xlsx"
df.to_excel(file_path, index=False)

print(f"Data saved to {file_path}")
