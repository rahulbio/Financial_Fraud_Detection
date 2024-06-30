from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.joblib')

# Expected columns based on the training data
expected_columns = [
    'Amount', 'Initial_Balance_Sender', 'Initial_Balance_Receiver', 
    'Final_Balance_Sender', 'Final_Balance_Receiver',
    'Transaction_Type_Cash-In', 'Transaction_Type_Cash-Out', 
    'Transaction_Type_Debit', 'Transaction_Type_Payment', 'Transaction_Type_Transfer'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    transaction_type = request.form['transaction_type']
    transaction_amount = float(request.form['transaction_amount'])
    initial_balance_sender = float(request.form['initial_balance_sender'])
    initial_balance_receiver = float(request.form['initial_balance_receiver'])
    final_balance_sender = float(request.form['final_balance_sender'])
    final_balance_receiver = float(request.form['final_balance_receiver'])

    # Create a DataFrame from the form data
    input_data = pd.DataFrame({
        'Transaction_Type': [transaction_type],
        'Amount': [transaction_amount],
        'Initial_Balance_Sender': [initial_balance_sender],
        'Initial_Balance_Receiver': [initial_balance_receiver],
        'Final_Balance_Sender': [final_balance_sender],
        'Final_Balance_Receiver': [final_balance_receiver]
    })

    # One-hot encode the Transaction_Type feature
    input_data = pd.get_dummies(input_data, columns=['Transaction_Type'])

    # Ensure the input data has the same columns as the training data
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match the training data
    input_data = input_data[expected_columns]

    # Make prediction
    prediction = model.predict(input_data)

    # Convert prediction to readable text and set is_fraud flag
    if prediction[0] == 1:
        prediction_text = 'Transaction is: Fraud'
        is_fraud = True
    else:
        prediction_text = 'Transaction is: Not Fraud'
        is_fraud = False

    return render_template('index.html', prediction_text=prediction_text, is_fraud=is_fraud)

if __name__ == '__main__':
    app.run(debug=True)