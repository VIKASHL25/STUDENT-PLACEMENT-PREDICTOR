from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('column_transformer.pkl', 'rb') as f:
    ct = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # taking input from form 
    input_data = pd.DataFrame([{
        'IQ': float(request.form['IQ']),
        'Prev_Sem_Result': float(request.form['Prev_Sem_Result']),
        'CGPA': float(request.form['CGPA']),
        'Academic_Performance': float(request.form['Academic_Performance']),
        'Internship_Experience': request.form['Internship_Experience'],
        'Extra_Curricular_Score': float(request.form['Extra_Curricular_Score']),
        'Communication_Skills': float(request.form['Communication_Skills']),
        'Projects_Completed': int(request.form['Projects_Completed'])
    }])

    
    input_transformed = ct.transform(input_data)

    
    prediction = model.predict(input_transformed)[0]

    # prediction
    result = "Yes! You are eligible for placement." if prediction == 1 else " No, you are not eligible for placement."

  
    return f"""
    <div style='
        font-family: Arial, sans-serif;
        background: linear-gradient(to right, #89f7fe, #66a6ff);
        padding: 100px 0;
        text-align: center;
        min-height: 100vh;
    '>
        <div style='
            background: white;
            display: inline-block;
            padding: 30px 50px;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        '>
            <h2 style='font-size: 28px; color: #333;'>{result}</h2>
            <a href='/' style='
                display: inline-block;
                margin-top: 20px;
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 8px;
                font-weight: bold;
            '>🔙 Back to Form</a>
        </div>
    </div>
    """

if __name__ == '__main__':
    app.run(debug=True)
