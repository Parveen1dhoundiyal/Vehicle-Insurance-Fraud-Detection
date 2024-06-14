from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Convert numeric fields to int and handle possible missing values
        user_input = {
            'months_as_customer': int(request.form.get('months_as_customer', 0)),
            'policy_deductable': int(request.form.get('policy_deductable', 0)),
            'umbrella_limit': int(request.form.get('umbrella_limit', 0)),
            'capital_gains': int(request.form.get('capital_gains', 0)),
            'capital_loss': int(request.form.get('capital_loss', 0)),
            'incident_hour_of_the_day': int(request.form.get('incident_hour_of_the_day', 0)),
            'number_of_vehicles_involved': int(request.form.get('number_of_vehicles_involved', 0)),
            'bodily_injuries': int(request.form.get('bodily_injuries', 0)),
            'witnesses': int(request.form.get('witnesses', 0)),
            'injury_claim': int(request.form.get('injury_claim', 0)),
            'property_claim': int(request.form.get('property_claim', 0)),
            'vehicle_claim': int(request.form.get('vehicle_claim', 0)),
            'policy_csl': request.form.get('policy_csl'),
            'insured_sex': request.form.get('insured_sex'),
            'insured_education_level': request.form.get('insured_education_level'),
            'insured_occupation': request.form.get('insured_occupation'),
            'insured_relationship': request.form.get('insured_relationship'),
            'incident_type': request.form.get('incident_type'),
            'collision_type': request.form.get('collision_type'),
            'incident_severity': request.form.get('incident_severity'),
            'authorities_contacted': request.form.get('authorities_contacted'),
            'property_damage': request.form.get('property_damage'),
            'police_report_available': request.form.get('police_report_available')
        }
        print("User Input Dictionary:")
        for key, value in user_input.items():
            print(f"{key}: {value}")

        # Handle missing categorical values by replacing them with np.nan
        for key, value in user_input.items():
            if value in [None,'NA']:
                user_input[key] = np.nan
        
        # Example: Print them or process them further
        print("User Input Dictionary:")
        for key, value in user_input.items():
            print(f"{key}: {value}")
        
        # Construct num_df DataFrame from form input
        num_df = pd.DataFrame({
            'months_as_customer': [user_input['months_as_customer']],
            'policy_deductable': [user_input['policy_deductable']],
            'umbrella_limit': [user_input['umbrella_limit']],
            'capital-gains': [user_input['capital_gains']],
            'capital-loss': [user_input['capital_loss']],
            'incident_hour_of_the_day': [user_input['incident_hour_of_the_day']],
            'number_of_vehicles_involved': [user_input['number_of_vehicles_involved']],
            'bodily_injuries': [user_input['bodily_injuries']],
            'witnesses': [user_input['witnesses']],
            'injury_claim': [user_input['injury_claim']],
            'property_claim': [user_input['property_claim']],
            'vehicle_claim': [user_input['vehicle_claim']]
        })

        # Load the saved StandardScaler
        with open('Models/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)

        # Scale the numerical values
        scaled_data = scaler.transform(num_df)

        # Optionally, you can print or log the scaled_data
        print("Scaled Data:")
        print(scaled_data)

        # Load the saved LabelEncoders
        categorical_cols = [
            'policy_csl', 'insured_sex', 'insured_education_level',
            'insured_occupation', 'insured_relationship', 'incident_type',
            'collision_type', 'incident_severity', 'authorities_contacted',
            'property_damage', 'police_report_available'
        ]

        label_encoders = {}
        for col in categorical_cols:
            with open(f'Models/{col}_encoder.pkl', 'rb') as file:
                label_encoders[col] = pickle.load(file)

        # Encode the categorical values in the sample input
        encoded_input = user_input.copy()
        for col in categorical_cols:
            encoded_input[col] = label_encoders[col].transform([user_input[col]])[0]

        # Define the correct column order (excluding policy_annual_premium)
        column_order = [
            'months_as_customer', 'policy_deductable', 'umbrella_limit', 
            'capital-gains', 'capital-loss', 'incident_hour_of_the_day', 
            'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 
            'injury_claim', 'property_claim', 'vehicle_claim', 
            'policy_csl', 'insured_sex', 'insured_education_level', 
            'insured_occupation', 'insured_relationship', 'incident_type', 
            'collision_type', 'incident_severity', 'authorities_contacted', 
            'property_damage', 'police_report_available'
        ]

        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([encoded_input], columns=column_order)

        # Display the transformed input
        print("Transformed Input:")
        print(input_df)

        # Load the saved model
        with open('Models/model.pkl', 'rb') as file:
            model = pickle.load(file)

        # Since the numerical features are already scaled, concatenate them with encoded features
        input_for_prediction = pd.concat([pd.DataFrame(scaled_data, columns=num_df.columns), input_df[categorical_cols]], axis=1)

        # Make a prediction using the loaded model
        prediction = model.predict(input_for_prediction)

        # Print the prediction result
        print("\nPrediction result:", prediction)

        # Determine redirection based on prediction result
        if prediction[0] == 'Y':
            return redirect('/fraud')
        else:
            return redirect('/not-fraud')

    else:
        return render_template('index.html')

@app.route('/fraud')
def fraud():
    return render_template('fraud.html')

@app.route('/not-fraud')
def not_fraud():
    return render_template('not-fraud.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
