import gradio as gr
import joblib
import pandas as pd

# Load pipeline
pipeline = joblib.load("outputs/models/random_forest_pipeline.pkl")

# Get expected feature names
expected_features = pipeline.named_steps["preprocess"].get_feature_names_out()

def predict_default(person_age, person_income, person_emp_length, loan_amnt, loan_int_rate,
                    loan_percent_income, cb_person_cred_hist_length, loan_intent,
                    loan_grade, cb_person_default_on_file, land_title_type, nbc_status):

    # Engineered features
    stability_score = person_emp_length + cb_person_cred_hist_length
    interest_burden = (loan_amnt * (loan_int_rate/100)) / person_income

    # Initialize all expected features to 0
    input_data = {feature: 0 for feature in expected_features}

    # Set numeric features
    input_data["person_age"] = person_age
    input_data["person_income"] = person_income
    input_data["person_emp_length"] = person_emp_length
    input_data["loan_amnt"] = loan_amnt
    input_data["loan_int_rate"] = loan_int_rate
    input_data["loan_percent_income"] = loan_percent_income
    input_data["cb_person_cred_hist_length"] = cb_person_cred_hist_length
    input_data["stability_score"] = stability_score
    input_data["interest_burden"] = interest_burden

    # Set categorical dummy columns to 1 based on user input
    # Adjust names to match your dataset’s dummy column naming
    input_data[f"loan_intent_{loan_intent}"] = 1
    input_data[f"land_title_type_{land_title_type}"] = 1
    input_data[f"cb_person_default_on_file_{cb_person_default_on_file}"] = 1
    input_data[f"nbc_status_{nbc_status}"] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    result = "Default Risk" if prediction == 1 else "Safe Loan"
    return result, round(probability, 2)

# Gradio interface
interface = gr.Interface(
    fn=predict_default,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Income (KHR)"),
        gr.Number(label="Employment Length (years)"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Interest Rate (%)"),
        gr.Number(label="Loan Percent Income"),
        gr.Number(label="Credit History Length (years)"),
        gr.Dropdown(["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"], label="Loan Intent"),
        gr.Dropdown(["A", "B", "C", "D", "E", "F", "G"], label="Loan Grade"),
        gr.Dropdown(["Y", "N"], label="Previous Default on File"),
        gr.Dropdown(["Hard Title", "Soft Title", "No Title"], label="Land Title Type"),
        gr.Dropdown(["Normal (Standard)", "Special Mention", "Substandard", "Doubtful", "Loss"], label="NBC Status")
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Number(label="Probability of Default")
    ],
    title="Alternative Credit Scoring Model (Random Forest)",
    description="Enter borrower details to predict loan default risk in Cambodian context."
)

interface.launch()