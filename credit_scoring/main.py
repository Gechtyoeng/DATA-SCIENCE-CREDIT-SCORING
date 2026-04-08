import gradio as gr
import joblib
import pandas as pd

# Load model directly
try:
    pipeline = joblib.load("Final_project/DATA-SCIENCE-CREDIT-SCORING/credit_scoring/outputs/models/random_forest.pkl")
except FileNotFoundError:
    raise RuntimeError("Model file not found. Please train and save the model first.")

# Expected features from the trained model
expected_features = list(pipeline.feature_names_in_)

def predict_default(person_age, person_income, person_emp_length, loan_amnt, loan_int_rate,
                    loan_percent_income, cb_person_cred_hist_length, loan_intent,
                    loan_grade, cb_person_default_on_file, land_title_type, nbc_status):

    # Base dataset stats for scaling (from data_feature.ipynb)
    min_income_orig = 4000
    max_income_orig = 6000000
    min_cam_inc = 2400
    max_cam_inc = 18000
    
    # Engineered features
    stability_score = person_emp_length + cb_person_cred_hist_length
    interest_burden = (loan_amnt * (loan_int_rate / 100)) / (person_income + 1e-9)
    person_income_kh = ((person_income - min_income_orig) / (max_income_orig - min_income_orig) * (max_cam_inc - min_cam_inc) + min_cam_inc)
    loan_to_income_ratio = loan_amnt / (person_income + 1e-9)

    # Override nbc_status based on loan_grade to automatically reflect dataset behavior,
    # though the user can also see their input for nbc_status fallback.
    # grade_map = {
    #     'A': 'Normal (Standard)',
    #     'B': 'Normal (Standard)',
    #     'C': 'Special Mention',
    #     'D': 'Substandard',
    #     'E': 'Doubtful',
    #     'F': 'Loss',
    #     'G': 'Loss'
    # }
    # nbc_status = grade_map.get(loan_grade, nbc_status)

    # Initialize all expected features to 0
    input_data = {feature: 0 for feature in expected_features}

    # Set numeric features
    if "person_age" in input_data: input_data["person_age"] = person_age
    if "person_income" in input_data: input_data["person_income"] = person_income
    if "person_emp_length" in input_data: input_data["person_emp_length"] = person_emp_length
    if "loan_amnt" in input_data: input_data["loan_amnt"] = loan_amnt
    if "loan_int_rate" in input_data: input_data["loan_int_rate"] = loan_int_rate
    if "loan_percent_income" in input_data: input_data["loan_percent_income"] = loan_percent_income
    if "cb_person_cred_hist_length" in input_data: input_data["cb_person_cred_hist_length"] = cb_person_cred_hist_length
    if "person_income_kh" in input_data: input_data["person_income_kh"] = person_income_kh
    if "loan_to_income_ratio" in input_data: input_data["loan_to_income_ratio"] = loan_to_income_ratio
    if "stability_score" in input_data: input_data["stability_score"] = stability_score
    if "interest_burden" in input_data: input_data["interest_burden"] = interest_burden

    # Set categorical dummy columns to 1 if they exist in the model
    intent_col = f"loan_intent_{loan_intent}"
    if intent_col in input_data: input_data[intent_col] = 1

    title_col = f"land_title_type_{land_title_type}"
    if title_col in input_data: input_data[title_col] = 1

    default_col = f"cb_person_default_on_file_{cb_person_default_on_file}"
    if default_col in input_data: input_data[default_col] = 1

    nbc_col = f"nbc_status_{nbc_status}"
    if nbc_col in input_data: input_data[nbc_col] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure correct column order
    input_df = input_df[expected_features]

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
        gr.Number(label="Income (USD/Base unit)"),
        gr.Number(label="Employment Length (years)"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Interest Rate (%)"),
        gr.Number(label="Loan Percent Income"),
        gr.Number(label="Credit History Length (years)"),
        gr.Dropdown(["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"], label="Loan Intent"),
        # gr.Dropdown(["A", "B", "C", "D", "E", "F", "G"], label="Loan Grade"),
        gr.Dropdown(["Y", "N"], label="Previous Default on File"),
        gr.Dropdown(["Hard Title", "Soft Title", "No Title"], label="Land Title Type"),
        gr.Dropdown(["Normal (Standard)", "Special Mention", "Substandard", "Doubtful", "Loss"], label="NBC Status (overridden by grade)")
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Number(label="Probability of Default")
    ],
    title="Alternative Credit Scoring Model (Random Forest)",
    description="Enter borrower details to predict loan default risk in Cambodian context."
)

if __name__ == "__main__":
    interface.launch()