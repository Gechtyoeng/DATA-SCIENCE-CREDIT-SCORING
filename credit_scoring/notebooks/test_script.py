import pandas as pd
df = pd.read_csv("../data/processed/cleaned_credit_risk_dataset.csv")
print("after read_csv:", len(df.columns))
if 'loan_grade' in df.columns:
    print("loan_grade exists initially")

house_map = {
    'RENT': 'Soft Title',
    'MORTGAGE': 'Hard Title',
    'OWN': 'Hard Title',
    'OTHER': 'No Title'
}
df['land_title_type'] = df['person_home_ownership'].map(house_map)

min_cam_inc = 2400
max_cam_inc = 18000
df['person_income_kh'] = ((df['person_income'] - df['person_income'].min()) / 
                         (df['person_income'].max() - df['person_income'].min()) * (max_cam_inc - min_cam_inc) + min_cam_inc)

df['loan_to_income_ratio'] = df['loan_amnt'] / df['person_income']
df['stability_score'] = df['person_emp_length'] + df['cb_person_cred_hist_length']
df['interest_burden'] = (df['loan_amnt'] * (df['loan_int_rate']/100)) / df['person_income']

df = pd.get_dummies(df, columns=['loan_intent', 'land_title_type', 'cb_person_default_on_file'], drop_first=True)
print("after get_dummies:", len(df.columns))
if 'loan_grade' in df.columns:
    print("loan_grade exists after get_dummies")

df = df.drop(['person_home_ownership'], axis=1)
print("after drop:", len(df.columns))
if 'loan_grade' in df.columns:
    print("loan_grade exists after drop")
