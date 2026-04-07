import json
import uuid

def generate_id():
    return str(uuid.uuid4())[:8]

with open('data_feature.ipynb', 'r') as f:
    nb = json.load(f)

markdown_cell = {
 "cell_type": "markdown",
 "id": generate_id(),
 "metadata": {},
 "source": [
  "Mapping Global Loan Grade to NBC Status (Khmer)\n",
  "- A / B -> Normal (Standard)\n",
  "- C -> Special Mention\n",
  "- D -> Substandard\n",
  "- E -> Doubtful\n",
  "- F / G -> Loss"
 ]
}

code_cell = {
 "cell_type": "code",
 "execution_count": None,
 "id": generate_id(),
 "metadata": {},
 "outputs": [],
 "source": [
  "grade_map = {\n",
  "    'A': 'Normal (Standard)',\n",
  "    'B': 'Normal (Standard)',\n",
  "    'C': 'Special Mention',\n",
  "    'D': 'Substandard',\n",
  "    'E': 'Doubtful',\n",
  "    'F': 'Loss',\n",
  "    'G': 'Loss'\n",
  "}\n",
  "df['nbc_status'] = df['loan_grade'].map(grade_map)"
 ]
}

# Insert cells after cell index 7
nb['cells'].insert(8, markdown_cell)
nb['cells'].insert(9, code_cell)

# Now find the pd.get_dummies cell and df.drop cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "pd.get_dummies" in source:
            new_source = source.replace("columns=['loan_intent', 'land_title_type', 'cb_person_default_on_file']", 
                                        "columns=['loan_intent', 'land_title_type', 'cb_person_default_on_file', 'nbc_status']")
            cell['source'] = [new_source]
        elif "df.drop(['person_home_ownership']" in source:
            new_source = source.replace("['person_home_ownership']", "['person_home_ownership', 'loan_grade']")
            cell['source'] = [new_source]

with open('data_feature.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
