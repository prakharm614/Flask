# ------------------------------------------
# 📦 Import Libraries
# ------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
from flask import Flask, request, render_template

# ------------------------------------------
# 📂 Load Data
# ------------------------------------------
df = pd.read_csv('patient_data.csv')

if 'C' in df.columns:
    df.rename(columns={'C': 'Gender'}, inplace=True)

# Fix spelling issues in 'Stages'
df['Stages'] = df['Stages'].replace({
    'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS',
    'HYPERTENSION (Stage-2).': 'HYPERTENSION (Stage-2)'
})

# ------------------------------------------
# 🛠️ Encode Categorical Columns
# ------------------------------------------
if 'Age' in df.columns:
    age_order = ['15-25', '25-35', '35-45', '45-55', '55-65', '65+']
    age_map = {val: idx for idx, val in enumerate(age_order)}
    df['Age'] = df['Age'].map(age_map)
    print("✅ 'Age' column converted to numeric.")

label_encoder = LabelEncoder()
columns = ['Gender', 'Severity', 'History', 'Patient', 'TakeMedication',
           'Breathshortness', 'VisualChanges', 'NoseBleeding', 'ControlledDiet', 'Stages']

for col in columns:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])
    else:
        print(f"⚠️ Column '{col}' not found in dataset.")

# ------------------------------------------
# 📦 Model Training
# ------------------------------------------
X = df.drop('Stages', axis=1)
Y = df['Stages']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=30)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)
acc_lr = accuracy_score(y_test, lr.predict(x_test))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
acc_dt = accuracy_score(y_test, dt.predict(x_test))

# Random Forest
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
acc_rf = accuracy_score(y_test, rf.predict(x_test))

# Gaussian NB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
acc_nb = accuracy_score(y_test, gnb.predict(x_test))

# Multinomial NB
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
acc_mnb = accuracy_score(y_test, mnb.predict(x_test))

# Accuracy Table
model_scores = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Decision Tree Classifier',
        'Random Forest Classifier',
        'Gaussian Naive Bayes',
        'Multinomial Naive Bayes'
    ],
    'Score': [
        round(acc_lr, 6),
        round(acc_dt, 6),
        round(acc_rf, 6),
        round(acc_nb, 6),
        round(acc_mnb, 6)
    ]
})

print("\nModel Accuracy Table:")
print(model_scores)

# ------------------------------------------
# 🔍 Grid Search for Best Random Forest
# ------------------------------------------
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)
print("\n🛠️ Best Parameters from GridSearch:")
print(grid_search.best_params_)
print("📈 Best CV Accuracy:", grid_search.best_score_)

# Final Best Model
best_rf = grid_search.best_estimator_
print("\n✅ Tuned RF Test Accuracy:", accuracy_score(y_test, best_rf.predict(x_test)))

# ------------------------------------------
# 💾 Save Best Model
# ------------------------------------------
pickle.dump(best_rf, open("model.pkl", "wb"))

# ------------------------------------------
# 🚀 FLASK APP DEPLOYMENT
# ------------------------------------------
app = Flask(__name__, static_url_path='/Flask/static')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    Gender = float(request.form["Gender"])
    Age = float(request.form["Age"])
    Patient = float(request.form['Patient'])
    Severity = float(request.form['Severity'])
    BreathShortness = float(request.form['BreathShortness'])
    VisualChange = float(request.form['VisualChanges'])
    NoseBleeding = float(request.form['NoseBleeding'])
    Whendiagnoused = float(request.form['Whendiagnoused'])
    Systolic = float(request.form['Systolic'])
    Diastolic = float(request.form['Diastolic'])
    ControlledDiet = float(request.form['ControlledDiet'])

    # Prepare features
    features_values = np.array([[Gender, Age, Patient, Severity, BreathShortness, VisualChange,
                                 NoseBleeding, Whendiagnoused, Systolic, Diastolic, ControlledDiet]])

    df_input = pd.DataFrame(features_values, columns=['Gender', 'Age', 'Patient', 'Severity',
                            'BreathShortness', 'VisualChanges', 'NoseBleeding',
                            'Whendiagnoused', 'Systolic', 'Diastolic', 'ControlledDiet'])

    prediction = model.predict(df_input)
    stage = prediction[0]

    if stage == 0:
        result = "NORMAL"
    elif stage == 1:
        result = "HYPERTENSION (Stage-1)"
    elif stage == 2:
        result = "HYPERTENSION (Stage-2)"
    else:
        result = "HYPERTENSIVE CRISIS"

    text = "Your Blood Pressure stage is: "
    return render_template("predict.html", prediction_text=text + result)

if __name__ == "__main__":
    app.run(debug=False, port=5000)
