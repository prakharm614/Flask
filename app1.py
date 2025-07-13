# ------------------------------------------
# 📦 Import Libraries
# ------------------------------------------
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score

# ------------------------------------------
# 📂 Load and Preprocess Data
# ------------------------------------------
df = pd.read_csv('patient_data.csv')

# Rename column if needed
if 'C' in df.columns:
    df.rename(columns={'C': 'Gender'}, inplace=True)

# Fix spelling in 'Stages'
df['Stages'] = df['Stages'].replace({
    'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS',
    'HYPERTENSION (Stage-2).': 'HYPERTENSION (Stage-2)'
})

# Encode Age ranges
if 'Age' in df.columns:
    age_order = ['15-25', '25-35', '35-45', '45-55', '55-65', '65+']
    df['Age'] = df['Age'].map({val: idx for idx, val in enumerate(age_order)})
    print("✅ 'Age' column converted to numeric.")

# Encode categorical columns
columns = ['Gender', 'History', 'Patient', 'TakeMedication', 'Severity',
           'BreathShortness', 'VisualChanges', 'NoseBleeding',
           'Whendiagnoused', 'Systolic', 'Diastolic', 'ControlledDiet', 'Stages']

label_encoder = LabelEncoder()
for col in columns:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])
        print(f"✅ Encoded: {col}")
    else:
        print(f"⚠️ Column '{col}' not found.")

# Handle NaNs if any
df.fillna(0, inplace=True)
print("\n✅ After handling NaNs:\n", df.isnull().sum())

# ------------------------------------------
# 📈 Model Training
# ------------------------------------------
X = df.drop('Stages', axis=1)
Y = df['Stages']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=30)

# Train Models
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)
acc_lr = accuracy_score(y_test, lr.predict(x_test))

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
acc_dt = accuracy_score(y_test, dt.predict(x_test))

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
acc_rf = accuracy_score(y_test, rf.predict(x_test))

gnb = GaussianNB()
gnb.fit(x_train, y_train)
acc_nb = accuracy_score(y_test, gnb.predict(x_test))

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
acc_mnb = accuracy_score(y_test, mnb.predict(x_test))

# Show accuracy table
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

# Save best model (Random Forest)
pickle.dump(rf, open("model.pkl", "wb"))

# ------------------------------------------
# 🌐 Flask Web App
# ------------------------------------------
app = Flask(__name__, static_url_path='/static')
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read all form inputs
        Gender = float(request.form['Gender'])
        Age = float(request.form['Age'])
        History = float(request.form['History'])
        Patient = float(request.form['Patient'])
        TakeMedication = float(request.form['TakeMedication'])
        Severity = float(request.form['Severity'])
        BreathShortness = float(request.form['BreathShortness'])
        VisualChanges = float(request.form['VisualChanges'])
        NoseBleeding = float(request.form['NoseBleeding'])
        Whendiagnoused = float(request.form['Whendiagnoused'])
        Systolic = float(request.form['Systolic'])
        Diastolic = float(request.form['Diastolic'])
        ControlledDiet = float(request.form['ControlledDiet'])

        # Prepare input
        features = np.array([[Gender, Age, History, Patient, TakeMedication,
                              Severity, BreathShortness, VisualChanges,
                              NoseBleeding, Whendiagnoused, Systolic, Diastolic, ControlledDiet]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Decode result
        if prediction == 0:
            result = "NORMAL"
        elif prediction == 1:
            result = "HYPERTENSION (Stage-1)"
        elif prediction == 2:
            result = "HYPERTENSION (Stage-2)"
        else:
            result = "HYPERTENSIVE CRISIS"

        return render_template("result.html", prediction_text="Your Blood Pressure stage is: " + result)
    except Exception as e:
        return render_template("result.html", prediction_text=f"❌ Error: {str(e)}")

# ------------------------------------------
# 🚀 Run App
# ------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
