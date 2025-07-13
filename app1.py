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

# ------------------------------------------
# 📂 Load Data
# ------------------------------------------
df = pd.read_csv('patient_data.csv')

# Rename if needed
if 'C' in df.columns:
    df.rename(columns={'C': 'Gender'}, inplace=True)

# Fix spelling issues in Stages
df['Stages'] = df['Stages'].replace({
    'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS',
    'HYPERTENSION (Stage-2).': 'HYPERTENSION (Stage-2)'
})

# ------------------------------------------
# 🛠️ Encode Categorical Columns
# ------------------------------------------
# Convert Age ranges to numeric
if 'Age' in df.columns:
    age_order = ['15-25', '25-35', '35-45', '45-55', '55-65', '65+']
    age_map = {val: idx for idx, val in enumerate(age_order)}
    df['Age'] = df['Age'].map(age_map)
    print("✅ 'Age' column converted to numeric.")

# Encode remaining categorical columns
label_encoder = LabelEncoder()
columns = ['Gender', 'Severity', 'History', 'Patient', 'TakeMedication',
           'Breathshortness', 'VisualChanges', 'NoseBleeding', 'ControlledDiet', 'Stages']

for col in columns:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])
    else:
        print(f"⚠️ Column '{col}' not found in dataset.")

# ------------------------------------------
# 🔍 Dataset Summary
# ------------------------------------------
print("\n✅ Dataset Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDescriptive Statistics:\n", df.describe())

# ------------------------------------------
# 📊 EDA Plots
# ------------------------------------------
# Gender Distribution
if 'Gender' in df.columns:
    gender_counts = df['Gender'].value_counts()
    plt.pie(gender_counts, labels=gender_counts.index, autopct="%1.0f%%", startangle=140)
    plt.title('Gender Distribution')
    plt.axis('equal')
    plt.show()

# Stage Frequency
if 'Stages' in df.columns:
    plt.figure(figsize=(6,6))
    df['Stages'].value_counts().plot(kind='bar')
    plt.xlabel('Stages')
    plt.ylabel('Frequency')
    plt.title('Count of Stages')
    plt.show()

# TakeMedication vs Severity
if all(col in df.columns for col in ['TakeMedication', 'Severity']):
    sns.countplot(x='TakeMedication', hue='Severity', data=df)
    plt.title('TakeMedication vs Severity')
    plt.show()

# Pairplot (if columns exist)
if all(col in df.columns for col in ['Age', 'Systolic', 'Diastolic']):
    sns.pairplot(df[['Age', 'Systolic', 'Diastolic']])
    plt.show()
else:
    print("⚠️ Skipping pairplot — Columns 'Age', 'Systolic', or 'Diastolic' not found.")

# ------------------------------------------
# 📦 Model Training
# ------------------------------------------
X = df.drop('Stages', axis=1)
Y = df['Stages']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=30)

# Logistic Regression
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(x_train, y_train)
y_pred_lr = logistic_regression.predict(x_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

# Random Forest
random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)
y_pred_rf = random_forest.predict(x_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_pred_dt = decision_tree.predict(x_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred_nb = gnb.predict(x_test)
acc_nb = accuracy_score(y_test, y_pred_nb)

# Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred_mnb = mnb.predict(x_test)
acc_mnb = accuracy_score(y_test, y_pred_mnb)

# ------------------------------------------
# 📊 Model Accuracy Comparison
# ------------------------------------------
print("\n🔍 Model Accuracy Comparison:")
print(f"Logistic Regression:      {acc_lr:.4f}")
print(f"Decision Tree Classifier: {acc_dt:.4f}")
print(f"Random Forest Classifier: {acc_rf:.4f}")
print(f"Gaussian Naive Bayes:     {acc_nb:.4f}")
print(f"Multinomial Naive Bayes:  {acc_mnb:.4f}")

# ------------------------------------------
# 📊 Compare Scores in DataFrame
# ------------------------------------------
model_scores = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Decision Tree Classifier',
        'Random Forest Classifier',
        'Gaussian Naive Bayes',
        'Multinomial Naive Bayes'
    ],
    'Accuracy Score': [
        acc_lr,
        acc_dt,
        acc_rf,
        acc_nb,
        acc_mnb
    ]
})
# ------------------------------------------
# 📊 Display Final Model Accuracy Table
# ------------------------------------------
model_scores = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Decision Tree Classifier',
        'Random Forest Classifier',
        'Gaussian Navie Bayes',
        'Multinomial Navie Bayes'
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
# 🎯 GridSearchCV for Random Forest
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

# Evaluate Tuned Random Forest
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(x_test)
print("\n✅ Tuned RF Test Accuracy:", accuracy_score(y_test, y_pred_best))
print("\n📃 Classification Report:\n", classification_report(y_test, y_pred_best))

# ------------------------------------------
# 🔮 Prediction on Sample Row
# ------------------------------------------
sample = x_test.iloc[0].values.reshape(1, -1)
predicted_stage = best_rf.predict(sample)[0]
print("\n📌 Predicted Stage for Sample Input:", predicted_stage)

