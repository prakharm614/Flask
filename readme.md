🩺 Hypertension Stage Prediction Web App
=======================================

A machine learning-powered Flask web application that predicts the hypertension stage of a patient based on various health indicators and symptoms. The app features a user-friendly interface and uses a trained Random Forest Classifier for accurate predictions.

📸 Screenshots
--------------

- Home Page: screenshots/home.png
- Input Form: screenshots/form.png
- Prediction Result: screenshots/predict.png

🚀 Features
-----------
- Predicts 4 stages of blood pressure:
  • Normal
  • Hypertension (Stage-1)
  • Hypertension (Stage-2)
  • Hypertensive Crisis
- Encodes categorical data and handles missing values
- Compares performance of multiple ML models
- Uses best model for prediction (Random Forest)
- Responsive UI with HTML, CSS, and JavaScript
- Trained model saved using pickle

📂 Project Structure
--------------------
Flask/
│
├── app1.py                  -> Main Python Flask backend
├── model.pkl                -> Trained ML model
├── patient_data.csv         -> Sample dataset
│
├── static/
│   └── style.css            -> CSS file
│
├── templates/
│   ├── index.html           -> Home input form
│   ├── predict.html         -> Result page
│   └── details.html         -> Optional team/info page
│
└── README.md                -> Project README

⚙️ How to Run the App
---------------------
1. Install Python 3.10 or later

2. Install required libraries:
   pip install pandas numpy scikit-learn flask seaborn matplotlib

3. Or use:
   pip install -r requirements.txt

4. Run the Flask app:
   python app1.py

5. Visit in browser:
   http://127.0.0.1:5000

📊 Model Performance
--------------------
| Model                      | Accuracy   |
|---------------------------|------------|
| Logistic Regression       | 97.53%     |
| Decision Tree Classifier  | 100%       |
| Random Forest Classifier  | 100% ✅     |
| Gaussian Naive Bayes      | 100%       |
| Multinomial Naive Bayes   | 77.77%     |

✅ Final model used: Random Forest Classifier

🧪 Inputs for Prediction
-------------------------
- Gender (dropdown)
- Age Group (dropdown)
- History (Yes/No)
- Take Medication (Yes/No)
- Severity (Low/Medium/High)
- Shortness of Breath (Yes/No)
- Visual Changes (Yes/No)
- Nose Bleeding (Yes/No)
- When Diagnosed (Months Ago)
- Systolic BP (90–200)
- Diastolic BP (60–120)
- Controlled Diet (Yes/No)

📬 Contact / Contribution
--------------------------
- Feel free to fork or suggest improvements.
- Raise issues for bugs or suggestions.

👨‍💻 Developer
--------------
Prakhar Mishra  
Email: prakharmishra614@gmail.com  
LinkedIn: https://www.linkedin.com/in/prakharm614

📄 License
----------
MIT License – free to use, modify, and distribute.
