# 📊 Smart Customer Churn Analyzer

## 🚀 Overview

Smart Customer Churn Analyzer is a machine learning-based web application that predicts whether a customer is likely to leave (churn) or stay with a company. The system uses customer data such as credit score, age, balance, and activity status to provide real-time predictions.

This project is deployed using Streamlit and provides an interactive user interface for easy input and instant results.

---

## 🧠 Features

* 📥 User-friendly input form
* 🤖 Machine Learning-based prediction
* 📊 Real-time churn probability
* ⚠️ Risk classification (High / Low)
* 🎯 Clean and modern UI design
* 🌐 Deployed on Streamlit Cloud

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Joblib

---

## 📂 Project Structure

```
Churn_Prediction_App/
│
├── app.py
├── requirements.txt
├── artifacts/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature_columns.json
```

---

## ⚙️ How It Works

1. User enters customer details
2. Data is preprocessed (encoding + scaling)
3. Machine learning model predicts churn probability
4. Results are displayed with risk level

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## 🌐 Live Demo

👉 (Add your Streamlit app link here)

---

## 📌 Input Parameters

* Credit Score
* Age
* Gender
* Tenure
* Balance
* Number of Products
* Credit Card Status
* Active Member Status
* Estimated Salary
* Geography

---

## 📊 Output

* Churn Probability (%)
* Retention Probability (%)
* Risk Level (High / Low)

---

## 💡 Future Improvements

* Add advanced ML models
* Improve dataset size and accuracy
* Add visualization charts
* Integrate database support

---

## 👨‍💻 Author

**Nikhil Raj**

---

## ⭐ Acknowledgement

This project is developed as part of academic learning to understand machine learning deployment using Streamlit.
