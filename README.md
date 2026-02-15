# Telecom Churn Intelligence Platform

This project is an end-to-end machine learning system for predicting telecom customer churn and converting predictions into actionable business insights.

The focus of this project is not just churn prediction, but **decision support** for telecom companies.

---

## What the Project Does

- Predicts customer churn probability  
- Classifies customers into risk levels  
- Explains why a customer may churn  
- Ranks customers by potential revenue loss  
- Suggests retention actions  
- Simulates business decisions (what-if analysis)  
- Supports batch CSV analysis  
- Works with external datasets using schema mapping  

---

## Key Features

- Churn Prediction using Random Forest  
- Risk Classification (Low / Medium / High)  
- Explainable Churn Reasons (rule-based)  
- Revenue Risk Ranking  
- What-If Simulation Engine  
- Batch CSV Prediction  
- Schema Mapping for External Datasets  

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- IBM Telco Customer Churn Dataset  

---

## System Architecture

Customer Dataset (CSV)
↓
Schema Mapping (column alignment)
↓
Data Cleaning & Encoding
↓
Churn Prediction Model
↓
Risk Classification
↓
Explainable Churn Logic
↓
Revenue Risk Calculation
↓
What-If Simulation
↓
Retention Action Recommendation

---

## Project Structure

Telecom-Churn-Intelligence/
│
├── data/
│ └── telco_churn.csv
│
├── train_model.py # Model training
├── churn_system.py # Single customer prediction
├── batch_predict.py # Batch CSV churn analysis
├── schema_mapper.py # Dataset schema mapping
├── utils.py # Business logic & helpers
│
├── churn_model.pkl
├── feature_names.pkl
├── label_encoders.pkl
│
├── requirements.txt
└── README.md

---

## How to Run the Project

Install dependencies:
```bash
pip install -r requirements.txt

Train the churn prediction model:
python train_model.py


Run single customer churn analysis:
python churn_system.py


Run batch churn analysis on CSV data:
python batch_predict.py


The batch analysis output will be saved as:
churn_analysis_output.csv

```

## Sample Output
Churn Probability: 0.72
Risk Level: High Risk
Reasons: Short-term contract, High monthly charges, Low customer tenure
Recommended Action: Offer discount + long-term contract
What-If Simulation:
- 10% Price Reduction → 0.68
- Switch to Yearly Contract → 0.47

---

## Notes

The model is trained on telecom customer behavior

It expects telecom-related features

External company datasets are supported through schema mapping

Post-churn data is excluded to avoid data leakage

The project focuses on realistic ML usage rather than artificial accuracy

---

## Learning Outcomes

Building an end-to-end ML pipeline

Handling feature consistency between training and inference

Designing explainable ML systems

Applying ML outputs to real business decisions

Supporting multiple datasets using schema mapping

---

## Author

Gaurav  
B.Tech Student | AI Developer (Machine Learning)  
Built as a learning-focused project to apply machine learning concepts to real-world business problems.
