import streamlit as st
import pandas as pd
import joblib

# Load all models and encoders
category_model = joblib.load("svm_task_classifier (1).joblib")
category_vectorizer = joblib.load("task_tfidf_vectorizer (1).pkl")
category_label_encoder = joblib.load("task_label_encoder (2).pkl")

priority_model = joblib.load("priority_xgboost (2).pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer (2).pkl")
priority_label_encoder = joblib.load("priority_label_encoder (2).pkl")

# Load dataset for user assignment
df = pd.read_csv("final_task_dataset_balanced.csv")

# Helper: Assign task to user with least tasks
def assign_user():
    user_counts = df["assigned_to"].value_counts()
    return user_counts.idxmin() if not user_counts.empty else "User_1"

# Streamlit UI
st.title("🚀 AI Task Management System")

task_description = st.text_area("📝 Enter task description:")

if st.button("Predict & Assign"):
    if task_description.strip() == "":
        st.warning("Please enter a task description.")
    else:
        # Category Prediction
        task_vec_cat = category_vectorizer.transform([task_description])
        category_pred_encoded = category_model.predict(task_vec_cat)[0]
        category_pred = category_label_encoder.inverse_transform([category_pred_encoded])[0]

        # Priority Prediction
        task_vec_prio = priority_vectorizer.transform([task_description])
        priority_pred_encoded = priority_model.predict(task_vec_prio)[0]
        priority_pred = priority_label_encoder.inverse_transform([priority_pred_encoded])[0]

        # Assign user
        assigned_user = assign_user()

        # Show results
        st.success(f"✅ Predicted Category: **{category_pred}**")
        st.success(f"📌 Predicted Priority: **{priority_pred}**")
        st.success(f"👤 Assigned to: **{assigned_user}**")
