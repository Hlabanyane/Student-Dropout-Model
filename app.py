import streamlit as st
import pandas as pd
import numpy as np
import joblib

#Page Configuration Setup

st.set_page_config(page_title="Student_Dropout_Prediction", layout="wide")

#Loading The Trained Model And It's Preprocessors

@st.cache_resource
def load_artifacts():
    model = joblib.load("student_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    return model, scaler, feature_names, label_encoders

model, scaler, feature_names, label_encoders = load_artifacts()

#Setup the Application Header

st.title("Welcome to Student Dropout and Academic Success Prediction Dashboard")

#SideBar Options

mode = st.sidebar.radio("Select Prediction Mode", ["Single Prediction", "Batch Prediction"])

# Common mapping
outcomes = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
colors = {"Dropout": "red", "Enrolled": "orange", "Graduate": "green"}

#Setup for the single prediction mode 

if mode == "Single Prediction":
    st.subheader("Enter Student Details")

    user_inputs = {}
    for col in feature_names:
        lname = col.lower()
        if any(k in lname for k in ["grade", "age", "rate", "gdp", "inflation", "unemployment", "index", "average"]):
            user_inputs[col] = st.number_input(col, value=0.0, format="%.4f", step=0.1)
        elif any(k in lname for k in ["yes", "no", "tuition", "scholarship", "international", "debtor", "displaced", "special needs"]):
            user_inputs[col] = st.selectbox(col, options=[0, 1])
        elif any(k in lname for k in ["curricular", "units", "enrolled", "approved", "evaluations", "without evaluations", "credited"]):
            user_inputs[col] = st.number_input(col, min_value=0, value=0, step=1)
        else:
            user_inputs[col] = st.text_input(col, value="0")

    # Convert to DataFrame
    input_df = pd.DataFrame([user_inputs])

    # Apply label encoders
    for col, le in label_encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = le.transform(input_df[col].astype(str))
            except Exception:
                input_df[col] = 0

    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    if st.button("PREDICT OUTCOME"):
        pred = model.predict(input_scaled)[0]
        result = outcomes.get(pred, "Unknown")

        st.markdown(f"### **Predicted Academic Outcome:**")
        st.markdown(f"<h2 style='color:{colors[result]}'>{result}</h2>", unsafe_allow_html=True)

        # Add the prediction to DataFrame
        input_df["Predicted_Outcome"] = result

        # Allow download of single prediction
        csv = input_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Prediction as CSV",
            data=csv,
            file_name="single_student_prediction.csv",
            mime="text/csv",
        )

    with st.expander("Show Processed Input Data"):
        st.dataframe(input_df.T)

#Setting up the batch prediction mode 
else:
    st.subheader("Upload Dataset for Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                batch_data = pd.read_csv(uploaded_file)
            else:
                batch_data = pd.read_excel(uploaded_file)

            st.success(f"Dataset loaded successfully with {batch_data.shape[0]} rows.")

            # Ensure feature alignment
            missing_features = [f for f in feature_names if f not in batch_data.columns]
            if missing_features:
                st.warning(f"Missing features: {missing_features}")
                st.stop()

            batch_data = batch_data[feature_names]

            # Apply label encoding
            for col, le in label_encoders.items():
                if col in batch_data.columns:
                    try:
                        batch_data[col] = le.transform(batch_data[col].astype(str))
                    except Exception:
                        batch_data[col] = 0

            batch_scaled = scaler.transform(batch_data)

            # Predict
            predictions = model.predict(batch_scaled)
            batch_data["Predicted_Outcome"] = [outcomes.get(p, "Unknown") for p in predictions]

            st.subheader("Prediction Results (First 20 Rows)")
            st.dataframe(batch_data.head(20))

            # Download results
            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download All Predictions as CSV",
                data=csv,
                file_name="student_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error reading file: {e}")
