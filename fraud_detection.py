import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#Load Test Data (for ROC Curve)
x_test, y_test = joblib.load("test_data.pkl")

#App Title
st.title("Fraud Detection Prediction App")
st.markdown("Please enter the transaction details and select a model")

st.divider()

#Model Selection
model_choice = st.selectbox("Choose Model", ["Logistic", "Random Forest"])

if model_choice == "Logistic":
    model = joblib.load("fraud_detection_pipeline.pkl")
else:
    model = joblib.load("random_forest_pipeline.pkl")

#User Input
transaction_type = st.selectbox("Transaction Type",["PAYMENT","TRANSFER","CASH_OUT","DEPOSITE"])
amount = st.number_input("Amount",min_value = 0.0,value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value = 0.0,value=10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value= 0.0,value=9000.0)
oldbalanceDest = st. number_input("Old Balance (Receiver)", min_value=0.0,value=0.0)
newbalanceDest = st. number_input("New Balance (Receiver)", min_value= 0.0,value=0.0)

#Predict Button
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "type" : transaction_type,
        "amount" : amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "balanceDiffOrig": oldbalanceOrg - newbalanceOrig,
        "balanceDiffDest": newbalanceDest - oldbalanceDest
    }])

    #Make Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    st. subheader (f"Prediction: '{int(prediction)}'")
    st.write(f"Fraud Probability: {round(probability*100, 2)}%")

    #Progress Bar
    st.progress(int(probability * 100))
    if prediction == 1:
        st.error("⚠️ This transaction is likely FRAUD")
    else:
        st.success("✅ This transaction looks SAFE")
    #Feature Importance for Random Forest
    if model_choice == "Random Forest":
        st.subheader("Feature Importance")

        try:
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
            importances = model.named_steps["model"].feature_importances_

            fig, ax = plt.subplots()
            ax.barh(feature_names, importances)
            ax.set_title("Feature Importance")
            st.pyplot(fig)

        except:
            st.warning("Feature importance not available")
    


