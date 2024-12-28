import streamlit as st
import pandas as pd
import numpy as np
import dill
import plotly.graph_objects as go
from src.preprocessing import preprocess_single_transaction
from src.visualization import create_confusion_matrix_plot, create_feature_importance_plot

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_models():
    with open('models/random_forest_model.pkl', 'rb') as f:
        rf_model = dill.load(f)
    with open('models/decision_tree_model.pkl', 'rb') as f:
        dt_model = dill.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = dill.load(f)
    return rf_model, dt_model, feature_names

rf_model, dt_model, feature_names = load_models()

st.title('Fraud Detection System')
st.markdown("""
This application uses machine learning to detect fraudulent transactions based on various features.
The models used are Decision Tree and Random Forest classifiers.
""")

tab1, tab2 = st.tabs(["Predict Single Transaction", "Model Information"])

with tab1:
    st.header("Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_type = st.selectbox(
            "Transaction Type",
            ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
        )
        amount = st.number_input("Amount", min_value=0.0, format="%.2f")
        old_balance_orig = st.number_input("Origin Account Old Balance", min_value=0.0, format="%.2f")
        new_balance_orig = st.number_input("Origin Account New Balance", min_value=0.0, format="%.2f")
        
    with col2:
        old_balance_dest = st.number_input("Destination Account Old Balance", min_value=0.0, format="%.2f")
        new_balance_dest = st.number_input("Destination Account New Balance", min_value=0.0, format="%.2f")
        is_merchant = st.checkbox("Destination is Merchant")
        
    # Create transaction data
    transaction_data = {
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': old_balance_orig,
        'newbalanceOrig': new_balance_orig,
        'oldbalanceDest': old_balance_dest,
        'newbalanceDest': new_balance_dest,
        'nameDest': 'M' if is_merchant else 'C'
    }
    
    if st.button("Predict"):
        X = preprocess_single_transaction(transaction_data, feature_names)
        
        dt_pred = dt_model.predict(X)[0]
        rf_pred = rf_model.predict(X)[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Decision Tree Prediction")
            if dt_pred == 1:
                st.error("⚠️ Fraudulent Transaction")
            else:
                st.success("✅ Legitimate Transaction")
                
        with col2:
            st.subheader("Random Forest Prediction")
            if rf_pred == 1:
                st.error("⚠️ Fraudulent Transaction")
            else:
                st.success("✅ Legitimate Transaction")

# In the "Model Information" tab section:
with tab2:
    st.header("Model Information")
    
    st.markdown("""
    ### Model Details
    - **Decision Tree**: A tree-based model that makes decisions based on feature thresholds
    - **Random Forest**: An ensemble of decision trees that provides more robust predictions
    
    ### Feature Importance
    The following plots show which features are most important for making predictions:
    """)
    
    try:
        # Show Decision Tree feature importance
        if hasattr(dt_model, 'feature_importances_') and len(dt_model.feature_importances_) == len(feature_names):
            dt_plot = create_feature_importance_plot(
                dt_model.feature_importances_,
                feature_names,
                "Decision Tree - Feature Importance"
            )
            st.plotly_chart(dt_plot, use_container_width=True)
        else:
            st.warning("Decision Tree feature importances not available")
            
        # Show Random Forest feature importance
        if hasattr(rf_model, 'feature_importances_') and len(rf_model.feature_importances_) == len(feature_names):
            rf_plot = create_feature_importance_plot(
                rf_model.feature_importances_,
                feature_names,
                "Random Forest - Feature Importance"
            )
            st.plotly_chart(rf_plot, use_container_width=True)
        else:
            st.warning("Random Forest feature importances not available")
            
    except Exception as e:
        st.error(f"Error creating feature importance plots: {str(e)}")

st.sidebar.markdown("""
### About
This fraud detection system uses custom-implemented machine learning models trained on financial transaction data.
The models analyze various transaction features to identify potentially fraudulent activities.

### Features Used
- Transaction type
- Transaction amount
- Account balances
- Balance changes
- Merchant information
""")
