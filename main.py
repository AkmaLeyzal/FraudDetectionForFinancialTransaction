# main.py
import streamlit as st
import pandas as pd
import numpy as np
import dill
from src.preprocessing import TransactionPreprocessor
from src.visualization import ModelVisualizer
from src.model_evaluation import ModelEvaluator, display_model_evaluation


preprocessor = TransactionPreprocessor()
visualizer = ModelVisualizer()
evaluator = ModelEvaluator()

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_models():
    try:
        with open('models/random_forest_model.pkl', 'rb') as f:
            rf_model = dill.load(f)
        with open('models/decision_tree_model.pkl', 'rb') as f:
            dt_model = dill.load(f)
        with open('data/test_data.pkl', 'rb') as f:
            test_data = dill.load(f)
 
        return rf_model, dt_model, test_data
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

rf_model, dt_model, test_data = load_models()

st.title('Fraud Detection System')
st.markdown("""
This application uses machine learning to detect fraudulent transactions based on various features.
The models used are Decision Tree and Random Forest classifiers.
""")

tab1, tab2, tab3 = st.tabs(["Transaction Analysis", "Model Performance", "Model Information"])

with tab1:
    st.header("Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_type = st.selectbox(
            "Transaction Type",
            ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
        )
        old_balance_orig = st.number_input("Origin Account Old Balance", min_value=0.0, format="%.2f")
        old_balance_dest = st.number_input("Destination Account Old Balance", min_value=0.0, format="%.2f")
        is_merchant = st.checkbox("Destination is Merchant")
    
    with col2:
        amount = st.number_input("Amount", min_value=0.0, format="%.2f")
        new_balance_orig = st.number_input("Origin Account New Balance", min_value=0.0, format="%.2f")
        new_balance_dest = st.number_input("Destination Account New Balance", min_value=0.0, format="%.2f")
        
    transaction_data = {
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': old_balance_orig,
        'newbalanceOrig': new_balance_orig,
        'oldbalanceDest': old_balance_dest,
        'newbalanceDest': new_balance_dest,
        'nameDest': 'M' if is_merchant else 'C'
    }
    
    if st.button("Analyze Transaction", help="Click button to start analyze):
        try:
            X = preprocessor.transform_single(transaction_data)
            
            dt_pred = dt_model.predict(X)[0] if dt_model else None
            rf_pred = rf_model.predict(X)[0] if rf_model else None
            
            st.subheader("Analysis Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Decision Tree Prediction")
                if dt_pred is not None:
                    if dt_pred == 1:
                        st.error("⚠️ Potential Fraud Detected")
                    else:
                        st.success("✅ Transaction appears legitimate")
                else:
                    st.warning("Model not available")
            
            with col2:
                st.markdown("#### Random Forest Prediction")
                if rf_pred is not None:
                    if rf_pred == 1:
                        st.error("⚠️ Potential Fraud Detected")
                    else:
                        st.success("✅ Transaction appears legitimate")
                else:
                    st.warning("Model not available")
            
            st.markdown("#### Transaction Risk Factors")
            risk_factors = []
            
            if amount > 100_000_000:
                risk_factors.append("High transaction amount")
            
            if abs(old_balance_orig - new_balance_orig) != amount:
                risk_factors.append("Balance change mismatch")
            
            if new_balance_orig == 0 and old_balance_orig > 0:
                risk_factors.append("Account emptied")
            
            if risk_factors:
                st.warning("Identified risk factors:\n- " + "\n- ".join(risk_factors))
            else:
                st.info("No specific risk factors identified")
            
        except Exception as e:
            st.error(f"Error analyzing transaction: {str(e)}")
with tab2:
    st.header("Model Performance Analysis")
        
    if test_data is not None:
        X_test, y_test = test_data['X'], test_data['y']
        
        if dt_model is not None:
            display_model_evaluation(dt_model, X_test, y_test, "Decision Tree")
            
        st.markdown("---")
        if rf_model is not None:
            display_model_evaluation(rf_model, X_test, y_test, "Random Forest")
    else:
        st.warning("Test data not available for model evaluation")
with tab3:
    st.header("Model Information")
    
    st.markdown("""
    ### Model Architecture
    The system uses two complementary models:
    
    1. **Decision Tree**
        - Simple, interpretable model
        - Makes decisions based on feature thresholds
        - Provides clear decision paths
    
    2. **Random Forest**
        - Ensemble of decision trees
        - More robust predictions
        - Better handles complex patterns
    """)
    
    st.markdown("### Feature Importance")
    try:
        feature_names = preprocessor.feature_names
        
        if dt_model and hasattr(dt_model, 'feature_importances_'):
            st.plotly_chart(
                visualizer.create_feature_importance_plot(
                    dt_model.feature_importances_,
                    feature_names,
                    "Decision Tree - Feature Importance"
                ),
                use_container_width=True
            )
        
        if rf_model and hasattr(rf_model, 'feature_importances_'):
            st.plotly_chart(
                visualizer.create_feature_importance_plot(
                    rf_model.feature_importances_,
                    feature_names,
                    "Random Forest - Feature Importance"
                ),
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"Error displaying feature importance: {str(e)}")
        
    st.markdown("### Model Performance")
    st.info("""
    The models were trained on a dataset of financial transactions with the following characteristics:
    - Balance between fraud and legitimate transactions
    - Various transaction types and amounts
    - Temporal patterns across days and hours
    """)

st.sidebar.markdown("""
### Disclaimer
This system is provided "as is" without any warranties or guarantees. Use at your own risk.

### About
This fraud detection system uses custom-implemented machine learning models trained on financial transaction data.
The models analyze various transaction features to identify potentially fraudulent activities.

### Features Used
- Transaction type
- Transaction amount
- Account balances
- Balance changes
- Merchant information
                    
### Dataset
The dataset used for training the models is a subset of a larger dataset of financial transactions  
[Link Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1).

### Acknowledgments
Special thanks to Akmal Rizal, Reva Deshinta Isyana, and Yasyifa Sastiya Nabali for their contributions to this project.

### Contact
For any inquiries or feedback, please contact our email akmal.23078@mhs.unesa.ac.id""")


