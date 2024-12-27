import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Load the saved models
@st.cache_resource
def load_models():
    with open('decision_tree_model.pkl', 'rb') as f:
        dt_model = pickle.load(f)
    with open('random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return dt_model, rf_model, feature_names

def preprocess_input(data):
    df = pd.DataFrame([data])
    
    df_encoded = pd.get_dummies(df, columns=['type'])
    
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    df_encoded = df_encoded[feature_names]
    
    return df_encoded.values.astype(np.float64)

def main():
    st.title('Fraud Detection System')
    st.write("""
    This application predicts whether a transaction is fraudulent or not using 
    both Decision Tree and Random Forest models.
    """)
    
    dt_model, rf_model, feature_names = load_models()
    
    st.header('Transaction Details')
    
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_type = st.selectbox('Transaction Type', 
                                      ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'])
        amount = st.number_input('Amount', min_value=0.0)
        old_balance_orig = st.number_input('Original Old Balance', min_value=0.0)
        new_balance_orig = st.number_input('Original New Balance', min_value=0.0)
        
    with col2:
        old_balance_dest = st.number_input('Destination Old Balance', min_value=0.0)
        new_balance_dest = st.number_input('Destination New Balance', min_value=0.0)
    
    if st.button('Predict'):
        input_data = {
            'type': transaction_type,
            'amount': amount,
            'oldbalanceOrg': old_balance_orig,
            'newbalanceOrig': new_balance_orig,
            'oldbalanceDest': old_balance_dest,
            'newbalanceDest': new_balance_dest
        }
        
        processed_input = preprocess_input(input_data)
        
        dt_pred = dt_model.predict(processed_input)[0]
        rf_pred = rf_model.predict(processed_input)[0]
        
        st.header('Prediction Results')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Decision Tree Prediction')
            if dt_pred == 1:
                st.error('⚠️ Fraudulent Transaction')
            else:
                st.success('✅ Legitimate Transaction')
                
        with col2:
            st.subheader('Random Forest Prediction')
            if rf_pred == 1:
                st.error('⚠️ Fraudulent Transaction')
            else:
                st.success('✅ Legitimate Transaction')
        
        # Add explanation
        st.write("""
        ### Model Interpretation
        - If both models predict fraud, the transaction is highly suspicious
        - If models disagree, further investigation might be needed
        - Consider the transaction amount and balance changes when evaluating results
        """)
        
        st.subheader('Transaction Flow Visualization')
        
        fig = go.Figure(data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=["Origin Account", "Transaction", "Destination Account"],
                    color=["blue", "gray", "green"]
                ),
                link=dict(
                    source=[0, 1],
                    target=[1, 2],
                    value=[amount, amount],
                    color=["rgba(0,0,255,0.4)", "rgba(0,255,0,0.4)"]
                )
            )
        ])
        
        fig.update_layout(title_text="Money Flow", font_size=10)
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()