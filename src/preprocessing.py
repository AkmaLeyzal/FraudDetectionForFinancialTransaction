# src/preprocessing.py

import numpy as np
import pandas as pd

def engineer_features(df):
    df_engineered = df.copy()
    
    # Balance differences
    df_engineered['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df_engineered['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # Balance ratios
    eps = 1e-10
    df_engineered['balance_ratio_orig'] = (df['newbalanceOrig'] + eps) / (df['oldbalanceOrg'] + eps)
    df_engineered['balance_ratio_dest'] = (df['newbalanceDest'] + eps) / (df['oldbalanceDest'] + eps)
    
    # Error balances
    df_engineered['error_balance_orig'] = df['amount'] - (df['oldbalanceOrg'] - df['newbalanceOrig'])
    df_engineered['error_balance_dest'] = df['amount'] - (df['newbalanceDest'] - df['oldbalanceDest'])
    
    # Amount ratios
    df_engineered['amount_ratio_orig'] = df['amount'] / (df['oldbalanceOrg'] + eps)
    df_engineered['amount_ratio_dest'] = df['amount'] / (df['oldbalanceDest'] + eps)
    
    # Binary flags
    df_engineered['zero_balance_orig'] = ((df['oldbalanceOrg'] == 0) | (df['newbalanceOrig'] == 0)).astype(int)
    df_engineered['zero_balance_dest'] = ((df['oldbalanceDest'] == 0) | (df['newbalanceDest'] == 0)).astype(int)
    df_engineered['full_transfer'] = (df['newbalanceOrig'] == 0).astype(int)
    df_engineered['balance_mismatch'] = (abs(df['oldbalanceOrg'] - df['newbalanceOrig']) != df['amount']).astype(int)
    
    df_engineered['dest_is_merchant'] = (df['nameDest'].str.startswith('M')).astype(int)
    
    return df_engineered

def preprocess_single_transaction(transaction_data, feature_names):
    
    df = pd.DataFrame([transaction_data])
    
    # Engineer features
    df = engineer_features(df)
    
    # Select and encode features
    base_features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                    'oldbalanceDest', 'newbalanceDest']
    
    additional_features = [
        'balance_diff_orig', 'balance_diff_dest',
        'balance_ratio_orig', 'balance_ratio_dest',
        'error_balance_orig', 'error_balance_dest',
        'amount_ratio_orig', 'amount_ratio_dest',
        'zero_balance_orig', 'zero_balance_dest',
        'full_transfer', 'balance_mismatch',
        'dest_is_merchant'
    ]
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df[base_features + additional_features], columns=['type'])
    
    # Ensure all features from training are present
    for feature in feature_names:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
            
    # Reorder columns to match training data
    df_encoded = df_encoded[feature_names]
    
    # Convert to numpy array
    X = df_encoded.values.astype(np.float64)
    
    return X