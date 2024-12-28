# src/preprocessing.py
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from collections import Counter

class TransactionPreprocessor:
    """Preprocessor for financial transaction data"""
    
    def __init__(self):
        self.type_categories = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
        self.base_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        self.feature_names = self._get_feature_names()

    def _get_feature_names(self) -> List[str]:
        type_features = [f'type_{cat}' for cat in self.type_categories]
        
        engineered_features = [
            'balance_diff_orig', 'balance_diff_dest',
            'balance_ratio_orig', 'balance_ratio_dest',
            'error_balance_orig', 'error_balance_dest',
            'amount_ratio_orig', 'amount_ratio_dest',
            'zero_balance_orig', 'zero_balance_dest',
            'full_transfer', 'balance_mismatch',
            'hour', 'day', 'dest_is_merchant'
        ]
        
        return type_features + self.base_features + engineered_features

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to the dataset"""
        df = df.copy()
        
        eps = 1e-10  
        df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        df['balance_ratio_orig'] = (df['newbalanceOrig'] + eps) / (df['oldbalanceOrg'] + eps)
        df['balance_ratio_dest'] = (df['newbalanceDest'] + eps) / (df['oldbalanceDest'] + eps)
        
        df['error_balance_orig'] = df['amount'] - (df['oldbalanceOrg'] - df['newbalanceOrig'])
        df['error_balance_dest'] = df['amount'] - (df['newbalanceDest'] - df['oldbalanceDest'])
        
        df['amount_ratio_orig'] = df['amount'] / (df['oldbalanceOrg'] + eps)
        df['amount_ratio_dest'] = df['amount'] / (df['oldbalanceDest'] + eps)
        
        df['zero_balance_orig'] = ((df['oldbalanceOrg'] == 0) | (df['newbalanceOrig'] == 0)).astype(int)
        df['zero_balance_dest'] = ((df['oldbalanceDest'] == 0) | (df['newbalanceDest'] == 0)).astype(int)
        df['full_transfer'] = (df['newbalanceOrig'] == 0).astype(int)
        df['balance_mismatch'] = (abs(df['oldbalanceOrg'] - df['newbalanceOrig']) != df['amount']).astype(int)
        
        if 'step' in df.columns:
            df['hour'] = df['step'] % 24
            df['day'] = df['step'] // 24
        else:
            df['hour'] = 0
            df['day'] = 0
            
        df['dest_is_merchant'] = df['nameDest'].str.startswith('M').astype(int)
        
        return df

    def fit_transform(self, df: pd.DataFrame, sampling_strategy: str = 'none') -> Tuple[np.ndarray, np.ndarray]:
        df_processed = self._engineer_features(df)
        
        df_encoded = pd.get_dummies(df_processed, columns=['type'], prefix='type')
        
        for type_col in [f'type_{cat}' for cat in self.type_categories]:
            if type_col not in df_encoded.columns:
                df_encoded[type_col] = 0
                
        X = df_encoded[self.feature_names].values
        y = df['isFraud'].values
        
        X = np.nan_to_num(X)
        
        if sampling_strategy != 'none':
            X, y = self._handle_imbalanced_data(X, y, sampling_strategy)
        
        return X, y

    def transform_single(self, transaction: Dict) -> np.ndarray:
        df = pd.DataFrame([transaction])
        df_processed = self._engineer_features(df)
        
        df_encoded = pd.get_dummies(df_processed, columns=['type'], prefix='type')
        
        for type_col in [f'type_{cat}' for cat in self.type_categories]:
            if type_col not in df_encoded.columns:
                df_encoded[type_col] = 0
        
        X = df_encoded[self.feature_names].values
        
        X = np.nan_to_num(X)
        
        return X

    def _handle_imbalanced_data(self, X: np.ndarray, y: np.ndarray, strategy: str) -> Tuple[np.ndarray, np.ndarray]:
        fraud_indices = np.where(y == 1)[0]
        non_fraud_indices = np.where(y == 0)[0]
        
        n_fraud = len(fraud_indices)
        n_non_fraud = len(non_fraud_indices)
        
        if strategy == 'undersample':
            selected_non_fraud = np.random.choice(non_fraud_indices, size=n_fraud, replace=False)
            selected_indices = np.concatenate([fraud_indices, selected_non_fraud])
            
        elif strategy == 'oversample':
            fraud_oversampled = np.random.choice(fraud_indices, size=n_non_fraud, replace=True)
            selected_indices = np.concatenate([fraud_oversampled, non_fraud_indices])
            
        elif strategy == 'combine':
            target_size = int(n_non_fraud * 0.5)
            selected_non_fraud = np.random.choice(non_fraud_indices, size=target_size, replace=False)
            fraud_oversampled = np.random.choice(fraud_indices, size=target_size, replace=True)
            selected_indices = np.concatenate([fraud_oversampled, selected_non_fraud])
            
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        return X[selected_indices], y[selected_indices]