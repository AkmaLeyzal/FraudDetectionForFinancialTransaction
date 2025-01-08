# src/preprocessing.py
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
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
    def process_single_transaction(
        self,
        transaction: Dict[str, Union[str, float, int]],
        scaler
    ) -> np.ndarray:
        """
        Process a single transaction for fraud detection.
        
        Args:
            transaction: Dictionary containing transaction data with keys:
                - type: Transaction type (CASH_OUT, TRANSFER, etc.)
                - amount: Transaction amount
                - oldbalanceOrg: Original account's previous balance
                - newbalanceOrig: Original account's new balance
                - oldbalanceDest: Destination account's previous balance
                - newbalanceDest: Destination account's new balance
                - nameDest: Destination account name
            scaler: Fitted StandardScaler for amount normalization
            
        Returns:
            np.ndarray: Processed features ready for model prediction
        """
        # Convert single transaction to DataFrame for consistent processing
        df_single = pd.DataFrame([transaction])
        df_single = df_single.drop(['isFraud', 'isFlaggedFraud'], axis=1, errors='ignore')
        # Engineer features
        df_engineered = self.engineer_single_transaction(df_single, scaler)
        # X = df_engineered
        # Convert to array
        X = df_engineered.values.astype(np.float64)
        
        # Handle any NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X

    def engineer_single_transaction(
        self,
        df: pd.DataFrame,
        scaler
    ) -> pd.DataFrame:
        
        df_engineered = df.copy()
        eps = 1e-10  # Small epsilon to prevent division by zero
        
        # 1. Transaction Amount Features
        df_engineered['amount_normalized'] = scaler.transform(df[['amount']])
        df_engineered['amount_log'] = np.log1p(df['amount'])
        df_engineered['is_large_transaction'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        
        # 2. Balance-Based Features
        # Basic balance changes
        df_engineered['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        df_engineered['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        
        # Balance ratios and patterns
        df_engineered['balance_ratio_orig'] = (df['newbalanceOrig'] + eps) / (df['oldbalanceOrg'] + eps)
        df_engineered['balance_ratio_dest'] = (df['newbalanceDest'] + eps) / (df['oldbalanceDest'] + eps)
        
        # Account emptying patterns
        df_engineered['orig_zero_after_tx'] = (df['newbalanceOrig'] == 0).astype(int)
        df_engineered['orig_zero_before_tx'] = (df['oldbalanceOrg'] == 0).astype(int)
        df_engineered['dest_zero_after_tx'] = (df['newbalanceDest'] == 0).astype(int)
        df_engineered['dest_zero_before_tx'] = (df['oldbalanceDest'] == 0).astype(int)
        
        # 3. Error Detection Features
        df_engineered['balance_error_orig'] = abs(df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig'])
        df_engineered['balance_error_dest'] = abs(df['newbalanceDest'] - df['oldbalanceDest'] - df['amount'])
        df_engineered['has_balance_error'] = ((df_engineered['balance_error_orig'] > eps) | 
                                            (df_engineered['balance_error_dest'] > eps)).astype(int)
        
        # 4. Transaction Pattern Features
        df_engineered['amount_to_oldbalance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + eps)
        df_engineered['amount_to_newbalance_ratio'] = df['amount'] / (df['newbalanceOrig'] + eps)
        
        # One-hot encode transaction types
        tx_types = ['CASH_OUT', 'TRANSFER', 'CASH_IN', 'DEBIT', 'PAYMENT']
        for tx_type in tx_types:
            df_engineered[f'type_{tx_type}'] = (df['type'] == tx_type).astype(int)
        
        # 5. Temporal Features
        df_engineered['hour'] = df['step'] % 24
        df_engineered['day'] = df['step'] // 24
        
        # 6. Destination Features
        df_engineered['is_merchant'] = (df['nameDest'].str.startswith('M')).astype(int)
        
        # 7. Risk Scoring Features
        risk_factors = [
            df_engineered['is_large_transaction'],
            df_engineered['orig_zero_after_tx'],
            df_engineered['has_balance_error'],
            ((df_engineered['type_CASH_OUT'] == 1) & (df_engineered['amount_to_oldbalance_ratio'] > 0.9)).astype(int),
            (df_engineered['amount_to_oldbalance_ratio'] > 0.9).astype(int)
        ]
        df_engineered['risk_score'] = sum(risk_factors)
        
        # 8. Fraud Pattern Recognition Features
        df_engineered['suspicious_pattern'] = (
            (df_engineered['type_TRANSFER'] & df_engineered['orig_zero_after_tx']) |  # Complete balance transfer
            (df_engineered['type_CASH_OUT'] & (df_engineered['amount_to_oldbalance_ratio'] > 0.9)) |  # Large cash out
            (df_engineered['is_large_transaction'] & df_engineered['has_balance_error']) |  # Suspicious large transaction
            (df_engineered['risk_score'] >= 3)  # Multiple risk factors
        ).astype(int)

        df_engineered['fraud_pattern'] = (df['amount'] == df['oldbalanceOrg']).astype(int)
        
        # Drop unnecessary columns
        columns_to_drop = ['nameDest', 'nameOrig', 'isFlaggedFraud', 'type', 'step']
        df_engineered = df_engineered.drop(columns=[col for col in columns_to_drop if col in df_engineered.columns])
        
        # Store feature names if not already stored
        
        feature_names = df_engineered.columns.tolist()
            
        return df_engineered
    
    # def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """Add engineered features to the dataset"""
    #     df = df.copy()
        
    #     eps = 1e-10  
    #     df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    #     df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    #     df['balance_ratio_orig'] = (df['newbalanceOrig'] + eps) / (df['oldbalanceOrg'] + eps)
    #     df['balance_ratio_dest'] = (df['newbalanceDest'] + eps) / (df['oldbalanceDest'] + eps)
        
    #     df['error_balance_orig'] = df['amount'] - (df['oldbalanceOrg'] - df['newbalanceOrig'])
    #     df['error_balance_dest'] = df['amount'] - (df['newbalanceDest'] - df['oldbalanceDest'])
        
    #     df['amount_ratio_orig'] = df['amount'] / (df['oldbalanceOrg'] + eps)
    #     df['amount_ratio_dest'] = df['amount'] / (df['oldbalanceDest'] + eps)
        
    #     df['zero_balance_orig'] = ((df['oldbalanceOrg'] == 0) | (df['newbalanceOrig'] == 0)).astype(int)
    #     df['zero_balance_dest'] = ((df['oldbalanceDest'] == 0) | (df['newbalanceDest'] == 0)).astype(int)
    #     df['full_transfer'] = (df['newbalanceOrig'] == 0).astype(int)
    #     df['balance_mismatch'] = (abs(df['oldbalanceOrg'] - df['newbalanceOrig']) != df['amount']).astype(int)
        
    #     if 'step' in df.columns:
    #         df['hour'] = df['step'] % 24
    #         df['day'] = df['step'] // 24
    #     else:
    #         df['hour'] = 0
    #         df['day'] = 0
            
    #     df['dest_is_merchant'] = df['nameDest'].str.startswith('M').astype(int)
        
    #     return df

    # def fit_transform(self, df: pd.DataFrame, sampling_strategy: str = 'none') -> Tuple[np.ndarray, np.ndarray]:
    #     df_processed = self._engineer_features(df)
        
    #     df_encoded = pd.get_dummies(df_processed, columns=['type'], prefix='type')
        
    #     for type_col in [f'type_{cat}' for cat in self.type_categories]:
    #         if type_col not in df_encoded.columns:
    #             df_encoded[type_col] = 0
                
    #     X = df_encoded[self.feature_names].values
    #     y = df['isFraud'].values
        
    #     X = np.nan_to_num(X)
        
    #     if sampling_strategy != 'none':
    #         X, y = self._handle_imbalanced_data(X, y, sampling_strategy)
        
    #     return X, y

    # def transform_single(self, transaction: Dict) -> np.ndarray:
    #     df = pd.DataFrame([transaction])
    #     df_processed = self._engineer_features(df)
        
    #     df_encoded = pd.get_dummies(df_processed, columns=['type'], prefix='type')
        
    #     for type_col in [f'type_{cat}' for cat in self.type_categories]:
    #         if type_col not in df_encoded.columns:
    #             df_encoded[type_col] = 0
        
    #     X = df_encoded[self.feature_names].values
        
    #     X = np.nan_to_num(X)
        
    #     return X

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