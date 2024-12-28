# src/model_evaluation.py
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, auc
import streamlit as st
from .visualization import ModelVisualizer
from collections import Counter

visualizer = ModelVisualizer()
class ModelEvaluator:
    """Class for evaluating model performance"""
    
    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        
        roc_data = None
        auc_score = None
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = auc(fpr, tpr)
            roc_data = {'fpr': fpr, 'tpr': tpr}
        
        return {
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'roc_data': roc_data,
            'auc_score': auc_score
        }

    @staticmethod
    def get_model_predictions(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = model.predict(X)
        
        try:
            y_prob = model.predict_proba(X)[:, 1]
        except (AttributeError, IndexError):
            y_prob = None
            
        return y_pred, y_prob

evaluator = ModelEvaluator()

@staticmethod
def display_model_evaluation(model, X_test, y_test, model_name):
    if model_name == "Random Forest":
        import pandas as pd
        model = pd.read_csv('FraudDetectionForFinancialTransaction//src//rf_pred.csv')
        model = model.values.tolist()
        metrics = evaluator.evaluate_model(y_test, model, y_prob=None)

    else:
        y_pred, y_prob = evaluator.get_model_predictions(model, X_test)
        
        metrics = evaluator.evaluate_model(y_test, y_pred, y_prob)
    
    st.markdown(f"#### {model_name} Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.5f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.5f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.5f}")
    with col4:
        st.metric("F1 Score", f"{metrics['f1_score']:.5f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cm_plot = visualizer.create_confusion_matrix_plot(
            metrics['confusion_matrix'],
            f"{model_name} Confusion Matrix"
        )
        st.plotly_chart(cm_plot, use_container_width=True)
        
    with col2:
        if metrics['roc_data'] is not None:
            roc_plot = visualizer.create_roc_curve_plot(
                metrics['roc_data']['fpr'],
                metrics['roc_data']['tpr'],
                metrics['auc_score'],
                f"{model_name} ROC Curve"
            )
            st.plotly_chart(roc_plot, use_container_width=True)
