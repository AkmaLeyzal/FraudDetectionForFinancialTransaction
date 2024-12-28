# src/visualization.py
import plotly.graph_objects as go
import numpy as np
from typing import List
from collections import Counter

class ModelVisualizer:
    """Visualization tools for model analysis"""
    
    @staticmethod
    def create_feature_importance_plot(importances: np.ndarray, feature_names: List[str], title: str = "Feature Importance") -> go.Figure:
        if len(importances) != len(feature_names):
            raise ValueError(
                f"Length mismatch: importances ({len(importances)}) != "
                f"feature_names ({len(feature_names)})"
            )
        
        indices = np.argsort(importances)
        sorted_importances = importances[indices]
        sorted_features = [feature_names[i] for i in indices]
        
        fig = go.Figure(data=[
            go.Bar(
                x=sorted_importances,
                y=sorted_features,
                orientation='h',
                marker_color='royalblue'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=max(400, len(feature_names) * 20),
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        
        return fig

    @staticmethod
    def create_confusion_matrix_plot(confusion_matrix: np.ndarray, title: str = "Confusion Matrix") -> go.Figure:
        labels = ['Not Fraud', 'Fraud']
        
        annotations = []
        for i in range(2):
            for j in range(2):
                color = 'white' if confusion_matrix[i, j] > confusion_matrix.max()/2 else 'black'
                annotations.append(dict(
                    x=j,
                    y=i,
                    text=str(confusion_matrix[i, j]),
                    font=dict(color=color),
                    showarrow=False,
                    align='center'
                ))
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=labels,
            y=labels,
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Predicted label',
            yaxis_title='True label',
            xaxis=dict(tickmode='array', ticktext=labels, tickvals=[0, 1]),
            yaxis=dict(tickmode='array', ticktext=labels, tickvals=[0, 1]),
            annotations=annotations,
            height=400,
            width=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig

    @staticmethod
    def create_roc_curve_plot(fpr: np.ndarray, tpr: np.ndarray, auc: float, title: str = "ROC Curve") -> go.Figure:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, 
            y=tpr,
            mode='lines',
            name=f'ROC (AUC = {auc:.3f})',
            line=dict(color='royalblue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            height=400,
            width=400,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            legend=dict(x=0.1, y=0.9)
        )
        
        return fig