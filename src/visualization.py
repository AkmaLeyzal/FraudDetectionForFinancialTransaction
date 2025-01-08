import plotly.graph_objects as go
import numpy as np
from typing import List, Optional
import plotly.express as px

class ModelVisualizer:
    """Visualization tools for sklearn model analysis"""
    
    @staticmethod
    def create_feature_importance_plot(
        importances: np.ndarray,
        feature_names: List[str],
        title: str = "Feature Importance"
    ) -> go.Figure:
        """Create feature importance plot using plotly"""
        if len(importances) != len(feature_names):
            raise ValueError(
                f"Length mismatch: importances ({len(importances)}) != "
                f"feature_names ({len(feature_names)})"
            )
        
        # Sort features by importance
        sorted_idx = np.argsort(importances)
        pos = np.arange(len(sorted_idx)) + .5
        
        fig = go.Figure([
            go.Bar(
                x=importances[sorted_idx],
                y=[feature_names[i] for i in sorted_idx],
                orientation='h',
                marker=dict(
                    color='royalblue',
                    line=dict(color='rgb(8,48,107)', width=1.5)
                )
            )
        ])
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(feature_names) * 20),
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            yaxis=dict(autorange="reversed")
        )
        
        return fig

    @staticmethod
    def create_confusion_matrix_plot(
        confusion_matrix: np.ndarray,
        title: str = "Confusion Matrix"
    ) -> go.Figure:
        """Create confusion matrix plot using plotly"""
        labels = ['Not Fraud', 'Fraud']
        
        # Create annotations
        annotations = []
        for i in range(2):
            for j in range(2):
                value = confusion_matrix[i, j]
                text_color = 'white' if value > confusion_matrix.max()/2 else 'black'
                annotations.append(dict(
                    x=j,
                    y=i,
                    text=str(value),
                    font=dict(color=text_color, size=14),
                    showarrow=False,
                    align='center'
                ))
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=labels,
            y=labels,
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Predicted label",
            yaxis_title="True label",
            xaxis=dict(tickmode='array', ticktext=labels, tickvals=[0, 1]),
            yaxis=dict(tickmode='array', ticktext=labels, tickvals=[0, 1]),
            annotations=annotations,
            height=400,
            width=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig

    @staticmethod
    def create_roc_curve_plot(
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float,
        title: str = "ROC Curve"
    ) -> go.Figure:
        """Create ROC curve plot using plotly"""
        fig = go.Figure()
        
        # Add ROC curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC (AUC = {auc:.3f})',
            line=dict(color='royalblue', width=2)
        ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1], gridcolor='lightgray'),
            yaxis=dict(range=[0, 1], gridcolor='lightgray'),
            height=400,
            width=400,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            legend=dict(x=0.1, y=0.9),
            plot_bgcolor='white'
        )
        
        return fig
    
    @staticmethod
    def create_prediction_distribution_plot(
        y_prob: np.ndarray,
        y_true: np.ndarray,
        title: str = "Prediction Score Distribution"
    ) -> go.Figure:
        """Create prediction distribution plot using plotly"""
        fig = go.Figure()
        
        # Add histograms for fraud and non-fraud predictions
        fig.add_trace(go.Histogram(
            x=y_prob[y_true == 0],
            name='Not Fraud',
            opacity=0.75,
            nbinsx=50,
            marker_color='blue'
        ))
        
        fig.add_trace(go.Histogram(
            x=y_prob[y_true == 1],
            name='Fraud',
            opacity=0.75,
            nbinsx=50,
            marker_color='red'
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Prediction Score",
            yaxis_title="Count",
            barmode='overlay',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            legend=dict(x=0.7, y=0.9),
            plot_bgcolor='white'
        )
        
        return fig