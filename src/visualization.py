import plotly.graph_objects as go
import numpy as np
import pandas as pd

def create_feature_importance_plot(importances, feature_names, title):
    if len(importances) != len(feature_names):
        raise ValueError(f"Length mismatch: importances ({len(importances)}) != feature_names ({len(feature_names)})")
    
    sorted_idx = np.argsort(importances)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]
    
    fig = go.Figure([
        go.Bar(
            x=sorted_importances,
            y=sorted_features,
            orientation='h'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title='Importance Score',
        yaxis_title='Features',
        height=600,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_confusion_matrix_plot(confusion_matrix, title):
    labels = ['Not Fraud', 'Fraud']
    
    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    x=i,
                    y=j,
                    text=str(confusion_matrix[j, i]),
                    font=dict(color='white' if confusion_matrix[j, i] > confusion_matrix.max()/2 else 'black'),
                    showarrow=False
                )
            )
    
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
        annotations=annotations,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig
