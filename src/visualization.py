# src/visualization.py

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def create_confusion_matrix_plot(confusion_matrix, title="Confusion Matrix"):
    labels = ['Not Fraud', 'Fraud']
    
    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=labels,
        y=labels,
        text=confusion_matrix,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False,
        colorscale='Blues'))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted label",
        yaxis_title="True label",
        width=500,
        height=400
    )
    
    return fig

def create_feature_importance_plot(importances, feature_names, title="Feature Importance"):
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    fig = px.bar(
        importance_df,
        x='Feature',
        y='Importance',
        title=title
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        width=800,
        height=400
    )
    
    return fig