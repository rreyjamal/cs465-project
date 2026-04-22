"""
Explainability utilities for machine learning models.

This module provides functions for model explainability including
feature importance extraction and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from sklearn.pipeline import Pipeline

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def extract_feature_importance(
    pipeline: Pipeline,
    feature_names: List[str],
    model_name: str
) -> pd.DataFrame:
    """
    Extract feature importance from a trained pipeline.
    
    Args:
        pipeline: Trained model pipeline
        feature_names: List of feature names after preprocessing
        model_name: Name of the model
        
    Returns:
        DataFrame with feature importance
    """
    model = pipeline.named_steps['classifier']
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models (Random Forest, Gradient Boosting, Decision Tree)
        importances = model.feature_importances_
        importance_type = 'feature_importance'
        method = 'Tree-based Feature Importance'
        
    elif hasattr(model, 'coef_'):
        # Linear models (Logistic Regression)
        # Use absolute coefficients for importance ranking
        importances = np.abs(model.coef_[0])
        importance_type = 'absolute_coefficient'
        method = 'Absolute Coefficient Importance'
        
        # Also store actual coefficients for interpretation
        coefficients = model.coef_[0]
        
    else:
        print(f"Model {model_name} does not have standard feature importance or coefficients")
        return pd.DataFrame()
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'importance_type': importance_type,
        'method': method
    }).sort_values('importance', ascending=False)
    
    # Add actual coefficients for linear models
    if hasattr(model, 'coef_'):
        importance_df['coefficient'] = coefficients
    
    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    top_n: int = 15,
    save_path: str = None
) -> plt.Figure:
    """
    Plot feature importance for a model.
    
    Args:
        importance_df: DataFrame with feature importance
        model_name: Name of the model
        top_n: Number of top features to show
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(12, 8))
    
    # Get top features
    top_features = importance_df.head(top_n)
    
    # Determine colors based on importance type
    if 'coefficient' in top_features.columns:
        # Linear model - color by coefficient sign
        colors = ['red' if x < 0 else 'green' for x in top_features['coefficient']]
        label = 'Absolute Coefficient Value'
        
        # Create legend
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Negative coefficient (bad credit)')
        green_patch = mpatches.Patch(color='green', label='Positive coefficient (good credit)')
        has_legend = True
        
    else:
        # Tree-based model - single color
        colors = 'skyblue'
        label = 'Feature Importance'
        has_legend = False
    
    # Create horizontal bar plot
    if isinstance(colors, list):
        plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    else:
        plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel(label, fontsize=12)
    plt.title(f'{model_name} - Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Show most important at top
    plt.grid(True, alpha=0.3)
    
    # Add legend for linear models
    if has_legend:
        plt.legend(handles=[green_patch, red_patch])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    return plt.gcf()


if __name__ == "__main__":
    # Example usage
    print("Testing explainability module...")
    
    if SHAP_AVAILABLE:
        print("SHAP is available for explainability analysis")
    else:
        print("SHAP is not available - some explainability features will be limited")
    
    print("Explainability module ready for use!")
