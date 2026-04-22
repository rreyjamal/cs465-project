"""
Evaluation utilities for machine learning models.

This module provides functions for evaluating models, generating
metrics, and creating visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# seaborn not used to minimize dependencies
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.pipeline import Pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> Dict[str, Any]:
    """
    Evaluate a single model on test data.
    
    Args:
        pipeline: Trained model pipeline
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
        
    Returns:
        Dictionary containing evaluation results
    """
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm,
        'roc_curve': {'fpr': fpr, 'tpr': tpr}
    }
    
    # Print results
    print(f"\n=== {model_name} EVALUATION ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Good Credit', 'Bad Credit']))
    
    return results


def evaluate_all_models(
    trained_pipelines: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all trained models.
    
    Args:
        trained_pipelines: Dictionary of trained model pipelines
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary containing evaluation results for all models
    """
    print("=== MODEL EVALUATION ===")
    
    all_results = {}
    
    for name, pipeline in trained_pipelines.items():
        results = evaluate_model(pipeline, X_test, y_test, name)
        all_results[name] = results
    
    return all_results


def create_comparison_table(evaluation_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a comparison table of all model metrics.
    
    Args:
        evaluation_results: Dictionary of evaluation results
        
    Returns:
        DataFrame with model comparison
    """
    comparison_data = []
    
    for model_name, results in evaluation_results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Balanced Accuracy': results.get('balanced_accuracy', None),
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score'],
            'ROC-AUC': results['roc_auc']
        })
    
    comparison_df = pd.DataFrame(comparison_data).round(4)
    
    print("\n=== MODEL COMPARISON TABLE ===")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def plot_confusion_matrices(
    evaluation_results: Dict[str, Dict[str, Any]],
    save_path: str = None
) -> plt.Figure:
    """
    Plot confusion matrices for all models.
    
    Args:
        evaluation_results: Dictionary of evaluation results
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    n_models = len(evaluation_results)
    cols = 2
    rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, (model_name, results) in enumerate(evaluation_results.items()):
        cm = results['confusion_matrix']
        
        axes[i].imshow(cm, cmap='Blues')
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(['Good', 'Bad'])
        axes[i].set_yticklabels(['Good', 'Bad'])
        
        # Add text annotations
        for text_row in range(len(cm)):
            for text_col in range(len(cm[text_row])):
                axes[i].text(text_col, text_row, str(cm[text_row][text_col]),
                            ha='center', va='center', fontweight='bold')
        
        axes[i].set_title(f'{model_name} Confusion Matrix', fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to {save_path}")
    
    return fig


def plot_roc_curves(
    evaluation_results: Dict[str, Dict[str, Any]],
    save_path: str = None
) -> plt.Figure:
    """
    Plot ROC curves for all models.
    
    Args:
        evaluation_results: Dictionary of evaluation results
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, results in evaluation_results.items():
        roc_data = results['roc_curve']
        auc_score = results['roc_auc']
        
        plt.plot(roc_data['fpr'], roc_data['tpr'], 
                label=f'{model_name} (AUC = {auc_score:.3f})', 
                linewidth=2)
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    return plt.gcf()


def save_evaluation_results(
    comparison_df: pd.DataFrame,
    evaluation_results: Dict[str, Dict[str, Any]],
    output_dir: str = 'results/metrics'
) -> None:
    """
    Save all evaluation results to files.
    
    Args:
        comparison_df: DataFrame with model comparison metrics
        evaluation_results: Dictionary of evaluation results
        output_dir: Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save comparison table
    comparison_df.to_csv(f'{output_dir}/model_comparison.csv', index=False)
    
    # Save detailed results
    detailed_results = []
    for model_name, results in evaluation_results.items():
        detailed_results.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1_Score': results['f1_score'],
            'ROC_AUC': results['roc_auc']
        })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(f'{output_dir}/detailed_results.csv', index=False)
    
    print(f"\nEvaluation results saved to {output_dir}/")


if __name__ == "__main__":
    # Example usage
    print("Testing evaluation module...")
    print("Evaluation module ready for use!")
