"""
Training utilities for machine learning models.

This module provides functions for training multiple models
with consistent preprocessing and evaluation setup.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def create_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Create a dictionary of models with reasonable default hyperparameters.
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary mapping model names to sklearn model instances
    """
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=random_state,
            class_weight='balanced',
            max_depth=10
        ),
        'Random Forest': RandomForestClassifier(
            random_state=random_state,
            class_weight='balanced',
            n_estimators=100,
            max_depth=10
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=random_state,
            n_estimators=100,
            max_depth=6
        )
    }
    
    print("Models created with the following configurations:")
    for name, model in models.items():
        print(f"- {name}: {model.__class__.__name__}")
        if hasattr(model, 'class_weight') and model.class_weight == 'balanced':
            print(f"  Using class_weight='balanced' to handle imbalance")
    
    return models


def create_model_pipelines(
    models: Dict[str, Any], 
    preprocessor
) -> Dict[str, Pipeline]:
    """
    Create full pipelines that combine preprocessing with models.
    
    Args:
        models: Dictionary of models
        preprocessor: Fitted preprocessing pipeline
        
    Returns:
        Dictionary mapping model names to sklearn pipelines
    """
    pipelines = {}
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipelines[name] = pipeline
    
    print(f"Created {len(pipelines)} model pipelines")
    return pipelines


def train_models(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Dict[str, Pipeline]:
    """
    Train all models on the training data.
    
    Args:
        pipelines: Dictionary of model pipelines
        X_train: Training features
        y_train: Training target
        
    Returns:
        Dictionary of trained model pipelines
    """
    trained_pipelines = {}
    training_scores = {}
    
    print("=== TRAINING MODELS ===")
    
    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Calculate training accuracy
        train_score = pipeline.score(X_train, y_train)
        
        # Store results
        trained_pipelines[name] = pipeline
        training_scores[name] = train_score
        
        print(f"{name} training accuracy: {train_score:.4f}")
    
    print(f"\nAll {len(trained_pipelines)} models trained successfully!")
    
    return trained_pipelines, training_scores


def perform_cross_validation(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Perform cross-validation on training data.
    
    Args:
        pipelines: Dictionary of trained model pipelines
        X_train: Training features
        y_train: Training target
        cv_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with cross-validation results for each model
    """
    print(f"=== {cv_folds}-FOLD CROSS-VALIDATION ===")
    
    cv_results = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    for name, pipeline in pipelines.items():
        print(f"\nCross-validating {name}...")
        
        # Perform cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        
        cv_results[name] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
        
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Individual scores: {[f'{score:.4f}' for score in cv_scores]}")
    
    return cv_results


def get_model_hyperparameters(models: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract hyperparameters from all models.
    
    Args:
        models: Dictionary of models
        
    Returns:
        Dictionary of model hyperparameters
    """
    hyperparameters = {}
    
    for name, model in models.items():
        hyperparameters[name] = model.get_params()
    
    return hyperparameters


def suggest_hyperparameter_grids() -> Dict[str, Dict[str, List[Any]]]:
    """
    Suggest hyperparameter grids for tuning.
    
    Returns:
        Dictionary of hyperparameter grids for each model
    """
    param_grids = {
        'Logistic Regression': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['liblinear', 'lbfgs']
        },
        'Decision Tree': {
            'classifier__max_depth': [5, 10, 15, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        },
        'Random Forest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, 10, 15, None],
            'classifier__min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 6, 10],
            'classifier__learning_rate': [0.01, 0.1, 0.2]
        }
    }
    
    return param_grids


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create train-test split with optional stratification.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of data for testing
        random_state: Random seed
        stratify: Whether to stratify by target
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    print("=== TRAIN-TEST SPLIT ===")
    print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    if stratify:
        print(f"\nTraining target distribution:")
        print(y_train.value_counts().sort_index())
        print(f"\nTest target distribution:")
        print(y_test.value_counts().sort_index())
    
    return X_train, X_test, y_train, y_test


def get_feature_importance_from_pipeline(
    pipeline: Pipeline,
    feature_names: List[str],
    model_name: str
) -> pd.DataFrame:
    """
    Extract feature importance from a trained pipeline.
    
    Args:
        pipeline: Trained pipeline
        feature_names: List of feature names after preprocessing
        model_name: Name of the model
        
    Returns:
        DataFrame with feature importance
    """
    model = pipeline.named_steps['classifier']
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
        importance_type = 'feature_importance'
    elif hasattr(model, 'coef_'):
        # Linear models
        importances = np.abs(model.coef_[0])
        importance_type = 'absolute_coefficient'
    else:
        print(f"Model {model_name} does not have feature importance or coefficients")
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'importance_type': importance_type
    }).sort_values('importance', ascending=False)
    
    return importance_df


if __name__ == "__main__":
    # Example usage
    print("Testing training module...")
    
    # Create models
    models = create_models()
    
    # Get hyperparameters
    hyperparams = get_model_hyperparameters(models)
    
    # Suggest parameter grids
    param_grids = suggest_hyperparameter_grids()
    
    print("Training module ready for use!")
