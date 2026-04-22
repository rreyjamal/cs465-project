"""
Data preprocessing utilities for German Credit Dataset.

This module provides functions for loading, cleaning, and preprocessing
the UCI German Credit dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def load_german_credit(file_path: str = "data/raw/german.data") -> pd.DataFrame:
    """
    Load the German Credit dataset with proper column names and target mapping.
    
    Args:
        file_path: Path to the german.data file
        
    Returns:
        DataFrame with loaded data and proper column names
    """
    # Define column names as specified in the project requirements
    column_names = [
        "status_checking_account",
        "duration_months",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings_account",
        "employment_since",
        "installment_rate",
        "personal_status_sex",
        "other_debtors_guarantors",
        "present_residence_since",
        "property",
        "age_years",
        "other_installment_plans",
        "housing",
        "existing_credits",
        "job",
        "num_dependents",
        "telephone",
        "foreign_worker",
        "target"
    ]
    
    # Load the dataset
    try:
        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=column_names)
        print(f"Dataset loaded successfully from {file_path}")
        print(f"Shape: {df.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the dataset file at {file_path}")
    
    # Map target values: 1 -> 0 (good credit), 2 -> 1 (bad credit)
    df['target'] = df['target'].map({1: 0, 2: 1})
    
    print("Target mapping applied: 1->0 (good), 2->1 (bad)")
    
    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Perform basic data quality checks.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing quality metrics
    """
    quality_report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'target_distribution': df['target'].value_counts().to_dict()
    }
    
    # Print summary
    print("=== DATA QUALITY REPORT ===")
    print(f"Shape: {quality_report['shape']}")
    print(f"Missing values: {sum(quality_report['missing_values'].values())}")
    print(f"Duplicate rows: {quality_report['duplicate_rows']}")
    print(f"Target distribution: {quality_report['target_distribution']}")
    
    return quality_report


def identify_column_types(df: pd.DataFrame, target_col: str = 'target') -> Tuple[List[str], List[str]]:
    """
    Identify categorical and numerical columns.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        Tuple of (categorical_columns, numerical_columns)
    """
    # Exclude target column from feature identification
    features_df = df.drop(columns=[target_col])
    
    categorical_columns = features_df.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Identified {len(categorical_columns)} categorical columns")
    print(f"Identified {len(numerical_columns)} numerical columns")
    
    return categorical_columns, numerical_columns


def split_features_target(df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        Tuple of (features_df, target_series)
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y


def clean_data(df: pd.DataFrame, remove_duplicates: bool = True) -> pd.DataFrame:
    """
    Perform basic data cleaning operations.
    
    Args:
        df: Input DataFrame
        remove_duplicates: Whether to remove duplicate rows
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Remove duplicates if requested
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_rows = initial_rows - len(df_clean)
        print(f"Removed {removed_rows} duplicate rows")
    
    # Check for any obvious data issues
    # (This can be extended based on specific dataset requirements)
    
    print(f"Data cleaning complete. Final shape: {df_clean.shape}")
    return df_clean


def get_target_info(y: pd.Series) -> dict:
    """
    Get information about target variable.
    
    Args:
        y: Target series
        
    Returns:
        Dictionary with target information
    """
    target_info = {
        'counts': y.value_counts().to_dict(),
        'percentages': (y.value_counts(normalize=True) * 100).to_dict(),
        'classes': y.unique().tolist(),
        'is_binary': len(y.unique()) == 2
    }
    
    print("=== TARGET INFORMATION ===")
    for class_val in sorted(target_info['counts'].keys()):
        label = "Good Credit (0)" if class_val == 0 else "Bad Credit (1)"
        count = target_info['counts'][class_val]
        percentage = target_info['percentages'][class_val]
        print(f"{label}: {count} ({percentage:.1f}%)")
    
    return target_info


def preprocess_pipeline_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the preprocessing pipeline.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with preprocessing summary
    """
    categorical_cols, numerical_cols = identify_column_types(df)
    target_info = get_target_info(df['target'])
    quality_report = check_data_quality(df)
    
    summary = {
        'total_samples': len(df),
        'total_features': len(df.columns) - 1,  # Exclude target
        'categorical_features': len(categorical_cols),
        'numerical_features': len(numerical_cols),
        'target_distribution': target_info,
        'data_quality': quality_report
    }
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("Testing data preprocessing module...")
    
    # Load data
    df = load_german_credit("../data/raw/german.data")
    
    # Check data quality
    quality_report = check_data_quality(df)
    
    # Identify column types
    cat_cols, num_cols = identify_column_types(df)
    
    # Split features and target
    X, y = split_features_target(df)
    
    print("Data preprocessing module test complete!")
