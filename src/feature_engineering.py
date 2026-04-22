"""
Feature engineering utilities for German Credit Dataset.

This module provides functions for building preprocessing pipelines
and feature engineering operations.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Any
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_preprocessing_pipeline(
    numerical_features: List[str], 
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Build a preprocessing pipeline for mixed data types.
    
    Args:
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        
    Returns:
        Configured ColumnTransformer for preprocessing
    """
    # Numerical preprocessing: Standardization
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing: One-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    print("Preprocessing pipeline created:")
    print(f"- Numeric features ({len(numerical_features)}): StandardScaler")
    print(f"- Categorical features ({len(categorical_features)}): OneHotEncoder")
    
    return preprocessor


def get_feature_names_after_preprocessing(
    preprocessor: ColumnTransformer,
    numerical_features: List[str],
    categorical_features: List[str]
) -> List[str]:
    """
    Get feature names after preprocessing (after one-hot encoding).
    
    Args:
        preprocessor: Fitted ColumnTransformer
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        
    Returns:
        List of feature names after preprocessing
    """
    # Get numerical feature names (unchanged)
    num_feature_names = numerical_features
    
    # Get categorical feature names (after one-hot encoding)
    cat_feature_names = (preprocessor
                        .named_transformers_['cat']
                        .named_steps['onehot']
                        .get_feature_names_out(categorical_features)
                        .tolist())
    
    # Combine all feature names
    all_feature_names = num_feature_names + cat_feature_names
    
    print(f"Total features after preprocessing: {len(all_feature_names)}")
    print(f"Numerical features: {len(num_feature_names)}")
    print(f"Categorical features (one-hot): {len(cat_feature_names)}")
    
    return all_feature_names


def create_feature_analysis_dataframe(
    df: pd.DataFrame,
    target_col: str = 'target'
) -> pd.DataFrame:
    """
    Create a DataFrame for feature analysis with basic statistics.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        DataFrame with feature analysis information
    """
    feature_analysis = []
    
    for col in df.columns:
        if col != target_col:
            info = {
                'feature': col,
                'dtype': str(df[col].dtype),
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': (df[col].nunique() / len(df)) * 100
            }
            
            # Add numeric-specific statistics
            if df[col].dtype in ['int64', 'float64']:
                info.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                })
            
            feature_analysis.append(info)
    
    analysis_df = pd.DataFrame(feature_analysis)
    return analysis_df


def analyze_feature_correlation(
    df: pd.DataFrame, 
    target_col: str = 'target',
    threshold: float = 0.7
) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    """
    Analyze correlations between numerical features.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        threshold: Correlation threshold for identifying highly correlated features
        
    Returns:
        Tuple of (correlation_matrix, highly_correlated_pairs)
    """
    # Get numerical features only
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Find highly correlated pairs
    highly_correlated = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value > threshold:
                highly_correlated.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_value
                ))
    
    print(f"Found {len(highly_correlated)} highly correlated feature pairs (threshold: {threshold})")
    
    return corr_matrix, highly_correlated


def suggest_feature_engineering_steps(
    df: pd.DataFrame,
    target_col: str = 'target'
) -> List[str]:
    """
    Suggest potential feature engineering steps based on data analysis.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        List of suggested feature engineering steps
    """
    suggestions = []
    
    # Analyze data characteristics
    categorical_cols, numerical_cols = identify_column_types(df, target_col)
    
    # Missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        suggestions.append(f"Handle missing values in {len(missing_cols)} columns: {missing_cols}")
    
    # High cardinality categorical features
    for col in categorical_cols:
        if df[col].nunique() > 20:
            suggestions.append(f"Consider feature grouping for high-cardinality feature: {col}")
    
    # Skewed numerical features
    for col in numerical_cols:
        skewness = df[col].skew()
        if abs(skewness) > 2:
            suggestions.append(f"Consider log transformation for skewed feature: {col} (skewness: {skewness:.2f})")
    
    # Correlated features
    _, highly_correlated = analyze_feature_correlation(df, target_col, 0.8)
    if highly_correlated:
        suggestions.append(f"Consider feature selection for correlated pairs: {highly_correlated}")
    
    if not suggestions:
        suggestions.append("No major feature engineering issues detected")
    
    return suggestions


def identify_column_types(df: pd.DataFrame, target_col: str = 'target') -> Tuple[List[str], List[str]]:
    """
    Identify categorical and numerical columns.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        Tuple of (categorical_columns, numerical_columns)
    """
    features_df = df.drop(columns=[target_col])
    
    categorical_columns = features_df.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    return categorical_columns, numerical_columns


def create_domain_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific features for credit risk analysis.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with additional engineered features
    """
    df_engineered = df.copy()
    
    # Credit amount to duration ratio (monthly payment proxy)
    if 'credit_amount' in df.columns and 'duration_months' in df.columns:
        df_engineered['credit_amount_per_month'] = df['credit_amount'] / df['duration_months']
    
    # Age groups (if age is available)
    if 'age_years' in df.columns:
        df_engineered['age_group'] = pd.cut(
            df['age_years'], 
            bins=[0, 25, 35, 50, 100], 
            labels=['Young', 'Adult', 'Middle-aged', 'Senior']
        )
    
    # Installment rate categories
    if 'installment_rate' in df.columns:
        df_engineered['installment_rate_category'] = pd.cut(
            df['installment_rate'],
            bins=[0, 2, 3, 4, 5],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
    
    print("Domain-specific features created:")
    new_features = set(df_engineered.columns) - set(df.columns)
    for feature in new_features:
        print(f"- {feature}")
    
    return df_engineered


if __name__ == "__main__":
    # Example usage
    print("Testing feature engineering module...")
    
    # This would require actual data to test
    print("Feature engineering module ready for use!")
