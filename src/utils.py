"""
General utility functions for the credit risk ML project.

This module provides common utility functions for file operations,
directory management, and other helper functions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings


def create_directories(directories: List[str]) -> None:
    """
    Create multiple directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created/verified: {directory}")


def save_dataframe(
    df: pd.DataFrame,
    file_path: str,
    index: bool = False,
    create_dir: bool = True
) -> None:
    """
    Save a DataFrame to CSV with error handling.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        index: Whether to save the index
        create_dir: Whether to create the directory if it doesn't exist
    """
    try:
        if create_dir:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        df.to_csv(file_path, index=index)
        print(f"DataFrame saved to: {file_path}")
    except Exception as e:
        print(f"Error saving DataFrame to {file_path}: {e}")


def load_dataframe(file_path: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Load a DataFrame from CSV with error handling.
    
    Args:
        file_path: Input file path
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        DataFrame or None if loading fails
    """
    try:
        df = pd.read_csv(file_path, **kwargs)
        print(f"DataFrame loaded from: {file_path}")
        return df
    except Exception as e:
        print(f"Error loading DataFrame from {file_path}: {e}")
        return None


def save_json(data: Any, file_path: str, create_dir: bool = True) -> None:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Data to save (must be JSON serializable)
        file_path: Output file path
        create_dir: Whether to create the directory if it doesn't exist
    """
    try:
        if create_dir:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"JSON data saved to: {file_path}")
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")


def load_json(file_path: str) -> Optional[Any]:
    """
    Load data from JSON file with error handling.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded data or None if loading fails
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"JSON data loaded from: {file_path}")
        return data
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return None


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    # Add other random seed settings if needed
    print(f"Random seed set to: {seed}")


def suppress_warnings(suppress: bool = True) -> None:
    """
    Suppress or enable warnings.
    
    Args:
        suppress: Whether to suppress warnings
    """
    if suppress:
        warnings.filterwarnings('ignore')
        print("Warnings suppressed")
    else:
        warnings.filterwarnings('default')
        print("Warnings enabled")


def setup_matplotlib_style(style: str = 'default', figsize: tuple = (10, 6), fontsize: int = 12) -> None:
    """
    Set up matplotlib plotting style.
    
    Args:
        style: Matplotlib style to use
        figsize: Default figure size
        fontsize: Default font size
    """
    plt.style.use(style)
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['axes.titlesize'] = fontsize + 2
    plt.rcParams['axes.labelsize'] = fontsize
    plt.rcParams['xtick.labelsize'] = fontsize - 2
    plt.rcParams['ytick.labelsize'] = fontsize - 2
    plt.rcParams['legend.fontsize'] = fontsize - 2
    print(f"Matplotlib style set to: {style}")


def save_figure(
    fig: plt.Figure,
    file_path: str,
    dpi: int = 300,
    bbox_inches: str = 'tight',
    create_dir: bool = True
) -> None:
    """
    Save a matplotlib figure with error handling.
    
    Args:
        fig: Matplotlib figure to save
        file_path: Output file path
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box setting
        create_dir: Whether to create the directory if it doesn't exist
    """
    try:
        if create_dir:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        fig.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Figure saved to: {file_path}")
    except Exception as e:
        print(f"Error saving figure to {file_path}: {e}")


def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Human-readable file size string
    """
    try:
        size_bytes = os.path.getsize(file_path)
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.1f} TB"
    except OSError:
        return "Unknown"


def list_files_in_directory(directory: str, pattern: str = "*") -> List[str]:
    """
    List files in a directory matching a pattern.
    
    Args:
        directory: Directory path
        pattern: File pattern to match
        
    Returns:
        List of file paths
    """
    try:
        path = Path(directory)
        files = list(path.glob(pattern))
        return [str(f) for f in files if f.is_file()]
    except Exception as e:
        print(f"Error listing files in {directory}: {e}")
        return []


def validate_file_path(file_path: str) -> bool:
    """
    Validate if a file path exists and is accessible.
    
    Args:
        file_path: Path to validate
        
    Returns:
        True if file exists and is accessible, False otherwise
    """
    try:
        return os.path.isfile(file_path) and os.access(file_path, os.R_OK)
    except Exception:
        return False


def create_project_summary(
    project_dir: str,
    results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Create a summary of the project structure and files.
    
    Args:
        project_dir: Root project directory
        results_dir: Results directory name
        
    Returns:
        Dictionary with project summary
    """
    summary = {
        'project_directory': project_dir,
        'directories': {},
        'files': {},
        'total_size': 0
    }
    
    try:
        # Walk through project directory
        for root, dirs, files in os.walk(project_dir):
            # Skip hidden directories and common cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', '.ipynb_checkpoints']]
            
            rel_root = os.path.relpath(root, project_dir)
            
            # Add directories
            for dir_name in dirs:
                dir_path = os.path.join(rel_root, dir_name)
                summary['directories'][dir_path] = {
                    'type': 'directory',
                    'path': dir_path
                }
            
            # Add files
            for file_name in files:
                if not file_name.startswith('.'):
                    file_path = os.path.join(root, file_name)
                    rel_file_path = os.path.relpath(file_path, project_dir)
                    
                    file_size = os.path.getsize(file_path)
                    summary['files'][rel_file_path] = {
                        'type': 'file',
                        'path': rel_file_path,
                        'size_bytes': file_size,
                        'size_readable': get_file_size(file_path)
                    }
                    summary['total_size'] += file_size
    
    except Exception as e:
        print(f"Error creating project summary: {e}")
    
    return summary


def print_project_summary(summary: Dict[str, Any]) -> None:
    """
    Print a formatted project summary.
    
    Args:
        summary: Project summary dictionary
    """
    print("=== PROJECT SUMMARY ===")
    print(f"Project Directory: {summary['project_directory']}")
    print(f"Total Files: {len(summary['files'])}")
    print(f"Total Directories: {len(summary['directories'])}")
    print(f"Total Size: {get_file_size_by_bytes(summary['total_size'])}")
    
    print("\n=== KEY FILES ===")
    key_files = ['README.md', 'requirements.txt', '.gitignore']
    for key_file in key_files:
        if key_file in summary['files']:
            file_info = summary['files'][key_file]
            print(f"✓ {key_file}: {file_info['size_readable']}")
        else:
            print(f"✗ {key_file}: Not found")
    
    print("\n=== DIRECTORIES ===")
    for dir_path, dir_info in summary['directories'].items():
        print(f"📁 {dir_path}/")


def get_file_size_by_bytes(size_bytes: int) -> str:
    """
    Convert bytes to human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def backup_results(results_dir: str, backup_suffix: str = "_backup") -> None:
    """
    Create a backup of the results directory.
    
    Args:
        results_dir: Directory to backup
        backup_suffix: Suffix to add to backup directory
    """
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist")
        return
    
    backup_dir = results_dir + backup_suffix
    try:
        import shutil
        shutil.copytree(results_dir, backup_dir)
        print(f"Results backed up to: {backup_dir}")
    except Exception as e:
        print(f"Error creating backup: {e}")


if __name__ == "__main__":
    # Example usage
    print("Testing utils module...")
    
    # Test directory creation
    test_dirs = ["test_dir1", "test_dir2/subdir"]
    create_directories(test_dirs)
    
    # Test random seed setting
    set_random_seed(42)
    
    # Test matplotlib setup
    setup_matplotlib_style()
    
    print("Utils module test complete!")
