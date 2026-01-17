"""
Module to load raw personality classification dataset.
Assumes data is stored as a CSV file with at least two columns:
- 'type': MBTI personality type (e.g., 'INFJ', 'ENTP')
- 'posts': Raw text (multiple posts separated by '|||')
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional

# Optional: configure data directory via environment or constants
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw"


def load_raw_data(
    filepath: Optional[str] = None,
    data_dir: str = str(DEFAULT_DATA_DIR),
    filename: str = "mbti_1.csv"
) -> pd.DataFrame:
    """
    Load the raw MBTI dataset from a CSV file.

    Parameters
    ----------
    filepath : str, optional
        Full path to the CSV file. If provided, overrides data_dir + filename.
    data_dir : str, optional
        Directory containing the raw data (default: project_root/data/raw).
    filename : str, optional
        Name of the CSV file (default: 'mbti_1.csv').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['type', 'posts'].

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If required columns are missing.
    """
    if filepath is None:
        filepath = os.path.join(data_dir, filename)

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found at {filepath}. "
            "Please download the MBTI dataset from Kaggle and place it in the data/raw/ directory."
        )

    df = pd.read_csv(filepath)

    # Validate expected columns
    required_cols = {"type", "posts"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns in dataset: {missing}")

    return df


if __name__ == "__main__":
    # Quick test
    try:
        data = load_raw_data()
        print("✅ Data loaded successfully!")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print("\nFirst few rows:")
        print(data.head())
    except Exception as e:
        print(f"❌ Error: {e}")