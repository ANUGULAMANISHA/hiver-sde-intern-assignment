"""
Data loading utilities for the Hiver SDE Intern assignment.
"""

from pathlib import Path
import pandas as pd


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def basic_dataset_info(df: pd.DataFrame) -> dict:
    """Return basic information about a dataset."""
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
    }
