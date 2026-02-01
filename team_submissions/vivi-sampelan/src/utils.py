"""
Utility functions for data validation and conversion.
"""

import json
from typing import Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np


def safe_float(value, default=None):
    """Safely convert value to float."""
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_round(value, decimals=2):
    """Safely round a value to specified decimals."""
    val = safe_float(value)
    return round(val, decimals) if val is not None else None


def format_percent(value, decimals=1):
    """Format value as percentage string."""
    val = safe_float(value)
    return f"{val:.{decimals}f}%" if val is not None else "N/A"


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """Validate ticker symbol format."""
    if not ticker or not isinstance(ticker, str):
        return False, "Invalid ticker: must be a non-empty string"
    ticker = ticker.upper().strip()
    if len(ticker) > 10 or not ticker.replace('.', '').isalnum():
        return False, "Invalid ticker format"
    return True, ticker


def get_current_date_formatted():
    """Get the current date formatted properly."""
    return datetime.now().strftime("%d %b %Y")


def convert_to_serializable(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif obj is None or (isinstance(obj, float) and np.isnan(obj)):
        return None
    elif pd.isna(obj):
        return None
    else:
        return obj


def safe_json_dumps(obj, **kwargs):
    """Safely dump object to JSON string."""
    return json.dumps(convert_to_serializable(obj), **kwargs)


def pick_series(df: pd.DataFrame, candidates: list) -> Optional[pd.Series]:
    """Pick first available series from list of candidates."""
    if df is None or df.empty:
        return None
    for col in candidates:
        if col in df.index:
            return df.loc[col]
    return None
