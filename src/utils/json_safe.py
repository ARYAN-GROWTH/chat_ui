import numpy as np
import pandas as pd
from datetime import datetime, date
from decimal import Decimal
from typing import Any


def make_json_safe(obj: Any):
    """
    Recursively convert NumPy / pandas / Python objects 
    into JSON-safe types (int, float, bool, str, list, dict).
    """

    # --- Handle NumPy numeric types ---
    numpy_int_types = (
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
    )
    numpy_float_types = (
        np.float16, np.float32, np.float64,
    )

    if isinstance(obj, numpy_int_types):
        return int(obj)
    if isinstance(obj, numpy_float_types):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)

    # --- Handle pandas.Timestamp safely ---
    # Avoid passing it directly to isinstance to satisfy Pylance
    try:
        if "Timestamp" in str(type(obj)):
            return str(obj)
    except Exception:
        pass

    # --- Handle datetime and date ---
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # --- Handle Decimal values ---
    if isinstance(obj, Decimal):
        return float(obj)

    # --- Handle None and base Python primitives ---
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # --- Handle lists, tuples, sets ---
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]

    # --- Handle dictionaries ---
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    # --- Fallback: convert to string safely ---
    try:
        return str(obj)
    except Exception:
        return repr(obj)
