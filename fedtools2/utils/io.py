# -*- coding: utf-8 -*-
"""Small IO helpers for loading/saving."""

from pathlib import Path
import pandas as pd

def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)