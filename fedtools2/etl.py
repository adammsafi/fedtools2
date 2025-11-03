# -*- coding: utf-8 -*-
"""
fedtools2.etl — CLI entrypoint.
Consolidates FEDFUNDS, DFEDTAR, DFEDTARL, DFEDTARU into a unified daily dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

from fedtools2.utils.io import read_csv, ensure_dir
from fedtools2.utils.consolidation import combine_timeframes, missing_ranges

def _load_config(cfg_path: Path | None) -> dict:
    if cfg_path is None:
        cfg_path = Path(__file__).with_suffix("").parent / "config" / "default.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def build_dataset(cfg: dict) -> pd.DataFrame:
    # (S1) Paths and toggles
    FED_DATA_DIR = Path(cfg["fed_data_dir"])
    OUTPUT_DIR   = Path(cfg["output_dir"])
    ensure_dir(OUTPUT_DIR)

    # (S2) Load CSVs
    fedfunds = read_csv(FED_DATA_DIR / "FEDFUNDS.csv")   # ['observation_date','FEDFUNDS']
    dfedtaru = read_csv(FED_DATA_DIR / "DFEDTARU.csv")   # ['observation_date','DFEDTARU']
    dfedtarl = read_csv(FED_DATA_DIR / "DFEDTARL.csv")   # ['observation_date','DFEDTARL']
    dfedtar  = read_csv(FED_DATA_DIR / "DFEDTAR.csv")    # ['observation_date','DFEDTAR']

    # (S3) Merge targets (single-era + range-era)
    merged_targets = combine_timeframes(
        dfs=[dfedtar, dfedtarl, dfedtaru],
        names=["DFEDTAR", "DFEDTARL", "DFEDTARU"],
        persist=True
    )

    # (S4) Cutoff DFEDTAR ffill after 2008-12-15 using np.nan to keep numeric dtype
    cutoff = pd.Timestamp(cfg["dfedtar_cutoff"])
    col = "DFEDTAR_DFEDTAR"
    merged_targets.loc[merged_targets.index > cutoff, col] = np.nan

    # (S5) TARGET_MID = DFEDTAR where available, else midpoint of bounds
    merged_targets["TARGET_MID"] = merged_targets[col].where(
        ~merged_targets[col].isna(),
        (merged_targets["DFEDTARL_DFEDTARL"] + merged_targets["DFEDTARU_DFEDTARU"]) / 2.0
    ).astype(float)

    # (S6) TARGET_SPREAD
    merged_targets["TARGET_SPREAD"] = (
        merged_targets["DFEDTARU_DFEDTARU"] - merged_targets["DFEDTARL_DFEDTARL"]
    )

    # (S7) FEDFUNDS monthly → daily ffill
    fedfunds["observation_date"] = pd.to_datetime(fedfunds["observation_date"])
    fedfunds = fedfunds.set_index("observation_date").sort_index()
    fedfunds_daily = fedfunds.resample("D").ffill()

    # (S8) Merge everything and fix has_* flags
    merged_all = merged_targets.join(fedfunds_daily, how="outer")
    has_cols = [c for c in merged_all.columns if c.startswith("has_")]
    if has_cols:
        merged_all[has_cols] = merged_all[has_cols].astype("boolean").fillna(False)

    # (S9) Regime labels
    merged_all["regime"] = pd.cut(
        merged_all.index,
        bins=[
            pd.Timestamp("1954-01-01"),
            pd.Timestamp("1982-09-26"),
            pd.Timestamp("2008-12-15"),
            pd.Timestamp("2025-12-31")
        ],
        labels=["pre-target", "single-target", "target-range"]
    )

    return merged_all

def save_outputs(df: pd.DataFrame, cfg: dict) -> Path:
    out_dir = Path(cfg["output_dir"])
    ts_path = out_dir / f"FED_Merged_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(ts_path, index=True)

    if cfg.get("write_latest_copy", True):
        latest = out_dir / cfg.get("latest_filename", "FED_Merged_latest.csv")
        df.to_csv(latest, index=True)
        print(f"Also wrote: {latest}")

    print(f"Saved: {ts_path}")
    return ts_path

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Consolidate Federal Reserve rate datasets into a unified daily CSV."
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config.")
    parser.add_argument("--plot", action="store_true", help="Show quick validation plot.")
    parser.add_argument("--verbose-missing", action="store_true", help="Print missing-range diagnostics.")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    # CLI overrides config toggles
    if args.plot:
        cfg["plot_quicklook"] = True
    if args.verbose_missing:
        cfg["verbose_missing"] = True

    df = build_dataset(cfg)

    # Optional: diagnostics
    if cfg.get("verbose_missing", False):
        for name in ["DFEDTAR", "DFEDTARL", "DFEDTARU"]:
            mask_col = f"has_{name}"
            if mask_col in df.columns:
                mask = ~df[mask_col]
                gaps = missing_ranges(mask)
                if gaps:
                    print(f"\n{name} missing ranges (up to 5):")
                    for s, e in gaps[:5]:
                        print(f"  {s.date()} → {e.date()}")
                    if len(gaps) > 5:
                        print(f"  ... {len(gaps)-5} more omitted.")

    # Optional: quicklook plot
    if cfg.get("plot_quicklook", False):
        try:
            import matplotlib.pyplot as plt
            ax = df[["TARGET_MID", "FEDFUNDS"]].plot(figsize=(11, 5), lw=1.1)
            ax.set_title("Fed Policy Target (Mid) vs Effective Fed Funds Rate")
            ax.set_xlabel("")
            ax.set_ylabel("Percent")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("Plot skipped:", e)

    save_outputs(df, cfg)

if __name__ == "__main__":
    main()