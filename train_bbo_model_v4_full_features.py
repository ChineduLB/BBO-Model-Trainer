import os
import json
import datetime as dt

import numpy as np
import pandas as pd
import pyodbc
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, precision_score
from sklearn.linear_model import LogisticRegression

from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType


# ====================================================================================
# CONFIG
# ====================================================================================

CONN_STR = (
    "Driver={ODBC Driver 18 for SQL Server};"
    "Server=LBLENOVOLEGION;"
    "Database=TradingDb;"
    "Trusted_Connection=Yes;"
    "TrustServerCertificate=Yes"
)

# BBO symbols only (no OBI L2 symbols here)
SYMBOLS = ["MARA", "RIOT"]

# Labels are 10-minute horizon with ±0.5% TP/SL
LOOKAHEAD_MIN = 10
MOVE_FRAC = 0.005

# Target precision for choosing thresholds on calibrated probabilities
TARGET_PRECISION = 0.75

# How much history for training vs recent test window
TRAIN_WINDOW_DAYS = 730   # ~2 years
RECENT_TEST_DAYS  = 60    # last 60 days as "recent test"

# Output locations
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "ml_model_v4_bbo_full_features")
ARTIFACT_DIR = os.path.join(MODEL_DIR, "artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

ONNX_LONG_PATH  = os.path.join(ARTIFACT_DIR, "bbo_v4_long.onnx")
ONNX_SHORT_PATH = os.path.join(ARTIFACT_DIR, "bbo_v4_short.onnx")
META_PATH       = os.path.join(MODEL_DIR, "bbo_v4_meta.json")


# Canonical feature list for this model (order matters!)
FEATURE_NAMES = [
    # Raw / regime
    "ATR",
    "RVOL",
    "RSI",
    "SpreadFraction_Final",
    "AverageSpread",
    "AvgSpreadFrac",
    "MaxSpreadFrac",
    "Volume",

    # VWAP & bands
    "VWAP",
    "VWAPBandWidth",
    "VWAPBandPercent",
    "VwapBandPos",

    # EMAs
    "EmaShort",
    "EmaMid",
    "EmaLong",

    # Bar shape
    "Return1",
    "RangeFrac",
    "BodyFrac",
    "ClosePosInRange",
    "UpperWickFrac",
    "LowerWickFrac",

    # Distance to anchors
    "CloseVsVwapFrac",
    "CloseVsEmaShortFrac",
    "CloseVsEmaMidFrac",
    "CloseVsEmaLongFrac",
    "EmaShortAboveMidFlag",
    "EmaMidAboveLongFlag",

    # ATR-normalized shape
    "RangeVsAtr",
    "BodyVsAtr",

    # Time-of-day
    "TimeOfDayFrac",

    # Liquidity tier numeric
    "LiquidityTierCode",
]


# ====================================================================================
# DATA LOADING
# ====================================================================================

def load_dataset() -> pd.DataFrame:
    """
    Load MARA/RIOT rows from dbo.ML_AllSymbolsTrainingDataset_Rebuild
    and prepare features + per-side labels.

    Assumes ML_AllSymbolsTrainingDataset_Rebuild includes:
      - Symbol, BarTime
      - OHLCV, VWAP, VWAP bands
      - ATR, RVOL, RSI
      - EmaShort, EmaMid, EmaLong
      - SpreadFraction_Final, AverageSpread, AvgSpreadFrac, MaxSpreadFrac
      - Derived features: Return1, RangeFrac, BodyFrac, ClosePosInRange,
        UpperWickFrac, LowerWickFrac, CloseVsVwapFrac, CloseVsEmaShortFrac,
        CloseVsEmaMidFrac, CloseVsEmaLongFrac, EmaShortAboveMidFlag,
        EmaMidAboveLongFlag, RangeVsAtr, BodyVsAtr, VwapBandPos, TimeOfDayFrac
      - LiquidityTier
      - Label3 (0=DownFirst, 1=NoTouch, 2=UpFirst)
    """
    symbol_list = "', '".join(SYMBOLS)
    query = f"""
        SELECT *
        FROM dbo.ML_AllSymbolsTrainingDataset_Rebuild
        WHERE Symbol IN ('{symbol_list}')
          AND LookAheadMin = {LOOKAHEAD_MIN}
          AND MoveUpFrac   = {MOVE_FRAC}
          AND MoveDownFrac = {MOVE_FRAC};
    """

    with pyodbc.connect(CONN_STR) as conn:
        df = pd.read_sql(query, conn)

    print(f"Raw rows from ML_AllSymbolsTrainingDataset_Rebuild (MARA/RIOT): {len(df)}")
    if df.empty:
        raise RuntimeError(
            "ML_AllSymbolsTrainingDataset_Rebuild returned 0 rows for MARA/RIOT. "
            "Check the view / labels."
        )

    # Ensure BarTime is datetime
    if "BarTime" not in df.columns:
        raise RuntimeError("Expected 'BarTime' column in ML_AllSymbolsTrainingDataset_Rebuild.")
    df["BarTime"] = pd.to_datetime(df["BarTime"])

    # Label sanity
    if "Label3" not in df.columns:
        raise RuntimeError("Expected 'Label3' column in ML_AllSymbolsTrainingDataset_Rebuild.")

    before_label = len(df)
    df = df[df["Label3"].notnull()].copy()
    print(f"Rows after Label3 NOT NULL filter: {len(df)} (dropped {before_label - len(df)})")

    invalid = df[~df["Label3"].isin([0, 1, 2])]
    if not invalid.empty:
        raise RuntimeError(
            f"Unexpected Label3 values: {invalid['Label3'].unique().tolist()}"
        )

    # Build long/short binary labels from Label3
    df["Label_Long"] = np.where(df["Label3"] == 2, 1, 0)   # UpFirst -> long positive
    df["Label_Short"] = np.where(df["Label3"] == 0, 1, 0)  # DownFirst -> short positive

    # Map LiquidityTier to numeric code
    def map_liquidity_tier(tier: str) -> int:
        if tier is None:
            return 0
        t = tier.strip().lower()
        if t == "veryhigh":
            return 3
        if t == "high":
            return 2
        if t == "medium":
            return 1
        return 0

    if "LiquidityTier" not in df.columns:
        raise RuntimeError("Expected 'LiquidityTier' column in ML_AllSymbolsTrainingDataset_Rebuild.")

    df["LiquidityTierCode"] = df["LiquidityTier"].apply(map_liquidity_tier).astype(np.float32)

    # Check required feature columns
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns in dataset: {missing}")

    # Null counts for debugging
    print("\nNull counts for core feature columns (before drop):")
    print(df[FEATURE_NAMES].isna().sum())

    # Drop rows with any NaNs in the chosen feature set
    before_drop = len(df)
    df = df.dropna(subset=FEATURE_NAMES + ["Label_Long", "Label_Short"]).copy()
    print(f"Rows after dropping NaNs in feature set: {len(df)} (dropped {before_drop - len(df)})")

    # Distribution by symbol
    print("\nRows per symbol (after NaN drop):")
    print(df["Symbol"].value_counts())

    # Label3 distribution by symbol
    print("\nLabel3 distribution by symbol:")
    print(
        df.pivot_table(
            index="Symbol",
            columns="Label3",
            values="BarTime",
            aggfunc="count",
            fill_value=0,
        )
    )

    # Positive rate per side
    print("\nPositive rate per side (Label_Long / Label_Short) by symbol:")
    for side, col in [("Long", "Label_Long"), ("Short", "Label_Short")]:
        side_rate = df.groupby("Symbol")[col].mean()
        print(f"  {side} side:")
        print((side_rate * 100).round(2).astype(str) + "%")

    return df


# ====================================================================================
# TIME SPLIT + FEATURE MATRIX
# ====================================================================================

def build_feature_matrix(df_side: pd.DataFrame) -> np.ndarray:
    """
    Build X matrix using FEATURE_NAMES order.
    """
    X = df_side[FEATURE_NAMES].to_numpy(dtype=np.float32)
    return X


def time_split(df_side: pd.DataFrame, train_window_days: int, recent_test_days: int):
    """
    Time-based split into train_val / test with a fallback for short ranges.
    """
    df_side = df_side.sort_values("BarTime").copy()
    min_time = df_side["BarTime"].min()
    max_time = df_side["BarTime"].max()
    print(f"Time span for side: {min_time} -> {max_time} (n={len(df_side)})")

    df_train_val = None
    df_test = None
    train_start = None
    test_start = None

    # Preferred: last recent_test_days as test, earlier as train_val
    if recent_test_days is not None and recent_test_days > 0:
        test_start_candidate = max_time - pd.Timedelta(days=recent_test_days)
        df_test = df_side[df_side["BarTime"] >= test_start_candidate]
        df_train_val = df_side[df_side["BarTime"] < test_start_candidate]
        test_start = test_start_candidate

    # Fallback: 80/20 time split if we don't get enough rows in either bucket
    if df_train_val is None or df_test is None or df_train_val.empty or df_test.empty:
        print("Date-based split not viable. Falling back to 80/20 time split.")
        n = len(df_side)
        if n < 10:
            raise RuntimeError(f"Not enough rows ({n}) to split into train/test.")
        split_idx = int(n * 0.8)
        df_train_val = df_side.iloc[:split_idx].copy()
        df_test = df_side.iloc[split_idx:].copy()
        test_start = df_test["BarTime"].min()

    # Apply training window limit
    if train_window_days is not None and train_window_days > 0:
        train_end = df_train_val["BarTime"].max()
        train_start_candidate = train_end - pd.Timedelta(days=train_window_days)
        df_train_val = df_train_val[df_train_val["BarTime"] >= train_start_candidate]
        train_start = train_start_candidate
    else:
        train_start = df_train_val["BarTime"].min()

    print(
        f"Final time split: "
        f"train_val [{train_start} -> {df_train_val['BarTime'].max()}] (n={len(df_train_val)}), "
        f"test [{test_start} -> {df_test['BarTime'].max()}] (n={len(df_test)})"
    )

    return df_train_val, df_test, train_start, test_start, max_time


# ====================================================================================
# TRAINING / CALIBRATION
# ====================================================================================

def choose_threshold_for_precision(
    y_true,
    p_cal,
    target_precision: float,
    default_thr: float = 0.5,
):
    precision, recall, thresholds = precision_recall_curve(y_true, p_cal)
    best_thr = default_thr
    best_prec = 0.0

    for thr, prec in zip(thresholds, precision[:-1]):  # last precision has no threshold
        if prec >= target_precision:
            best_thr = float(thr)
            best_prec = float(prec)
            break

    if best_prec == 0.0:
        print("WARNING: Could not reach target precision; using default threshold.")
        precision_default = precision_score(y_true, (p_cal >= default_thr).astype(int))
        return default_thr, float(precision_default)

    return best_thr, best_prec


def train_side_model(df: pd.DataFrame, side: str):
    """
    Train one binary LightGBM model for a given side ("Long" or "Short").

    Label mapping:
      - Label3 = 2 -> Label_Long=1 (UpFirst long)
      - Label3 = 0 -> Label_Short=1 (DownFirst short)
      - Label3 = 1 -> both 0 (NoTouch)
    """
    assert side in ("Long", "Short")
    label_col = "Label_Long" if side == "Long" else "Label_Short"

    print(f"\n=== {side.upper()} side ===")
    print(f"[{side}] Total rows before side filtering: {len(df)}")

    df_side = df[df[label_col].notnull()].copy()
    print(f"[{side}] Rows after label NOT NULL filter: {len(df_side)}")
    if df_side.empty:
        raise RuntimeError(f"No rows left for {side} side after label filtering.")

    df_train_val, df_test, train_start, test_start, max_time = time_split(
        df_side,
        train_window_days=TRAIN_WINDOW_DAYS,
        recent_test_days=RECENT_TEST_DAYS,
    )

    y_train_val = df_train_val[label_col].astype(int).to_numpy()
    X_train_val = build_feature_matrix(df_train_val)
    y_test = df_test[label_col].astype(int).to_numpy()
    X_test = build_feature_matrix(df_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.3,
        random_state=42,
        stratify=y_train_val,
    )

    print(
        f"[{side}] train={len(y_train)}, val={len(y_val)}, test={len(y_test)} "
        f"(train window start={train_start.date()}, test start={test_start.date()}, "
        f"max_time={max_time.date()})"
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # Raw probabilities
    p_val_raw = model.predict_proba(X_val)[:, 1]
    p_test_raw = model.predict_proba(X_test)[:, 1]

    # Platt calibration on validation
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(p_val_raw.reshape(-1, 1), y_val)
    A = float(lr.coef_[0, 0])
    B = float(lr.intercept_[0])

    def platt(p):
        z = A * p + B
        return 1.0 / (1.0 + np.exp(-z))

    p_val_cal = platt(p_val_raw)
    p_test_cal = platt(p_test_raw)

    thr, val_prec = choose_threshold_for_precision(y_val, p_val_cal, TARGET_PRECISION)
    y_test_pred = (p_test_cal >= thr).astype(int)
    test_prec = precision_score(y_test, y_test_pred)

    print(
        f"[{side}] A={A:.6f}, B={B:.6f}, thr={thr:.6f}, "
        f"val_prec={val_prec:.4f}, test_prec_recent={test_prec:.4f}"
    )

    return model, A, B, thr, val_prec, test_prec


# ====================================================================================
# ONNX EXPORT + META
# ====================================================================================

def export_to_onnx(model: lgb.LGBMClassifier, onnx_path: str):
    n_features = len(FEATURE_NAMES)
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_type)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Saved ONNX: {onnx_path}")


def write_meta_json(long_params, short_params):
    A_long, B_long, thr_long, val_prec_long, test_prec_long = long_params
    A_short, B_short, thr_short, val_prec_short, test_prec_short = short_params

    meta = {
        "trained_at": dt.datetime.utcnow().isoformat(timespec="microseconds") + "Z",
        "description": "BBO v4 model — MARA/RIOT, full price+VWAP+shape feature set, "
                       "10-min horizon, ±0.5% TP/SL.",
        "target_precision": TARGET_PRECISION,
        "models": [
            {
                "side": "Long",
                "features": FEATURE_NAMES,
                "onnx": r"artifacts\\bbo_v4_long.onnx",
                "calibration": {
                    "A": float(A_long),
                    "B": float(B_long),
                    "type": "platt_prob_on_proba",
                },
                "threshold": float(thr_long),
                "val_target_precision": float(val_prec_long),
                "test_precision_at_threshold": float(test_prec_long),
            },
            {
                "side": "Short",
                "features": FEATURE_NAMES,
                "onnx": r"artifacts\\bbo_v4_short.onnx",
                "calibration": {
                    "A": float(A_short),
                    "B": float(B_short),
                    "type": "platt_prob_on_proba",
                },
                "threshold": float(thr_short),
                "val_target_precision": float(val_prec_short),
                "test_precision_at_threshold": float(test_prec_short),
            },
        ],
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote meta: {META_PATH}")


# ====================================================================================
# MAIN
# ====================================================================================

def main():
    print("Loading MARA/RIOT dataset from ML_AllSymbolsTrainingDataset_Rebuild...")
    df = load_dataset()
    print(f"\nLoaded {len(df)} usable rows after feature/label filtering.\n")

    # Train Long / Short
    model_long, A_long, B_long, thr_long, val_prec_long, test_prec_long = train_side_model(df, "Long")
    model_short, A_short, B_short, thr_short, val_prec_short, test_prec_short = train_side_model(df, "Short")

    # Export ONNX artifacts
    export_to_onnx(model_long, ONNX_LONG_PATH)
    export_to_onnx(model_short, ONNX_SHORT_PATH)

    # Write meta JSON
    write_meta_json(
        (A_long, B_long, thr_long, val_prec_long, test_prec_long),
        (A_short, B_short, thr_short, val_prec_short, test_prec_short),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
