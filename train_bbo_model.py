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

# Use the connection string that already works for you
CONN_STR = (
    "Driver={ODBC Driver 18 for SQL Server};"
    "Server=LBLENOVOLEGION;"
    "Database=TradingDb;"
    "Trusted_Connection=Yes;"
    "TrustServerCertificate=Yes"
)

# Target precision when choosing probability thresholds
TARGET_PRECISION = 0.75

# How much history for training vs recent test window
TRAIN_WINDOW_DAYS = 730   # ~2 years; set 180 for ~6 months, etc.
RECENT_TEST_DAYS  = 60    # recent window; script falls back if range is shorter

# Hard lower bound on BarTime for training (dense BBO region only)
# Adjust this date if your coverage analysis says the dense region starts earlier/later.
MIN_TRAIN_BARTIME = dt.datetime(2025, 6, 1, 9, 30, 0)

# Output locations (relative to this script)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "ml_model")
ARTIFACT_DIR = os.path.join(MODEL_DIR, "artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

ONNX_LONG_PATH = os.path.join(ARTIFACT_DIR, "bbo_long.onnx")
ONNX_SHORT_PATH = os.path.join(ARTIFACT_DIR, "bbo_short.onnx")
META_PATH = os.path.join(MODEL_DIR, "bbo_meta.json")

# Order must match OnnxBboScorer.BuildVector9
FEATURE_NAMES = [
    "Rsi",
    "Rvol",
    "Atr",
    "SpreadFraction",
    "QueueImbalance",
    "MicroPrice",
    "SignedSpread",
    "EmaShort",
    "EmaMid",
]

# ====================================================================================
# DATA LOADING / RSI
# ====================================================================================

def add_rsi(df: pd.DataFrame, price_col: str, group_col: str, time_col: str, window: int = 14) -> pd.DataFrame:
    """
    Compute a simple rolling RSI per symbol (used only if RSI isn't already in the view).
    """
    df = df.sort_values([group_col, time_col]).copy()

    def _compute_group_rsi(g: pd.DataFrame) -> pd.DataFrame:
        delta = g[price_col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        roll_up = gain.rolling(window=window, min_periods=window).mean()
        roll_down = loss.rolling(window=window, min_periods=window).mean()

        rs = roll_up / roll_down.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        g["RSI"] = rsi
        return g

    df = df.groupby(group_col, group_keys=False).apply(_compute_group_rsi)
    return df


def load_dataset() -> pd.DataFrame:
    """
    Load MARA + RIOT rows from dbo.ML_BboTrainingDataset and prepare features.
    """
    query = """
        SELECT *
        FROM dbo.ML_BboTrainingDataset
        WHERE Symbol IN ('MARA', 'RIOT');
    """

    with pyodbc.connect(CONN_STR) as conn:
        df = pd.read_sql(query, conn)

    print(f"Raw rows from ML_BboTrainingDataset (MARA/RIOT): {len(df)}")
    if df.empty:
        raise RuntimeError(
            "ML_BboTrainingDataset returned 0 rows for MARA/RIOT. "
            "Check the view / backfill."
        )

    if "BarTime" not in df.columns:
        raise RuntimeError("Expected column 'BarTime' in ML_BboTrainingDataset.")
    df["BarTime"] = pd.to_datetime(df["BarTime"])

    # Restrict to dense BBO region only
    if MIN_TRAIN_BARTIME is not None:
        before = len(df)
        df = df[df["BarTime"] >= MIN_TRAIN_BARTIME].copy()
        print(
            f"Filtered to dense BBO region: BarTime >= {MIN_TRAIN_BARTIME} "
            f"-> {len(df)} rows (dropped {before - len(df)})"
        )
        if df.empty:
            raise RuntimeError(
                f"No rows remain after applying MIN_TRAIN_BARTIME={MIN_TRAIN_BARTIME}. "
                "Check coverage or adjust the date."
            )

    # Add RSI if not present
    if "RSI" not in df.columns:
        if "ClosePrice" not in df.columns:
            raise RuntimeError("RSI not present and ClosePrice missing; cannot compute RSI.")
        df = add_rsi(df, price_col="ClosePrice", group_col="Symbol", time_col="BarTime", window=14)

    required_cols = [
        "ATR", "RVOL", "EmaShort", "EmaMid",
        "SpreadFraction", "QueueImbalance", "MicroPrice", "SignedSpread",
        "Direction", "Label_IsTpFirst", "RSI"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in ML_BboTrainingDataset: {missing}")

    # Debug: show null counts for key columns
    print("Null counts for key columns:")
    print(df[required_cols].isna().sum())

    # Only keep rows with a defined TP/SL-first label
    before_label = len(df)
    df = df[df["Label_IsTpFirst"].notnull()].copy()
    print(f"Rows after Label_IsTpFirst NOT NULL filter: {len(df)} (dropped {before_label - len(df)})")

    # Drop rows with NaNs in core fields
    df = df.dropna(subset=required_cols).copy()
    print(f"Rows after dropping NaNs in required columns: {len(df)}")

    return df

# ====================================================================================
# FEATURES + TIME SPLIT
# ====================================================================================

def build_feature_matrix(df_side: pd.DataFrame) -> np.ndarray:
    """
    Build X as a numpy array in the exact order expected by BuildVector9.
    """
    col_map = {
        "Rsi": "RSI",
        "Rvol": "RVOL",
        "Atr": "ATR",
        "SpreadFraction": "SpreadFraction",
        "QueueImbalance": "QueueImbalance",
        "MicroPrice": "MicroPrice",
        "SignedSpread": "SignedSpread",
        "EmaShort": "EmaShort",
        "EmaMid": "EmaMid",
    }
    cols = [col_map[name] for name in FEATURE_NAMES]
    X = df_side[cols].to_numpy(dtype=np.float32)
    return X


def time_split(df_side: pd.DataFrame, train_window_days: int, recent_test_days: int):
    """
    Time-based split into train_val / test with fallback for short ranges.
    """
    df_side = df_side.sort_values("BarTime").copy()
    min_time = df_side["BarTime"].min()
    max_time = df_side["BarTime"].max()
    print(f"Time span for side: {min_time} -> {max_time} (n={len(df_side)})")

    df_train_val = None
    df_test = None
    train_start = None
    test_start = None

    # Preferred split: "recent_test_days" worth of data for test, rest for train_val
    if recent_test_days is not None and recent_test_days > 0:
        test_start_candidate = max_time - pd.Timedelta(days=recent_test_days)
        df_test = df_side[df_side["BarTime"] >= test_start_candidate]
        df_train_val = df_side[df_side["BarTime"] < test_start_candidate]
        test_start = test_start_candidate

    # Fallback: if that yields empty train or test, do an 80/20 time split instead
    if df_train_val is None or df_test is None or df_train_val.empty or df_test.empty:
        print("Date-based split not viable (empty train or test). Falling back to 80/20 time split.")
        n = len(df_side)
        if n < 10:
            raise RuntimeError(f"Not enough rows ({n}) to split into train/test.")
        split_idx = int(n * 0.8)
        df_train_val = df_side.iloc[:split_idx].copy()
        df_test = df_side.iloc[split_idx:].copy()
        test_start = df_test["BarTime"].min()

    # Apply training window limit if requested
    if train_window_days is not None and train_window_days > 0:
        train_end = df_train_val["BarTime"].max()
        train_start_candidate = train_end - pd.Timedelta(days=train_window_days)
        df_train_val = df_train_val[df_train_val["BarTime"] >= train_start_candidate]
        train_start = train_start_candidate
    else:
        train_start = df_train_val["BarTime"].min()

    print(
        f"Final time split: train_val [{train_start} -> {df_train_val['BarTime'].max()}] "
        f"(n={len(df_train_val)}), test [{test_start} -> {df_test['BarTime'].max()}] (n={len(df_test)})"
    )

    return df_train_val, df_test, train_start, test_start, max_time

# ====================================================================================
# TRAINING / CALIBRATION
# ====================================================================================

def choose_threshold_for_precision(y_true, p_cal, target_precision: float, default_thr: float = 0.5):
    precision, recall, thresholds = precision_recall_curve(y_true, p_cal)
    best_thr = default_thr
    best_prec = 0.0

    for thr, prec in zip(thresholds, precision[:-1]):
        if prec >= target_precision:
            best_thr = float(thr)
            best_prec = float(prec)
            break

    if best_prec == 0.0 and len(thresholds) > 0:
        idx = np.argmax(precision[:-1])
        best_thr = float(thresholds[idx])
        best_prec = float(precision[idx])

    return best_thr, best_prec


def train_side_model(df: pd.DataFrame, side: str):
    """
    Train one binary LightGBM model for a given side ("Long" or "Short").

    - Direction is numeric: 1 = Long, -1 = Short
    - Time-based split with fallback
    - Platt calibration on validation
    """
    assert side in ("Long", "Short")
    print(f"\n=== {side.upper()} side ===")
    print(f"[{side}] Total rows before side filter: {len(df)}")

    dir_series = df["Direction"]
    if not np.issubdtype(dir_series.dtype, np.number):
        dir_series = pd.to_numeric(dir_series, errors="coerce")

    if side == "Long":
        mask = dir_series == 1
    else:
        mask = dir_series == -1

    df_side = df[mask].copy()
    print(f"[{side}] Rows after Direction filter: {len(df_side)}")
    if df_side.empty:
        raise RuntimeError(
            f"No rows found for Direction={1 if side=='Long' else -1}. "
            "Check ML_BboTrainingDataset.Direction values."
        )

    df_train_val, df_test, train_start, test_start, max_time = time_split(
        df_side,
        train_window_days=TRAIN_WINDOW_DAYS,
        recent_test_days=RECENT_TEST_DAYS,
    )

    y_train_val = df_train_val["Label_IsTpFirst"].astype(int).to_numpy()
    X_train_val = build_feature_matrix(df_train_val)
    y_test = df_test["Label_IsTpFirst"].astype(int).to_numpy()
    X_test = build_feature_matrix(df_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.3, random_state=42, stratify=y_train_val
    )

    print(
        f"[{side}] train={len(y_train)}, val={len(y_val)}, test={len(y_test)} "
        f"(train window start={train_start.date()}, test start={test_start.date()}, max_time={max_time.date()})"
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

    # Raw model probabilities
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
        "target_precision": TARGET_PRECISION,
        "models": [
            {
                "side": "Long",
                "features": FEATURE_NAMES,
                "onnx": r"artifacts\\bbo_long.onnx",
                "calibration": {
                    "A": float(A_long),
                    "B": float(B_long),
                    "type": "platt_prob_on_proba"
                },
                "threshold": float(thr_long),
                "val_target_precision": float(val_prec_long),
                "test_precision_at_threshold": float(test_prec_long)
            },
            {
                "side": "Short",
                "features": FEATURE_NAMES,
                "onnx": r"artifacts\\bbo_short.onnx",
                "calibration": {
                    "A": float(A_short),
                    "B": float(B_short),
                    "type": "platt_prob_on_proba"
                },
                "threshold": float(thr_short),
                "val_target_precision": float(val_prec_short),
                "test_precision_at_threshold": float(test_prec_short)
            }
        ]
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote meta: {META_PATH}")

# ====================================================================================
# MAIN
# ====================================================================================

def main():
    print("Loading ML_BboTrainingDataset...")
    df = load_dataset()
    print(f"Loaded {len(df)} usable rows after filtering.\n")

    # Train Long / Short
    model_long, A_long, B_long, thr_long, val_prec_long, test_prec_long = train_side_model(df, "Long")
    model_short, A_short, B_short, thr_short, val_prec_short, test_prec_short = train_side_model(df, "Short")

    # Export ONNX artifacts
    export_to_onnx(model_long, ONNX_LONG_PATH)
    export_to_onnx(model_short, ONNX_SHORT_PATH)

    # Write V2 meta JSON for OnnxBboScorer
    write_meta_json(
        (A_long, B_long, thr_long, val_prec_long, test_prec_long),
        (A_short, B_short, thr_short, val_prec_short, test_prec_short),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
