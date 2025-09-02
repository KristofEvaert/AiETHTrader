"""
AI Crypto Trading Pipeline for ETH/USDC using multiple timeframes.
- Downloads up to 2 years of historical klines from Binance (Spot)
  for intervals: 1m,5m,15m,30m,1h,4h,1d.
- Builds a multi-timeframe feature set aligned to a base timeframe (15m by default).
- Trains a simple PyTorch classifier to predict next-bar direction.
- Backtests a threshold strategy on the validation set.
- Optional live trading loop (default DRY_RUN / testnet).

DISCLAIMER: Educational template. Crypto is risky. Test on TESTNET first.
"""

import os
import time
import math
import json
import joblib
import yaml
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from binance.spot import Spot as SpotClient
from binance.error import ClientError

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -------------------- CONFIG --------------------

DEFAULT_CONFIG = {
    "symbol": "ETHUSDC",
    "timeframes": ["1m","5m","15m","30m","1h","4h","1d"],
    "base_timeframe": "15m",
    "lookback_years": 2,
    "label_horizon_bars": 1,     # predict next bar direction on base timeframe
    "min_rows_per_tf": 300,       # sanity check
    "train": {
        "test_size": 0.2,
        "random_state": 42,
        "epochs": 20,
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_size": 128,
        "dropout": 0.2
    },
    "backtest": {
        "fee_bps": 10,           # 0.10%
        "threshold": 0.55,       # probability to enter long
        "hold_bars": 1
    },
    "live": {
        "enabled": False,
        "poll_seconds": 30,
        "dry_run": True,
        "trade_quote_size": 50.0,
    },
    "storage": {
        "data_dir": "./data",
        "artifacts_dir": "./artifacts"
    },
    "use_testnet": True
}

# Load .env if present (very lightweight parser)
def load_env_file(path: str = ".env"):
    if not os.path.exists(path): 
        return {}
    env = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k,v = line.split("=",1)
            env[k.strip()] = v.strip()
    return env

# -------------------- BINANCE HELPERS --------------------

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 5*60_000,
    "15m": 15*60_000,
    "30m": 30*60_000,
    "1h": 60*60_000,
    "4h": 4*60*60_000,
    "1d": 24*60*60_000,
}

def make_client(api_key:str, api_secret:str, use_testnet: bool):
    base_url = "https://testnet.binance.vision" if use_testnet else "https://api.binance.com"
    return SpotClient(api_key=api_key, api_secret=api_secret, base_url=base_url)

def fetch_klines(client: SpotClient, symbol: str, interval: str, start_ts_ms: int, end_ts_ms: int, sleep_sec: float = 0.2) -> pd.DataFrame:
    """
    Paginate klines (max 1000 per request). Returns DataFrame with columns:
    open_time, open, high, low, close, volume, close_time, quote_asset_volume, trades, taker_buy_base, taker_buy_quote
    """
    all_rows = []
    limit = 1000
    cur = start_ts_ms
    step = INTERVAL_MS[interval]*limit
    while cur < end_ts_ms:
        try:
            res = client.klines(symbol=symbol, interval=interval, startTime=cur, endTime=min(end_ts_ms, cur+step-1), limit=limit)
        except ClientError as e:
            print(f"[WARN] Klines error {e}. Sleeping and retrying...")
            time.sleep(2)
            continue
        if not res:
            break
        all_rows.extend(res)
        last_open = res[-1][0]
        next_start = last_open + INTERVAL_MS[interval]
        if next_start <= cur:
            # safety to avoid infinite loop
            next_start = cur + INTERVAL_MS[interval]
        cur = next_start
        time.sleep(sleep_sec)
    if not all_rows:
        return pd.DataFrame()
    cols = ["open_time","open","high","low","close","volume","close_time","quote_asset_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df = df.drop(columns=["ignore"])
    # Convert types
    num_cols = ["open","high","low","close","volume","quote_asset_volume","taker_buy_base","taker_buy_quote"]
    df[num_cols] = df[num_cols].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    return df

# -------------------- FEATURES --------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100/(1+rs))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def build_tf_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Expects df indexed by open_time with columns close, high, low, volume, etc.
    Returns a feature dataframe with engineered columns prefixed.
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    out[f"{prefix}_ret1"] = close.pct_change()
    out[f"{prefix}_ret5"] = close.pct_change(5)
    out[f"{prefix}_vol"] = df["volume"].rolling(20).mean()
    out[f"{prefix}_rng"] = (df["high"]-df["low"]) / (df["open"]+1e-12)
    out[f"{prefix}_rsi14"] = rsi(close, 14)
    m, s, h = macd(close)
    out[f"{prefix}_macd"] = m
    out[f"{prefix}_macd_sig"] = s
    out[f"{prefix}_macd_hist"] = h
    out[f"{prefix}_ema20"] = ema(close, 20)
    out[f"{prefix}_ema50"] = ema(close, 50)
    out[f"{prefix}_ema200"] = ema(close, 200)
    return out

def merge_asof_multi(base: pd.DataFrame, others: dict) -> pd.DataFrame:
    """
    others: dict of name->feature_df (indexed by open_time)
    Performs asof merge to align other TF features onto base index.
    """
    merged = base.copy()
    merged = merged.sort_index().reset_index().rename(columns={"open_time":"ts"}) if "open_time" in merged.columns else merged.sort_index().reset_index().rename(columns={merged.index.name or "index":"ts"})
    for name, feat in others.items():
        f = feat.sort_index().reset_index().rename(columns={"open_time":"ts"})
        merged = pd.merge_asof(merged.sort_values("ts"), f.sort_values("ts"), on="ts", direction="backward")
    merged = merged.set_index("ts").sort_index()
    return merged

# -------------------- LABELS --------------------

def make_labels(base_candles: pd.DataFrame, horizon_bars: int = 1) -> pd.Series:
    future_close = base_candles["close"].shift(-horizon_bars)
    y = (future_close > base_candles["close"]).astype(int)
    return y

# -------------------- MODEL --------------------

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# -------------------- BACKTEST --------------------

def backtest_long_only(pred_proba: pd.Series, base_candles: pd.DataFrame, threshold: float = 0.55, hold_bars: int = 1, fee_bps: float = 10.0):
    """
    Very simple: enter long at open of next bar when prob>threshold, hold for N bars, exit at close, apply fees.
    """
    df = pd.DataFrame(index=pred_proba.index)
    df["prob"] = pred_proba
    df["close"] = base_candles.loc[df.index, "close"]
    df["next_open"] = base_candles["open"].shift(-1).reindex(df.index)
    df["future_close"] = base_candles["close"].shift(-hold_bars).shift(-(1)).reindex(df.index)  # close after hold bars
    fee = fee_bps / 10_000.0
    df["enter"] = (df["prob"] > threshold).astype(int)
    # Compute trade returns
    # If enter, buy at next_open, sell at future_close.
    trade_ret = np.where(df["enter"]==1,
                         ((df["future_close"] / df["next_open"]) - 1.0) - 2*fee,
                         0.0)
    df["trade_ret"] = trade_ret
    df["equity"] = (1.0 + df["trade_ret"].fillna(0)).cumprod()
    # Metrics
    total_return = df["equity"].iloc[-1] - 1.0 if len(df) else 0.0
    rets = df["trade_ret"].replace(0, np.nan).dropna()
    if len(rets) > 1:
        sharpe = np.sqrt(252*24*4) * (rets.mean() / (rets.std()+1e-12))  # rough: 15m bars ~ 4/h
    else:
        sharpe = np.nan
    # Max drawdown
    roll_max = df["equity"].cummax()
    drawdown = (df["equity"]/roll_max - 1.0)
    max_dd = drawdown.min() if len(drawdown) else 0.0
    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe) if not math.isnan(sharpe) else None,
        "max_drawdown": float(max_dd),
        "trades": int(df["enter"].sum())
    }, df

# -------------------- LIVE --------------------

def place_market_buy(client: SpotClient, symbol: str, quote_qty: float):
    return client.new_order(symbol=symbol, side="BUY", type="MARKET", quoteOrderQty=round(quote_qty,2))

def place_market_sell(client: SpotClient, symbol: str, base_qty: float):
    return client.new_order(symbol=symbol, side="SELL", type="MARKET", quantity=base_qty)

# -------------------- MAIN PIPELINE --------------------

def run_pipeline(config: dict):
    os.makedirs(config["storage"]["data_dir"], exist_ok=True)
    os.makedirs(config["storage"]["artifacts_dir"], exist_ok=True)

    # API keys
    env = load_env_file()
    api_key = env.get("BINANCE_API_KEY", os.environ.get("BINANCE_API_KEY", ""))
    api_secret = env.get("BINANCE_API_SECRET", os.environ.get("BINANCE_API_SECRET", ""))
    use_testnet = env.get("USE_TESTNET", str(config.get("use_testnet", True))).lower() == "true"

    if not api_key or not api_secret:
        print("[WARN] No API keys found in .env or environment. You can still download public klines, but live trading will be disabled.")

    client = make_client(api_key, api_secret, use_testnet=use_testnet)

    # Time window
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=365*config["lookback_years"])

    print(f"Fetching klines for {config['symbol']} from {start_dt} to {end_dt}...")

    # Download TF data
    tf_data = {}
    for tf in config["timeframes"]:
        print(f"  -> {tf}")
        df = fetch_klines(client, config["symbol"], tf, int(start_dt.timestamp()*1000), int(end_dt.timestamp()*1000))
        if len(df) < config["min_rows_per_tf"]:
            raise RuntimeError(f"Not enough data for {tf}, got {len(df)} rows")
        # Save raw
        df.to_csv(os.path.join(config["storage"]["data_dir"], f"{config['symbol']}_{tf}.csv"))
        tf_data[tf] = df

    # Build features per TF
    feats = {}
    for tf, df in tf_data.items():
        feats[tf] = build_tf_features(df, prefix=tf)

    # Define base TF candles
    base_tf = config["base_timeframe"]
    base_candles = tf_data[base_tf][["open","high","low","close","volume"]].copy()

    # Merge all features as-of onto base index
    others = {tf: feat for tf, feat in feats.items() if tf != base_tf}
    X_all = merge_asof_multi(feats[base_tf], others)

    # Drop rows with NaNs at the start
    X_all = X_all.dropna().copy()
    base_candles = base_candles.reindex(X_all.index).dropna()

    # Labels
    y = make_labels(base_candles, horizon_bars=config["label_horizon_bars"]).reindex(X_all.index).dropna()
    X_all = X_all.reindex(y.index).copy()

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X_all.values, y.values, test_size=config["train"]["test_size"], shuffle=False)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    # Torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_features=X_train_sc.shape[1], hidden=config["train"]["hidden_size"], dropout=config["train"]["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    loss_fn = nn.BCELoss()

    train_ds = TensorDataset(torch.tensor(X_train_sc, dtype=torch.float32), torch.tensor(y_train.reshape(-1,1), dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val_sc, dtype=torch.float32), torch.tensor(y_val.reshape(-1,1), dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    best_val = 1e9
    best_state = None
    for epoch in range(1, config["train"]["epochs"]+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item()*len(xb)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            va_loss = 0.0
            all_p, all_y = [], []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pr = model(xb)
                loss = loss_fn(pr, yb)
                va_loss += loss.item()*len(xb)
                all_p.append(pr.cpu().numpy())
                all_y.append(yb.cpu().numpy())
            va_loss /= len(val_loader.dataset)
            all_p = np.vstack(all_p).ravel()
            all_y = np.vstack(all_y).ravel()
            auc = roc_auc_score(all_y, all_p) if len(np.unique(all_y))==2 else float("nan")
        print(f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | AUC {auc:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save artifacts
    artifacts_dir = config["storage"]["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    features_path = os.path.join(artifacts_dir, "feature_columns.json")
    joblib.dump(X_all.columns.tolist(), features_path)
    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")

    # Backtest on validation fold (time-ordered split)
    # Recompute probabilities on entire X_all to align indices
    X_all_sc = scaler.transform(X_all.values)
    with torch.no_grad():
        proba_all = model(torch.tensor(X_all_sc, dtype=torch.float32, device=device)).cpu().numpy().ravel()
    proba_series = pd.Series(proba_all, index=X_all.index, name="proba")
    # Use last 20% for backtest
    split_idx = int(len(proba_series)*(1-config["train"]["test_size"]))
    proba_bt = proba_series.iloc[split_idx:]
    base_bt = base_candles.reindex(proba_bt.index)
    stats, curve = backtest_long_only(
        pred_proba=proba_bt,
        base_candles=base_bt,
        threshold=config["backtest"]["threshold"],
        hold_bars=config["backtest"]["hold_bars"],
        fee_bps=config["backtest"]["fee_bps"]
    )
    stats_path = os.path.join(artifacts_dir, "backtest_stats.json")
    curve_path = os.path.join(artifacts_dir, "backtest_curve.csv")
    with open(stats_path,"w") as f:
        json.dump(stats, f, indent=2)
    curve.to_csv(curve_path)
    print("Backtest:", stats)

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "features_path": features_path,
        "backtest_stats_path": stats_path,
        "backtest_curve_path": curve_path
    }

# -------------------- LIVE LOOP --------------------

def latest_features_and_signal(client: SpotClient, config: dict, scaler: StandardScaler, model: nn.Module):
    """
    Pulls the latest candles for each timeframe (about 400 bars per tf),
    rebuilds features, aligns, returns current proba signal on the latest base bar.
    """
    symbol = config["symbol"]
    base_tf = config["base_timeframe"]
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=14)  # minimal window for indicators

    tf_data = {}
    for tf in config["timeframes"]:
        df = fetch_klines(client, symbol, tf, int(start_dt.timestamp()*1000), int(end_dt.timestamp()*1000))
        tf_data[tf] = df

    feats = {tf: build_tf_features(df, prefix=tf) for tf, df in tf_data.items()}
    base_candles = tf_data[base_tf][["open","high","low","close","volume"]]
    others = {tf: feat for tf, feat in feats.items() if tf != base_tf}
    X_now = merge_asof_multi(feats[base_tf], others).dropna()
    if X_now.empty:
        return None, None, None
    cols = X_now.columns.tolist()
    X_sc = scaler.transform(X_now.values)
    device = next(model.parameters()).device
    with torch.no_grad():
        proba = model(torch.tensor(X_sc, dtype=torch.float32, device=device)).cpu().numpy().ravel()
    proba_series = pd.Series(proba, index=X_now.index, name="proba")
    last_ts = proba_series.index[-1]
    last_proba = float(proba_series.iloc[-1])
    last_price = float(base_candles.loc[last_ts, "close"])
    return last_ts, last_proba, last_price

def run_live_loop(config: dict):
    env = load_env_file()
    api_key = env.get("BINANCE_API_KEY", os.environ.get("BINANCE_API_KEY", ""))
    api_secret = env.get("BINANCE_API_SECRET", os.environ.get("BINANCE_API_SECRET", ""))
    use_testnet = env.get("USE_TESTNET", str(config.get("use_testnet", True))).lower() == "true"
    dry_run = env.get("DRY_RUN", str(config["live"]["dry_run"])).lower() == "true"
    trade_quote_size = float(env.get("TRADE_QUOTE_SIZE", config["live"]["trade_quote_size"]))

    client = make_client(api_key, api_secret, use_testnet=use_testnet)

    # Load artifacts
    artifacts_dir = config["storage"]["artifacts_dir"]
    model_path = os.path.join(artifacts_dir, "model.pt")
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("[ERROR] Train the model first to create artifacts.")
        return

    scaler = joblib.load(scaler_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # re-create model with correct input size
    sample_cols = joblib.load(os.path.join(artifacts_dir, "feature_columns.json"))
    model = MLP(in_features=len(sample_cols))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    print("Starting live loop. Press Ctrl+C to stop.")
    position_file = os.path.join(artifacts_dir, "position_state.json")
    # Simple position state
    if os.path.exists(position_file):
        with open(position_file,"r") as f:
            state = json.load(f)
    else:
        state = {"has_position": False, "base_qty": 0.0}

    threshold = config["backtest"]["threshold"]

    try:
        while True:
            ts, proba, price = latest_features_and_signal(client, config, scaler, model)
            if ts is None:
                print("Waiting for enough data...")
                time.sleep(config["live"]["poll_seconds"])
                continue
            print(f"[{ts}] proba_up={proba:.3f} price={price:.2f}")
            if (proba is not None) and (proba > threshold) and not state["has_position"]:
                # Enter long
                if dry_run:
                    print(f"DRY_RUN: would BUY ~{trade_quote_size} USDC of {config['symbol']}")
                    state["has_position"] = True
                    state["base_qty"] = (trade_quote_size / price) * 0.999  # fee slippage estimate
                else:
                    try:
                        order = place_market_buy(client, config["symbol"], trade_quote_size)
                        fills_qty = sum(float(f["qty"]) for f in order.get("fills", [])) if "fills" in order else 0.0
                        state["has_position"] = True
                        state["base_qty"] = fills_qty if fills_qty>0 else (trade_quote_size/price)
                        print("BUY order executed:", order)
                    except ClientError as e:
                        print("BUY error:", e)
            elif state["has_position"]:
                # Naive exit: after one base bar, just sell on next loop iteration
                if dry_run:
                    print(f"DRY_RUN: would SELL {state['base_qty']:.6f} {config['symbol'][:-4]}")
                    state["has_position"] = False
                    state["base_qty"] = 0.0
                else:
                    try:
                        order = place_market_sell(client, config["symbol"], state["base_qty"])
                        state["has_position"] = False
                        state["base_qty"] = 0.0
                        print("SELL order executed:", order)
                    except ClientError as e:
                        print("SELL error:", e)

            with open(position_file,"w") as f:
                json.dump(state, f)

            time.sleep(config["live"]["poll_seconds"])
    except KeyboardInterrupt:
        print("Stopped.")

# -------------------- CLI --------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Crypto Trading Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (optional)")
    parser.add_argument("--train", action="store_true", help="Run data download + train + backtest")
    parser.add_argument("--live", action="store_true", help="Run live loop (loads artifacts)")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config, "r") as f:
            user_conf = yaml.safe_load(f)
        # deep update
        def deep_update(d,u):
            for k,v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    deep_update(d[k], v)
                else:
                    d[k]=v
        deep_update(config, user_conf)

    if args.train:
        paths = run_pipeline(config)
        print("Artifacts:")
        for k,v in paths.items():
            print(f" - {k}: {v}")

    if args.live:
        run_live_loop(config)

    if not args.train and not args.live:
        parser.print_help()

if __name__ == "__main__":
    main()
