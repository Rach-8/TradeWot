import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import ta
import yfinance as yf
import vectorbt as vbt
from ta import add_all_ta_features


def rf_engineer_features(data_df, ema_short, ema_long, thresh):

    data = data_df.copy()

    # =====================
    # Basic returns
    # =====================
    data["returns"] = data["Close"].pct_change()
    data["Return_1d"] = data["returns"]

    # =====================
    # Volatility
    # =====================
    data["std_5"] = data["Return_1d"].rolling(5).std()
    data["Vol_21"] = data["Return_1d"].rolling(21).std()
    data["Vol_Z"] = (data["Vol_21"] - data["Vol_21"].rolling(252).mean()) / data[
        "Vol_21"
    ].rolling(252).std()

    # =====================
    # Volume
    # =====================
    data["Vol_MA20"] = data["Volume"].rolling(20).mean()

    # =====================
    # VPT
    # =====================
    data["VPT"] = data["Volume"] * (1 + data["Return_1d"])
    data["VPT_cum"] = data["VPT"].cumsum()
    data["VPT_Momentum"] = data["VPT_cum"].diff()

    # =====================
    # ATR
    # =====================
    atr = ta.volatility.AverageTrueRange(
        high=data["High"], low=data["Low"], close=data["Close"], window=14
    ).average_true_range()
    data["ATR_pct"] = atr / data["Close"]
    data["ATR_Change_5"] = data["ATR_pct"].diff(5)

    # =====================
    # Momentum
    # =====================
    data["ROC_5"] = data["Close"].pct_change(5)
    data["Ret_10"] = data["Close"].pct_change(10)

    # =====================
    # Bollinger
    # =====================
    bb = ta.volatility.BollingerBands(data["Close"], window=20)
    data["BB_width"] = bb.bollinger_hband() - bb.bollinger_lband()

    # =====================
    # EMA + Trend regime
    # =====================
    data["EMA20"] = data["Close"].ewm(span=20).mean()
    data["EMA50"] = data["Close"].ewm(span=50).mean()
    data["EMA100"] = data["Close"].ewm(span=100).mean()
    data["EMA200"] = data["Close"].ewm(span=200).mean()

    data["EMA_Slope"] = data["EMA20"] - data["EMA50"]
    data["EMA_Slope_Long"] = data["EMA20"] - data["EMA100"]
    data["EMA_fast"] = data["Close"].ewm(span=ema_short, adjust=False).mean()
    data["EMA_slow"] = data["Close"].ewm(span=ema_long, adjust=False).mean()

    data["Trend_Regime"] = (
        (data["Close"] > data["EMA200"]) & (data["EMA20"] > data["EMA50"])
    ).astype(int)

    # =====================
    # ADX
    # =====================
    adx = ta.trend.ADXIndicator(
        high=data["High"], low=data["Low"], close=data["Close"], window=14
    )
    data["ADX_14"] = adx.adx()
    data["ADX_Slope"] = data["ADX_14"].diff()

    # =====================
    # Gap
    # =====================
    data["Gap"] = (data["Open"] - data["Close"].shift(1)) / data["Close"].shift(1)

    # ======================================================
    # BREADTH — download large-cap universe
    # ======================================================
    tickers = [
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "META",
        "NVDA",
        "TSLA",
        "JPM",
        "UNH",
        "XOM",
        "JNJ",
        "V",
        "PG",
        "MA",
        "HD",
        "AVGO",
        "LLY",
        "MRK",
        "PEP",
        "ABBV",
        "COST",
        "KO",
        "CVX",
        "WMT",
        "BAC",
        "AMD",
        "DIS",
        "CSCO",
        "PFE",
        "ORCL",
    ]

    prices = yf.download(tickers, start=data.index.min(), auto_adjust=True)["Close"]
    returns = prices.pct_change()

    advancers = (returns > 0).sum(axis=1)
    decliners = (returns <= 0).sum(axis=1)
    data["AD_Line"] = (advancers - decliners).reindex(data.index).fillna(0).cumsum()

    pct_above_50 = (prices > prices.rolling(50).mean()).sum(axis=1) / prices.shape[1]
    data["Pct_Above_50MA"] = pct_above_50.reindex(data.index)

    data["Breadth_Strength"] = data["Pct_Above_50MA"]

    # ======================================================
    # CROSS-ASSET — VIX, Rates, Dollar
    # ======================================================
    macro = yf.download(
        ["^VIX", "^TNX", "DX-Y.NYB"], start=data.index.min(), auto_adjust=True
    )["Close"]

    data["VIX"] = macro["^VIX"].reindex(data.index)
    data["VIX_Change"] = data["VIX"].pct_change(3)
    data["VIX_Regime"] = (data["VIX"] > data["VIX"].rolling(252).median()).astype(int)

    data["Rates_Change"] = macro["^TNX"].diff(5).reindex(data.index)
    data["DXY_Change"] = macro["DX-Y.NYB"].pct_change(5).reindex(data.index)

    RF_SELECTED_TA_FEATURES = [
        "EMA_slow",
        "EMA_fast",
        "volatility_dcw",
        "volume_sma_em",
        "trend_adx",
        "volatility_bbh",
        "trend_visual_ichimoku_a",
        "volume_cmf",
        "trend_macd",
        "volatility_bbm",
        "volume_adi",
        "volume_vpt",
        "volume_obv",
        "trend_stc",
        "momentum_ppo_hist",
        "momentum_pvo_signal",
        "volatility_kcw",
        "trend_ichimoku_a",
        "momentum_pvo",
    ]
    ta_input = data[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    ta_features = add_all_ta_features(
        ta_input,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True,
    )

    # Remove raw OHLCV
    ta_features = ta_features.drop(
        columns=["Open", "High", "Low", "Volume"], errors="ignore"
    )

    rf_ta_features = ta_features[
        [c for c in RF_SELECTED_TA_FEATURES if c in ta_features.columns]
    ]

    # Merge back
    data = pd.concat([data, rf_ta_features], axis=1)

    # =====================
    # BASE FEATURES
    # =====================
    base_features = [
        "ROC_5",
        "BB_width",
        "EMA_Slope_Long",
        "ADX_14",
        "Rates_Change",
        "EMA_fast",
        "EMA_slow",
    ]

    final_feature_columns = base_features + list(rf_ta_features.columns)

    # =====================
    # Target (NO LEAK)
    # =====================
    fwd_ret_3d = data["Close"].shift(-3) / data["Close"] - 1
    data["y_signal"] = (fwd_ret_3d > thresh).astype(int)

    return data, final_feature_columns


start_date = "2018-01-01"
end_date = "2022-01-01"


# ---------------------------------------------------

# RF Control Panel


RF_n_months_training = 14
RF_retrain_every_weeks = 4
RF_ema_short = 20
RF_ema_long = 50
RF_sig_thresh = 0.001

rf_params = {
    "n_estimators": 300,
    "max_depth": 5,
    "min_samples_leaf": 10,
    "min_samples_split": 15,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


spx = yf.download("SPY", start_date, end_date, auto_adjust=True)
if isinstance(spx.columns, pd.MultiIndex):
    spx.columns = spx.columns.get_level_values(0)


features, feature_cols = rf_engineer_features(
    spx, RF_ema_short, RF_ema_long, RF_sig_thresh
)


features = features.dropna().reset_index(drop=True)
X = features[feature_cols]
y = features["y_signal"]
training_end = RF_n_months_training * 20
test_start = training_end + 1
step_days = 5 * RF_retrain_every_weeks
test_end = len(X)
X_train_full = X.iloc[:training_end]
y_train_full = y.iloc[:training_end]
X_test_full = X.iloc[test_start:]
test_index = features.index[test_start:]

# ======================================================
# WALK-FORWARD TRAINING
# ======================================================

signals_list = []
start_idx = 0

feature_importance_list = []

while start_idx < len(X_test_full):

    end_idx = min(start_idx + step_days, len(X_test_full))

    # Past-only training
    X_train = pd.concat([X_train_full, X_test_full.iloc[:start_idx]])
    y_train = pd.concat([y_train_full, y.iloc[test_start : test_start + start_idx]])

    base_rf = RandomForestClassifier(**rf_params)
    rf = CalibratedClassifierCV(base_rf, method="isotonic", cv=3)
    rf.fit(X_train, y_train)

    # ----------------------------
    # FEATURE IMPORTANCE (FIXED)
    # ----------------------------
    fitted_rf = rf.calibrated_classifiers_[0].estimator

    imp = pd.Series(fitted_rf.feature_importances_, index=X_train.columns)

    feature_importance_list.append(imp)

    # Predict on CURRENT window only
    X_test = X_test_full.iloc[start_idx:end_idx]
    y_prob_raw = rf.predict_proba(X_test)[:, 1]

    y_prob_series = pd.Series(y_prob_raw, index=X_test.index)  # keep proper dates/index
    y_prob_smooth = y_prob_series.rolling(window=3, min_periods=1).mean()
    y_prob = y_prob_smooth.values

    # Adaptive probability thresholds
    ema_fast = X_test["EMA_fast"].values
    ema_slow = X_test["EMA_slow"].values

    trend_long = ema_fast > ema_slow
    trend_short = ema_fast < ema_slow

    pos_signal = np.zeros(len(y_prob))

    hi = np.quantile(y_prob, 0.80)
    lo = np.quantile(y_prob, 0.20)

    # -------- LONG --------
    pos_signal[(y_prob > hi) | ((y_prob > 0.50) & (y_prob <= hi) & trend_long)] = 1

    # -------- SHORT --------
    pos_signal[(y_prob < lo) | ((y_prob >= lo) & (y_prob < 0.50) & trend_short)] = -1

    signals_list.append(pd.Series(pos_signal, index=test_index[start_idx:end_idx]))

    start_idx = end_idx

signals = pd.concat(signals_list)
feature_importances_df = pd.concat(feature_importance_list, axis=1)

# Feature Ranking
avg_importance = feature_importances_df.mean(axis=1)
avg_importance = avg_importance / avg_importance.sum()
feature_ranking = avg_importance.sort_values(ascending=False)
print(feature_ranking.to_string())


# ======================================================
# VECTORBT PORTFOLIO (FIXED 3-DAY HOLD)
# ======================================================


# Align close to signals
close_series = features["Close"].reindex(signals.index)

# Long / short entries
long_entries = signals == 1
short_entries = signals == -1

# Fixed 3-day holding
hold_days = 3
long_exits = long_entries.vbt.fshift(hold_days, fill_value=False)
short_exits = short_entries.vbt.fshift(hold_days, fill_value=False)

# Portfolio
pf = vbt.Portfolio.from_signals(
    close=close_series,
    entries=long_entries,
    exits=long_exits,
    short_entries=short_entries,
    short_exits=short_exits,
    init_cash=10_000,
    fees=0.0005,
)


print(pf.stats())

import plotly.graph_objects as go

# Portfolio equity curve
equity = pf.value()  # portfolio value over time

# Buy & Hold equity curve
buy_hold_equity = (
    features["Close"].iloc[test_start:test_end]
    / features["Close"].iloc[test_start]
    * 10000
)

# Create figure
fig = go.Figure()

# Add RF + EMA strategy
fig.add_trace(
    go.Scatter(
        x=features.index[test_start:test_end],
        y=equity,
        mode="lines",
        name="RF + EMA Strategy",
    )
)

# Add Buy & Hold SPY
fig.add_trace(
    go.Scatter(
        x=features.index[test_start:test_end],
        y=buy_hold_equity,
        mode="lines",
        name="Buy & Hold SPY",
    )
)

# Layout
fig.update_layout(
    title="Random Forest + EMA Strategy vs Buy & Hold",
    xaxis_title="Date",
    yaxis_title="Portfolio Value ($)",
    template="plotly_dark",
)

fig.show()
