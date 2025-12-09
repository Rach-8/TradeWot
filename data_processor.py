import pandas as pd
import ta
import os
import numpy as np

# -----------------------------
# Load & clean raw data
# -----------------------------
spy = pd.read_csv("RawData/Spy_raw.csv")

# Remove unnecessary columns
if 'Sno' in spy.columns:
    spy = spy.drop(columns=['Sno'])

# Keep consistent ordering
spy = spy[['Date','Close','High','Low','Open','Volume']]

spy['Date'] = pd.to_datetime(spy['Date'])

# -----------------------------
# Technical Indicators
# -----------------------------
spy['EMA20'] = ta.trend.ema_indicator(spy['Close'], window=20)
spy['EMA50'] = ta.trend.ema_indicator(spy['Close'], window=50)
spy['EMA200'] = ta.trend.ema_indicator(spy['Close'], window=200)
spy['SMA50'] = ta.trend.sma_indicator(spy['Close'], window=50)


spy['RSI14'] = ta.momentum.rsi(spy['Close'], window=14)
spy['Stoch_k'] = ta.momentum.stoch(
    high=spy['High'],
    low=spy['Low'],
    close=spy['Close'],
    window=14,
    smooth_window=3
)

spy['ROC5'] = ta.momentum.roc(spy['Close'], window=5)
spy['ATR14'] = ta.volatility.average_true_range(
    high=spy['High'],
    low=spy['Low'],
    close=spy['Close'],
    window=14
)
spy['BB_width'] = (
    ta.volatility.bollinger_hband(spy['Close'], window=20, window_dev=2)
    - ta.volatility.bollinger_lband(spy['Close'], window=20, window_dev=2)
)

# -----------------------------
# Volume features
# -----------------------------
spy['Volume_MA5'] = spy['Volume'].rolling(5).mean()
spy['Volume_Ratio5'] = spy['Volume'] / spy['Volume_MA5']
spy['Volume_MA10'] = spy['Volume'].rolling(10).mean()
spy['Volume_Ratio10'] = spy['Volume'] / spy['Volume_MA10']
spy['Volume_log'] = np.log1p(spy['Volume'])
spy['Volume_std5'] = spy['Volume'].rolling(5).std()

# -----------------------------
# Trend differences
# -----------------------------
spy['EMA20_EMA50_diff'] = spy['EMA20'] - spy['EMA50']
spy['EMA50_EMA200_diff'] = spy['EMA50'] - spy['EMA200']

# -----------------------------
# Lagged features
# -----------------------------
for lag in [1, 2, 3, 5]:
    spy[f'Close_lag{lag}'] = spy['Close'].shift(lag)
    spy[f'EMA20_lag{lag}'] = spy['EMA20'].shift(lag)
    spy[f'RSI14_lag{lag}'] = spy['RSI14'].shift(lag)

# -----------------------------
# Future returns (targets)
# -----------------------------
spy['Return1d'] = spy['Close'].shift(-1) / spy['Close'] - 1
spy['Return5d'] = spy['Close'].shift(-5) / spy['Close'] - 1
spy['Target_1d'] = (spy['Return1d'] > 0).astype(int)
spy['Target_5d'] = (spy['Return5d'] > 0).astype(int)

# Drop last 5 rows (cannot compute 5d target)
spy.dropna(inplace=True)

# -----------------------------
# Optional: encode date
# -----------------------------
spy['DayOfWeek'] = spy['Date'].dt.dayofweek
spy['Month'] = spy['Date'].dt.month

# -----------------------------
# Feature list
# -----------------------------
feature_cols = ['Date',
    'EMA20','EMA50','EMA200','SMA50',
    'RSI14','Stoch_k','ROC5',
    'ATR14','BB_width',
    'Volume_MA5','Volume_Ratio5','Volume_MA10','Volume_Ratio10',
    'Volume_log','Volume_std5',
    'EMA20_EMA50_diff','EMA50_EMA200_diff',
    'Close_lag1','Close_lag2','Close_lag3','Close_lag5',
    'EMA20_lag1','EMA20_lag2','EMA20_lag3','EMA20_lag5',
    'RSI14_lag1','RSI14_lag2','RSI14_lag3','RSI14_lag5',
    'DayOfWeek','Month',
    'Target_1d','Target_5d'
]

# -----------------------------
# Save processed features & targets
# -----------------------------
os.makedirs("ProcessedData", exist_ok=True)
spy[feature_cols].to_csv("ProcessedData/SPY_features_targets.csv", index=False)
print("ProcessedData/SPY_features_targets.csv saved with improved features.")
