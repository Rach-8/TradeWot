import pandas as pd
import ta
import os


spy = pd.read_csv("RawData/Spy_raw.csv")


if 'Sno' in spy.columns:
    spy = spy.drop(columns=['Sno'])


spy = spy[['Date','Close','High','Low','Open','Volume']]


spy['Date'] = pd.to_datetime(spy['Date'])


spy['EMA20'] = ta.trend.ema_indicator(spy['Close'], window=20)
spy['EMA50'] = ta.trend.ema_indicator(spy['Close'], window=50)
spy['EMA200'] = ta.trend.ema_indicator(spy['Close'], window=200)
spy['SMA50'] = ta.trend.sma_indicator(spy['Close'], window=50)

spy['RSI14'] = ta.momentum.rsi(spy['Close'], window=14)
spy['Stoch_k'] = ta.momentum.stoch(high=spy['High'], low=spy['Low'], close=spy['Close'], window=14, smooth_window=3)
spy['ROC5'] = ta.momentum.roc(spy['Close'], window=5)

spy['ATR14'] = ta.volatility.average_true_range(high=spy['High'], low=spy['Low'], close=spy['Close'], window=14)
spy['BB_width'] = ta.volatility.bollinger_hband(spy['Close'], window=20, window_dev=2) - ta.volatility.bollinger_lband(spy['Close'], window=20, window_dev=2)


spy['Return1d'] = spy['Close'].pct_change(1)
spy['Return5d'] = spy['Close'].pct_change(5)


spy['Volume_MA5'] = spy['Volume'].rolling(window=5).mean()
spy['Volume_Ratio'] = spy['Volume'] / spy['Volume_MA5']


spy['Target_1d'] = (spy['Return1d'] > 0).astype(int)  
spy['Target_5d'] = (spy['Return5d'] > 0).astype(int)  


spy.dropna(inplace=True)


os.makedirs("ProcessedData", exist_ok=True)  

feature_cols = ['Date','EMA20','EMA50','EMA200','SMA50','RSI14','Stoch_k','ROC5',
                'ATR14','BB_width','Return1d','Return5d','Volume_MA5','Volume_Ratio',
                'Target_1d','Target_5d']

spy[feature_cols].to_csv("ProcessedData/SPY_features_targets.csv", index=False)

print("Features + targets CSV saved in ProcessedData/SPY_features_targets.csv")
