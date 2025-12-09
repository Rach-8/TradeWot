import pandas as pd
from backtesting import Backtest
from StrategyClasses.Sideways import Sideways_ADX_RSI_ATR
from StrategyClasses.Trending import Trending_BB_SMA



df_feat = pd.read_csv("ProcessedData/SPY_features_targets.csv")   
df_raw = pd.read_csv("RawData/spy_raw.csv")          

df_feat["Date"] = pd.to_datetime(df_feat["Date"])
df_raw["Date"] = pd.to_datetime(df_raw["Date"])

df = df_raw.merge(df_feat, on="Date", how="inner")
df = df.sort_values("Date").reset_index(drop=True)

start_date = '2021-11-01'
end_date = '2023-12-01'
df_subset = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
df_subset.set_index('Date', inplace=True) 

bh_return = df_subset['Close'].iloc[-1] / df_subset['Close'].iloc[0] - 1




bt = Backtest(df_subset, Sideways_ADX_RSI_ATR, commission=0.0002, cash=10_000,finalize_trades=True)
results = bt.run()
print("Manual Buy & Hold Return:", bh_return)
print(results)
bt.plot()
