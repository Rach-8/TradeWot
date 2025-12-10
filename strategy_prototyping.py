import pandas as pd
from backtesting import Backtest

from StrategyClasses.Sideways import Sideways_ADX_RSI_ATR
from StrategyClasses.Trending import Trending_BB_SMA

start = '2021-10-01'
end = '2023-11-30'

df_raw = pd.read_csv("./Data/SPY.csv")
df_raw["Adj Close"] = pd.to_numeric(df_raw["Adj Close"], errors="coerce")
df_raw["Date"] = pd.to_datetime(df_raw["Date"])
df_raw = df_raw.sort_values("Date").reset_index(drop=True)

df_subset = df_raw[(df_raw['Date'] >= start) & (df_raw['Date'] <= end)].copy()
df_subset.set_index('Date', inplace=True)

df_subset["Returns"] = df_subset["Adj Close"].pct_change()
df_subset.dropna(inplace=True)


# Run backtest
bt = Backtest(df_subset, Sideways_ADX_RSI_ATR, cash=100000, commission=0.001)
stats = bt.run()
print(stats)
bt.plot()
