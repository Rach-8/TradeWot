import pandas as pd
from backtesting import Backtest
from StrategyClasses.Sideways import Sideways_ADX_RSI_ATR
from StrategyClasses.Trending import Trending_BB_SMA

# Load data
df_feat = pd.read_csv("ProcessedData/SPY_features_targets.csv")   
df_raw = pd.read_csv("RawData/spy_raw.csv")          

df_feat["Date"] = pd.to_datetime(df_feat["Date"])
df_raw["Date"] = pd.to_datetime(df_raw["Date"])

df = df_raw.merge(df_feat, on="Date", how="inner").sort_values("Date").reset_index(drop=True)



choppy_periods = [
    ('1997-08-01', '1998-02-28'),
    ('2000-09-01', '2001-04-30'),
    ('2003-02-01', '2003-07-31'),
    ('2007-10-01', '2008-08-31'),
    ('2015-05-01', '2016-12-31'),
    ('2020-01-01', '2020-10-31'),
    ('2021-10-01', '2023-11-30')
]

trending_periods = [
    ('1993-03-01', '1997-07-30'),
    ('1999-03-01', '2000-08-30'),
    ('2003-07-01', '2007-10-30'),
    ('2009-03-01', '2020-01-30'),
    ('2023-12-01', '2025-10-30')
]

sideways_results = []

for start, end in choppy_periods:
    df_subset = df[(df['Date'] >= start) & (df['Date'] <= end)].copy()
    df_subset.set_index('Date', inplace=True)

    bt = Backtest(df_subset, Sideways_ADX_RSI_ATR,  cash=10000000000000, finalize_trades=True)

    # Optimize parameters: RSI thresholds and ATR multiplier
    result = bt.optimize(
    adx_period=range(10, 21, 2),        # test RSI period from 10 to 20
    atr_period=range(10, 21, 2),          # when to buy
    adx_threshold=range(10, 21, 2),          # when to sell


    maximize='Sharpe Ratio'
)

    sideways_results.append(result)

# Average optimized parameters for sideways strategy
#adx_period = sum(r._strategy.rsi_period for r in sideways_results) / len(sideways_results)
avg_atr_period = sum(r._strategy.atr_period for r in sideways_results) / len(sideways_results)
#avg_rsi_long = sum(r._strategy.rsi_low for r in sideways_results)/len(sideways_results)
#avg_rsi_short = sum(r._strategy.rsi_high for r in sideways_results)/len(sideways_results)
#avg_atr_mult = sum(r._strategy.atr_mult for r in sideways_results)/len(sideways_results)
adx_threshold = sum(r._strategy.adx_threshold for r in sideways_results)/len(sideways_results)
avg_adx_period = sum(r._strategy.adx_period for r in sideways_results) / len(sideways_results)

#print("Optimized Sideways Strategy Parameters:")
#print("RSI Period:", avg_rsi_period)
print("ATR Period:", avg_atr_period)
print("ATR Period:", avg_adx_period)
#print("RSI Long Threshold:", avg_rsi_long)
#print("RSI Short Threshold:", avg_rsi_short)
#print("ATR Multiplier:", avg_atr_mult)
print("ADX Threshold:", adx_threshold)

