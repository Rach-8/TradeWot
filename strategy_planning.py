import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib



df_feat = pd.read_csv("ProcessedData/SPY_features_targets.csv")   
df_raw = pd.read_csv("RawData/spy_raw.csv")          

df_feat["Date"] = pd.to_datetime(df_feat["Date"])
df_raw["Date"] = pd.to_datetime(df_raw["Date"])

df = df_raw.merge(df_feat, on="Date", how="inner")
df = df.sort_values("Date").reset_index(drop=True)




class BBRSI(Strategy):
    def init(self):
        close = self.data.Close
        self.rsi = self.I(talib.RSI, close, 14)
        
        upper, middle, lower = talib.BBANDS(close, timeperiod=30)
        self.upper = self.I(lambda: upper)
        self.lower = self.I(lambda: lower)

    def next(self):
        price = self.data.Close[-1]

        # Long
        if price <= self.lower[-1] :
            self.position.close()
            self.buy()

        # Short
        elif price >= self.upper[-1] :
            self.position.close()
            self.sell()

bt = Backtest(df, BBRSI, commission=0.0002, cash=10_000,)
results = bt.run()
print(results)
bt.plot()
