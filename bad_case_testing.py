import pandas as pd
from backtesting import Backtest, Strategy
import talib


df_feat = pd.read_csv("ProcessedData/SPY_features_targets.csv")   
df_raw = pd.read_csv("RawData/spy_raw.csv")          

df_feat["Date"] = pd.to_datetime(df_feat["Date"])
df_raw["Date"] = pd.to_datetime(df_raw["Date"])

df = df_raw.merge(df_feat, on="Date", how="inner")
df = df.sort_values("Date").reset_index(drop=True)


periods = [
    ("1997-08-01", "1998-02-28"),
    ("2000-09-01", "2001-04-30"),
    ("2003-02-01", "2003-07-31"),
    ("2007-10-01", "2008-08-31"),
    ("2015-05-01", "2016-12-31"),
    ("2020-01-01", "2020-10-31"),
    ("2021-10-01", "2023-11-30")
]


class Sideways_ADX_RSI_ATR(Strategy):
    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.rsi = self.I(talib.RSI, close, timeperiod=14)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=14)
        self.atr_mult = 2
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=14)

    def next(self):
        price = self.data.Close[-1]

        for trade in self.trades:
            if trade.is_long:
                stop_price = trade.entry_price - self.atr[-1] * self.atr_mult
                if price <= stop_price:
                    trade.close()
            elif trade.is_short:
                stop_price = trade.entry_price + self.atr[-1] * self.atr_mult
                if price >= stop_price:
                    trade.close()

        if self.adx[-1] >= 15:
            return

        if self.rsi[-1] < 45:
            if not any(trade.is_long for trade in self.trades):
                self.buy()
        elif self.rsi[-1] > 55:
            if not any(trade.is_short for trade in self.trades):
                self.sell()


class Trending_BB_SMA(Strategy):
    def init(self):
        close = self.data.Close
        self.rsi = self.I(talib.RSI, close, 14)
        
        upper, middle, lower = talib.BBANDS(close, timeperiod=30)
        self.upper = self.I(lambda: upper)
        self.lower = self.I(lambda: lower)

        sma200 = talib.SMA(close, timeperiod=200)
        self.sma200 = self.I(lambda: sma200)

    def next(self):
        price = self.data.Close[-1]

        if price > self.sma200[-1] and price <= self.lower[-1]:
            self.position.close()
            self.buy()
        elif price < self.sma200[-1] and price >= self.upper[-1]:
            self.position.close()
            self.sell()


sideways_returns = []
trending_returns = []

for start_date, end_date in periods:
    df_subset = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    if df_subset.empty:
        continue
    df_subset.set_index('Date', inplace=True)


    bt_sideways = Backtest(df_subset, Sideways_ADX_RSI_ATR, commission=0.0002, cash=10_000, finalize_trades=True)
    res_sideways = bt_sideways.run()
    sideways_returns.append(res_sideways['Return [%]'])


    bt_trending = Backtest(df_subset, Trending_BB_SMA, commission=0.0002, cash=10_000, finalize_trades=True)
    res_trending = bt_trending.run()
    trending_returns.append(res_trending['Return [%]'])


avg_sideways_return = sum(sideways_returns) / len(sideways_returns)
avg_trending_return = sum(trending_returns) / len(trending_returns)

print(f"Average Sideways Strategy Return [%]: {avg_sideways_return:.2f}")
print(f"Average Trending Strategy Return [%]: {avg_trending_return:.2f}")
