from backtesting import Strategy
import talib

class Trending_BB_SMA(Strategy):

    rsi_period = 10
    bb_period = 30
    sma_period = 192

    def init(self):
        close = self.data.Close

        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        upper, middle, lower = talib.BBANDS(close, timeperiod=self.bb_period)
        self.upper = self.I(lambda: upper)
        self.lower = self.I(lambda: lower)
        sma200 = talib.SMA(close, timeperiod=self.sma_period)
        self.sma200 = self.I(lambda: sma200)

    def next(self):
        price = self.data.Close[-1]

        if price > self.sma200[-1] and price <= self.lower[-1]:
            self.position.close()
            self.buy()
        elif price < self.sma200[-1] and price >= self.upper[-1]:
            self.position.close()
            self.sell()
