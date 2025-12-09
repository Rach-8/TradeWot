from backtesting import  Strategy
import talib


class Trending_BB_SMA(Strategy):
    def init(self):
        close = self.data.Close
        self.rsi = self.I(talib.RSI, close, 14)
        
        upper, middle, lower = talib.BBANDS(close, timeperiod=30)
        self.upper = self.I(lambda: upper)
        self.lower = self.I(lambda: lower)

        sma200 = talib.SMA(close,timeperiod=200)
        self.sma200 = self.I(lambda:sma200)

  


    def next(self):
        price = self.data.Close[-1]


        if price > self.sma200[-1] and price <= self.lower[-1]:
            self.position.close()
            self.buy()

   
        elif price < self.sma200[-1] and price >= self.upper[-1]:
            self.position.close()
            self.sell()

