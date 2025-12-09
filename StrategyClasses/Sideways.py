from backtesting import  Strategy
import talib




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
                stop_price = trade.entry_price - self.atr[-1]*self.atr_mult
                if price <= stop_price:
                    trade.close()
            elif trade.is_short:
                stop_price = trade.entry_price + self.atr[-1]*self.atr_mult
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
