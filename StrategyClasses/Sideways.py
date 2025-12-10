from backtesting import Strategy
import talib

class Sideways_ADX_RSI_ATR(Strategy):

    atr_period = 14
    atr_mult = 2
    rsi_period = 14
    rsi_low = 45
    rsi_high = 55
    adx_period = 14
    adx_threshold = 20

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low


        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period)

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


        if self.adx[-1] >= self.adx_threshold:
            return

    
        if self.rsi[-1] < self.rsi_low:
            if not any(trade.is_long for trade in self.trades):
                self.buy()
        elif self.rsi[-1] > self.rsi_high:
            if not any(trade.is_short for trade in self.trades):
                self.sell()
