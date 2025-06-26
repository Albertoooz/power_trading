import backtrader as bt


class WeightedTechnicalStrategy(bt.Strategy):
    params = (
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("rsi_period", 14),
        ("bb_period", 20),
        ("bb_dev", 2),
        ("stoch_period", 14),
        ("stoch_k", 3),
        ("stoch_d", 3),
        ("buy_threshold", 0.3),
        ("sell_threshold", -0.3),
        ("risk_pct", 0.02),  # Risk 2% per trade
    )

    def __init__(self):
        # MACD
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal,
        )

        # RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

        # Bollinger Bands
        self.bb = bt.indicators.BollingerBands(
            self.data.close, period=self.p.bb_period, devfactor=self.p.bb_dev
        )

        # Stochastic
        self.stoch = bt.indicators.Stochastic(
            self.data,
            period=self.p.stoch_period,
            period_dfast=self.p.stoch_k,
            period_dslow=self.p.stoch_d,
        )

        self.equity_curve = []
        self.trades = []

    def normalize_macd(self):
        # Normalize MACD between -1 and 1 using recent history
        macd_hist = self.macd.macd - self.macd.signal
        if len(macd_hist) > 0:
            max_hist = max(abs(macd_hist[-20:]))  # Use last 20 periods
            return macd_hist[0] / max_hist if max_hist != 0 else 0
        return 0

    def normalize_rsi(self):
        # Convert RSI to -1 to 1 scale
        return (self.rsi[0] - 50) / 50

    def normalize_bb(self):
        # Calculate position within Bollinger Bands
        mid = self.bb.mid[0]
        top = self.bb.top[0]
        bot = self.bb.bot[0]
        price = self.data.close[0]

        if price > mid:
            return (price - mid) / (top - mid) if (top - mid) != 0 else 0
        else:
            return (price - mid) / (mid - bot) if (mid - bot) != 0 else 0

    def normalize_stoch(self):
        # Convert Stochastic to -1 to 1 scale
        return (self.stoch.percK[0] - 50) / 50

    def calculate_position_size(self):
        # Calculate position size based on risk percentage
        price = self.data.close[0]
        atr = bt.indicators.ATR(self.data, period=14)[0]

        if atr == 0:  # Avoid division by zero
            return 0

        risk_amount = self.broker.getvalue() * self.p.risk_pct
        stop_distance = atr * 2  # Use 2 ATR for stop loss

        return int(risk_amount / (price * stop_distance))

    def next(self):
        self.equity_curve.append(self.broker.getvalue())

        # Calculate weighted signal
        macd_signal = self.normalize_macd() * 1.0  # Weight: 1.0
        rsi_signal = self.normalize_rsi() * 1.0  # Weight: 1.0
        bb_signal = self.normalize_bb() * 0.5  # Weight: 0.5
        stoch_signal = self.normalize_stoch() * 0.5  # Weight: 0.5

        # Combined signal
        total_signal = (
            macd_signal + rsi_signal + bb_signal + stoch_signal
        ) / 3.0  # Sum of weights = 3.0

        # Debug prints
        if len(self.equity_curve) % 20 == 0:  # Print every 20 bars
            print(f"\nSignal Components at {self.data.datetime.date(0)}:")
            print(f"MACD: {macd_signal:.3f}")
            print(f"RSI: {rsi_signal:.3f}")
            print(f"BB: {bb_signal:.3f}")
            print(f"Stoch: {stoch_signal:.3f}")
            print(f"Total: {total_signal:.3f}")

        position_size = self.calculate_position_size()

        if not self.position:
            if total_signal > self.p.buy_threshold:
                self.buy(size=position_size)
                self.trades.append(
                    {
                        "date": self.data.datetime.date(0),
                        "price": self.data.close[0],
                        "size": position_size,
                    }
                )
        else:
            if total_signal < self.p.sell_threshold:
                self.close()
                self.trades.append(
                    {
                        "date": self.data.datetime.date(0),
                        "price": self.data.close[0],
                        "size": -position_size,
                    }
                )
