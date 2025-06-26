import backtrader as bt  # type: ignore


class OptimizedTechnicalStrategy(bt.Strategy):
    """Optimized technical strategy based on optimization results.

    Uses MACD, RSI, Bollinger Bands and Stochastic oscillator for generating trading
    signals. Parameters have been optimized using historical data.
    """

    params = (
        # MACD parameters (optimized)
        ("macd_fast", 9),
        ("macd_slow", 39),
        ("macd_signal", 15),
        # RSI parameters (optimized)
        ("rsi_period", 16),
        ("rsi_oversold", 35),
        ("rsi_overbought", 63),
        # Bollinger Bands parameters (optimized)
        ("bb_period", 15),
        ("bb_dev", 1.62),  # Rounded from 1.6201634445305326
        # Stochastic parameters (optimized)
        ("stoch_period", 20),
        ("stoch_smooth", 9),
        ("stoch_oversold", 28),
        ("stoch_overbought", 73),
        # Risk management
        ("risk_per_trade", 0.02),  # 2% risk per trade
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
            period_dfast=self.p.stoch_smooth,
            period_dslow=self.p.stoch_smooth,
        )

        # ATR for position sizing
        self.atr = bt.indicators.ATR(self.data, period=14)

        # Track performance
        self.equity_curve = []
        self.trades = []

    def normalize_macd(self):
        """Normalize MACD between -1 and 1 using recent history."""
        if len(self.data) < 20:  # Need at least 20 periods
            return 0

        # Calculate current MACD histogram
        current_hist = self.macd.macd[0] - self.macd.signal[0]

        # Find max absolute value in recent history
        max_hist = 0
        for i in range(20):
            hist = self.macd.macd[-i] - self.macd.signal[-i]
            max_hist = max(max_hist, abs(hist))

        if max_hist > 0:
            return current_hist / max_hist
        return 0

    def get_rsi_signal(self):
        """Convert RSI to a trading signal based on optimized thresholds."""
        if self.rsi[0] < self.p.rsi_oversold:
            return 1.0  # Strong buy signal
        elif self.rsi[0] > self.p.rsi_overbought:
            return -1.0  # Strong sell signal
        return 0  # Neutral

    def get_bb_signal(self):
        """Calculate position within Bollinger Bands."""
        mid = self.bb.mid[0]
        top = self.bb.top[0]
        bot = self.bb.bot[0]
        price = self.data.close[0]

        if price > top:
            return -1.0  # Strong sell signal
        elif price < bot:
            return 1.0  # Strong buy signal
        elif price > mid:
            return -0.5 * (price - mid) / (top - mid) if (top - mid) != 0 else 0
        else:
            return 0.5 * (mid - price) / (mid - bot) if (mid - bot) != 0 else 0

    def get_stoch_signal(self):
        """Get Stochastic signal based on optimized thresholds."""
        if (
            self.stoch.percK[0] < self.p.stoch_oversold
            and self.stoch.percD[0] < self.p.stoch_oversold
        ):
            return 1.0  # Strong buy signal
        elif (
            self.stoch.percK[0] > self.p.stoch_overbought
            and self.stoch.percD[0] > self.p.stoch_overbought
        ):
            return -1.0  # Strong sell signal
        return 0  # Neutral

    def calculate_position_size(self):
        """Calculate position size based on risk percentage and ATR."""
        price = self.data.close[0]
        atr = self.atr[0]

        if atr == 0 or price == 0:  # Avoid division by zero
            return 0

        risk_amount = self.broker.getvalue() * self.p.risk_per_trade
        stop_distance = atr * 2  # Use 2 ATR for stop loss
        position_size = risk_amount / (stop_distance * price)

        # Ensure minimum position size of 1 share if we want to trade
        if position_size > 0:
            return max(1, int(position_size))
        return 0

    def next(self):
        """Main strategy logic."""
        # Calculate signals from each indicator
        macd_signal = self.normalize_macd() * 1.0  # Weight: 1.0
        rsi_signal = self.get_rsi_signal() * 1.0  # Weight: 1.0
        bb_signal = self.get_bb_signal() * 0.5  # Weight: 0.5
        stoch_signal = self.get_stoch_signal() * 0.5  # Weight: 0.5

        # Combine signals
        total_signal = (
            macd_signal + rsi_signal + bb_signal + stoch_signal
        ) / 3.0  # Sum of weights

        # Calculate position size
        size = self.calculate_position_size()

        # Entry/exit logic
        if not self.position:  # No position
            if total_signal > 0.5:  # Strong buy signal
                self.buy(size=size)
                self.trades.append(
                    {
                        "date": self.data.datetime.date(),
                        "price": self.data.close[0],
                        "size": size,
                    }
                )
                self.log(f"BUY: Price={self.data.close[0]:.2f}, Size={size}")
            elif total_signal < -0.5:  # Strong sell signal
                self.sell(size=size)
                self.trades.append(
                    {
                        "date": self.data.datetime.date(),
                        "price": self.data.close[0],
                        "size": -size,
                    }
                )
                self.log(f"SELL: Price={self.data.close[0]:.2f}, Size={size}")
        else:  # Have position
            if self.position.size > 0:  # Long position
                if total_signal < -0.2:  # Exit long
                    self.close()
                    self.trades.append(
                        {
                            "date": self.data.datetime.date(),
                            "price": self.data.close[0],
                            "size": -self.position.size,
                        }
                    )
                    self.log(f"CLOSE LONG: Price={self.data.close[0]:.2f}")
            else:  # Short position
                if total_signal > 0.2:  # Exit short
                    self.close()
                    self.trades.append(
                        {
                            "date": self.data.datetime.date(),
                            "price": self.data.close[0],
                            "size": -self.position.size,
                        }
                    )
                    self.log(f"CLOSE SHORT: Price={self.data.close[0]:.2f}")

        # Track equity curve
        self.equity_curve.append(self.broker.getvalue())

        # Print status every month (approximately)
        if len(self.data) % 20 == 0:  # Every ~20 trading days
            self.log(
                f"\nStatus at {self.data.datetime.date()}:\n"
                f"Portfolio Value: {self.broker.getvalue():.2f}\n"
                f"MACD Histogram: {self.macd.macd[0] - self.macd.signal[0]:.4f}\n"
                f"RSI: {self.rsi[0]:.2f}\n"
                f"Stochastic K%: {self.stoch.percK[0]:.2f}\n"
                f"BB Position: {(self.data.close[0] - self.bb.mid[0]) / (self.bb.top[0] - self.bb.mid[0]):.2f}"  # noqa: E501
            )

    def log(self, txt, dt=None):
        """Logging function."""
        dt = dt or self.data.datetime.date()
        print(f"{dt}: {txt}")
