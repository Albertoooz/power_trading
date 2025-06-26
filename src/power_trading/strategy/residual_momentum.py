from typing import Any

import backtrader as bt


class ResidualMomentumStrategy(bt.Strategy):
    """Residual momentum strategy with ATR-based stops and Bollinger Bands
    positioning."""

    params = (
        ("sma_period", 200),  # SMA period for trend following
        ("momentum_period", 90),  # Lookback period for momentum calculation
        ("atr_period", 90),  # ATR period for volatility-based stops
        ("atr_multiplier", 2.0),  # ATR multiplier for stop loss
        ("bb_period", 20),  # Bollinger Bands period
        ("bb_dev", 2.0),  # Number of standard deviations for Bollinger Bands
        ("pyramid_pct", 0.02),  # % move required for pyramid position
        ("profit_target_pct", 0.30),  # Take profit at 30% gain
        ("position_size", 0.95),  # Position size as fraction of portfolio
        ("max_pyramids", 3),  # Maximum number of pyramid entries
    )

    def __init__(self) -> None:
        """Initialize strategy components."""
        super().__init__()

        # Keep track of trades for visualization
        self.trades: list[dict[str, Any]] = []
        self.equity_curve: list[float] = []

        # Strategy components
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.sma_period)
        self.momentum = bt.indicators.ROC(self.data.close, period=self.p.momentum_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.bb = bt.indicators.BollingerBands(
            self.data.close, period=self.p.bb_period, devfactor=self.p.bb_dev
        )

        # Trading state
        self.entry_price = 0.0
        self.pyramid_target = 0.0
        self.pyramid_count = 0
        self.trailing_high = 0.0
        self.days_from_high = 0
        self.in_downtrend = False

    def next(self) -> None:
        """Execute trading logic for the next bar."""
        # Record equity curve
        self.equity_curve.append(self.broker.getvalue())

        # Update trailing high and days count
        if self.position:
            if self.data.close[0] > self.trailing_high:
                self.trailing_high = self.data.close[0]
                self.days_from_high = 0
            else:
                self.days_from_high += 1

        # Monthly check for market trend (first day of month)
        if len(self.data) > self.p.sma_period and self.data.datetime.date(0).day == 1:
            self._check_market_trend()

        # Check for profit target or stop loss if we have a position
        if self.position:
            self._manage_position()
            return

        # Entry logic
        if not self.position and not self.in_downtrend:
            self._check_entry()

    def _check_market_trend(self) -> None:
        """Check overall market trend and manage positions accordingly."""
        prev_downtrend = self.in_downtrend
        self.in_downtrend = self.data.close[0] < self.sma[0]

        # If entering downtrend, exit positions
        if self.in_downtrend and not prev_downtrend and self.position:
            self.close()
            self._record_trade("SELL", self.data.close[0])
            self._reset_trade_state()

        # If exiting downtrend, look for new entry
        elif not self.in_downtrend and prev_downtrend:
            self._check_entry()

    def _check_entry(self) -> None:
        """Check for entry conditions using Bollinger Bands."""
        # Only enter if price is near lower Bollinger Band
        if self.data.close[0] <= self.bb.lines.bot[0]:
            self._enter_position()

    def _manage_position(self) -> None:
        """Manage existing position - check stops, targets and pyramid opportunities."""
        if self.entry_price <= 0:
            return

        # Calculate current return
        current_return = (self.data.close[0] - self.entry_price) / self.entry_price

        # Check ATR-based stop loss
        atr_stop = self.trailing_high - (
            self.p.atr_multiplier * self.atr[0] * min(self.days_from_high / 6.0 + 1, 2.0)
        )

        if self.data.close[0] < atr_stop:
            self.close()
            self._record_trade("SELL", self.data.close[0])
            self._reset_trade_state()
            return

        # Check profit target
        if current_return >= self.p.profit_target_pct:
            self.close()
            self._record_trade("SELL", self.data.close[0])
            self._reset_trade_state()
            return

        # Check for pyramid opportunity
        if (
            self.pyramid_target > 0
            and self.data.close[0] >= self.pyramid_target
            and self.pyramid_count < self.p.max_pyramids
        ):
            self._add_pyramid_position()

    def _enter_position(self) -> None:
        """Enter initial position."""
        # cash = self.broker.getcash()
        value = self.broker.getvalue()

        # Calculate position size
        size = (value * self.p.position_size) / self.data.close[0]

        # Enter position
        self.buy(size=size)
        self._record_trade("BUY", self.data.close[0])

        # Set up for pyramid entries
        self.entry_price = self.data.close[0]
        self.pyramid_target = self.entry_price * (1 + self.p.pyramid_pct)
        self.pyramid_count = 1
        self.trailing_high = self.data.close[0]
        self.days_from_high = 0

    def _add_pyramid_position(self) -> None:
        """Add a pyramid position."""
        cash = self.broker.getcash()
        if cash <= 0:
            return

        # Calculate pyramid position size (half of initial)
        size = (cash * self.p.position_size) / self.data.close[0] / 2

        # Enter pyramid position
        self.buy(size=size)
        self._record_trade("BUY", self.data.close[0])

        # Update pyramid target
        self.pyramid_target = self.data.close[0] * (1 + self.p.pyramid_pct)
        self.pyramid_count += 1

    def _reset_trade_state(self) -> None:
        """Reset all trade-related state variables."""
        self.entry_price = 0.0
        self.pyramid_target = 0.0
        self.pyramid_count = 0
        self.trailing_high = 0.0
        self.days_from_high = 0

    def _record_trade(self, action: str, price: float) -> None:
        """Record trade for visualization."""
        self.trades.append(
            {
                "date": self.data.datetime.date(0),
                "price": price,
                "size": 1 if action == "BUY" else -1,
            }
        )
