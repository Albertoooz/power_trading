from typing import Any

import backtrader as bt


class MomentumGtaaStrategy(bt.Strategy):
    """Momentum strategy with Global Tactical Asset Allocation and pyramid
    positioning."""

    params = (
        ("sma_period", 200),  # SMA period for trend following
        ("momentum_period", 90),  # Lookback period for momentum calculation
        ("pyramid_pct", 0.02),  # % move required for pyramid position
        ("profit_target_pct", 0.30),  # Take profit at 30% gain
        ("stop_loss_pct", 0.25),  # Stop loss at 25% drawdown
        ("position_size", 0.95),  # Position size as fraction of portfolio
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

        # Trading state
        self.entry_price = 0.0
        self.pyramid_target = 0.0
        self.pyramid_count = 0
        self.max_pyramids = 3  # Maximum number of pyramid entries

    def next(self) -> None:
        """Execute trading logic for the next bar."""
        # Record equity curve
        self.equity_curve.append(self.broker.getvalue())

        # Monthly check for market trend (first day of month)
        if len(self.data) > self.p.sma_period and self.data.datetime.date(0).day == 1:
            self._check_market_trend()

        # Check for profit target or stop loss if we have a position
        if self.position:
            self._manage_position()
            return

        # Entry logic - only enter if price is above SMA (uptrend)
        if not self.position and self.data.close[0] > self.sma[0]:
            self._enter_position()

    def _check_market_trend(self) -> None:
        """Check overall market trend and exit if bearish."""
        if self.data.close[0] < self.sma[0] and self.position:
            # Market in downtrend - exit position
            self.close()
            self._record_trade("SELL", self.data.close[0])
            self.entry_price = 0.0
            self.pyramid_target = 0.0
            self.pyramid_count = 0

    def _manage_position(self) -> None:
        """Manage existing position - check stops, targets and pyramid opportunities."""
        # Calculate current return
        if self.entry_price <= 0:
            return

        current_return = (self.data.close[0] - self.entry_price) / self.entry_price

        # Check stop loss
        if current_return <= -self.p.stop_loss_pct:
            self.close()
            self._record_trade("SELL", self.data.close[0])
            self.entry_price = 0.0
            self.pyramid_target = 0.0
            self.pyramid_count = 0
            return

        # Check profit target
        if current_return >= self.p.profit_target_pct:
            self.close()
            self._record_trade("SELL", self.data.close[0])
            self.entry_price = 0.0
            self.pyramid_target = 0.0
            self.pyramid_count = 0
            return

        # Check for pyramid opportunity
        if (
            self.pyramid_target > 0
            and self.data.close[0] >= self.pyramid_target
            and self.pyramid_count < self.max_pyramids
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

    def _record_trade(self, action: str, price: float) -> None:
        """Record trade for visualization."""
        self.trades.append(
            {
                "date": self.data.datetime.date(0),
                "price": price,
                "size": 1 if action == "BUY" else -1,
            }
        )
