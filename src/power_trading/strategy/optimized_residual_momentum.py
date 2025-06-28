from .residual_momentum import ResidualMomentumStrategy


class OptimizedResidualMomentumStrategy(ResidualMomentumStrategy):
    """Residual momentum strategy with parameters optimized using genetic algorithm."""

    params = (
        ("sma_period", 89),  # SMA period for trend following
        ("momentum_period", 180),  # Lookback period for momentum calculation
        ("atr_period", 39),  # ATR period for volatility-based stops
        ("atr_multiplier", 3.47),  # ATR multiplier for stop loss
        ("bb_period", 36),  # Bollinger Bands period
        ("bb_dev", 2.04),  # Number of standard deviations for Bollinger Bands
        ("pyramid_pct", 0.012),  # % move required for pyramid position
        ("profit_target_pct", 0.48),  # Take profit at 48% gain
        ("position_size", 0.98),  # Position size as fraction of portfolio
        ("max_pyramids", 5),  # Maximum number of pyramid entries
    )

    def __init__(self) -> None:
        """Initialize strategy - parent class handles all the setup."""
        super().__init__()
