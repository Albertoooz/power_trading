import backtrader as bt


class BreakoutStrategy(bt.Strategy):
    params = (
        ("period", 20),  # okres do szukania maksimum/minimum
    )

    def __init__(self):
        self.highest = bt.indicators.Highest(
            self.data.close(-1), period=self.params.period
        )
        # najniższa cena zamknięcia z ostatnich n okresów, z wyłączeniem bieżącego
        self.lowest = bt.indicators.Lowest(self.data.close(-1), period=self.params.period)

    def next(self):
        if not self.position:
            # jeśli aktualna cena przebija maksimum z ostatnich n okresów, kupujemy
            if self.data.close[0] > self.highest[0]:
                self.buy()
            # (opcjonalnie możesz dodać short, jeśli przebijamy minimum)
            elif self.data.close[0] < self.lowest[0]:
                self.sell()
        else:
            # wyjście z pozycji — np. przy zamknięciu poniżej minimum dla longów
            if self.position.size > 0 and self.data.close[0] < self.lowest[0]:
                self.close()
            elif self.position.size < 0 and self.data.close[0] > self.highest[0]:
                self.close()
