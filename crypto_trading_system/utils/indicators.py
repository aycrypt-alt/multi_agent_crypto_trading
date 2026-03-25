"""
Technical Indicators — Math utilities for market analysis.

Pure-Python implementations (no numpy dependency required, but
uses it when available for performance).
"""

from __future__ import annotations

import math
from collections import deque


def sma(data: list[float], period: int) -> list[float]:
    """Simple Moving Average."""
    if len(data) < period:
        return []
    result = []
    window_sum = sum(data[:period])
    result.append(window_sum / period)
    for i in range(period, len(data)):
        window_sum += data[i] - data[i - period]
        result.append(window_sum / period)
    return result


def ema(data: list[float], period: int) -> list[float]:
    """Exponential Moving Average."""
    if not data:
        return []
    multiplier = 2 / (period + 1)
    result = [data[0]]
    for price in data[1:]:
        result.append((price - result[-1]) * multiplier + result[-1])
    return result


def rsi(data: list[float], period: int = 14) -> list[float]:
    """Relative Strength Index."""
    if len(data) < period + 1:
        return []
    deltas = [data[i] - data[i - 1] for i in range(1, len(data))]
    gains = [max(d, 0) for d in deltas]
    losses = [-min(d, 0) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    result = []
    for i in range(period, len(deltas)):
        if avg_loss == 0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100 - (100 / (1 + rs)))
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    return result


def macd(data: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """MACD (Moving Average Convergence Divergence)."""
    fast_ema = ema(data, fast)
    slow_ema = ema(data, slow)

    # Align lengths
    offset = len(fast_ema) - len(slow_ema)
    macd_line = [fast_ema[i + offset] - slow_ema[i] for i in range(len(slow_ema))]
    signal_line = ema(macd_line, signal)
    histogram = [macd_line[i + len(macd_line) - len(signal_line)] - signal_line[i]
                 for i in range(len(signal_line))]

    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def bollinger_bands(data: list[float], period: int = 20, std_dev: float = 2.0) -> dict:
    """Bollinger Bands."""
    if len(data) < period:
        return {"upper": [], "middle": [], "lower": []}

    middle = sma(data, period)
    upper = []
    lower = []

    for i in range(len(middle)):
        window = data[i:i + period]
        mean = middle[i]
        variance = sum((x - mean) ** 2 for x in window) / period
        sd = math.sqrt(variance)
        upper.append(mean + std_dev * sd)
        lower.append(mean - std_dev * sd)

    return {"upper": upper, "middle": middle, "lower": lower}


def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> list[float]:
    """Average True Range — key for position sizing and volatility measurement."""
    if len(highs) < 2:
        return []
    true_ranges = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)
    return sma(true_ranges, period)


def vwap(prices: list[float], volumes: list[float]) -> list[float]:
    """Volume Weighted Average Price."""
    if not prices or not volumes or len(prices) != len(volumes):
        return []
    cumulative_pv = 0.0
    cumulative_v = 0.0
    result = []
    for p, v in zip(prices, volumes):
        cumulative_pv += p * v
        cumulative_v += v
        result.append(cumulative_pv / cumulative_v if cumulative_v > 0 else p)
    return result


def fibonacci_levels(high: float, low: float) -> dict[str, float]:
    """Fibonacci retracement levels."""
    diff = high - low
    return {
        "0.0": high,
        "0.236": high - diff * 0.236,
        "0.382": high - diff * 0.382,
        "0.5": high - diff * 0.5,
        "0.618": high - diff * 0.618,
        "0.786": high - diff * 0.786,
        "1.0": low,
    }


def sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe Ratio (assumes daily returns)."""
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    daily_sharpe = (mean_r - risk_free_rate / 365) / std
    return daily_sharpe * math.sqrt(365)  # Annualize for crypto (365 days)


def sortino_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
    """Sortino Ratio — like Sharpe but only penalizes downside volatility."""
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    downside = [min(r - risk_free_rate / 365, 0) ** 2 for r in returns]
    downside_dev = math.sqrt(sum(downside) / len(downside))
    if downside_dev == 0:
        return 0.0
    return (mean_r - risk_free_rate / 365) / downside_dev * math.sqrt(365)


def max_drawdown(equity_curve: list[float]) -> float:
    """Maximum drawdown as a fraction (e.g., 0.15 = 15% drawdown)."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


class StreamingIndicator:
    """Efficient streaming indicator calculator for real-time data."""

    def __init__(self, period: int):
        self.period = period
        self._window: deque[float] = deque(maxlen=period)
        self._sum = 0.0

    def update(self, value: float) -> float | None:
        if len(self._window) == self.period:
            self._sum -= self._window[0]
        self._window.append(value)
        self._sum += value
        if len(self._window) == self.period:
            return self._sum / self.period
        return None
