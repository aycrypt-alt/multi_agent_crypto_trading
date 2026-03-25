"""
Market Analysis Agents

Agents dedicated to analyzing market conditions and providing
context to strategy agents:
- Technical Analysis (support/resistance, chart patterns)
- Volatility Analysis (regime detection)
- Correlation Analysis (cross-asset relationships)
- Sentiment Analysis (funding rates, open interest)
"""

import math
import time

from ...core.agent_base import Agent, AgentPriority
from ...core.message_bus import Message, MessageBus, MessageType
from ...utils.indicators import atr, bollinger_bands, rsi, sma, ema


class TechnicalAnalysisAgent(Agent):
    """Identifies support/resistance levels and key price zones."""

    def __init__(self, message_bus: MessageBus, symbol: str):
        super().__init__(
            name=f"tech_analysis_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.LOW,
        )
        self.symbol = symbol
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._closes: list[float] = []
        self._volumes: list[float] = []
        self._analysis_interval = 60  # Analyze every 60 data points
        self._counter = 0

    async def on_start(self):
        await self.subscribe(f"market_data.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type != MessageType.MARKET_DATA:
            return None

        self._highs.append(message.payload.get("high", 0.0))
        self._lows.append(message.payload.get("low", 0.0))
        self._closes.append(message.payload.get("close", 0.0))
        self._volumes.append(message.payload.get("volume", 0.0))
        self._counter += 1

        if self._counter % self._analysis_interval != 0:
            return None
        if len(self._closes) < 100:
            return None

        # Find support/resistance via pivot points
        support_levels = self._find_support_resistance(self._lows[-200:], mode="support")
        resistance_levels = self._find_support_resistance(self._highs[-200:], mode="resistance")

        # Trend analysis
        sma_20 = sma(self._closes, 20)
        sma_50 = sma(self._closes, 50)
        trend = "up" if sma_20 and sma_50 and sma_20[-1] > sma_50[-1] else "down"

        # Volume profile
        avg_vol = sum(self._volumes[-50:]) / 50 if len(self._volumes) >= 50 else 0
        current_vol = self._volumes[-1]
        vol_trend = "increasing" if current_vol > avg_vol * 1.2 else (
            "decreasing" if current_vol < avg_vol * 0.8 else "normal"
        )

        analysis = {
            "symbol": self.symbol,
            "support_levels": support_levels[:3],
            "resistance_levels": resistance_levels[:3],
            "trend": trend,
            "volume_trend": vol_trend,
            "current_price": self._closes[-1],
            "timestamp": time.time(),
        }

        await self.emit(MessageType.ANALYSIS_RESULT, f"analysis.{self.symbol}", analysis)
        return analysis

    def _find_support_resistance(self, prices: list[float], mode: str, window: int = 10) -> list[float]:
        """Find local minima (support) or maxima (resistance)."""
        levels = []
        for i in range(window, len(prices) - window):
            segment = prices[i - window:i + window + 1]
            if mode == "support" and prices[i] == min(segment):
                levels.append(prices[i])
            elif mode == "resistance" and prices[i] == max(segment):
                levels.append(prices[i])
        # Cluster nearby levels
        return self._cluster_levels(levels)

    def _cluster_levels(self, levels: list[float], tolerance: float = 0.005) -> list[float]:
        """Group nearby price levels together."""
        if not levels:
            return []
        sorted_levels = sorted(levels)
        clusters: list[list[float]] = [[sorted_levels[0]]]
        for level in sorted_levels[1:]:
            if abs(level - clusters[-1][-1]) / clusters[-1][-1] < tolerance:
                clusters[-1].append(level)
            else:
                clusters.append([level])
        # Return average of each cluster, sorted by frequency (most touches first)
        result = [(sum(c) / len(c), len(c)) for c in clusters]
        result.sort(key=lambda x: -x[1])
        return [r[0] for r in result]


class VolatilityRegimeAgent(Agent):
    """Detects current volatility regime to help other agents adapt."""

    def __init__(self, message_bus: MessageBus, symbol: str):
        super().__init__(
            name=f"volatility_regime_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
        )
        self.symbol = symbol
        self._closes: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._current_regime = "normal"

    async def on_start(self):
        await self.subscribe(f"market_data.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type != MessageType.MARKET_DATA:
            return None

        self._closes.append(message.payload.get("close", 0.0))
        self._highs.append(message.payload.get("high", 0.0))
        self._lows.append(message.payload.get("low", 0.0))

        if len(self._closes) < 50:
            return None

        # Calculate realized volatility
        returns = [(self._closes[i] / self._closes[i - 1]) - 1 for i in range(-20, 0)]
        realized_vol = math.sqrt(sum(r ** 2 for r in returns) / len(returns)) * math.sqrt(365) * 100

        # ATR-based volatility
        atr_values = atr(self._highs, self._lows, self._closes, 14)
        current_atr = atr_values[-1] if atr_values else 0
        avg_atr = sum(atr_values[-50:]) / min(len(atr_values), 50) if atr_values else 0
        atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

        # Classify regime
        if realized_vol > 100 or atr_ratio > 2.0:
            regime = "extreme"
        elif realized_vol > 60 or atr_ratio > 1.5:
            regime = "high"
        elif realized_vol < 20 or atr_ratio < 0.5:
            regime = "low"
        else:
            regime = "normal"

        if regime != self._current_regime:
            self._current_regime = regime
            await self.emit(
                MessageType.REGIME_CHANGE,
                "broadcast",
                {
                    "symbol": self.symbol,
                    "regime": regime,
                    "realized_vol": realized_vol,
                    "atr_ratio": atr_ratio,
                },
                priority=AgentPriority.HIGH.value,
            )

        await self.emit(
            MessageType.ANALYSIS_RESULT,
            f"analysis.{self.symbol}",
            {
                "type": "volatility",
                "symbol": self.symbol,
                "regime": regime,
                "realized_vol_annualized": round(realized_vol, 2),
                "atr_ratio": round(atr_ratio, 2),
            },
        )
        return {"regime": regime, "vol": realized_vol}


class CorrelationAgent(Agent):
    """
    Monitors cross-asset correlations.
    Detects when correlations break down (potential trading opportunity)
    or when correlation spikes (risk event / contagion).
    """

    def __init__(self, message_bus: MessageBus, symbols: list[str]):
        super().__init__(
            name="correlation_monitor",
            message_bus=message_bus,
            priority=AgentPriority.LOW,
        )
        self.symbols = symbols
        self._price_data: dict[str, list[float]] = {s: [] for s in symbols}
        self._lookback = 30

    async def on_start(self):
        for symbol in self.symbols:
            await self.subscribe(f"market_data.{symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type != MessageType.MARKET_DATA:
            return None

        symbol = message.payload.get("symbol", "")
        if symbol not in self._price_data:
            return None

        self._price_data[symbol].append(message.payload.get("close", 0.0))

        # Only compute when all symbols have enough data
        min_len = min(len(v) for v in self._price_data.values())
        if min_len < self._lookback:
            return None

        # Compute correlation matrix
        correlations = {}
        for i, s1 in enumerate(self.symbols):
            for s2 in self.symbols[i + 1:]:
                corr = self._pearson_correlation(
                    self._price_data[s1][-self._lookback:],
                    self._price_data[s2][-self._lookback:],
                )
                correlations[f"{s1}/{s2}"] = round(corr, 3)

        await self.emit(
            MessageType.ANALYSIS_RESULT,
            "analysis.correlation",
            {"correlations": correlations, "lookback": self._lookback},
        )
        return correlations

    @staticmethod
    def _pearson_correlation(x: list[float], y: list[float]) -> float:
        n = len(x)
        if n < 2:
            return 0.0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)
        if std_x == 0 or std_y == 0:
            return 0.0
        return cov / (std_x * std_y)


class SentimentAgent(Agent):
    """
    Monitors on-chain / exchange sentiment indicators:
    - Funding rates (positive = longs pay shorts, market bullish)
    - Open interest changes
    - Long/short ratio
    """

    def __init__(self, message_bus: MessageBus, symbol: str):
        super().__init__(
            name=f"sentiment_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.LOW,
        )
        self.symbol = symbol
        self._funding_history: list[float] = []
        self._oi_history: list[float] = []

    async def on_start(self):
        await self.subscribe(f"funding.{self.symbol}")
        await self.subscribe(f"open_interest.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        payload = message.payload

        if "funding_rate" in payload:
            self._funding_history.append(payload["funding_rate"])
        if "open_interest" in payload:
            self._oi_history.append(payload["open_interest"])

        if len(self._funding_history) < 3:
            return None

        avg_funding = sum(self._funding_history[-8:]) / min(len(self._funding_history), 8)

        # Extreme funding = potential reversal signal
        sentiment = "neutral"
        if avg_funding > 0.01:
            sentiment = "extreme_greed"
        elif avg_funding > 0.005:
            sentiment = "bullish"
        elif avg_funding < -0.01:
            sentiment = "extreme_fear"
        elif avg_funding < -0.005:
            sentiment = "bearish"

        result = {
            "symbol": self.symbol,
            "sentiment": sentiment,
            "avg_funding_rate": avg_funding,
            "funding_trend": "rising" if len(self._funding_history) > 1
                            and self._funding_history[-1] > self._funding_history[-2]
                            else "falling",
        }

        await self.emit(MessageType.SENTIMENT_UPDATE, f"sentiment.{self.symbol}", result)
        return result
