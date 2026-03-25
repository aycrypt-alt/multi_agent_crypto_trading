"""
Mean Reversion Strategy Agents

Agents that detect overextended price moves and trade the snap-back:
- Bollinger Band Reversion
- RSI Extremes
- Z-Score Reversion
"""

import math

from ...core.agent_base import Agent, AgentPriority
from ...core.message_bus import Message, MessageBus, MessageType
from ...utils.indicators import bollinger_bands, rsi, sma


class BollingerReversionAgent(Agent):
    """Trade mean reversion when price touches Bollinger Bands."""

    def __init__(self, message_bus: MessageBus, symbol: str, period: int = 20, std_dev: float = 2.0):
        super().__init__(
            name=f"bollinger_reversion_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
        )
        self.symbol = symbol
        self.period = period
        self.std_dev = std_dev
        self._prices: list[float] = []

    async def on_start(self):
        await self.subscribe(f"market_data.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type != MessageType.MARKET_DATA:
            return None

        self._prices.append(message.payload.get("close", 0.0))
        if len(self._prices) < self.period + 1:
            return None

        bands = bollinger_bands(self._prices, self.period, self.std_dev)
        if not bands["upper"]:
            return None

        price = self._prices[-1]
        upper = bands["upper"][-1]
        lower = bands["lower"][-1]
        middle = bands["middle"][-1]
        band_width = upper - lower

        signal = None
        strength = 0.0

        if price <= lower:
            signal = "long"
            strength = min(abs(lower - price) / (band_width + 1e-10), 1.0)
        elif price >= upper:
            signal = "short"
            strength = min(abs(price - upper) / (band_width + 1e-10), 1.0)

        if signal:
            self.metrics.signals_generated += 1
            await self.emit(
                MessageType.STRATEGY_SIGNAL,
                "signals",
                {
                    "symbol": self.symbol,
                    "direction": signal,
                    "strength": strength,
                    "confidence": self.confidence * 0.9,  # Slightly lower confidence for mean reversion
                    "strategy": "bollinger_reversion",
                },
            )
            return {"signal": signal, "strength": strength}
        return None


class RSIReversionAgent(Agent):
    """Trade RSI extremes — oversold bounces and overbought reversals."""

    def __init__(
        self, message_bus: MessageBus, symbol: str,
        period: int = 14, oversold: float = 30, overbought: float = 70,
    ):
        super().__init__(
            name=f"rsi_reversion_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
        )
        self.symbol = symbol
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self._prices: list[float] = []
        self._prev_rsi: float = 50.0

    async def on_start(self):
        await self.subscribe(f"market_data.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type != MessageType.MARKET_DATA:
            return None

        self._prices.append(message.payload.get("close", 0.0))
        rsi_values = rsi(self._prices, self.period)
        if not rsi_values:
            return None

        current_rsi = rsi_values[-1]
        signal = None
        strength = 0.0

        # Signal on RSI crossing back from extreme
        if self._prev_rsi < self.oversold <= current_rsi:
            signal = "long"
            strength = (self.oversold - self._prev_rsi) / self.oversold
        elif self._prev_rsi > self.overbought >= current_rsi:
            signal = "short"
            strength = (self._prev_rsi - self.overbought) / (100 - self.overbought)

        self._prev_rsi = current_rsi

        if signal:
            self.metrics.signals_generated += 1
            await self.emit(
                MessageType.STRATEGY_SIGNAL,
                "signals",
                {
                    "symbol": self.symbol,
                    "direction": signal,
                    "strength": min(strength, 1.0),
                    "confidence": self.confidence,
                    "strategy": "rsi_reversion",
                    "rsi": current_rsi,
                },
            )
            return {"signal": signal, "rsi": current_rsi}
        return None


class ZScoreReversionAgent(Agent):
    """Statistical mean reversion based on Z-Score of price relative to its mean."""

    def __init__(self, message_bus: MessageBus, symbol: str, lookback: int = 50, threshold: float = 2.0):
        super().__init__(
            name=f"zscore_reversion_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
        )
        self.symbol = symbol
        self.lookback = lookback
        self.threshold = threshold
        self._prices: list[float] = []

    async def on_start(self):
        await self.subscribe(f"market_data.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type != MessageType.MARKET_DATA:
            return None

        self._prices.append(message.payload.get("close", 0.0))
        if len(self._prices) < self.lookback:
            return None

        window = self._prices[-self.lookback:]
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std = math.sqrt(variance) if variance > 0 else 1e-10
        z_score = (self._prices[-1] - mean) / std

        signal = None
        if z_score < -self.threshold:
            signal = "long"
        elif z_score > self.threshold:
            signal = "short"

        if signal:
            strength = min(abs(z_score) / (self.threshold * 2), 1.0)
            self.metrics.signals_generated += 1
            await self.emit(
                MessageType.STRATEGY_SIGNAL,
                "signals",
                {
                    "symbol": self.symbol,
                    "direction": signal,
                    "strength": strength,
                    "confidence": self.confidence,
                    "strategy": "zscore_reversion",
                    "z_score": z_score,
                },
            )
            return {"signal": signal, "z_score": z_score}
        return None
