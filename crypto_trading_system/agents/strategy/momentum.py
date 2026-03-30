"""
Momentum Strategy Agents

Agents that detect and ride strong directional moves:
- Rate of Change (ROC)
- Volume-Weighted Momentum
- Multi-Timeframe Momentum
"""

import math

from ...core.agent_base import Agent, AgentPriority
from ...core.message_bus import Message, MessageBus, MessageType
from ...utils.indicators import ema, rsi, adx, volume_ratio


class ROCMomentumAgent(Agent):
    """Rate of Change momentum — enters when ROC accelerates.
    Enhanced with higher threshold, ADX confirmation, and longer cooldown."""

    SIGNAL_COOLDOWN = 15

    def __init__(self, message_bus: MessageBus, symbol: str, period: int = 12, threshold: float = 3.0):
        super().__init__(
            name=f"roc_momentum_{symbol}_{period}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
        )
        self.symbol = symbol
        self.period = period
        self.threshold = threshold
        self._prices: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._candle_count = 0
        self._last_signal_candle = -self.SIGNAL_COOLDOWN

    async def on_start(self):
        await self.subscribe(f"market_data.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type != MessageType.MARKET_DATA:
            return None

        self._prices.append(message.payload.get("close", 0.0))
        self._highs.append(message.payload.get("high", 0.0))
        self._lows.append(message.payload.get("low", 0.0))
        self._candle_count += 1
        if len(self._prices) < self.period + 1:
            return None

        if self._candle_count - self._last_signal_candle < self.SIGNAL_COOLDOWN:
            return None

        roc = ((self._prices[-1] - self._prices[-self.period - 1])
               / self._prices[-self.period - 1]) * 100

        if abs(roc) < self.threshold:
            return None

        # Require ADX > 20 to confirm momentum is backed by trend
        adx_values = adx(self._highs, self._lows, self._prices, 14)
        if adx_values and adx_values[-1] < 20:
            return None

        signal = "long" if roc > 0 else "short"
        strength = min(abs(roc) / (self.threshold * 3), 1.0)

        self._last_signal_candle = self._candle_count
        self.metrics.signals_generated += 1
        await self.emit(
            MessageType.STRATEGY_SIGNAL,
            "signals",
            {
                "symbol": self.symbol,
                "direction": signal,
                "strength": strength,
                "confidence": self.confidence,
                "strategy": "roc_momentum",
                "roc_pct": roc,
            },
        )
        return {"signal": signal, "roc": roc}


class VolumeWeightedMomentumAgent(Agent):
    """Momentum weighted by volume — stronger volume = stronger conviction."""

    SIGNAL_COOLDOWN = 15  # Minimum candles between signals

    def __init__(self, message_bus: MessageBus, symbol: str, period: int = 14):
        super().__init__(
            name=f"vol_momentum_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
        )
        self.symbol = symbol
        self.period = period
        self._prices: list[float] = []
        self._volumes: list[float] = []
        self._candle_count = 0
        self._last_signal_candle = -self.SIGNAL_COOLDOWN

    async def on_start(self):
        await self.subscribe(f"market_data.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type != MessageType.MARKET_DATA:
            return None

        self._prices.append(message.payload.get("close", 0.0))
        self._volumes.append(message.payload.get("volume", 0.0))
        self._candle_count += 1

        if len(self._prices) < self.period + 1:
            return None

        # Cooldown: skip if too soon since last signal
        if self._candle_count - self._last_signal_candle < self.SIGNAL_COOLDOWN:
            return None

        # Volume-weighted price momentum
        recent_prices = self._prices[-self.period:]
        recent_volumes = self._volumes[-self.period:]
        total_vol = sum(recent_volumes) or 1.0

        weighted_change = sum(
            (recent_prices[i] - recent_prices[i - 1]) * (recent_volumes[i] / total_vol)
            for i in range(1, len(recent_prices))
        )
        avg_price = sum(recent_prices) / len(recent_prices)
        normalized = weighted_change / avg_price * 100

        # Volume spike detection (current vol vs average)
        avg_vol = sum(self._volumes[-self.period:]) / self.period
        current_vol = self._volumes[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        if abs(normalized) < 0.5 or vol_ratio < 1.5:
            return None

        signal = "long" if normalized > 0 else "short"
        strength = min(abs(normalized) * vol_ratio / 10, 1.0)

        self._last_signal_candle = self._candle_count
        self.metrics.signals_generated += 1
        await self.emit(
            MessageType.STRATEGY_SIGNAL,
            "signals",
            {
                "symbol": self.symbol,
                "direction": signal,
                "strength": strength,
                "confidence": self.confidence,
                "strategy": "volume_momentum",
                "vol_ratio": vol_ratio,
            },
        )
        return {"signal": signal, "vol_ratio": vol_ratio}


class MultiTimeframeMomentumAgent(Agent):
    """
    Checks momentum alignment across multiple timeframes.
    Signal only fires when short, medium, and long-term momentum align.
    """

    SIGNAL_COOLDOWN = 15
    MIN_MOMENTUM = 0.005  # Minimum 0.5% move required per timeframe

    def __init__(self, message_bus: MessageBus, symbol: str):
        super().__init__(
            name=f"mtf_momentum_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
        )
        self.symbol = symbol
        self._prices: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self.timeframes = [10, 30, 60]  # Longer lookbacks for less noise
        self._candle_count = 0
        self._last_signal_candle = -self.SIGNAL_COOLDOWN

    async def on_start(self):
        await self.subscribe(f"market_data.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type != MessageType.MARKET_DATA:
            return None

        self._prices.append(message.payload.get("close", 0.0))
        self._highs.append(message.payload.get("high", 0.0))
        self._lows.append(message.payload.get("low", 0.0))
        self._candle_count += 1
        if len(self._prices) < max(self.timeframes) + 1:
            return None

        if self._candle_count - self._last_signal_candle < self.SIGNAL_COOLDOWN:
            return None

        # Calculate momentum for each timeframe
        momentums = []
        for tf in self.timeframes:
            roc = (self._prices[-1] - self._prices[-tf - 1]) / self._prices[-tf - 1]
            momentums.append(roc)

        # Check alignment — all same direction AND minimum magnitude
        all_positive = all(m > self.MIN_MOMENTUM for m in momentums)
        all_negative = all(m < -self.MIN_MOMENTUM for m in momentums)

        if not (all_positive or all_negative):
            return None

        # Confirm with ADX
        adx_values = adx(self._highs, self._lows, self._prices, 14)
        if adx_values and adx_values[-1] < 20:
            return None

        signal = "long" if all_positive else "short"
        abs_momentums = [abs(m) for m in momentums]
        strength = math.exp(sum(math.log(max(m, 1e-10)) for m in abs_momentums) / len(abs_momentums))
        strength = min(strength * 100, 1.0)

        self._last_signal_candle = self._candle_count
        self.metrics.signals_generated += 1
        await self.emit(
            MessageType.STRATEGY_SIGNAL,
            "signals",
            {
                "symbol": self.symbol,
                "direction": signal,
                "strength": strength,
                "confidence": min(self.confidence * 1.1, 1.0),
                "strategy": "mtf_momentum",
                "timeframe_momentums": dict(zip(self.timeframes, momentums)),
            },
        )
        return {"signal": signal, "aligned_timeframes": len(self.timeframes)}
