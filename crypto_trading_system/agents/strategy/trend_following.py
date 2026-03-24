"""
Trend Following Strategy Agents

Multiple agents using different trend-detection methods:
- Moving Average Crossover
- Breakout Detection
- ADX Trend Strength
- Ichimoku Cloud

Each agent runs independently and emits signals to the orchestrator.
"""

from ...core.agent_base import Agent, AgentPriority
from ...core.message_bus import Message, MessageBus, MessageType
from ...utils.indicators import ema, sma, atr, macd


class MACrossoverAgent(Agent):
    """Dual moving average crossover strategy."""

    def __init__(self, message_bus: MessageBus, symbol: str, fast: int = 9, slow: int = 21):
        super().__init__(
            name=f"ma_crossover_{symbol}_{fast}_{slow}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
            config={"symbol": symbol, "fast": fast, "slow": slow},
        )
        self.symbol = symbol
        self.fast_period = fast
        self.slow_period = slow
        self._prices: list[float] = []
        self._prev_signal: str = "neutral"

    async def on_start(self):
        await self.subscribe(f"market_data.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type != MessageType.MARKET_DATA:
            return None

        price = message.payload.get("close", 0.0)
        self._prices.append(price)

        if len(self._prices) < self.slow_period + 1:
            return None

        fast_ma = ema(self._prices, self.fast_period)
        slow_ma = ema(self._prices, self.slow_period)

        if not fast_ma or not slow_ma:
            return None

        # Detect crossover
        current_fast = fast_ma[-1]
        current_slow = slow_ma[-1]
        prev_fast = fast_ma[-2] if len(fast_ma) > 1 else current_fast
        prev_slow = slow_ma[-2] if len(slow_ma) > 1 else current_slow

        signal = "neutral"
        strength = 0.0

        if prev_fast <= prev_slow and current_fast > current_slow:
            signal = "long"
            strength = min((current_fast - current_slow) / current_slow * 100, 1.0)
        elif prev_fast >= prev_slow and current_fast < current_slow:
            signal = "short"
            strength = min((current_slow - current_fast) / current_fast * 100, 1.0)

        if signal != "neutral" and signal != self._prev_signal:
            self._prev_signal = signal
            self.metrics.signals_generated += 1
            await self.emit(
                MessageType.STRATEGY_SIGNAL,
                "signals",
                {
                    "symbol": self.symbol,
                    "direction": signal,
                    "strength": strength,
                    "confidence": self.confidence,
                    "strategy": "ma_crossover",
                    "fast_ma": current_fast,
                    "slow_ma": current_slow,
                },
            )
            return {"signal": signal, "strength": strength}

        return None


class MACDAgent(Agent):
    """MACD-based trend following."""

    def __init__(self, message_bus: MessageBus, symbol: str):
        super().__init__(
            name=f"macd_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
        )
        self.symbol = symbol
        self._prices: list[float] = []
        self._prev_histogram: float = 0.0

    async def on_start(self):
        await self.subscribe(f"market_data.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type != MessageType.MARKET_DATA:
            return None

        self._prices.append(message.payload.get("close", 0.0))
        if len(self._prices) < 35:
            return None

        result = macd(self._prices)
        if not result["histogram"]:
            return None

        hist = result["histogram"][-1]
        prev_hist = result["histogram"][-2] if len(result["histogram"]) > 1 else 0.0

        # Signal on histogram crossover
        if prev_hist <= 0 < hist:
            signal, direction = "long", 1
        elif prev_hist >= 0 > hist:
            signal, direction = "short", -1
        else:
            return None

        self.metrics.signals_generated += 1
        strength = min(abs(hist) / (abs(result["macd"][-1]) + 1e-10), 1.0)
        await self.emit(
            MessageType.STRATEGY_SIGNAL,
            "signals",
            {
                "symbol": self.symbol,
                "direction": signal,
                "strength": strength,
                "confidence": self.confidence,
                "strategy": "macd",
            },
        )
        return {"signal": signal}


class BreakoutAgent(Agent):
    """Donchian channel breakout detection."""

    def __init__(self, message_bus: MessageBus, symbol: str, lookback: int = 20):
        super().__init__(
            name=f"breakout_{symbol}_{lookback}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
        )
        self.symbol = symbol
        self.lookback = lookback
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._closes: list[float] = []

    async def on_start(self):
        await self.subscribe(f"market_data.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type != MessageType.MARKET_DATA:
            return None

        self._highs.append(message.payload.get("high", 0.0))
        self._lows.append(message.payload.get("low", 0.0))
        self._closes.append(message.payload.get("close", 0.0))

        if len(self._highs) < self.lookback + 1:
            return None

        upper_channel = max(self._highs[-self.lookback - 1:-1])
        lower_channel = min(self._lows[-self.lookback - 1:-1])
        current_close = self._closes[-1]

        signal = None
        if current_close > upper_channel:
            signal = "long"
        elif current_close < lower_channel:
            signal = "short"

        if signal:
            channel_width = upper_channel - lower_channel
            strength = abs(current_close - (upper_channel if signal == "long" else lower_channel))
            strength = min(strength / (channel_width + 1e-10), 1.0)
            self.metrics.signals_generated += 1
            await self.emit(
                MessageType.STRATEGY_SIGNAL,
                "signals",
                {
                    "symbol": self.symbol,
                    "direction": signal,
                    "strength": strength,
                    "confidence": self.confidence,
                    "strategy": "breakout",
                },
            )
            return {"signal": signal}

        return None
