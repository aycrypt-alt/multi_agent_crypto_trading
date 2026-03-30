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
from ...utils.indicators import ema, sma, atr, macd, adx, volume_ratio


class MACrossoverAgent(Agent):
    """Dual moving average crossover strategy.
    Enhanced with ADX trend strength confirmation — only signals when trend is developing."""

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
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._prev_signal: str = "neutral"

    async def on_start(self):
        await self.subscribe(f"market_data.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type != MessageType.MARKET_DATA:
            return None

        self._prices.append(message.payload.get("close", 0.0))
        self._highs.append(message.payload.get("high", 0.0))
        self._lows.append(message.payload.get("low", 0.0))

        if len(self._prices) < self.slow_period + 1:
            return None

        fast_ma = ema(self._prices, self.fast_period)
        slow_ma = ema(self._prices, self.slow_period)

        if not fast_ma or not slow_ma:
            return None

        # Require ADX > 20 to confirm a trend is present
        adx_values = adx(self._highs, self._lows, self._prices, 14)
        if adx_values and adx_values[-1] < 20:
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

        # Boost strength by ADX value
        if adx_values and signal != "neutral":
            adx_boost = min(adx_values[-1] / 50, 1.0)
            strength = min(strength * (1 + adx_boost), 1.0)

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
    """MACD-based trend following.
    Enhanced: requires histogram to exceed a minimum threshold relative to price,
    and confirms with ADX trend presence."""

    SIGNAL_COOLDOWN = 15

    def __init__(self, message_bus: MessageBus, symbol: str):
        super().__init__(
            name=f"macd_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
        )
        self.symbol = symbol
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

        if len(self._prices) < 35:
            return None

        if self._candle_count - self._last_signal_candle < self.SIGNAL_COOLDOWN:
            return None

        result = macd(self._prices)
        if not result["histogram"] or len(result["histogram"]) < 2:
            return None

        hist = result["histogram"][-1]
        prev_hist = result["histogram"][-2]

        # Require minimum histogram magnitude relative to price (filter noise)
        price = self._prices[-1]
        min_hist = price * 0.0002  # 0.02% of price
        if abs(hist) < min_hist:
            return None

        # Signal on histogram crossover
        if prev_hist <= 0 < hist:
            signal = "long"
        elif prev_hist >= 0 > hist:
            signal = "short"
        else:
            return None

        # Confirm with ADX — prefer trending environments
        adx_values = adx(self._highs, self._lows, self._prices, 14)
        confidence = self.confidence
        if adx_values and adx_values[-1] > 25:
            confidence *= 1.1  # Boost in trending market
        elif adx_values and adx_values[-1] < 15:
            return None  # Skip in very weak trend

        self._last_signal_candle = self._candle_count
        self.metrics.signals_generated += 1
        strength = min(abs(hist) / (abs(result["macd"][-1]) + 1e-10), 1.0)
        await self.emit(
            MessageType.STRATEGY_SIGNAL,
            "signals",
            {
                "symbol": self.symbol,
                "direction": signal,
                "strength": strength,
                "confidence": min(confidence, 1.0),
                "strategy": "macd",
            },
        )
        return {"signal": signal}


class BreakoutAgent(Agent):
    """Donchian channel breakout detection.
    Enhanced with volume confirmation and cooldown — only signals on high-volume breakouts."""

    SIGNAL_COOLDOWN = 18

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
        self._volumes: list[float] = []
        self._candle_count = 0
        self._last_signal_candle = -self.SIGNAL_COOLDOWN

    async def on_start(self):
        await self.subscribe(f"market_data.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type != MessageType.MARKET_DATA:
            return None

        self._highs.append(message.payload.get("high", 0.0))
        self._lows.append(message.payload.get("low", 0.0))
        self._closes.append(message.payload.get("close", 0.0))
        self._volumes.append(message.payload.get("volume", 0.0))
        self._candle_count += 1

        if len(self._highs) < self.lookback + 1:
            return None

        if self._candle_count - self._last_signal_candle < self.SIGNAL_COOLDOWN:
            return None

        upper_channel = max(self._highs[-self.lookback - 1:-1])
        lower_channel = min(self._lows[-self.lookback - 1:-1])
        current_close = self._closes[-1]

        signal = None
        if current_close > upper_channel:
            signal = "long"
        elif current_close < lower_channel:
            signal = "short"

        if not signal:
            return None

        # Volume confirmation — require above-average volume on breakout
        vol_r = volume_ratio(self._volumes, min(self.lookback, 20))
        if vol_r < 1.3:
            return None  # Breakout on low volume = likely false breakout

        channel_width = upper_channel - lower_channel
        strength = abs(current_close - (upper_channel if signal == "long" else lower_channel))
        strength = min(strength / (channel_width + 1e-10), 1.0)

        # Boost strength by volume
        strength = min(strength * min(vol_r / 1.5, 1.5), 1.0)

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
                "strategy": "breakout",
                "volume_ratio": vol_r,
            },
        )
        return {"signal": signal}

        return None
