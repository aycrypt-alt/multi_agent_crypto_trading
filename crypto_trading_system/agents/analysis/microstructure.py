"""
Market Microstructure Agents — Leading indicators from market structure data.

These agents analyze data that pure price-based TA misses:
- Funding Rate: Sentiment/positioning indicator (extreme funding = crowded trade)
- Open Interest: Capital flow and leverage buildup detection
- Liquidation Levels: Where forced selling/buying will cascade

These are leading indicators — they predict price moves before they happen.
"""

import math
import time
from collections import deque

from ...core.agent_base import Agent, AgentPriority
from ...core.message_bus import Message, MessageBus, MessageType


class FundingRateAgent(Agent):
    """
    Analyzes funding rates as a contrarian sentiment indicator.

    Key insight: Extreme positive funding = too many longs = price likely to drop.
    Extreme negative funding = too many shorts = price likely to bounce.

    Funding rate is a leading indicator because it reflects trader positioning
    BEFORE the liquidation cascade happens.
    """

    SIGNAL_COOLDOWN = 10

    # Thresholds for extreme funding (annualized %)
    EXTREME_POSITIVE = 0.03    # 0.03% per 8h = ~33% annual = very bullish crowd
    EXTREME_NEGATIVE = -0.03   # Very bearish crowd
    MODERATE_POSITIVE = 0.015
    MODERATE_NEGATIVE = -0.015

    def __init__(self, message_bus: MessageBus, symbol: str):
        super().__init__(
            name=f"funding_rate_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
        )
        self.symbol = symbol
        self._funding_history: deque[float] = deque(maxlen=100)
        self._prices: list[float] = []
        self._candle_count = 0
        self._last_signal_candle = -self.SIGNAL_COOLDOWN

    async def on_start(self):
        await self.subscribe(f"market_data.{self.symbol}")
        await self.subscribe(f"funding.{self.symbol}")
        await self.subscribe(f"microstructure.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type == MessageType.MARKET_DATA:
            self._prices.append(message.payload.get("close", 0.0))
            self._candle_count += 1

            # Simulate funding from price action when no real funding data
            if len(self._prices) > 50 and not self._funding_history:
                self._estimate_funding_from_price()
            return None

        # Real funding data from exchange or microstructure channel
        funding_rate = message.payload.get("funding_rate",
                       message.payload.get("fundingRate", None))
        if funding_rate is None:
            return None

        funding_rate = float(funding_rate)
        self._funding_history.append(funding_rate)

        if len(self._funding_history) < 3:
            return None

        if self._candle_count - self._last_signal_candle < self.SIGNAL_COOLDOWN:
            return None

        return self._evaluate_funding()

    def _estimate_funding_from_price(self):
        """Estimate synthetic funding from recent price momentum.
        In reality, strong uptrends have positive funding, downtrends negative."""
        if len(self._prices) < 50:
            return
        recent_return = (self._prices[-1] - self._prices[-20]) / self._prices[-20]
        # Scale to approximate funding rate range
        synthetic_funding = recent_return * 0.01
        self._funding_history.append(synthetic_funding)

    def _evaluate_funding(self) -> dict | None:
        current = self._funding_history[-1]
        avg_funding = sum(self._funding_history) / len(self._funding_history)

        signal = None
        strength = 0.0

        # Contrarian signals on extreme funding
        if current >= self.EXTREME_POSITIVE:
            # Too many longs — contrarian short
            signal = "short"
            strength = min((current - self.EXTREME_POSITIVE) / self.EXTREME_POSITIVE + 0.5, 1.0)
        elif current <= self.EXTREME_NEGATIVE:
            # Too many shorts — contrarian long
            signal = "long"
            strength = min((abs(current) - abs(self.EXTREME_NEGATIVE)) / abs(self.EXTREME_NEGATIVE) + 0.5, 1.0)
        elif current >= self.MODERATE_POSITIVE and avg_funding > self.MODERATE_POSITIVE:
            # Sustained positive funding — mild contrarian short
            signal = "short"
            strength = 0.3
        elif current <= self.MODERATE_NEGATIVE and avg_funding < self.MODERATE_NEGATIVE:
            signal = "long"
            strength = 0.3

        if signal:
            self._last_signal_candle = self._candle_count
            self.metrics.signals_generated += 1
            await_data = {
                "symbol": self.symbol,
                "direction": signal,
                "strength": strength,
                "confidence": self.confidence * 0.85,
                "strategy": "funding_rate",
                "funding_rate": current,
                "avg_funding": avg_funding,
            }
            return await_data

        return None

    async def _emit_signal(self, data: dict):
        await self.emit(MessageType.STRATEGY_SIGNAL, "signals", data)

    async def process(self, message: Message) -> dict | None:
        if message.type == MessageType.MARKET_DATA:
            self._prices.append(message.payload.get("close", 0.0))
            self._candle_count += 1

            # Generate synthetic funding estimate every 20 candles
            if self._candle_count % 20 == 0 and len(self._prices) > 50:
                self._estimate_funding_from_price()
                result = self._evaluate_funding()
                if result:
                    await self.emit(MessageType.STRATEGY_SIGNAL, "signals", result)
            return None

        # Real funding data
        funding_rate = message.payload.get("funding_rate",
                       message.payload.get("fundingRate", None))
        if funding_rate is not None:
            self._funding_history.append(float(funding_rate))

        if len(self._funding_history) < 3:
            return None

        if self._candle_count - self._last_signal_candle < self.SIGNAL_COOLDOWN:
            return None

        result = self._evaluate_funding()
        if result:
            await self.emit(MessageType.STRATEGY_SIGNAL, "signals", result)
            return result
        return None


class OpenInterestAgent(Agent):
    """
    Analyzes Open Interest changes as a leading indicator.

    Key insights:
    - Rising OI + rising price = new longs entering = trend continuation
    - Rising OI + falling price = new shorts entering = trend continuation down
    - Falling OI + rising price = shorts closing = weak rally (likely to reverse)
    - Falling OI + falling price = longs closing = capitulation (bottom forming)

    OI divergence from price is one of the strongest leading signals in crypto.
    """

    SIGNAL_COOLDOWN = 10
    OI_CHANGE_THRESHOLD = 0.02  # 2% OI change to be significant

    def __init__(self, message_bus: MessageBus, symbol: str):
        super().__init__(
            name=f"open_interest_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
        )
        self.symbol = symbol
        self._oi_history: deque[float] = deque(maxlen=100)
        self._prices: list[float] = []
        self._volumes: list[float] = []
        self._candle_count = 0
        self._last_signal_candle = -self.SIGNAL_COOLDOWN

    async def on_start(self):
        await self.subscribe(f"market_data.{self.symbol}")
        await self.subscribe(f"microstructure.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type == MessageType.MARKET_DATA:
            self._prices.append(message.payload.get("close", 0.0))
            self._volumes.append(message.payload.get("volume", 0.0))
            self._candle_count += 1

            # Estimate OI from volume patterns when no real OI data
            if self._candle_count % 10 == 0 and len(self._volumes) > 30:
                self._estimate_oi_from_volume()
                return await self._try_signal()
            return None

        # Real OI data
        oi = message.payload.get("open_interest", message.payload.get("openInterest", None))
        if oi is not None:
            self._oi_history.append(float(oi))
            return await self._try_signal()
        return None

    def _estimate_oi_from_volume(self):
        """Estimate OI changes from volume patterns.
        High volume with price movement suggests OI increase (new positions).
        High volume with small price movement suggests OI decrease (closing)."""
        if len(self._prices) < 20 or len(self._volumes) < 20:
            return

        avg_vol = sum(self._volumes[-20:]) / 20
        current_vol = self._volumes[-1]
        price_change = abs(self._prices[-1] - self._prices[-2]) / self._prices[-2] if len(self._prices) > 1 else 0

        # Synthetic OI: volume * price_efficiency as a proxy
        # High volume + big move = position building
        # High volume + small move = position closing
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        if price_change > 0.001:  # Meaningful move
            oi_proxy = vol_ratio * (1 + price_change * 100)
        else:
            oi_proxy = vol_ratio * (1 - vol_ratio * 0.1)  # Closing positions

        if not self._oi_history:
            self._oi_history.append(100.0)  # Base
        last_oi = self._oi_history[-1]
        self._oi_history.append(last_oi * (0.99 + oi_proxy * 0.01))

    async def _try_signal(self) -> dict | None:
        if len(self._oi_history) < 5 or len(self._prices) < 10:
            return None

        if self._candle_count - self._last_signal_candle < self.SIGNAL_COOLDOWN:
            return None

        # Calculate OI change
        oi_now = self._oi_history[-1]
        oi_prev = self._oi_history[-5]
        oi_change = (oi_now - oi_prev) / oi_prev if oi_prev > 0 else 0

        # Price change over same period
        price_now = self._prices[-1]
        price_prev = self._prices[-5] if len(self._prices) > 5 else self._prices[0]
        price_change = (price_now - price_prev) / price_prev if price_prev > 0 else 0

        signal = None
        strength = 0.0

        if abs(oi_change) < self.OI_CHANGE_THRESHOLD * 0.5:
            return None  # OI change too small to be meaningful

        if oi_change > self.OI_CHANGE_THRESHOLD:
            if price_change > 0.003:
                # Rising OI + rising price = bullish continuation
                signal = "long"
                strength = min(oi_change * 10, 1.0) * 0.7
            elif price_change < -0.003:
                # Rising OI + falling price = bearish continuation
                signal = "short"
                strength = min(oi_change * 10, 1.0) * 0.7
        elif oi_change < -self.OI_CHANGE_THRESHOLD:
            if price_change < -0.003:
                # Falling OI + falling price = capitulation = bottom forming
                signal = "long"
                strength = min(abs(oi_change) * 10, 1.0) * 0.6
            elif price_change > 0.003:
                # Falling OI + rising price = short squeeze ending = weak rally
                signal = "short"
                strength = min(abs(oi_change) * 10, 1.0) * 0.5

        if signal:
            self._last_signal_candle = self._candle_count
            self.metrics.signals_generated += 1
            data = {
                "symbol": self.symbol,
                "direction": signal,
                "strength": strength,
                "confidence": self.confidence * 0.8,
                "strategy": "open_interest",
                "oi_change_pct": round(oi_change * 100, 2),
                "price_change_pct": round(price_change * 100, 2),
            }
            await self.emit(MessageType.STRATEGY_SIGNAL, "signals", data)
            return data

        return None


class LiquidationLevelAgent(Agent):
    """
    Estimates liquidation clusters and predicts cascade moves.

    Key insight: When price approaches a cluster of liquidation levels,
    it tends to be magnetically attracted to them (market makers hunt stops).
    Once liquidations trigger, they cause a cascade in one direction.

    This agent estimates where liquidations are likely clustered based on
    recent price action and typical leverage levels in crypto.
    """

    SIGNAL_COOLDOWN = 10

    # Typical leverage levels in crypto and their liquidation distances
    LEVERAGE_LEVELS = [5, 10, 25, 50, 100]

    def __init__(self, message_bus: MessageBus, symbol: str):
        super().__init__(
            name=f"liquidation_levels_{symbol}",
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

        if len(self._prices) < 50:
            return None

        if self._candle_count - self._last_signal_candle < self.SIGNAL_COOLDOWN:
            return None

        return await self._analyze_liquidation_levels()

    async def _analyze_liquidation_levels(self) -> dict | None:
        current_price = self._prices[-1]

        # Find recent swing highs and lows (likely entry points)
        swing_highs = self._find_swing_points(self._highs, is_high=True)
        swing_lows = self._find_swing_points(self._lows, is_high=False)

        # Estimate liquidation levels for longs entered at swing lows
        long_liq_levels = []
        for low in swing_lows:
            for lev in self.LEVERAGE_LEVELS:
                # Liquidation price for long = entry * (1 - 1/leverage)
                liq_price = low * (1 - 0.95 / lev)
                if liq_price > 0:
                    long_liq_levels.append(liq_price)

        # Estimate liquidation levels for shorts entered at swing highs
        short_liq_levels = []
        for high in swing_highs:
            for lev in self.LEVERAGE_LEVELS:
                # Liquidation price for short = entry * (1 + 1/leverage)
                liq_price = high * (1 + 0.95 / lev)
                short_liq_levels.append(liq_price)

        # Find the nearest dense liquidation cluster
        signal = None
        strength = 0.0

        # Check if there's a cluster of long liquidations below current price
        nearby_long_liqs = [l for l in long_liq_levels
                            if 0.97 * current_price < l < current_price]
        # Check short liquidation cluster above
        nearby_short_liqs = [l for l in short_liq_levels
                             if current_price < l < 1.03 * current_price]

        # Dense cluster below = price might dip to trigger them, then bounce
        if len(nearby_long_liqs) >= 3:
            # Many longs about to be liquidated below — could cascade down then bounce
            cluster_density = len(nearby_long_liqs)
            avg_liq = sum(nearby_long_liqs) / len(nearby_long_liqs)
            distance_pct = (current_price - avg_liq) / current_price

            if distance_pct < 0.01:
                # Very close to liquidation cluster — expect volatility
                # After cascade, price usually reverses
                signal = "long"  # Buy the cascade
                strength = min(cluster_density / 10, 0.8)

        # Dense cluster above = price might spike to trigger them
        if len(nearby_short_liqs) >= 3 and not signal:
            cluster_density = len(nearby_short_liqs)
            avg_liq = sum(nearby_short_liqs) / len(nearby_short_liqs)
            distance_pct = (avg_liq - current_price) / current_price

            if distance_pct < 0.01:
                signal = "short"  # Sell after short squeeze
                strength = min(cluster_density / 10, 0.8)

        if signal:
            self._last_signal_candle = self._candle_count
            self.metrics.signals_generated += 1
            data = {
                "symbol": self.symbol,
                "direction": signal,
                "strength": strength,
                "confidence": self.confidence * 0.7,
                "strategy": "liquidation_levels",
                "long_liqs_nearby": len(nearby_long_liqs),
                "short_liqs_nearby": len(nearby_short_liqs),
            }
            await self.emit(MessageType.STRATEGY_SIGNAL, "signals", data)
            return data

        return None

    def _find_swing_points(self, data: list[float], is_high: bool, lookback: int = 30) -> list[float]:
        """Find recent swing highs or lows from the last `lookback` candles."""
        if len(data) < lookback:
            return []

        recent = data[-lookback:]
        points = []
        for i in range(2, len(recent) - 2):
            if is_high:
                if recent[i] > recent[i-1] and recent[i] > recent[i-2] and \
                   recent[i] > recent[i+1] and recent[i] > recent[i+2]:
                    points.append(recent[i])
            else:
                if recent[i] < recent[i-1] and recent[i] < recent[i-2] and \
                   recent[i] < recent[i+1] and recent[i] < recent[i+2]:
                    points.append(recent[i])

        return points if points else [recent[-1]]
