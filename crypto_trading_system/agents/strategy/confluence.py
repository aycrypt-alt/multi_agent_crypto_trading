"""
Multi-Indicator Confluence Strategy Agents

These agents require multiple independent indicators to align before signaling.
A single indicator has ~52% accuracy. Three independent indicators aligning
raises the probability significantly — each one acts as a confirmation filter.

Agents:
- MeanReversionConfluence: Bollinger + RSI + Z-Score + Volume
- TrendConfluence: MA Crossover + MACD + ADX + Breakout
- MomentumConfluence: ROC + Multi-TF alignment + Volume spike + RSI trend
"""

import math
from collections import deque

from ...core.agent_base import Agent, AgentPriority
from ...core.message_bus import Message, MessageBus, MessageType
from ...utils.indicators import (
    ema, sma, rsi, macd, bollinger_bands, adx, atr, volume_ratio,
)
from .mean_reversion import _in_strong_trend


class MeanReversionConfluenceAgent(Agent):
    """
    High-conviction mean reversion: requires 3+ of these to agree:
    1. Price at/beyond Bollinger Band (2 std)
    2. RSI at extreme (<30 or >70)
    3. Z-Score beyond 2.0
    4. Volume spike (confirms capitulation/exhaustion)
    5. ADX < 25 (confirming range-bound market)

    Only fires when >= 3 indicators align. Much fewer but much better signals.
    """

    SIGNAL_COOLDOWN = 8
    MIN_CONFIRMATIONS = 3

    def __init__(self, message_bus: MessageBus, symbol: str):
        super().__init__(
            name=f"confluence_mean_reversion_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
        )
        self.symbol = symbol
        self._prices: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._volumes: list[float] = []
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
        self._volumes.append(message.payload.get("volume", 0.0))
        self._candle_count += 1

        if len(self._prices) < 55:
            return None

        if self._candle_count - self._last_signal_candle < self.SIGNAL_COOLDOWN:
            return None

        # TREND FILTER: mean reversion confluence should not fire in strong trends
        if _in_strong_trend(self._prices, self._highs, self._lows):
            return None

        # Evaluate each indicator independently
        long_votes = 0
        short_votes = 0
        confirmations = []

        # 1. Bollinger Bands
        bands = bollinger_bands(self._prices, 20, 2.0)
        if bands["upper"]:
            price = self._prices[-1]
            if price <= bands["lower"][-1]:
                long_votes += 1
                confirmations.append("bollinger_lower")
            elif price >= bands["upper"][-1]:
                short_votes += 1
                confirmations.append("bollinger_upper")

        # 2. RSI
        rsi_values = rsi(self._prices, 14)
        if rsi_values:
            current_rsi = rsi_values[-1]
            if current_rsi < 30:
                long_votes += 1
                confirmations.append(f"rsi_{current_rsi:.0f}")
            elif current_rsi > 70:
                short_votes += 1
                confirmations.append(f"rsi_{current_rsi:.0f}")

        # 3. Z-Score
        window = self._prices[-50:]
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std = math.sqrt(variance) if variance > 0 else 1e-10
        z_score = (self._prices[-1] - mean) / std

        if z_score < -2.0:
            long_votes += 1
            confirmations.append(f"zscore_{z_score:.1f}")
        elif z_score > 2.0:
            short_votes += 1
            confirmations.append(f"zscore_{z_score:.1f}")

        # 4. Volume spike (confirms exhaustion move)
        vol_r = volume_ratio(self._volumes, 20)
        if vol_r > 1.5:
            # Volume spike — confirms whichever direction has more votes
            if long_votes > short_votes:
                long_votes += 1
                confirmations.append(f"volume_{vol_r:.1f}x")
            elif short_votes > long_votes:
                short_votes += 1
                confirmations.append(f"volume_{vol_r:.1f}x")

        # 5. ADX — confirm we're NOT in a strong trend (mean reversion works in ranges)
        adx_values = adx(self._highs, self._lows, self._prices, 14)
        if adx_values and adx_values[-1] < 25:
            # Range-bound market — boost whichever direction
            if long_votes > short_votes:
                long_votes += 1
                confirmations.append(f"adx_low_{adx_values[-1]:.0f}")
            elif short_votes > long_votes:
                short_votes += 1
                confirmations.append(f"adx_low_{adx_values[-1]:.0f}")

        # Check confluence threshold
        max_votes = max(long_votes, short_votes)
        if max_votes < self.MIN_CONFIRMATIONS:
            return None

        signal = "long" if long_votes > short_votes else "short"
        # Strength scales with number of confirmations
        strength = min(max_votes / 5, 1.0)
        # Confidence is high because multiple independent indicators agree
        confidence = min(self.confidence * (0.7 + max_votes * 0.1), 1.5)

        self._last_signal_candle = self._candle_count
        self.metrics.signals_generated += 1
        await self.emit(
            MessageType.STRATEGY_SIGNAL,
            "signals",
            {
                "symbol": self.symbol,
                "direction": signal,
                "strength": strength,
                "confidence": confidence,
                "strategy": "confluence_mean_reversion",
                "confirmations": confirmations,
                "num_confirmations": max_votes,
            },
        )
        return {"signal": signal, "confirmations": max_votes}


class TrendConfluenceAgent(Agent):
    """
    High-conviction trend following: requires 3+ of these to agree:
    1. EMA fast > slow (or vice versa) — trend direction
    2. MACD histogram positive/negative — momentum confirmation
    3. ADX > 25 — trend strength confirmation
    4. Price above/below recent breakout level — structural confirmation
    5. Volume above average — participation confirmation

    Only fires when >= 3 indicators align on direction.
    """

    SIGNAL_COOLDOWN = 8
    MIN_CONFIRMATIONS = 3

    def __init__(self, message_bus: MessageBus, symbol: str):
        super().__init__(
            name=f"confluence_trend_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
        )
        self.symbol = symbol
        self._prices: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._volumes: list[float] = []
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
        self._volumes.append(message.payload.get("volume", 0.0))
        self._candle_count += 1

        if len(self._prices) < 55:
            return None

        if self._candle_count - self._last_signal_candle < self.SIGNAL_COOLDOWN:
            return None

        long_votes = 0
        short_votes = 0
        confirmations = []

        # 1. EMA crossover state (not just crossover moment — current alignment)
        fast_ema = ema(self._prices, 12)
        slow_ema = ema(self._prices, 26)
        if fast_ema and slow_ema:
            if fast_ema[-1] > slow_ema[-1]:
                long_votes += 1
                confirmations.append("ema_bullish")
            elif fast_ema[-1] < slow_ema[-1]:
                short_votes += 1
                confirmations.append("ema_bearish")

        # 2. MACD histogram direction
        macd_result = macd(self._prices)
        if macd_result["histogram"] and len(macd_result["histogram"]) >= 2:
            hist = macd_result["histogram"][-1]
            prev_hist = macd_result["histogram"][-2]
            if hist > 0 and hist > prev_hist:
                long_votes += 1
                confirmations.append("macd_bullish")
            elif hist < 0 and hist < prev_hist:
                short_votes += 1
                confirmations.append("macd_bearish")

        # 3. ADX trend strength — must be > 25
        adx_values = adx(self._highs, self._lows, self._prices, 14)
        if adx_values and adx_values[-1] > 25:
            # ADX confirms trend exists — boost whichever direction
            if long_votes > short_votes:
                long_votes += 1
                confirmations.append(f"adx_{adx_values[-1]:.0f}")
            elif short_votes > long_votes:
                short_votes += 1
                confirmations.append(f"adx_{adx_values[-1]:.0f}")

        # 4. Donchian breakout — price above 20-bar high or below 20-bar low
        if len(self._highs) > 21:
            upper = max(self._highs[-21:-1])
            lower = min(self._lows[-21:-1])
            price = self._prices[-1]
            if price > upper:
                long_votes += 1
                confirmations.append("breakout_high")
            elif price < lower:
                short_votes += 1
                confirmations.append("breakout_low")

        # 5. Volume confirmation
        vol_r = volume_ratio(self._volumes, 20)
        if vol_r > 1.3:
            if long_votes > short_votes:
                long_votes += 1
                confirmations.append(f"volume_{vol_r:.1f}x")
            elif short_votes > long_votes:
                short_votes += 1
                confirmations.append(f"volume_{vol_r:.1f}x")

        max_votes = max(long_votes, short_votes)
        if max_votes < self.MIN_CONFIRMATIONS:
            return None

        signal = "long" if long_votes > short_votes else "short"
        strength = min(max_votes / 5, 1.0)
        confidence = min(self.confidence * (0.7 + max_votes * 0.1), 1.5)

        self._last_signal_candle = self._candle_count
        self.metrics.signals_generated += 1
        await self.emit(
            MessageType.STRATEGY_SIGNAL,
            "signals",
            {
                "symbol": self.symbol,
                "direction": signal,
                "strength": strength,
                "confidence": confidence,
                "strategy": "confluence_trend",
                "confirmations": confirmations,
                "num_confirmations": max_votes,
            },
        )
        return {"signal": signal, "confirmations": max_votes}


class MomentumConfluenceAgent(Agent):
    """
    High-conviction momentum: requires 3+ of these to agree:
    1. ROC > threshold across multiple periods (10, 20)
    2. RSI trending (between 50-70 for longs, 30-50 for shorts — not at extremes)
    3. Price > EMA 50 (for longs) or < EMA 50 (for shorts) — trend alignment
    4. Volume increasing over last 5 bars — accumulation/distribution
    5. ATR expanding — volatility confirming the move has energy

    Momentum signals work best when the move has just started, not at extremes.
    """

    SIGNAL_COOLDOWN = 8
    MIN_CONFIRMATIONS = 3
    ROC_THRESHOLD = 1.5  # percent

    def __init__(self, message_bus: MessageBus, symbol: str):
        super().__init__(
            name=f"confluence_momentum_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
        )
        self.symbol = symbol
        self._prices: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._volumes: list[float] = []
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
        self._volumes.append(message.payload.get("volume", 0.0))
        self._candle_count += 1

        if len(self._prices) < 55:
            return None

        if self._candle_count - self._last_signal_candle < self.SIGNAL_COOLDOWN:
            return None

        long_votes = 0
        short_votes = 0
        confirmations = []

        # 1. ROC alignment across periods
        roc_10 = (self._prices[-1] - self._prices[-11]) / self._prices[-11] * 100
        roc_20 = (self._prices[-1] - self._prices[-21]) / self._prices[-21] * 100

        if roc_10 > self.ROC_THRESHOLD and roc_20 > self.ROC_THRESHOLD:
            long_votes += 1
            confirmations.append(f"roc_bull_{roc_10:.1f}%")
        elif roc_10 < -self.ROC_THRESHOLD and roc_20 < -self.ROC_THRESHOLD:
            short_votes += 1
            confirmations.append(f"roc_bear_{roc_10:.1f}%")

        # 2. RSI in momentum zone (not at extremes — that's mean reversion territory)
        rsi_values = rsi(self._prices, 14)
        if rsi_values:
            current_rsi = rsi_values[-1]
            if 50 < current_rsi < 70:
                long_votes += 1
                confirmations.append(f"rsi_momentum_{current_rsi:.0f}")
            elif 30 < current_rsi < 50:
                short_votes += 1
                confirmations.append(f"rsi_momentum_{current_rsi:.0f}")

        # 3. Price relative to EMA 50
        ema_50 = ema(self._prices, 50)
        if ema_50:
            if self._prices[-1] > ema_50[-1] * 1.001:
                long_votes += 1
                confirmations.append("above_ema50")
            elif self._prices[-1] < ema_50[-1] * 0.999:
                short_votes += 1
                confirmations.append("below_ema50")

        # 4. Volume trend — increasing over last 5 bars
        if len(self._volumes) >= 6:
            recent_vols = self._volumes[-5:]
            vol_slope = sum(recent_vols[i] - recent_vols[i-1] for i in range(1, 5))
            avg_vol = sum(self._volumes[-20:]) / 20
            if vol_slope > 0 and recent_vols[-1] > avg_vol:
                if long_votes > short_votes:
                    long_votes += 1
                    confirmations.append("volume_increasing")
                elif short_votes > long_votes:
                    short_votes += 1
                    confirmations.append("volume_increasing")

        # 5. ATR expanding — move has energy
        atr_values = atr(self._highs, self._lows, self._prices, 14)
        if len(atr_values) >= 5:
            atr_now = atr_values[-1]
            atr_prev = sum(atr_values[-5:-1]) / 4
            if atr_now > atr_prev * 1.1:
                if long_votes > short_votes:
                    long_votes += 1
                    confirmations.append("atr_expanding")
                elif short_votes > long_votes:
                    short_votes += 1
                    confirmations.append("atr_expanding")

        max_votes = max(long_votes, short_votes)
        if max_votes < self.MIN_CONFIRMATIONS:
            return None

        signal = "long" if long_votes > short_votes else "short"
        strength = min(max_votes / 5, 1.0)
        confidence = min(self.confidence * (0.7 + max_votes * 0.1), 1.5)

        self._last_signal_candle = self._candle_count
        self.metrics.signals_generated += 1
        await self.emit(
            MessageType.STRATEGY_SIGNAL,
            "signals",
            {
                "symbol": self.symbol,
                "direction": signal,
                "strength": strength,
                "confidence": confidence,
                "strategy": "confluence_momentum",
                "confirmations": confirmations,
                "num_confirmations": max_votes,
            },
        )
        return {"signal": signal, "confirmations": max_votes}
