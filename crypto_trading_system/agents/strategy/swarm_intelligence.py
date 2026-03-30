"""
Swarm Intelligence Agents — MiroFish-Inspired Market Persona Simulation

Instead of running all agents through the same technical lens, these agents
simulate different market participant personas (whale, retail, institutional,
quant, contrarian). Each interprets the same price data through a different
behavioral lens, then they debate via an interaction round to produce an
emergent consensus signal.

Inspired by MiroFish's multi-agent swarm prediction engine.
"""

import math
import random
from collections import defaultdict

from ...core.agent_base import Agent, AgentPriority
from ...core.message_bus import Message, MessageBus, MessageType
from ...utils.indicators import ema, sma, atr, rsi


# ── Persona Definitions ──────────────────────────────────────

PERSONAS = {
    "whale": {
        "description": "Large holder, accumulates on dips, sells into strength",
        "risk_tolerance": 0.3,       # Conservative
        "time_horizon": 60,          # Looks 60 candles ahead
        "contrarian_bias": 0.6,      # Tends to buy when others sell
        "momentum_sensitivity": 0.3,  # Low — doesn't chase
        "volume_sensitivity": 0.9,    # Very sensitive to volume spikes
        "fear_greed_weight": 0.7,     # Heavily influenced by extreme sentiment
    },
    "retail": {
        "description": "Small trader, follows trends, panic sells on drops",
        "risk_tolerance": 0.7,       # Aggressive
        "time_horizon": 10,          # Short-term focus
        "contrarian_bias": 0.1,      # Follows the crowd
        "momentum_sensitivity": 0.9,  # Chases momentum
        "volume_sensitivity": 0.4,    # Doesn't watch volume much
        "fear_greed_weight": 0.9,     # Highly emotional
    },
    "institutional": {
        "description": "Fund manager, systematic entry, risk-adjusted positions",
        "risk_tolerance": 0.4,
        "time_horizon": 40,
        "contrarian_bias": 0.4,
        "momentum_sensitivity": 0.5,
        "volume_sensitivity": 0.7,
        "fear_greed_weight": 0.3,     # Less emotional
    },
    "quant": {
        "description": "Algorithmic trader, pure stats, no emotion",
        "risk_tolerance": 0.5,
        "time_horizon": 20,
        "contrarian_bias": 0.5,      # Neutral — follows the math
        "momentum_sensitivity": 0.6,
        "volume_sensitivity": 0.8,
        "fear_greed_weight": 0.1,     # Almost no emotional bias
    },
    "contrarian": {
        "description": "Fades the crowd, buys fear, sells greed",
        "risk_tolerance": 0.5,
        "time_horizon": 30,
        "contrarian_bias": 0.95,     # Strongly contrarian
        "momentum_sensitivity": 0.4,
        "volume_sensitivity": 0.6,
        "fear_greed_weight": 0.8,     # Uses sentiment inversely
    },
}


class SwarmPersonaAgent(Agent):
    """
    A trading agent that interprets market data through a specific persona lens.

    Each persona has different biases:
    - Whale: accumulates on dips, sells into rallies, watches volume
    - Retail: chases momentum, panic sells, follows crowd
    - Institutional: systematic, risk-adjusted, longer horizon
    - Quant: pure statistics, mean-reversion and momentum blended
    - Contrarian: fades extremes, buys fear, sells greed
    """

    def __init__(self, message_bus: MessageBus, symbol: str, persona: str):
        if persona not in PERSONAS:
            raise ValueError(f"Unknown persona: {persona}. Choose from: {list(PERSONAS.keys())}")

        self.persona = persona
        self.traits = PERSONAS[persona]
        super().__init__(
            name=f"swarm_{persona}_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
            config={"symbol": symbol, "persona": persona},
        )
        self.symbol = symbol
        self._prices: list[float] = []
        self._volumes: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._prev_signal: str = "neutral"

    async def on_start(self):
        await self.subscribe(f"market_data.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type != MessageType.MARKET_DATA:
            return None

        self._prices.append(message.payload.get("close", 0.0))
        self._volumes.append(message.payload.get("volume", 0.0))
        self._highs.append(message.payload.get("high", 0.0))
        self._lows.append(message.payload.get("low", 0.0))

        # Need enough data for all indicators
        if len(self._prices) < 60:
            return None

        signal = self._evaluate()
        if signal and signal["direction"] != "neutral" and signal["direction"] != self._prev_signal:
            self._prev_signal = signal["direction"]
            self.metrics.signals_generated += 1
            await self.emit(
                MessageType.STRATEGY_SIGNAL,
                "signals",
                {
                    "symbol": self.symbol,
                    "direction": signal["direction"],
                    "strength": signal["strength"],
                    "confidence": self.confidence,
                    "strategy": f"swarm_{self.persona}",
                    "persona": self.persona,
                    "components": signal["components"],
                },
            )
            return signal
        return None

    def _evaluate(self) -> dict | None:
        """Evaluate market through this persona's behavioral lens."""
        prices = self._prices
        volumes = self._volumes
        traits = self.traits

        # ── Component signals (all -1.0 to 1.0) ──

        # 1. Momentum signal
        momentum = self._calc_momentum()

        # 2. Volume signal
        volume_sig = self._calc_volume_signal()

        # 3. Fear/greed (RSI-based proxy)
        fear_greed = self._calc_fear_greed()

        # 4. Mean reversion signal
        reversion = self._calc_reversion()

        # 5. Trend signal
        trend = self._calc_trend()

        # ── Persona-weighted blend ──
        # Each persona weighs these components differently

        raw_score = 0.0

        # Momentum component (retail loves it, whale ignores it)
        raw_score += momentum * traits["momentum_sensitivity"]

        # Volume component (whale loves it, retail ignores it)
        raw_score += volume_sig * traits["volume_sensitivity"]

        # Fear/greed (contrarian inverts it, retail amplifies it)
        if traits["contrarian_bias"] > 0.7:
            raw_score += -fear_greed * traits["fear_greed_weight"]
        else:
            raw_score += fear_greed * traits["fear_greed_weight"]

        # Trend component (scaled by time horizon preference)
        horizon_factor = min(traits["time_horizon"] / 30, 1.0)
        raw_score += trend * horizon_factor

        # Mean reversion (contrarian loves it)
        raw_score += reversion * traits["contrarian_bias"]

        # Normalize by total weight
        total_weight = (
            traits["momentum_sensitivity"]
            + traits["volume_sensitivity"]
            + traits["fear_greed_weight"]
            + horizon_factor
            + traits["contrarian_bias"]
        )
        if total_weight == 0:
            return None

        normalized = raw_score / total_weight

        # Apply risk tolerance as a threshold
        threshold = 0.15 * (1.0 - traits["risk_tolerance"])  # More risk-tolerant = lower threshold
        if abs(normalized) < threshold:
            return None

        direction = "long" if normalized > 0 else "short"
        strength = min(abs(normalized), 1.0)

        return {
            "direction": direction,
            "strength": strength,
            "components": {
                "momentum": round(momentum, 3),
                "volume": round(volume_sig, 3),
                "fear_greed": round(fear_greed, 3),
                "reversion": round(reversion, 3),
                "trend": round(trend, 3),
            },
        }

    def _calc_momentum(self) -> float:
        """Rate of change over persona's time horizon."""
        horizon = self.traits["time_horizon"]
        if len(self._prices) < horizon + 1:
            return 0.0
        roc = (self._prices[-1] - self._prices[-horizon]) / self._prices[-horizon]
        return max(-1.0, min(1.0, roc * 20))  # Scale small % moves to -1..1

    def _calc_volume_signal(self) -> float:
        """Volume spike detection — high volume + price up = bullish."""
        if len(self._volumes) < 20:
            return 0.0
        avg_vol = sum(self._volumes[-20:]) / 20
        if avg_vol == 0:
            return 0.0
        vol_ratio = self._volumes[-1] / avg_vol
        price_change = (self._prices[-1] - self._prices[-2]) / self._prices[-2] if self._prices[-2] > 0 else 0

        if vol_ratio > 2.0:  # Volume spike
            return max(-1.0, min(1.0, price_change * 100))
        elif vol_ratio > 1.5:
            return max(-1.0, min(1.0, price_change * 50))
        return 0.0

    def _calc_fear_greed(self) -> float:
        """RSI-based fear/greed proxy. < 30 = fear, > 70 = greed."""
        rsi_values = rsi(self._prices, 14)
        if not rsi_values:
            return 0.0
        current_rsi = rsi_values[-1]
        # Map RSI 0-100 to -1..1 (fear to greed)
        return (current_rsi - 50) / 50

    def _calc_reversion(self) -> float:
        """Z-score based mean reversion signal."""
        if len(self._prices) < 30:
            return 0.0
        window = self._prices[-30:]
        mean = sum(window) / len(window)
        std = (sum((p - mean) ** 2 for p in window) / len(window)) ** 0.5
        if std == 0:
            return 0.0
        zscore = (self._prices[-1] - mean) / std
        # Invert: high z-score = overbought = sell signal
        return max(-1.0, min(1.0, -zscore / 3))

    def _calc_trend(self) -> float:
        """EMA-based trend signal over persona's time horizon."""
        fast_period = max(5, self.traits["time_horizon"] // 4)
        slow_period = self.traits["time_horizon"]
        if len(self._prices) < slow_period + 1:
            return 0.0
        fast = ema(self._prices, fast_period)
        slow = ema(self._prices, slow_period)
        if not fast or not slow:
            return 0.0
        diff = (fast[-1] - slow[-1]) / slow[-1]
        return max(-1.0, min(1.0, diff * 100))


class SwarmDebateAgent(Agent):
    """
    Collects signals from all SwarmPersonaAgents and runs an interaction round.

    Instead of simple weighted averaging, this agent simulates debate:
    1. Collect all persona signals for a symbol
    2. Check for strong disagreements (contrarian vs retail)
    3. Weight by conviction strength and persona reliability
    4. Apply crowd dynamics (herding amplification or contrarian dampening)
    5. Emit the emergent consensus signal

    This mimics MiroFish's approach of letting agents interact and observing
    emergent behavior rather than just averaging.
    """

    def __init__(self, message_bus: MessageBus, symbol: str):
        super().__init__(
            name=f"swarm_debate_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.NORMAL,
        )
        self.symbol = symbol
        self._persona_signals: dict[str, dict] = {}
        self._signal_count = 0
        self._debate_history: list[dict] = []

    async def on_start(self):
        await self.subscribe("signals")

    async def process(self, message: Message) -> dict | None:
        if message.type != MessageType.STRATEGY_SIGNAL:
            return None

        payload = message.payload
        if payload.get("symbol") != self.symbol:
            return None

        # Only collect swarm persona signals
        strategy = payload.get("strategy", "")
        if not strategy.startswith("swarm_") or strategy == "swarm_consensus":
            return None

        persona = payload.get("persona", "")
        if not persona:
            return None

        self._persona_signals[persona] = {
            "direction": payload["direction"],
            "strength": payload["strength"],
            "confidence": payload.get("confidence", 0.5),
            "components": payload.get("components", {}),
        }

        # Run debate when we have signals from at least 3 personas
        if len(self._persona_signals) >= 3:
            result = self._run_debate()
            if result:
                self._persona_signals.clear()
                self.metrics.signals_generated += 1
                await self.emit(
                    MessageType.STRATEGY_SIGNAL,
                    "signals",
                    {
                        "symbol": self.symbol,
                        "direction": result["direction"],
                        "strength": result["strength"],
                        "confidence": self.confidence,
                        "strategy": "swarm_consensus",
                        "debate_result": result["debate_summary"],
                    },
                )
                return result
        return None

    def _run_debate(self) -> dict | None:
        """
        Simulate agent interaction/debate and extract emergent consensus.

        Debate mechanics:
        1. Count bulls vs bears among personas
        2. Check for conviction clustering (strong agreement = amplify)
        3. Check for polarization (strong disagreement = reduce confidence)
        4. Apply crowd dynamics rules
        """
        signals = self._persona_signals
        if not signals:
            return None

        # ── Phase 1: Tally votes ──
        bull_score = 0.0
        bear_score = 0.0
        bull_count = 0
        bear_count = 0
        persona_votes = {}

        for persona, sig in signals.items():
            vote = sig["strength"] if sig["direction"] == "long" else -sig["strength"]
            persona_votes[persona] = vote

            if sig["direction"] == "long":
                bull_score += sig["strength"] * sig["confidence"]
                bull_count += 1
            elif sig["direction"] == "short":
                bear_score += sig["strength"] * sig["confidence"]
                bear_count += 1

        # ── Phase 2: Detect crowd dynamics ──
        total_participants = bull_count + bear_count
        if total_participants == 0:
            return None

        # Agreement ratio: how much consensus exists
        agreement_ratio = max(bull_count, bear_count) / total_participants

        # Polarization: strong opposing views
        polarization = min(bull_score, bear_score) / max(bull_score, bear_score, 0.01)

        # ── Phase 3: Apply interaction rules ──

        # Rule 1: Whale + Institutional agreement = strong signal
        whale_vote = persona_votes.get("whale", 0)
        inst_vote = persona_votes.get("institutional", 0)
        smart_money_aligned = (whale_vote > 0 and inst_vote > 0) or (whale_vote < 0 and inst_vote < 0)
        smart_money_bonus = 1.3 if smart_money_aligned else 1.0

        # Rule 2: If retail is on one side and contrarian on the other,
        #          weight contrarian more (retail tends to be wrong at extremes)
        retail_vote = persona_votes.get("retail", 0)
        contrarian_vote = persona_votes.get("contrarian", 0)
        retail_contrarian_divergence = (retail_vote > 0 and contrarian_vote < 0) or \
                                      (retail_vote < 0 and contrarian_vote > 0)

        # Rule 3: High agreement = herding = slight caution
        herding_factor = 1.0
        if agreement_ratio > 0.8:
            herding_factor = 0.85  # Too much agreement might mean crowded trade

        # Rule 4: High polarization = uncertainty = reduce strength
        polarization_factor = 1.0 - (polarization * 0.3)

        # ── Phase 4: Compute consensus ──
        net_score = bull_score - bear_score

        # Apply smart money bonus
        net_score *= smart_money_bonus

        # Apply contrarian override: if retail and contrarian disagree,
        # slightly favor contrarian direction
        if retail_contrarian_divergence and contrarian_vote != 0:
            contrarian_pull = contrarian_vote * 0.15
            net_score += contrarian_pull

        # Apply crowd dynamics
        net_score *= herding_factor * polarization_factor

        # Normalize
        max_possible = total_participants * 1.0  # Max if all vote 1.0
        normalized = net_score / max(max_possible, 1.0)
        normalized = max(-1.0, min(1.0, normalized))

        if abs(normalized) < 0.1:
            return None

        direction = "long" if normalized > 0 else "short"
        strength = min(abs(normalized), 1.0)

        # Confidence based on agreement and smart money alignment
        confidence_base = agreement_ratio * 0.6 + (0.4 if smart_money_aligned else 0.2)
        confidence = min(confidence_base * polarization_factor, 1.0)

        debate_summary = {
            "bull_count": bull_count,
            "bear_count": bear_count,
            "agreement_ratio": round(agreement_ratio, 2),
            "polarization": round(polarization, 2),
            "smart_money_aligned": smart_money_aligned,
            "retail_contrarian_divergence": retail_contrarian_divergence,
            "herding_factor": round(herding_factor, 2),
            "persona_votes": {k: round(v, 3) for k, v in persona_votes.items()},
        }

        self._debate_history.append(debate_summary)
        # Keep last 50 debates
        if len(self._debate_history) > 50:
            self._debate_history = self._debate_history[-50:]

        return {
            "direction": direction,
            "strength": strength,
            "confidence": confidence,
            "debate_summary": debate_summary,
        }


class NewsInjectorAgent(Agent):
    """
    Ingests news/event signals and broadcasts them to swarm personas.

    In backtest mode: generates synthetic news events based on price action
    (large moves trigger "news" that personas react to).

    In live mode: can be extended to consume real news feeds (CryptoPanic,
    Twitter, etc.) and convert them to sentiment signals.
    """

    def __init__(self, message_bus: MessageBus, symbol: str):
        super().__init__(
            name=f"news_injector_{symbol}",
            message_bus=message_bus,
            priority=AgentPriority.LOW,
        )
        self.symbol = symbol
        self._prices: list[float] = []
        self._last_news_idx = 0

    async def on_start(self):
        await self.subscribe(f"market_data.{self.symbol}")

    async def process(self, message: Message) -> dict | None:
        if message.type != MessageType.MARKET_DATA:
            return None

        self._prices.append(message.payload.get("close", 0.0))

        if len(self._prices) < 20:
            return None

        # Detect significant price moves as "news events"
        recent = self._prices[-10:]
        older = self._prices[-20:-10]

        if not older:
            return None

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        price_change_pct = (recent_avg - older_avg) / older_avg * 100

        # Only emit news for significant moves (>2% in 10 candles)
        if abs(price_change_pct) < 2.0:
            return None

        # Throttle: don't emit more than once every 20 candles
        if len(self._prices) - self._last_news_idx < 20:
            return None

        self._last_news_idx = len(self._prices)

        if price_change_pct > 0:
            sentiment = min(price_change_pct / 5.0, 1.0)
            event_type = "bullish_momentum"
        else:
            sentiment = max(price_change_pct / 5.0, -1.0)
            event_type = "bearish_momentum"

        await self.emit(
            MessageType.SENTIMENT_UPDATE,
            f"news.{self.symbol}",
            {
                "symbol": self.symbol,
                "sentiment_score": sentiment,
                "event_type": event_type,
                "price_change_pct": round(price_change_pct, 2),
            },
        )
        self.metrics.signals_generated += 1
        return {"event": event_type, "sentiment": sentiment}
