"""
Orchestrator — The brain that coordinates all agents.

Manages agent lifecycle, aggregates signals from strategy agents,
weighs them by confidence, detects market regime changes, and
routes final decisions to execution agents.
"""

import asyncio
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .agent_base import Agent, AgentPriority, AgentState
from .agent_registry import AgentRegistry
from .message_bus import Message, MessageBus, MessageType

logger = logging.getLogger(__name__)


@dataclass
class AggregatedSignal:
    symbol: str
    direction: str           # "long", "short", "neutral"
    strength: float          # -1.0 to 1.0
    confidence: float        # 0.0 to 1.0
    contributing_agents: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class MarketRegime:
    """Tracks the current market regime for adaptive behavior."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRASH = "crash"
    RECOVERY = "recovery"

    def __init__(self):
        self.current = self.RANGING
        self.confidence = 0.5
        self.history: list[tuple[str, float]] = []

    def update(self, regime: str, confidence: float):
        if regime != self.current:
            self.history.append((self.current, time.time()))
            self.current = regime
            self.confidence = confidence
            logger.info(f"Market regime change: {regime} (confidence: {confidence:.2f})")


class Orchestrator:
    """
    Central coordinator for the multi-agent trading system.

    Responsibilities:
    - Start/stop all agents
    - Aggregate signals from strategy agents using weighted voting
    - Detect market regime changes and notify agents
    - Route execution decisions through risk management
    - Monitor agent health and performance
    """

    # Strategies that perform well in trending markets
    TREND_STRATEGIES = {"ma_crossover", "macd", "breakout", "roc_momentum", "mtf_momentum"}
    # Strategies that perform well in ranging/mean-reverting markets
    REVERSION_STRATEGIES = {"bollinger_reversion", "rsi_reversion", "zscore_reversion"}
    # Swarm strategies are regime-agnostic (they have their own internal logic)
    SWARM_STRATEGIES = {"swarm_whale", "swarm_retail", "swarm_institutional", "swarm_quant",
                        "swarm_contrarian", "swarm_consensus"}

    def __init__(self, message_bus: MessageBus, registry: AgentRegistry):
        self.message_bus = message_bus
        self.registry = registry
        self.regime = MarketRegime()
        self._running = False
        self._pending_signals: dict[str, list[dict]] = defaultdict(list)
        self._signal_window_ms = 2000  # Aggregate signals within 2s windows
        self._last_aggregation: float = 0.0
        # Price tracking for regime detection
        self._price_history: dict[str, list[float]] = defaultdict(list)
        self._volume_history: dict[str, list[float]] = defaultdict(list)
        self._regime_by_symbol: dict[str, str] = {}
        self._volatility_by_symbol: dict[str, str] = {}  # "high", "normal", "low"

        # Meta-learner: track per-agent, per-regime signal outcomes
        self._agent_regime_scores: dict[str, dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=50))
        )
        # Track recent aggregated decisions and their outcomes
        self._decision_history: deque[dict] = deque(maxlen=100)
        self._decision_eval_horizon = 15  # candles to evaluate a decision

        # Entry confirmation buffer: signals wait here for price confirmation
        # {symbol: {"signal": AggregatedSignal, "entry_price": float, "candle_idx": int, "wait_candles": 0}}
        self._pending_confirmations: dict[str, dict] = {}
        self.CONFIRMATION_WAIT = 2      # Wait up to 2 candles for confirmation
        self.CONFIRMATION_MOVE = 0.001  # Price must move 0.1% in signal direction

    async def start(self):
        """Start the orchestrator and all registered agents."""
        self._running = True
        await self.message_bus.start()

        # Subscribe to key message types
        self.message_bus.subscribe_type(MessageType.STRATEGY_SIGNAL, self._on_strategy_signal)
        self.message_bus.subscribe_type(MessageType.RISK_ALERT, self._on_risk_alert)
        self.message_bus.subscribe_type(MessageType.ANOMALY_DETECTED, self._on_anomaly)
        self.message_bus.subscribe_type(MessageType.MARKET_DATA, self._on_market_data)

        # Start all agents
        agents = list(self.registry._agents.values())
        await asyncio.gather(*[a.start() for a in agents])

        # Background tasks
        asyncio.create_task(self._aggregation_loop())
        asyncio.create_task(self._health_monitor_loop())

        logger.info(f"Orchestrator started with {self.registry.count} agents")

    async def stop(self):
        self._running = False
        agents = list(self.registry._agents.values())
        await asyncio.gather(*[a.stop() for a in agents])
        await self.message_bus.stop()
        logger.info("Orchestrator stopped")

    # ── Signal Aggregation ─────────────────────────────────────

    async def _on_market_data(self, message: Message):
        """Track prices for regime detection and decision evaluation."""
        symbol = message.payload.get("symbol", "")
        close = message.payload.get("close", 0.0)
        volume = message.payload.get("volume", 0.0)
        if symbol and close > 0:
            self._price_history[symbol].append(close)
            self._volume_history[symbol].append(volume)
            # Keep last 200 prices for better regime detection
            if len(self._price_history[symbol]) > 200:
                self._price_history[symbol] = self._price_history[symbol][-200:]
                self._volume_history[symbol] = self._volume_history[symbol][-200:]
            self._detect_regime(symbol)
            self._evaluate_past_decisions(symbol, close)

    def _detect_regime(self, symbol: str):
        """Enhanced multi-factor regime detection: trend, volatility, and momentum alignment."""
        prices = self._price_history[symbol]
        if len(prices) < 60:
            self._regime_by_symbol[symbol] = "unknown"
            self._volatility_by_symbol[symbol] = "normal"
            return

        # 1. Trend detection via dual SMA slope
        sma_short = sum(prices[-20:]) / 20
        sma_long = sum(prices[-50:]) / 50
        sma_ratio = (sma_short - sma_long) / sma_long if sma_long > 0 else 0

        # Linear regression slope of last 30 prices
        recent = prices[-30:]
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n
        numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator > 0 else 0
        normalized_slope = slope / y_mean * 100 if y_mean > 0 else 0

        # 2. Volatility classification
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(-20, 0)]
        vol = math.sqrt(sum(r**2 for r in returns) / len(returns)) if returns else 0
        vol_annualized = vol * math.sqrt(365 * 96)  # 15-min candles

        if vol_annualized > 1.0:
            self._volatility_by_symbol[symbol] = "high"
        elif vol_annualized < 0.3:
            self._volatility_by_symbol[symbol] = "low"
        else:
            self._volatility_by_symbol[symbol] = "normal"

        # 3. Regime classification — combine trend slope + SMA alignment
        if abs(normalized_slope) > 0.05 and abs(sma_ratio) > 0.002:
            self._regime_by_symbol[symbol] = "trending"
        elif abs(normalized_slope) < 0.02 and abs(sma_ratio) < 0.001:
            self._regime_by_symbol[symbol] = "ranging"
        else:
            self._regime_by_symbol[symbol] = "transitioning"

    def _evaluate_past_decisions(self, symbol: str, current_price: float):
        """Evaluate past aggregated decisions to learn which agent combos work."""
        for decision in self._decision_history:
            if decision.get("evaluated") or decision["symbol"] != symbol:
                continue
            if decision["candle_idx"] + self._decision_eval_horizon > len(self._price_history[symbol]):
                continue

            entry_price = decision["price"]
            price_change = (current_price - entry_price) / entry_price if entry_price > 0 else 0

            correct = (
                (decision["direction"] == "long" and price_change > 0) or
                (decision["direction"] == "short" and price_change < 0)
            )
            decision["evaluated"] = True
            decision["correct"] = correct

            # Update per-agent regime scores
            regime = decision.get("regime", "unknown")
            for agent_name in decision.get("agents", []):
                self._agent_regime_scores[agent_name][regime].append(1.0 if correct else 0.0)

    def _get_agent_regime_weight(self, agent_name: str, regime: str) -> float:
        """Get learned weight for an agent in the current regime.
        Returns a multiplier: >1 if agent is good in this regime, <1 if bad."""
        scores = self._agent_regime_scores.get(agent_name, {}).get(regime, deque())
        if len(scores) < 5:
            # Not enough data — fall back to strategy-type heuristic
            return self._heuristic_regime_weight(agent_name, regime)

        # Exponentially weighted accuracy
        alpha = 0.1
        weighted = 0.0
        total_w = 0.0
        for i, score in enumerate(scores):
            w = math.exp(alpha * (i - len(scores)))
            weighted += w * score
            total_w += w
        accuracy = weighted / total_w if total_w > 0 else 0.5

        # Map accuracy to weight: 50% = 1.0, 60% = 1.3, 40% = 0.7
        return 0.4 + accuracy * 1.2

    def _heuristic_regime_weight(self, agent_name: str, regime: str) -> float:
        """Fallback heuristic before we have enough learning data."""
        name = agent_name.lower()
        is_trend = any(s in name for s in
                       ("ma_crossover", "macd", "breakout", "roc_momentum", "mtf_momentum"))
        is_reversion = any(s in name for s in
                           ("bollinger", "rsi_reversion", "zscore"))

        if regime == "trending":
            if is_reversion:
                return 0.3
            if is_trend:
                return 1.2
        elif regime == "ranging":
            if is_trend:
                return 0.3
            if is_reversion:
                return 1.2
        elif regime == "transitioning":
            # In transitions, slightly favor trend-following
            if is_trend:
                return 1.0
            if is_reversion:
                return 0.6
        return 1.0

    def _regime_gate_blocks(self, symbol: str) -> bool:
        """Hard regime gate: block ALL trading in unfavorable conditions.
        Returns True if trading should be BLOCKED."""
        regime = self._regime_by_symbol.get(symbol, "unknown")
        volatility = self._volatility_by_symbol.get(symbol, "normal")

        # Block in unknown regime (not enough data)
        if regime == "unknown":
            return True

        # Block in choppy/low-vol ranging markets — signals are noise
        if regime == "ranging" and volatility == "low":
            return True

        return False

    def _filter_by_regime(self, symbol: str, signals: list[dict]) -> list[dict]:
        """ML meta-learner: weight signals by learned agent-regime performance."""
        regime = self._regime_by_symbol.get(symbol, "unknown")
        volatility = self._volatility_by_symbol.get(symbol, "normal")

        filtered = []
        for sig in signals:
            agent = self.registry.get(sig["sender"])
            if not agent:
                filtered.append(sig)
                continue

            sig = sig.copy()

            # Determine strategy type from agent name
            agent_name = agent.name.lower()
            is_swarm = "swarm_" in agent_name
            if is_swarm:
                filtered.append(sig)
                continue  # Swarm agents handle regime internally

            # Apply learned regime weight
            regime_weight = self._get_agent_regime_weight(agent.name, regime)
            sig["confidence"] *= regime_weight

            # Volatility adjustment — reduce confidence in high-vol, signals are noisier
            if volatility == "high":
                sig["confidence"] *= 0.7
                sig["strength"] *= 0.8
            elif volatility == "low":
                sig["confidence"] *= 1.1  # Low vol = cleaner signals

            # Use agent's adaptive self-learned confidence
            sig["confidence"] *= agent.confidence

            filtered.append(sig)
        return filtered

    async def _on_strategy_signal(self, message: Message):
        """Collect strategy signals for aggregation."""
        symbol = message.payload.get("symbol", "UNKNOWN")
        self._pending_signals[symbol].append({
            "sender": message.sender_id,
            "direction": message.payload.get("direction", "neutral"),
            "strength": message.payload.get("strength", 0.0),
            "confidence": message.payload.get("confidence", 0.5),
            "timestamp": message.timestamp,
        })

    async def _aggregation_loop(self):
        """Periodically aggregate pending signals into consensus decisions."""
        while self._running:
            await asyncio.sleep(self._signal_window_ms / 1000)

            for symbol, signals in list(self._pending_signals.items()):
                if not signals:
                    continue
                await self._process_signals(symbol, signals)
                self._pending_signals[symbol] = []

            # Check pending confirmations
            await self._check_pending_confirmations()

    async def flush_signals(self):
        """Force-aggregate all pending signals immediately (used in backtest mode)."""
        for symbol, signals in list(self._pending_signals.items()):
            if not signals:
                continue
            await self._process_signals(symbol, signals)
            self._pending_signals[symbol] = []

        # Check pending confirmations each candle
        await self._check_pending_confirmations()

    async def _process_signals(self, symbol: str, signals: list[dict]):
        """Apply regime gate, aggregate, and buffer for entry confirmation."""
        # ── HARD REGIME GATE: block in unfavorable conditions ──
        if self._regime_gate_blocks(symbol):
            return

        signals = self._filter_by_regime(symbol, signals)
        aggregated = self._aggregate_signals(symbol, signals)

        if not aggregated or abs(aggregated.strength) <= 0.50 or len(aggregated.contributing_agents) < 2:
            return

        # ── ENTRY CONFIRMATION: buffer signal, wait for price to confirm ──
        prices = self._price_history.get(symbol, [])
        if prices:
            self._pending_confirmations[symbol] = {
                "signal": aggregated,
                "entry_price": prices[-1],
                "candle_idx": len(prices),
                "wait_candles": 0,
            }
        else:
            # No price history, send immediately
            await self._send_order(aggregated)

    async def _check_pending_confirmations(self):
        """Check buffered signals for price confirmation before entering."""
        expired = []
        for symbol, pending in list(self._pending_confirmations.items()):
            prices = self._price_history.get(symbol, [])
            if not prices:
                continue

            signal = pending["signal"]
            entry_price = pending["entry_price"]
            current_price = prices[-1]
            pending["wait_candles"] += 1

            # Check if price has confirmed the signal direction
            price_move = (current_price - entry_price) / entry_price if entry_price > 0 else 0

            confirmed = False
            if signal.direction == "long" and price_move > self.CONFIRMATION_MOVE:
                confirmed = True
            elif signal.direction == "short" and price_move < -self.CONFIRMATION_MOVE:
                confirmed = True

            if confirmed:
                await self._send_order(signal)
                expired.append(symbol)
            elif pending["wait_candles"] >= self.CONFIRMATION_WAIT:
                # Signal expired without confirmation — discard
                expired.append(symbol)

        for sym in expired:
            self._pending_confirmations.pop(sym, None)

    async def _send_order(self, aggregated: AggregatedSignal):
        """Send confirmed signal to risk management for execution."""
        await self.message_bus.publish(Message(
            type=MessageType.ORDER_REQUEST,
            channel="risk_check",
            payload={
                "symbol": aggregated.symbol,
                "direction": aggregated.direction,
                "strength": aggregated.strength,
                "confidence": aggregated.confidence,
                "contributing_agents": aggregated.contributing_agents,
            },
            sender_id="orchestrator",
            priority=AgentPriority.HIGH.value,
        ))

    def _aggregate_signals(self, symbol: str, signals: list[dict]) -> AggregatedSignal | None:
        """
        ML Meta-Learner Aggregation:
        1. Weight each signal by agent's regime-specific learned performance
        2. Calculate directional agreement ratio (what % of agents agree)
        3. Boost confidence when high-performing agents agree
        4. Record decision for future learning
        """
        if not signals:
            return None

        weighted_sum = 0.0
        total_weight = 0.0
        contributors = []
        long_count = 0
        short_count = 0

        for sig in signals:
            agent = self.registry.get(sig["sender"])

            # Weight already includes regime weight, volatility adjustment, and
            # agent self-learned confidence from _filter_by_regime
            weight = sig["confidence"] * sig["strength"]
            direction_value = 1.0 if sig["direction"] == "long" else (-1.0 if sig["direction"] == "short" else 0.0)

            if sig["direction"] == "long":
                long_count += 1
            elif sig["direction"] == "short":
                short_count += 1

            weighted_sum += direction_value * weight
            total_weight += abs(weight)

            if agent:
                contributors.append(agent.name)

        if total_weight == 0:
            return None

        net_strength = weighted_sum / total_weight
        direction = "long" if net_strength > 0 else ("short" if net_strength < 0 else "neutral")

        # Directional agreement bonus — strong consensus = higher confidence
        total_directional = long_count + short_count
        if total_directional > 0:
            majority = max(long_count, short_count)
            agreement_ratio = majority / total_directional
            # Scale: 50% agreement = 0.7x, 75% = 1.0x, 100% = 1.3x
            agreement_multiplier = 0.4 + agreement_ratio * 0.9
        else:
            agreement_multiplier = 0.7

        final_confidence = min(total_weight / len(signals) * agreement_multiplier, 1.5)
        final_strength = abs(net_strength) * agreement_multiplier

        # Record this decision for future meta-learning evaluation
        prices = self._price_history.get(symbol, [])
        if prices:
            self._decision_history.append({
                "symbol": symbol,
                "direction": direction,
                "strength": final_strength,
                "price": prices[-1],
                "candle_idx": len(prices),
                "regime": self._regime_by_symbol.get(symbol, "unknown"),
                "agents": contributors,
                "agreement_ratio": agreement_ratio if total_directional > 0 else 0,
                "evaluated": False,
            })

        return AggregatedSignal(
            symbol=symbol,
            direction=direction,
            strength=final_strength,
            confidence=final_confidence,
            contributing_agents=contributors,
        )

    # ── Risk & Anomaly Handling ────────────────────────────────

    async def _on_risk_alert(self, message: Message):
        """Handle risk alerts — may pause trading or reduce exposure."""
        severity = message.payload.get("severity", "low")
        if severity == "critical":
            logger.warning("CRITICAL risk alert — pausing all strategy agents")
            for agent in self.registry.get_by_category("strategy"):
                await agent.pause()
        elif severity == "high":
            logger.warning("High risk alert — reducing position sizes")
            await self.message_bus.publish(Message(
                type=MessageType.CONFIG_UPDATE,
                channel="execution",
                payload={"position_scale_factor": 0.5},
                sender_id="orchestrator",
                priority=AgentPriority.CRITICAL.value,
            ))

    async def _on_anomaly(self, message: Message):
        """Anomaly detected — trigger regime reassessment."""
        anomaly_type = message.payload.get("type", "unknown")
        logger.info(f"Anomaly detected: {anomaly_type}")
        await self.message_bus.publish(Message(
            type=MessageType.REGIME_CHANGE,
            channel="broadcast",
            payload={"trigger": anomaly_type, "action": "reassess"},
            sender_id="orchestrator",
            priority=AgentPriority.HIGH.value,
        ))

    # ── Health Monitoring ──────────────────────────────────────

    async def _health_monitor_loop(self):
        """Monitor agent health and restart failed agents."""
        while self._running:
            await asyncio.sleep(30)
            for agent in list(self.registry._agents.values()):
                if agent.state == AgentState.ERROR:
                    logger.warning(f"Restarting failed agent: {agent.name}")
                    await agent.stop()
                    await agent.start()
                elif agent.metrics.errors > 100:
                    logger.warning(f"Agent {agent.name} has too many errors, reducing confidence")
                    agent.set_confidence(agent.confidence * 0.8)

    def get_system_status(self) -> dict:
        return {
            "running": self._running,
            "regime": {"current": self.regime.current, "confidence": self.regime.confidence},
            "agents": self.registry.get_summary(),
            "message_bus": self.message_bus.get_stats(),
            "pending_signals": {k: len(v) for k, v in self._pending_signals.items()},
        }
