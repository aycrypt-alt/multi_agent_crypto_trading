"""
Orchestrator — The brain that coordinates all agents.

Manages agent lifecycle, aggregates signals from strategy agents,
weighs them by confidence, detects market regime changes, and
routes final decisions to execution agents.
"""

import asyncio
import logging
import time
from collections import defaultdict
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

    def __init__(self, message_bus: MessageBus, registry: AgentRegistry):
        self.message_bus = message_bus
        self.registry = registry
        self.regime = MarketRegime()
        self._running = False
        self._pending_signals: dict[str, list[dict]] = defaultdict(list)
        self._signal_window_ms = 2000  # Aggregate signals within 2s windows
        self._last_aggregation: float = 0.0

    async def start(self):
        """Start the orchestrator and all registered agents."""
        self._running = True
        await self.message_bus.start()

        # Subscribe to key message types
        self.message_bus.subscribe_type(MessageType.STRATEGY_SIGNAL, self._on_strategy_signal)
        self.message_bus.subscribe_type(MessageType.RISK_ALERT, self._on_risk_alert)
        self.message_bus.subscribe_type(MessageType.ANOMALY_DETECTED, self._on_anomaly)

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
            now = time.time()

            for symbol, signals in list(self._pending_signals.items()):
                if not signals:
                    continue

                aggregated = self._aggregate_signals(symbol, signals)
                if aggregated and abs(aggregated.strength) > 0.3:
                    # Send to risk management before execution
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

                self._pending_signals[symbol] = []

    def _aggregate_signals(self, symbol: str, signals: list[dict]) -> AggregatedSignal | None:
        """
        Weighted voting: each agent's signal is weighted by its confidence.
        This allows high-performing agents to have more influence.
        """
        if not signals:
            return None

        weighted_sum = 0.0
        total_weight = 0.0
        contributors = []

        for sig in signals:
            agent = self.registry.get(sig["sender"])
            agent_confidence = agent.confidence if agent else 0.5

            # Weight = agent's historical confidence * signal confidence
            weight = agent_confidence * sig["confidence"]
            direction_value = 1.0 if sig["direction"] == "long" else (-1.0 if sig["direction"] == "short" else 0.0)
            weighted_sum += direction_value * sig["strength"] * weight
            total_weight += weight

            if agent:
                contributors.append(agent.name)

        if total_weight == 0:
            return None

        net_strength = weighted_sum / total_weight
        direction = "long" if net_strength > 0 else ("short" if net_strength < 0 else "neutral")

        return AggregatedSignal(
            symbol=symbol,
            direction=direction,
            strength=abs(net_strength),
            confidence=min(total_weight / len(signals), 1.0),
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
