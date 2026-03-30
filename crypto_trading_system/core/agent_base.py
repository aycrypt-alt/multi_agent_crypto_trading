"""
Agent Base — Foundation for all agents in the system.

Every agent (strategy, analysis, risk, execution, etc.) inherits from
this base class, which provides lifecycle management, messaging, state
tracking, and adaptive behavior hooks.
"""

import asyncio
import logging
import math
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .message_bus import Message, MessageBus, MessageType

logger = logging.getLogger(__name__)


class AgentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class AgentPriority(Enum):
    CRITICAL = 100   # Risk management, stop-loss
    HIGH = 75        # Order execution
    NORMAL = 50      # Strategy signals
    LOW = 25         # Analysis, logging
    BACKGROUND = 10  # Simulation, backtesting


@dataclass
class AgentMetrics:
    messages_received: int = 0
    messages_sent: int = 0
    errors: int = 0
    last_active: float = 0.0
    avg_processing_ms: float = 0.0
    total_processing_ms: float = 0.0
    signals_generated: int = 0
    uptime_seconds: float = 0.0


class Agent(ABC):
    """Base class for all trading system agents."""

    def __init__(
        self,
        name: str,
        message_bus: MessageBus,
        priority: AgentPriority = AgentPriority.NORMAL,
        config: dict | None = None,
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.message_bus = message_bus
        self.priority = priority
        self.config = config or {}
        self.state = AgentState.IDLE
        self.metrics = AgentMetrics()
        self._start_time = 0.0
        self._subscriptions: list[str] = []
        self._confidence: float = 1.0  # Agent's self-assessed confidence
        # Auto-initialize adaptive learning for all agents
        self._init_learning()

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self):
        """Initialize and begin processing."""
        self.state = AgentState.RUNNING
        self._start_time = time.time()
        await self.on_start()
        logger.info(f"Agent started: {self.name} [{self.id[:8]}]")

    async def stop(self):
        self.state = AgentState.STOPPED
        await self.on_stop()
        logger.info(f"Agent stopped: {self.name} [{self.id[:8]}]")

    async def pause(self):
        self.state = AgentState.PAUSED

    async def resume(self):
        self.state = AgentState.RUNNING

    # ── Abstract methods agents must implement ─────────────────

    @abstractmethod
    async def on_start(self):
        """Called when the agent starts. Subscribe to channels here."""
        ...

    @abstractmethod
    async def process(self, message: Message) -> dict | None:
        """Process an incoming message. Return a result dict or None."""
        ...

    async def on_stop(self):
        """Cleanup hook."""
        pass

    # ── Messaging helpers ──────────────────────────────────────

    async def subscribe(self, channel: str):
        self.message_bus.subscribe(channel, self._handle_message)
        self._subscriptions.append(channel)

    async def subscribe_type(self, msg_type: MessageType):
        self.message_bus.subscribe_type(msg_type, self._handle_message)

    async def emit(self, msg_type: MessageType, channel: str, payload: dict, priority: int | None = None):
        # Auto-inject agent name into payload for signal tracking
        if "agent_name" not in payload:
            payload["agent_name"] = self.name

        # Auto-record strategy signals for adaptive learning
        if msg_type == MessageType.STRATEGY_SIGNAL and self._price_buffer:
            direction = payload.get("direction", "neutral")
            if direction in ("long", "short"):
                self.record_signal(direction, self._price_buffer[-1])
            # Inject adaptive confidence into the signal
            payload["confidence"] = payload.get("confidence", 1.0) * self._confidence

        msg = Message(
            type=msg_type,
            channel=channel,
            payload=payload,
            sender_id=self.id,
            priority=priority if priority is not None else self.priority.value,
        )
        await self.message_bus.publish(msg)
        self.metrics.messages_sent += 1

    async def _handle_message(self, message: Message):
        if self.state != AgentState.RUNNING:
            return
        self.metrics.messages_received += 1
        self.metrics.last_active = time.time()

        # Auto-track prices from market data for adaptive learning
        if message.type == MessageType.MARKET_DATA:
            close = message.payload.get("close", 0.0)
            if close > 0:
                self.record_price(close)

        start = time.monotonic()
        try:
            result = await self.process(message)
            elapsed = (time.monotonic() - start) * 1000
            self.metrics.total_processing_ms += elapsed
            count = self.metrics.messages_received
            self.metrics.avg_processing_ms = self.metrics.total_processing_ms / count
            return result
        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"Agent {self.name} error: {e}", exc_info=True)
            return None

    # ── Adaptive Self-Learning ───────────────────────────────────

    LEARNING_WINDOW = 50       # Track last N signals for rolling accuracy
    EVAL_HORIZON = 15          # Candles to wait before evaluating a signal
    MIN_SIGNALS_TO_ADAPT = 10  # Need this many evaluated signals before adapting
    CONFIDENCE_FLOOR = 0.2     # Never go below this confidence
    CONFIDENCE_CEILING = 1.5   # Allow above 1.0 for consistently accurate agents
    ADAPT_RATE = 0.15          # How fast confidence adjusts (0=never, 1=instant)

    def _init_learning(self):
        """Initialize adaptive learning state. Call from subclass __init__ after super().__init__."""
        self._signal_history: deque[dict] = deque(maxlen=self.LEARNING_WINDOW)
        self._price_buffer: deque[float] = deque(maxlen=500)
        self._candle_idx: int = 0
        self._base_confidence: float = self._confidence
        self._rolling_accuracy: float = 0.5
        self._streak: int = 0  # positive = wins, negative = losses
        self._regime_accuracy: dict[str, list[bool]] = {
            "trending": [], "ranging": [], "unknown": []
        }

    def record_price(self, price: float):
        """Called each candle to track prices for signal evaluation."""
        self._price_buffer.append(price)
        self._candle_idx += 1
        self._evaluate_pending_signals()

    def record_signal(self, direction: str, price: float, regime: str = "unknown"):
        """Record a signal for later accuracy evaluation."""
        if not hasattr(self, '_signal_history'):
            return
        self._signal_history.append({
            "direction": direction,
            "price": price,
            "candle_idx": self._candle_idx,
            "regime": regime,
            "evaluated": False,
            "correct": None,
        })

    def _evaluate_pending_signals(self):
        """Check if enough candles have passed to evaluate old signals."""
        if not hasattr(self, '_signal_history'):
            return
        current_price = self._price_buffer[-1] if self._price_buffer else 0
        evaluated_any = False

        for sig in self._signal_history:
            if sig["evaluated"]:
                continue
            if self._candle_idx - sig["candle_idx"] < self.EVAL_HORIZON:
                continue

            # Evaluate: was the signal direction correct?
            price_change = (current_price - sig["price"]) / sig["price"] if sig["price"] > 0 else 0
            if sig["direction"] == "long":
                sig["correct"] = price_change > 0
            elif sig["direction"] == "short":
                sig["correct"] = price_change < 0
            else:
                sig["correct"] = False
            sig["evaluated"] = True
            evaluated_any = True

            # Track regime-specific accuracy
            regime = sig.get("regime", "unknown")
            if regime in self._regime_accuracy:
                self._regime_accuracy[regime].append(sig["correct"])
                # Keep only last 30 per regime
                if len(self._regime_accuracy[regime]) > 30:
                    self._regime_accuracy[regime] = self._regime_accuracy[regime][-30:]

            # Update streak
            if sig["correct"]:
                self._streak = max(self._streak + 1, 1)
            else:
                self._streak = min(self._streak - 1, -1)

        if evaluated_any:
            self._update_confidence()

    def _update_confidence(self):
        """Adapt confidence based on rolling accuracy using exponential smoothing."""
        evaluated = [s for s in self._signal_history if s["evaluated"]]
        if len(evaluated) < self.MIN_SIGNALS_TO_ADAPT:
            return

        # Exponentially weighted accuracy — recent signals matter more
        alpha = 0.1  # Decay factor
        weighted_correct = 0.0
        total_weight = 0.0
        for i, sig in enumerate(evaluated):
            w = math.exp(alpha * (i - len(evaluated)))
            weighted_correct += w * (1.0 if sig["correct"] else 0.0)
            total_weight += w

        self._rolling_accuracy = weighted_correct / total_weight if total_weight > 0 else 0.5

        # Map accuracy to confidence: 50% accuracy = base, above = boost, below = reduce
        # Using a sigmoid-like curve centered at 0.5
        accuracy_edge = self._rolling_accuracy - 0.5
        target_confidence = self._base_confidence * (1.0 + accuracy_edge * 3.0)

        # Streak bonus/penalty — consecutive wins/losses amplify adjustment
        streak_factor = 1.0 + min(abs(self._streak), 5) * 0.05 * (1 if self._streak > 0 else -1)
        target_confidence *= streak_factor

        # Smooth adjustment
        self._confidence = self._confidence + self.ADAPT_RATE * (target_confidence - self._confidence)
        self._confidence = max(self.CONFIDENCE_FLOOR, min(self.CONFIDENCE_CEILING, self._confidence))

    def get_regime_accuracy(self, regime: str) -> float:
        """Get this agent's accuracy in a specific market regime."""
        if not hasattr(self, '_regime_accuracy'):
            return 0.5
        results = self._regime_accuracy.get(regime, [])
        if len(results) < 5:
            return 0.5
        return sum(results) / len(results)

    def set_confidence(self, confidence: float):
        """Agents can adjust their own confidence based on recent accuracy."""
        self._confidence = max(0.0, min(1.0, confidence))

    @property
    def confidence(self) -> float:
        return self._confidence

    def get_status(self) -> dict:
        accuracy = self._rolling_accuracy if hasattr(self, '_rolling_accuracy') else 0.5
        streak = self._streak if hasattr(self, '_streak') else 0
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "priority": self.priority.value,
            "confidence": round(self._confidence, 3),
            "rolling_accuracy": round(accuracy, 3),
            "streak": streak,
            "metrics": {
                "messages_received": self.metrics.messages_received,
                "messages_sent": self.metrics.messages_sent,
                "errors": self.metrics.errors,
                "avg_processing_ms": round(self.metrics.avg_processing_ms, 2),
                "signals_generated": self.metrics.signals_generated,
            },
        }
