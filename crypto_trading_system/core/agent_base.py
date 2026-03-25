"""
Agent Base — Foundation for all agents in the system.

Every agent (strategy, analysis, risk, execution, etc.) inherits from
this base class, which provides lifecycle management, messaging, state
tracking, and adaptive behavior hooks.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
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

    # ── Adaptive behavior ──────────────────────────────────────

    def set_confidence(self, confidence: float):
        """Agents can adjust their own confidence based on recent accuracy."""
        self._confidence = max(0.0, min(1.0, confidence))

    @property
    def confidence(self) -> float:
        return self._confidence

    def get_status(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "priority": self.priority.value,
            "confidence": self._confidence,
            "metrics": {
                "messages_received": self.metrics.messages_received,
                "messages_sent": self.metrics.messages_sent,
                "errors": self.metrics.errors,
                "avg_processing_ms": round(self.metrics.avg_processing_ms, 2),
                "signals_generated": self.metrics.signals_generated,
            },
        }
