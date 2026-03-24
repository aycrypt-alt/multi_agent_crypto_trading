"""
Message Bus — Communication backbone for all agents.

Agents communicate through typed messages on named channels.
Supports pub/sub, request/response, and broadcast patterns.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine


class MessageType(Enum):
    # Market data
    MARKET_DATA = "market_data"
    ORDERBOOK_UPDATE = "orderbook_update"
    TRADE_TICK = "trade_tick"

    # Signals
    SIGNAL = "signal"
    STRATEGY_SIGNAL = "strategy_signal"
    RISK_ALERT = "risk_alert"
    ANOMALY_DETECTED = "anomaly_detected"

    # Orders
    ORDER_REQUEST = "order_request"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_UPDATE = "position_update"

    # System
    AGENT_HEARTBEAT = "agent_heartbeat"
    AGENT_STATUS = "agent_status"
    CONFIG_UPDATE = "config_update"
    REGIME_CHANGE = "regime_change"

    # Analysis
    ANALYSIS_RESULT = "analysis_result"
    PREDICTION = "prediction"
    SENTIMENT_UPDATE = "sentiment_update"

    # Risk
    RISK_ASSESSMENT = "risk_assessment"
    EXPOSURE_UPDATE = "exposure_update"
    DRAWDOWN_ALERT = "drawdown_alert"

    # Simulation
    BACKTEST_RESULT = "backtest_result"
    SIMULATION_STATE = "simulation_state"


@dataclass
class Message:
    type: MessageType
    channel: str
    payload: dict[str, Any]
    sender_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    priority: int = 0
    correlation_id: str | None = None


class MessageBus:
    """Async message bus supporting thousands of concurrent agent subscriptions."""

    def __init__(self, max_queue_size: int = 100_000):
        self._subscribers: dict[str, list[Callable[[Message], Coroutine]]] = {}
        self._type_subscribers: dict[MessageType, list[Callable[[Message], Coroutine]]] = {}
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._running = False
        self._message_count = 0
        self._stats: dict[str, int] = {}

    async def start(self):
        """Start processing messages."""
        self._running = True
        asyncio.create_task(self._process_loop())

    async def stop(self):
        self._running = False

    def subscribe(self, channel: str, handler: Callable[[Message], Coroutine]):
        """Subscribe to a named channel."""
        if channel not in self._subscribers:
            self._subscribers[channel] = []
        self._subscribers[channel].append(handler)

    def subscribe_type(self, msg_type: MessageType, handler: Callable[[Message], Coroutine]):
        """Subscribe to all messages of a given type."""
        if msg_type not in self._type_subscribers:
            self._type_subscribers[msg_type] = []
        self._type_subscribers[msg_type].append(handler)

    def unsubscribe(self, channel: str, handler: Callable[[Message], Coroutine]):
        if channel in self._subscribers:
            self._subscribers[channel] = [h for h in self._subscribers[channel] if h != handler]

    async def publish(self, message: Message):
        """Publish a message — dispatched by priority."""
        await self._queue.put((-message.priority, self._message_count, message))
        self._message_count += 1
        self._stats[message.type.value] = self._stats.get(message.type.value, 0) + 1

    async def _process_loop(self):
        while self._running:
            try:
                _, _, message = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                await self._dispatch(message)
            except asyncio.TimeoutError:
                continue

    async def _dispatch(self, message: Message):
        tasks = []

        # Channel subscribers
        for handler in self._subscribers.get(message.channel, []):
            tasks.append(handler(message))

        # Type subscribers
        for handler in self._type_subscribers.get(message.type, []):
            tasks.append(handler(message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_stats(self) -> dict:
        return {
            "total_messages": self._message_count,
            "queue_size": self._queue.qsize(),
            "channels": len(self._subscribers),
            "type_subscriptions": len(self._type_subscribers),
            "by_type": dict(self._stats),
        }
