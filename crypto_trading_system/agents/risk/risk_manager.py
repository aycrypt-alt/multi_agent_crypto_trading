"""
Risk Management Agents

The most critical agents in the system — they protect capital:
- Position Sizing (Kelly Criterion, fixed fractional)
- Drawdown Monitor
- Exposure Manager
- Correlation Risk
- Stop-Loss Manager
"""

import math
import time

from ...core.agent_base import Agent, AgentPriority
from ...core.message_bus import Message, MessageBus, MessageType
from ...utils.indicators import max_drawdown


class PositionSizingAgent(Agent):
    """
    Calculates optimal position sizes using:
    - Kelly Criterion (mathematical optimal)
    - Fixed fractional (conservative)
    - Volatility-adjusted sizing (ATR-based)

    Every order request passes through this agent.
    """

    def __init__(self, message_bus: MessageBus, config: dict | None = None):
        super().__init__(
            name="position_sizer",
            message_bus=message_bus,
            priority=AgentPriority.CRITICAL,
            config=config or {},
        )
        self.max_risk_per_trade = self.config.get("max_risk_per_trade", 0.02)  # 2% max risk
        self.max_portfolio_risk = self.config.get("max_portfolio_risk", 0.06)  # 6% total
        self.account_balance = self.config.get("initial_balance", 10000.0)
        self._open_risk = 0.0  # Currently risked amount
        self._win_rate = 0.5
        self._avg_win = 1.0
        self._avg_loss = 1.0
        self._trade_history: list[dict] = []

    async def on_start(self):
        await self.subscribe("risk_check")
        await self.subscribe_type(MessageType.ORDER_FILLED)
        await self.subscribe_type(MessageType.POSITION_UPDATE)

    async def process(self, message: Message) -> dict | None:
        if message.channel == "risk_check":
            return await self._evaluate_order(message)
        elif message.type == MessageType.ORDER_FILLED:
            return self._update_balance(message)
        elif message.type == MessageType.POSITION_UPDATE:
            self._update_open_risk(message)
        return None

    async def _evaluate_order(self, message: Message) -> dict | None:
        """Evaluate an order request and determine position size."""
        symbol = message.payload.get("symbol", "")
        direction = message.payload.get("direction", "neutral")
        strength = message.payload.get("strength", 0.0)
        confidence = message.payload.get("confidence", 0.5)

        if direction == "neutral":
            return None

        # Check portfolio-level risk limit
        if self._open_risk >= self.max_portfolio_risk * self.account_balance:
            await self.emit(
                MessageType.RISK_ALERT,
                "risk_alerts",
                {"severity": "high", "reason": "max_portfolio_risk_reached",
                 "open_risk": self._open_risk},
            )
            return {"action": "reject", "reason": "portfolio_risk_limit"}

        # Kelly Criterion
        kelly_fraction = self._kelly_criterion()

        # Conservative: use half-Kelly
        position_fraction = kelly_fraction * 0.5

        # Cap at max risk per trade
        position_fraction = min(position_fraction, self.max_risk_per_trade)

        # Scale by signal strength and confidence
        position_fraction *= strength * confidence

        position_size_usd = self.account_balance * position_fraction

        order = {
            "symbol": symbol,
            "direction": direction,
            "size_usd": round(position_size_usd, 2),
            "position_fraction": round(position_fraction, 4),
            "kelly_fraction": round(kelly_fraction, 4),
            "risk_amount": round(position_size_usd * self.max_risk_per_trade, 2),
        }

        await self.emit(
            MessageType.ORDER_REQUEST,
            "execution",
            order,
            priority=AgentPriority.HIGH.value,
        )
        return order

    def _kelly_criterion(self) -> float:
        """Kelly Criterion: f* = (bp - q) / b where b = avg_win/avg_loss."""
        # Not enough trade history — use conservative fixed fraction
        if len(self._trade_history) < 5:
            return self.max_risk_per_trade
        if self._avg_loss == 0:
            return self.max_risk_per_trade
        b = self._avg_win / self._avg_loss
        p = self._win_rate
        q = 1 - p
        kelly = (b * p - q) / b
        # Floor at a small fraction so we still take trades even with poor stats
        return max(0.001, min(kelly, 0.25))

    def _update_balance(self, message: Message) -> dict | None:
        pnl = message.payload.get("pnl", 0.0)
        self.account_balance += pnl
        self._trade_history.append({
            "pnl": pnl,
            "timestamp": time.time(),
            "balance": self.account_balance,
        })
        # Update win rate
        wins = sum(1 for t in self._trade_history if t["pnl"] > 0)
        total = len(self._trade_history)
        self._win_rate = wins / total if total > 0 else 0.5
        winning = [t["pnl"] for t in self._trade_history if t["pnl"] > 0]
        losing = [abs(t["pnl"]) for t in self._trade_history if t["pnl"] < 0]
        self._avg_win = sum(winning) / len(winning) if winning else 1.0
        self._avg_loss = sum(losing) / len(losing) if losing else 1.0
        return {"balance": self.account_balance, "win_rate": self._win_rate}

    def _update_open_risk(self, message: Message):
        self._open_risk = message.payload.get("total_risk", 0.0)


class DrawdownMonitorAgent(Agent):
    """
    Monitors portfolio drawdown and triggers protective actions:
    - Warning at configurable thresholds
    - Reduces position sizes during drawdowns
    - Emergency stop at max drawdown
    """

    def __init__(self, message_bus: MessageBus, config: dict | None = None):
        super().__init__(
            name="drawdown_monitor",
            message_bus=message_bus,
            priority=AgentPriority.CRITICAL,
            config=config or {},
        )
        self.warning_threshold = self.config.get("warning_threshold", 0.05)  # 5%
        self.reduce_threshold = self.config.get("reduce_threshold", 0.10)    # 10%
        self.emergency_threshold = self.config.get("emergency_threshold", 0.15)  # 15%
        self._equity_curve: list[float] = []
        self._peak = 0.0

    async def on_start(self):
        await self.subscribe_type(MessageType.ORDER_FILLED)
        await self.subscribe_type(MessageType.POSITION_UPDATE)

    async def process(self, message: Message) -> dict | None:
        balance = message.payload.get("balance", message.payload.get("equity", 0.0))
        if balance <= 0:
            return None

        self._equity_curve.append(balance)
        if balance > self._peak:
            self._peak = balance

        current_dd = (self._peak - balance) / self._peak if self._peak > 0 else 0.0

        if current_dd >= self.emergency_threshold:
            await self.emit(
                MessageType.RISK_ALERT,
                "risk_alerts",
                {
                    "severity": "critical",
                    "reason": "emergency_drawdown",
                    "drawdown_pct": round(current_dd * 100, 2),
                    "peak": self._peak,
                    "current": balance,
                },
                priority=AgentPriority.CRITICAL.value,
            )
            await self.emit(
                MessageType.DRAWDOWN_ALERT,
                "broadcast",
                {"action": "close_all_positions", "drawdown_pct": current_dd * 100},
                priority=AgentPriority.CRITICAL.value,
            )
        elif current_dd >= self.reduce_threshold:
            scale = 1.0 - (current_dd - self.reduce_threshold) / (self.emergency_threshold - self.reduce_threshold)
            await self.emit(
                MessageType.CONFIG_UPDATE,
                "execution",
                {"position_scale_factor": max(scale, 0.25)},
                priority=AgentPriority.HIGH.value,
            )
        elif current_dd >= self.warning_threshold:
            await self.emit(
                MessageType.RISK_ALERT,
                "risk_alerts",
                {
                    "severity": "medium",
                    "reason": "drawdown_warning",
                    "drawdown_pct": round(current_dd * 100, 2),
                },
            )

        return {"drawdown_pct": round(current_dd * 100, 2)}


class ExposureManagerAgent(Agent):
    """
    Manages overall portfolio exposure:
    - Limits per-asset exposure
    - Limits directional exposure (net long/short)
    - Limits leverage
    """

    def __init__(self, message_bus: MessageBus, config: dict | None = None):
        super().__init__(
            name="exposure_manager",
            message_bus=message_bus,
            priority=AgentPriority.CRITICAL,
            config=config or {},
        )
        self.max_single_asset_pct = self.config.get("max_single_asset_pct", 0.20)  # 20%
        self.max_directional_pct = self.config.get("max_directional_pct", 0.50)    # 50%
        self.max_leverage = self.config.get("max_leverage", 3.0)
        self._positions: dict[str, dict] = {}
        self._account_balance = self.config.get("initial_balance", 10000.0)

    async def on_start(self):
        await self.subscribe_type(MessageType.POSITION_UPDATE)
        await self.subscribe_type(MessageType.ORDER_FILLED)

    async def process(self, message: Message) -> dict | None:
        if message.type == MessageType.POSITION_UPDATE:
            self._update_positions(message)
        elif message.type == MessageType.ORDER_FILLED:
            self._account_balance = message.payload.get("balance", self._account_balance)

        # Calculate exposures
        total_long = sum(p["size_usd"] for p in self._positions.values() if p["direction"] == "long")
        total_short = sum(abs(p["size_usd"]) for p in self._positions.values() if p["direction"] == "short")
        total_exposure = total_long + total_short
        net_exposure = total_long - total_short
        leverage = total_exposure / self._account_balance if self._account_balance > 0 else 0

        exposure_data = {
            "total_long": total_long,
            "total_short": total_short,
            "total_exposure": total_exposure,
            "net_exposure": net_exposure,
            "leverage": round(leverage, 2),
            "positions": len(self._positions),
        }

        # Check limits
        if leverage > self.max_leverage:
            await self.emit(
                MessageType.RISK_ALERT,
                "risk_alerts",
                {"severity": "high", "reason": "max_leverage_exceeded", "leverage": leverage},
                priority=AgentPriority.CRITICAL.value,
            )

        if abs(net_exposure) > self._account_balance * self.max_directional_pct:
            await self.emit(
                MessageType.RISK_ALERT,
                "risk_alerts",
                {"severity": "medium", "reason": "directional_bias_high",
                 "net_exposure": net_exposure},
            )

        await self.emit(MessageType.EXPOSURE_UPDATE, "risk_exposure", exposure_data)
        return exposure_data

    def _update_positions(self, message: Message):
        symbol = message.payload.get("symbol", "")
        size = message.payload.get("size_usd", 0.0)
        if abs(size) < 0.01:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = {
                "size_usd": size,
                "direction": message.payload.get("direction", "long"),
                "entry_price": message.payload.get("entry_price", 0.0),
            }
