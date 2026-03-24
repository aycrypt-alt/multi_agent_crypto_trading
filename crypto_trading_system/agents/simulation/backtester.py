"""
Backtesting & Simulation Engine

Runs the full multi-agent system against historical data:
- Replay historical candles through the message bus
- Track virtual portfolio, fills, and PnL
- Track per-agent signal accuracy and attribution
- Calculate performance metrics (Sharpe, Sortino, max drawdown)
- Monte Carlo simulation for strategy robustness testing
"""

import asyncio
import logging
import math
import random
import time
from dataclasses import dataclass, field

from ...core.agent_base import Agent, AgentPriority
from ...core.message_bus import Message, MessageBus, MessageType
from ...utils.indicators import max_drawdown, sharpe_ratio, sortino_ratio

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    size_usd: float
    pnl: float
    entry_time: float
    exit_time: float
    contributing_agents: list[str] = field(default_factory=list)


@dataclass
class AgentSignalRecord:
    """Tracks a single signal emitted by an agent."""
    agent_name: str
    strategy: str
    symbol: str
    direction: str
    strength: float
    confidence: float
    timestamp: float
    price_at_signal: float
    # Filled after we know the outcome
    price_after_10: float = 0.0
    price_after_30: float = 0.0
    price_after_60: float = 0.0
    was_correct_10: bool = False
    was_correct_30: bool = False
    was_correct_60: bool = False


@dataclass
class BacktestResult:
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    best_trade: float
    worst_trade: float
    equity_curve: list[float] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)
    agent_signals: list[AgentSignalRecord] = field(default_factory=list)


class BacktestEngine:
    """
    Replays historical data through the agent system and collects results.
    Enhanced with per-agent signal tracking for performance attribution.
    """

    def __init__(self, message_bus: MessageBus, initial_balance: float = 10000.0, orchestrator=None):
        self.message_bus = message_bus
        self.orchestrator = orchestrator  # Optional: for flushing signals in backtest
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity_curve: list[float] = [initial_balance]
        self.trades: list[BacktestTrade] = []
        self._open_positions: dict[str, dict] = {}
        self._daily_returns: list[float] = []
        self._prev_balance = initial_balance

        # Per-agent signal tracking
        self._agent_signals: list[AgentSignalRecord] = []
        self._price_history: list[float] = []
        self._candle_index = 0

        # Track which agents contributed to each trade
        self._last_contributing_agents: list[str] = []

    async def run(self, historical_data: list[dict], symbol: str) -> BacktestResult:
        """
        Run backtest on historical candle data.

        Args:
            historical_data: List of dicts with keys: open, high, low, close, volume, timestamp
            symbol: Trading pair symbol
        """
        # Subscribe to execution channel (after risk sizing) for trade tracking
        self.message_bus.subscribe("execution", self._on_order)
        # Subscribe to strategy signals for per-agent performance attribution
        self.message_bus.subscribe("signals", self._on_strategy_signal)

        logger.info(f"Starting backtest: {len(historical_data)} candles for {symbol}")

        for i, candle in enumerate(historical_data):
            self._candle_index = i
            self._price_history.append(candle["close"])

            # Publish market data
            await self.message_bus.publish(Message(
                type=MessageType.MARKET_DATA,
                channel=f"market_data.{symbol}",
                payload={
                    "symbol": symbol,
                    "open": candle["open"],
                    "high": candle["high"],
                    "low": candle["low"],
                    "close": candle["close"],
                    "volume": candle.get("volume", 0),
                    "timestamp": candle.get("timestamp", i),
                },
                sender_id="backtester",
            ))

            # Drain all pending messages so agents process the candle
            await self.message_bus.drain()

            # In backtest mode, force the orchestrator to aggregate signals immediately
            if self.orchestrator and hasattr(self.orchestrator, 'flush_signals'):
                await self.orchestrator.flush_signals()
                # Drain the resulting risk check → position sizing → execution chain
                await self.message_bus.drain()
                await self.message_bus.drain()
                await self.message_bus.drain()

            # Mark-to-market open positions
            self._mark_to_market(candle["close"])

            # Track daily returns (assume each candle = 1 period)
            if i > 0:
                ret = (self.balance - self._prev_balance) / self._prev_balance if self._prev_balance > 0 else 0
                self._daily_returns.append(ret)
                self._prev_balance = self.balance

            self.equity_curve.append(self.balance)

            # Evaluate past signals now that we have forward data
            self._evaluate_signals_forward(i)

        # Close remaining positions at last price
        if historical_data:
            last_price = historical_data[-1]["close"]
            for sym, pos in list(self._open_positions.items()):
                self._close_position(sym, last_price, historical_data[-1].get("timestamp", 0))

        # Final evaluation of remaining signals
        self._evaluate_signals_forward(len(historical_data) - 1, force=True)

        return self._compute_results()

    async def _on_strategy_signal(self, message: Message):
        """Track individual agent signals for performance attribution."""
        # Use strategy+symbol as a readable fallback name
        strategy = message.payload.get("strategy", "unknown")
        symbol = message.payload.get("symbol", "")
        agent_name = message.payload.get("agent_name", f"{strategy}_{symbol}_{message.sender_id[:8]}")
        record = AgentSignalRecord(
            agent_name=agent_name,
            strategy=strategy,
            symbol=message.payload.get("symbol", ""),
            direction=message.payload.get("direction", "neutral"),
            strength=message.payload.get("strength", 0.0),
            confidence=message.payload.get("confidence", 0.5),
            timestamp=self._candle_index,
            price_at_signal=self._price_history[-1] if self._price_history else 0.0,
        )
        self._agent_signals.append(record)

    async def _on_order(self, message: Message):
        """Simulate order execution."""
        symbol = message.payload.get("symbol", "")
        direction = message.payload.get("direction", "")
        size_usd = message.payload.get("size_usd", 0.0)
        contributing = message.payload.get("contributing_agents", [])

        if size_usd <= 0 or not direction or direction == "neutral":
            return

        self._last_contributing_agents = contributing

        if symbol in self._open_positions:
            existing = self._open_positions[symbol]
            if existing["direction"] != direction:
                self._close_position(symbol, existing.get("current_price", existing["entry_price"]), time.time())

        self._open_positions[symbol] = {
            "direction": direction,
            "entry_price": message.payload.get("entry_price", 0.0) or (self._price_history[-1] if self._price_history else size_usd),
            "size_usd": size_usd,
            "current_price": 0.0,
            "entry_time": time.time(),
            "contributing_agents": contributing,
        }

    def _mark_to_market(self, current_price: float):
        """Update balance based on current positions."""
        for sym, pos in self._open_positions.items():
            if pos["current_price"] > 0:
                prev = pos["current_price"]
                if pos["direction"] == "long":
                    pnl = (current_price - prev) / prev * pos["size_usd"]
                else:
                    pnl = (prev - current_price) / prev * pos["size_usd"]
                self.balance += pnl
            pos["current_price"] = current_price

    def _close_position(self, symbol: str, exit_price: float, exit_time: float):
        pos = self._open_positions.pop(symbol, None)
        if not pos:
            return
        entry = pos["entry_price"]
        if entry <= 0:
            return
        if pos["direction"] == "long":
            pnl = (exit_price - entry) / entry * pos["size_usd"]
        else:
            pnl = (entry - exit_price) / entry * pos["size_usd"]

        self.balance += pnl
        self.trades.append(BacktestTrade(
            symbol=symbol, direction=pos["direction"],
            entry_price=entry, exit_price=exit_price,
            size_usd=pos["size_usd"], pnl=pnl,
            entry_time=pos["entry_time"], exit_time=exit_time,
            contributing_agents=pos.get("contributing_agents", []),
        ))

    def _evaluate_signals_forward(self, current_index: int, force: bool = False):
        """
        Look back at past signals and check if they were correct
        using forward price data (10, 30, 60 candles later).
        """
        for sig in self._agent_signals:
            if sig.price_at_signal <= 0:
                continue

            sig_idx = int(sig.timestamp)

            # Check 10-candle forward
            if (not sig.was_correct_10 and sig.price_after_10 == 0.0
                    and (current_index >= sig_idx + 10 or force)):
                fwd_idx = min(sig_idx + 10, len(self._price_history) - 1)
                sig.price_after_10 = self._price_history[fwd_idx]
                price_change = (sig.price_after_10 - sig.price_at_signal) / sig.price_at_signal
                sig.was_correct_10 = (
                    (sig.direction == "long" and price_change > 0) or
                    (sig.direction == "short" and price_change < 0)
                )

            # Check 30-candle forward
            if (not sig.was_correct_30 and sig.price_after_30 == 0.0
                    and (current_index >= sig_idx + 30 or force)):
                fwd_idx = min(sig_idx + 30, len(self._price_history) - 1)
                sig.price_after_30 = self._price_history[fwd_idx]
                price_change = (sig.price_after_30 - sig.price_at_signal) / sig.price_at_signal
                sig.was_correct_30 = (
                    (sig.direction == "long" and price_change > 0) or
                    (sig.direction == "short" and price_change < 0)
                )

            # Check 60-candle forward
            if (not sig.was_correct_60 and sig.price_after_60 == 0.0
                    and (current_index >= sig_idx + 60 or force)):
                fwd_idx = min(sig_idx + 60, len(self._price_history) - 1)
                sig.price_after_60 = self._price_history[fwd_idx]
                price_change = (sig.price_after_60 - sig.price_at_signal) / sig.price_at_signal
                sig.was_correct_60 = (
                    (sig.direction == "long" and price_change > 0) or
                    (sig.direction == "short" and price_change < 0)
                )

    def _compute_results(self) -> BacktestResult:
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]

        return BacktestResult(
            total_return_pct=round(total_return, 2),
            sharpe_ratio=round(sharpe_ratio(self._daily_returns), 2) if self._daily_returns else 0.0,
            sortino_ratio=round(sortino_ratio(self._daily_returns), 2) if self._daily_returns else 0.0,
            max_drawdown_pct=round(max_drawdown(self.equity_curve) * 100, 2),
            total_trades=len(self.trades),
            win_rate=round(len(wins) / len(pnls) * 100, 1) if pnls else 0.0,
            profit_factor=round(sum(wins) / sum(losses), 2) if losses and sum(losses) > 0 else 0.0,
            avg_trade_pnl=round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
            best_trade=round(max(pnls), 2) if pnls else 0.0,
            worst_trade=round(min(pnls), 2) if pnls else 0.0,
            equity_curve=self.equity_curve,
            trades=self.trades,
            agent_signals=self._agent_signals,
        )


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy robustness.
    Shuffles trade sequence to test if results are robust or luck-dependent.
    """

    def __init__(self, trades: list[BacktestTrade], initial_balance: float = 10000.0):
        self.trades = trades
        self.initial_balance = initial_balance

    def run(self, num_simulations: int = 1000) -> dict:
        """Run Monte Carlo simulation by shuffling trade order."""
        results = []
        pnls = [t.pnl for t in self.trades]

        for _ in range(num_simulations):
            shuffled = pnls.copy()
            random.shuffle(shuffled)

            balance = self.initial_balance
            peak = balance
            max_dd = 0.0
            curve = [balance]

            for pnl in shuffled:
                balance += pnl
                curve.append(balance)
                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak if peak > 0 else 0.0
                max_dd = max(max_dd, dd)

            total_return = (balance - self.initial_balance) / self.initial_balance * 100
            results.append({
                "total_return_pct": total_return,
                "max_drawdown_pct": max_dd * 100,
                "final_balance": balance,
            })

        # Statistics
        returns = [r["total_return_pct"] for r in results]
        drawdowns = [r["max_drawdown_pct"] for r in results]

        return {
            "num_simulations": num_simulations,
            "return_mean": round(sum(returns) / len(returns), 2),
            "return_median": round(sorted(returns)[len(returns) // 2], 2),
            "return_5th_pct": round(sorted(returns)[int(len(returns) * 0.05)], 2),
            "return_95th_pct": round(sorted(returns)[int(len(returns) * 0.95)], 2),
            "max_dd_mean": round(sum(drawdowns) / len(drawdowns), 2),
            "max_dd_worst": round(max(drawdowns), 2),
            "probability_profitable": round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1),
        }
