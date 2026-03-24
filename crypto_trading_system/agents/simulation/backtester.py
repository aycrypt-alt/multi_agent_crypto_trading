"""
Backtesting & Simulation Engine

Runs the full multi-agent system against historical data:
- Replay historical candles through the message bus
- Track virtual portfolio, fills, and PnL
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


class BacktestEngine:
    """
    Replays historical data through the agent system and collects results.
    """

    def __init__(self, message_bus: MessageBus, initial_balance: float = 10000.0):
        self.message_bus = message_bus
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity_curve: list[float] = [initial_balance]
        self.trades: list[BacktestTrade] = []
        self._open_positions: dict[str, dict] = {}
        self._daily_returns: list[float] = []
        self._prev_balance = initial_balance

    async def run(self, historical_data: list[dict], symbol: str) -> BacktestResult:
        """
        Run backtest on historical candle data.

        Args:
            historical_data: List of dicts with keys: open, high, low, close, volume, timestamp
            symbol: Trading pair symbol
        """
        # Subscribe to execution events
        self.message_bus.subscribe_type(MessageType.ORDER_REQUEST, self._on_order)

        logger.info(f"Starting backtest: {len(historical_data)} candles for {symbol}")

        for i, candle in enumerate(historical_data):
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

            # Process pending messages
            await asyncio.sleep(0)

            # Mark-to-market open positions
            self._mark_to_market(candle["close"])

            # Track daily returns (assume each candle = 1 period)
            if i > 0:
                ret = (self.balance - self._prev_balance) / self._prev_balance if self._prev_balance > 0 else 0
                self._daily_returns.append(ret)
                self._prev_balance = self.balance

            self.equity_curve.append(self.balance)

        # Close remaining positions at last price
        if historical_data:
            last_price = historical_data[-1]["close"]
            for sym, pos in list(self._open_positions.items()):
                self._close_position(sym, last_price, historical_data[-1].get("timestamp", 0))

        return self._compute_results()

    async def _on_order(self, message: Message):
        """Simulate order execution."""
        symbol = message.payload.get("symbol", "")
        direction = message.payload.get("direction", "")
        size_usd = message.payload.get("size_usd", 0.0)

        if size_usd <= 0 or not direction or direction == "neutral":
            return

        # Simple fill at current price (in a real backtest, add slippage model)
        # For now, assume the order gets filled at the requested price
        if symbol in self._open_positions:
            existing = self._open_positions[symbol]
            if existing["direction"] != direction:
                # Close existing position
                self._close_position(symbol, existing.get("current_price", existing["entry_price"]), time.time())

        self._open_positions[symbol] = {
            "direction": direction,
            "entry_price": message.payload.get("entry_price", 0.0) or size_usd,  # Simplified
            "size_usd": size_usd,
            "current_price": 0.0,
            "entry_time": time.time(),
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
        ))

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
            best_trade=max(pnls) if pnls else 0.0,
            worst_trade=min(pnls) if pnls else 0.0,
            equity_curve=self.equity_curve,
            trades=self.trades,
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
