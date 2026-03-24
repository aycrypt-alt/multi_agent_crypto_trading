"""
Multi-Agent Crypto Trading System — Main Entry Point

This script demonstrates how to:
1. Create the message bus and agent registry
2. Spawn hundreds/thousands of agents across multiple symbols
3. Wire them together through the orchestrator
4. Run in live mode (Bybit) or backtest mode

Usage:
    # Backtest mode (no API keys needed)
    python -m crypto_trading_system.main --mode backtest

    # Paper trading on Bybit testnet
    python -m crypto_trading_system.main --mode paper

    # Live trading (requires BYBIT_API_KEY and BYBIT_API_SECRET env vars)
    python -m crypto_trading_system.main --mode live
"""

import argparse
import asyncio
import logging
import sys

from .core.message_bus import MessageBus, MessageType, Message
from .core.agent_registry import AgentRegistry
from .core.orchestrator import Orchestrator

# Strategy agents
from .agents.strategy.trend_following import MACrossoverAgent, MACDAgent, BreakoutAgent
from .agents.strategy.mean_reversion import BollingerReversionAgent, RSIReversionAgent, ZScoreReversionAgent
from .agents.strategy.momentum import ROCMomentumAgent, VolumeWeightedMomentumAgent, MultiTimeframeMomentumAgent

# Analysis agents
from .agents.analysis.market_analyzer import (
    TechnicalAnalysisAgent, VolatilityRegimeAgent,
    CorrelationAgent, SentimentAgent,
)

# Risk agents
from .agents.risk.risk_manager import (
    PositionSizingAgent, DrawdownMonitorAgent, ExposureManagerAgent,
)

# Execution
from .agents.execution.order_executor import OrderExecutorAgent

# Exchange
from .exchange.bybit_client import BybitClient

# Simulation
from .agents.simulation.backtester import BacktestEngine, MonteCarloSimulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Symbols to trade ──────────────────────────────────────────

TRADING_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "MATICUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT", "OPUSDT",
    "ARBUSDT", "SUIUSDT", "APTUSDT", "INJUSDT", "TIAUSDT",
]

# ── MA parameter combinations for scaling agents ─────────────

MA_PARAMS = [
    (5, 13), (7, 21), (9, 21), (10, 30), (12, 26),
    (15, 50), (20, 50), (20, 100), (50, 100), (50, 200),
]

BREAKOUT_LOOKBACKS = [10, 14, 20, 30, 55]  # Turtle trading inspired
ROC_PERIODS = [5, 10, 14, 21]


def create_agents(message_bus: MessageBus, registry: AgentRegistry, symbols: list[str]):
    """
    Create a full suite of agents for each symbol.
    With 20 symbols and ~30 agent types per symbol, this creates ~600+ agents.
    Add more parameter variations to scale to thousands.
    """
    agent_count = 0

    for symbol in symbols:
        # ── Strategy: Trend Following ──
        for fast, slow in MA_PARAMS:
            agent = MACrossoverAgent(message_bus, symbol, fast, slow)
            registry.register(agent, "strategy", tags=["trend", symbol])
            agent_count += 1

        agent = MACDAgent(message_bus, symbol)
        registry.register(agent, "strategy", tags=["trend", symbol])
        agent_count += 1

        for lookback in BREAKOUT_LOOKBACKS:
            agent = BreakoutAgent(message_bus, symbol, lookback)
            registry.register(agent, "strategy", tags=["trend", symbol])
            agent_count += 1

        # ── Strategy: Mean Reversion ──
        agent = BollingerReversionAgent(message_bus, symbol)
        registry.register(agent, "strategy", tags=["mean_reversion", symbol])
        agent_count += 1

        agent = RSIReversionAgent(message_bus, symbol)
        registry.register(agent, "strategy", tags=["mean_reversion", symbol])
        agent_count += 1

        agent = ZScoreReversionAgent(message_bus, symbol)
        registry.register(agent, "strategy", tags=["mean_reversion", symbol])
        agent_count += 1

        # ── Strategy: Momentum ──
        for period in ROC_PERIODS:
            agent = ROCMomentumAgent(message_bus, symbol, period)
            registry.register(agent, "strategy", tags=["momentum", symbol])
            agent_count += 1

        agent = VolumeWeightedMomentumAgent(message_bus, symbol)
        registry.register(agent, "strategy", tags=["momentum", symbol])
        agent_count += 1

        agent = MultiTimeframeMomentumAgent(message_bus, symbol)
        registry.register(agent, "strategy", tags=["momentum", symbol])
        agent_count += 1

        # ── Analysis ──
        agent = TechnicalAnalysisAgent(message_bus, symbol)
        registry.register(agent, "analysis", tags=["technical", symbol])
        agent_count += 1

        agent = VolatilityRegimeAgent(message_bus, symbol)
        registry.register(agent, "analysis", tags=["volatility", symbol])
        agent_count += 1

        agent = SentimentAgent(message_bus, symbol)
        registry.register(agent, "analysis", tags=["sentiment", symbol])
        agent_count += 1

    # ── Cross-asset analysis ──
    agent = CorrelationAgent(message_bus, symbols)
    registry.register(agent, "analysis", tags=["correlation"])
    agent_count += 1

    # ── Risk Management (system-wide) ──
    agent = PositionSizingAgent(message_bus, {"max_risk_per_trade": 0.02, "initial_balance": 10000})
    registry.register(agent, "risk", tags=["position_sizing"])
    agent_count += 1

    agent = DrawdownMonitorAgent(message_bus, {
        "warning_threshold": 0.05,
        "reduce_threshold": 0.10,
        "emergency_threshold": 0.15,
    })
    registry.register(agent, "risk", tags=["drawdown"])
    agent_count += 1

    agent = ExposureManagerAgent(message_bus, {
        "max_leverage": 3.0,
        "max_single_asset_pct": 0.20,
    })
    registry.register(agent, "risk", tags=["exposure"])
    agent_count += 1

    logger.info(f"Created {agent_count} agents across {len(symbols)} symbols")
    return agent_count


async def run_live(symbols: list[str], testnet: bool = True):
    """Run the system in live/paper trading mode with Bybit."""
    message_bus = MessageBus()
    registry = AgentRegistry()

    # Create exchange client
    exchange = BybitClient(testnet=testnet)

    # Create all agents
    create_agents(message_bus, registry, symbols)

    # Add execution agent with exchange client
    executor = OrderExecutorAgent(message_bus, exchange_client=exchange, config={
        "use_limit_orders": True,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
    })
    registry.register(executor, "execution", tags=["executor"])

    # Create and start orchestrator
    orchestrator = Orchestrator(message_bus, registry)
    await orchestrator.start()

    logger.info("=" * 60)
    logger.info(f"  MULTI-AGENT CRYPTO TRADING SYSTEM")
    logger.info(f"  Mode: {'TESTNET' if testnet else 'LIVE'}")
    logger.info(f"  Agents: {registry.count}")
    logger.info(f"  Symbols: {len(symbols)}")
    logger.info("=" * 60)

    # Main data feed loop
    try:
        while True:
            for symbol in symbols:
                try:
                    # Fetch latest kline
                    klines = await exchange.get_klines(symbol, interval="1", limit=1)
                    if klines:
                        candle = klines[-1]
                        await message_bus.publish(Message(
                            type=MessageType.MARKET_DATA,
                            channel=f"market_data.{symbol}",
                            payload={
                                "symbol": symbol,
                                "open": candle["open"],
                                "high": candle["high"],
                                "low": candle["low"],
                                "close": candle["close"],
                                "volume": candle["volume"],
                            },
                            sender_id="data_feed",
                        ))

                    # Fetch funding rate periodically
                    funding = await exchange.get_funding_rate(symbol)
                    if funding:
                        await message_bus.publish(Message(
                            type=MessageType.MARKET_DATA,
                            channel=f"funding.{symbol}",
                            payload=funding,
                            sender_id="data_feed",
                        ))
                except Exception as e:
                    logger.error(f"Data feed error for {symbol}: {e}")

            # Wait for next candle
            await asyncio.sleep(60)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await orchestrator.stop()
        await exchange.close()


async def run_backtest(symbols: list[str]):
    """Run backtest with synthetic data (replace with real historical data)."""
    import math
    import random

    message_bus = MessageBus()
    registry = AgentRegistry()
    create_agents(message_bus, registry, symbols[:3])  # Backtest with fewer symbols

    executor = OrderExecutorAgent(message_bus, config={"stop_loss_pct": 0.02, "take_profit_pct": 0.04})
    registry.register(executor, "execution")

    orchestrator = Orchestrator(message_bus, registry)

    # Generate synthetic price data (replace with real data from Bybit API)
    logger.info("Generating synthetic data for backtest...")
    num_candles = 5000
    price = 50000.0  # Starting BTC price
    historical = []
    for i in range(num_candles):
        # Random walk with slight upward drift + occasional volatility spikes
        drift = 0.0001
        vol = 0.02 * (1 + 0.5 * math.sin(i / 200))  # Varying volatility
        change = random.gauss(drift, vol)
        price *= (1 + change)
        high = price * (1 + abs(random.gauss(0, 0.005)))
        low = price * (1 - abs(random.gauss(0, 0.005)))
        volume = random.uniform(100, 1000) * (1 + abs(change) * 50)
        historical.append({
            "timestamp": i,
            "open": price * (1 + random.gauss(0, 0.001)),
            "high": high,
            "low": low,
            "close": price,
            "volume": volume,
        })

    # Run backtest
    engine = BacktestEngine(message_bus, initial_balance=10000.0)
    await orchestrator.start()
    result = await engine.run(historical, "BTCUSDT")
    await orchestrator.stop()

    # Print results
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Total Return:    {result.total_return_pct:>8.2f}%")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:>8.2f}")
    print(f"  Sortino Ratio:   {result.sortino_ratio:>8.2f}")
    print(f"  Max Drawdown:    {result.max_drawdown_pct:>8.2f}%")
    print(f"  Total Trades:    {result.total_trades:>8d}")
    print(f"  Win Rate:        {result.win_rate:>8.1f}%")
    print(f"  Profit Factor:   {result.profit_factor:>8.2f}")
    print(f"  Avg Trade PnL:  ${result.avg_trade_pnl:>8.2f}")
    print(f"  Best Trade:     ${result.best_trade:>8.2f}")
    print(f"  Worst Trade:    ${result.worst_trade:>8.2f}")
    print("=" * 60)

    # Monte Carlo
    if result.trades:
        print("\n  Running Monte Carlo simulation (1000 iterations)...")
        mc = MonteCarloSimulator(result.trades, initial_balance=10000.0)
        mc_result = mc.run(1000)
        print(f"  MC Mean Return:       {mc_result['return_mean']:>8.2f}%")
        print(f"  MC 5th Percentile:    {mc_result['return_5th_pct']:>8.2f}%")
        print(f"  MC 95th Percentile:   {mc_result['return_95th_pct']:>8.2f}%")
        print(f"  MC Avg Max Drawdown:  {mc_result['max_dd_mean']:>8.2f}%")
        print(f"  MC Prob Profitable:   {mc_result['probability_profitable']:>8.1f}%")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Crypto Trading System")
    parser.add_argument("--mode", choices=["live", "paper", "backtest"], default="backtest",
                        help="Trading mode")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Trading symbols (default: top 20)")
    args = parser.parse_args()

    symbols = args.symbols or TRADING_SYMBOLS

    if args.mode == "backtest":
        asyncio.run(run_backtest(symbols))
    elif args.mode == "paper":
        asyncio.run(run_live(symbols, testnet=True))
    elif args.mode == "live":
        print("\n⚠️  LIVE TRADING MODE — REAL MONEY AT RISK")
        print("  Make sure BYBIT_API_KEY and BYBIT_API_SECRET are set.")
        confirm = input("  Type 'YES' to confirm: ")
        if confirm != "YES":
            print("  Aborted.")
            sys.exit(0)
        asyncio.run(run_live(symbols, testnet=False))


if __name__ == "__main__":
    main()
