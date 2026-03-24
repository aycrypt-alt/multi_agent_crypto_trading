"""
Multi-Agent Crypto Trading System — Main Entry Point

This script demonstrates how to:
1. Create the message bus and agent registry
2. Spawn hundreds/thousands of agents across multiple symbols
3. Wire them together through the orchestrator
4. Run in live mode (Bybit) or backtest mode
5. Optimize agent weights via backtesting and performance analysis

Usage:
    # Backtest mode (no API keys needed)
    python -m crypto_trading_system.main --mode backtest

    # Optimize mode — backtest + analyze agents + adjust weights
    python -m crypto_trading_system.main --mode optimize

    # Optimize with real Bybit data (no API keys needed, uses public endpoints)
    python -m crypto_trading_system.main --mode optimize --use-real-data

    # Optimize with parameter grid search
    python -m crypto_trading_system.main --mode optimize --grid-search

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
from .agents.simulation.data_fetcher import HistoricalDataFetcher, generate_synthetic_data
from .agents.simulation.optimizer import (
    AgentPerformanceAnalyzer, ParameterOptimizer,
    compute_optimized_weights, apply_optimized_weights,
    print_performance_report, save_optimization_results,
    load_optimized_weights,
)

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


def create_strategy_agents(message_bus: MessageBus, registry: AgentRegistry, symbols: list[str]):
    """
    Create strategy and analysis agents only (no risk/execution).
    Used by both backtest and optimize modes.
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

    logger.info(f"Created {agent_count} strategy/analysis agents across {len(symbols)} symbols")
    return agent_count


def create_agents(message_bus: MessageBus, registry: AgentRegistry, symbols: list[str]):
    """
    Create a full suite of agents for each symbol.
    With 20 symbols and ~30 agent types per symbol, this creates ~600+ agents.
    """
    agent_count = create_strategy_agents(message_bus, registry, symbols)

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

    # Load optimized weights if available
    weights = load_optimized_weights()
    if weights:
        apply_optimized_weights(weights, registry)
        logger.info("Loaded and applied optimized agent weights from previous optimization")

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
    logger.info(f"  Optimized weights: {'YES' if weights else 'NO (default)'}")
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
    historical = generate_synthetic_data("BTCUSDT", num_candles=5000)

    # Run backtest
    engine = BacktestEngine(message_bus, initial_balance=10000.0, orchestrator=orchestrator)
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


async def run_optimize(symbols: list[str], use_real_data: bool = False, grid_search: bool = False):
    """
    Run backtest + agent performance analysis + weight optimization.

    Steps:
    1. Fetch historical data (real or synthetic)
    2. Run backtest with per-agent signal tracking
    3. Analyze each agent's prediction accuracy
    4. Generate optimized confidence weights
    5. Optionally run parameter grid search
    6. Save results and optimized weights to disk
    """
    backtest_symbols = symbols[:3]  # Use fewer symbols for faster optimization

    # ── Step 1: Get historical data ──
    if use_real_data:
        print("\n  Fetching real historical data from Bybit...")
        fetcher = HistoricalDataFetcher()
        try:
            historical_data = await fetcher.fetch_multi_symbol(
                backtest_symbols, interval="15", num_candles=5000
            )
        finally:
            await fetcher.close()

        if not any(historical_data.values()):
            print("  Failed to fetch real data, falling back to synthetic...")
            use_real_data = False

    if not use_real_data:
        print("\n  Generating synthetic historical data...")
        historical_data = {}
        prices = {"BTCUSDT": 50000, "ETHUSDT": 3000, "SOLUSDT": 100}
        for sym in backtest_symbols:
            start_price = prices.get(sym, 100)
            historical_data[sym] = generate_synthetic_data(sym, num_candles=5000, start_price=start_price)

    # ── Step 2: Run backtest with signal tracking ──
    print("\n  Running backtest with agent signal tracking...")
    message_bus = MessageBus()
    registry = AgentRegistry()

    create_agents(message_bus, registry, backtest_symbols)

    executor = OrderExecutorAgent(message_bus, config={"stop_loss_pct": 0.02, "take_profit_pct": 0.04})
    registry.register(executor, "execution")

    orchestrator = Orchestrator(message_bus, registry)
    engine = BacktestEngine(message_bus, initial_balance=10000.0, orchestrator=orchestrator)

    await orchestrator.start()
    primary_symbol = backtest_symbols[0]
    result = await engine.run(historical_data[primary_symbol], primary_symbol)
    await orchestrator.stop()

    # ── Step 3: Analyze agent performance ──
    print(f"\n  Analyzing {len(result.agent_signals)} agent signals...")
    analyzer = AgentPerformanceAnalyzer()
    agent_perfs, strategy_perfs = analyzer.analyze(result.agent_signals)

    # ── Step 4: Compute optimized weights ──
    weights = compute_optimized_weights(agent_perfs, registry)

    # ── Step 5: Optional parameter grid search ──
    grid_results = []
    if grid_search:
        print("\n  Running parameter grid search...")
        optimizer = ParameterOptimizer(create_strategy_agents, backtest_symbols)

        param_grid = {
            "signal_threshold": [0.2, 0.3, 0.4, 0.5],
            "stop_loss_pct": [0.015, 0.02, 0.03],
            "take_profit_pct": [0.03, 0.04, 0.06],
        }

        grid_results = await optimizer.grid_search(
            historical_data,
            param_grid,
        )

    # ── Step 6: Print report and save results ──
    print_performance_report(result, agent_perfs, strategy_perfs, grid_results)

    filepath = save_optimization_results(
        result, agent_perfs, strategy_perfs, weights, grid_results
    )

    # Monte Carlo
    if result.trades:
        print("\n  Running Monte Carlo simulation (1000 iterations)...")
        mc = MonteCarloSimulator(result.trades, initial_balance=10000.0)
        mc_result = mc.run(1000)
        print(f"  MC Mean Return:       {mc_result['return_mean']:>8.2f}%")
        print(f"  MC 5th Percentile:    {mc_result['return_5th_pct']:>8.2f}%")
        print(f"  MC 95th Percentile:   {mc_result['return_95th_pct']:>8.2f}%")
        print(f"  MC Prob Profitable:   {mc_result['probability_profitable']:>8.1f}%")

    print(f"\n  Results saved to: {filepath}")
    print(f"  Optimized weights saved — will be auto-loaded in paper/live mode.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Crypto Trading System")
    parser.add_argument("--mode", choices=["live", "paper", "backtest", "optimize"], default="backtest",
                        help="Trading mode")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Trading symbols (default: top 20)")
    parser.add_argument("--use-real-data", action="store_true",
                        help="Fetch real historical data from Bybit (optimize mode)")
    parser.add_argument("--grid-search", action="store_true",
                        help="Run parameter grid search (optimize mode)")
    args = parser.parse_args()

    symbols = args.symbols or TRADING_SYMBOLS

    if args.mode == "backtest":
        asyncio.run(run_backtest(symbols))
    elif args.mode == "optimize":
        asyncio.run(run_optimize(symbols, use_real_data=args.use_real_data, grid_search=args.grid_search))
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
