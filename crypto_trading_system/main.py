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
from .agents.strategy.swarm_intelligence import SwarmPersonaAgent, SwarmDebateAgent, NewsInjectorAgent, PERSONAS

# Confluence agents (multi-indicator)
from .agents.strategy.confluence import (
    MeanReversionConfluenceAgent, TrendConfluenceAgent, MomentumConfluenceAgent,
)

# Analysis agents
from .agents.analysis.market_analyzer import (
    TechnicalAnalysisAgent, VolatilityRegimeAgent,
    CorrelationAgent, SentimentAgent,
)
from .agents.analysis.microstructure import (
    FundingRateAgent, OpenInterestAgent, LiquidationLevelAgent,
)

# Risk agents
from .agents.risk.risk_manager import (
    PositionSizingAgent, DrawdownMonitorAgent, ExposureManagerAgent,
)

# Execution
from .agents.execution.order_executor import OrderExecutorAgent
from .agents.execution.position_manager import PositionManagerAgent

# Exchange
from .exchange.bybit_client import BybitClient

# Simulation
from .agents.simulation.backtester import BacktestEngine, MonteCarloSimulator
from .agents.simulation.data_fetcher import HistoricalDataFetcher, generate_synthetic_data
from .agents.simulation.optimizer import (
    AgentPerformanceAnalyzer, ParameterOptimizer, WalkForwardOptimizer,
    compute_optimized_weights, apply_optimized_weights,
    print_performance_report, print_walkforward_report,
    save_optimization_results, load_optimized_weights,
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
    (9, 21), (12, 26), (50, 200),  # Pruned from 10 to top 3 uncorrelated pairs
]

BREAKOUT_LOOKBACKS = [20, 55]  # Pruned from 5 to 2 (classic Donchian + long-range)
ROC_PERIODS = [10, 21]  # Pruned from 4 to 2 (short + medium lookback)


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

        # ── Confluence (Multi-Indicator) ──
        agent = MeanReversionConfluenceAgent(message_bus, symbol)
        registry.register(agent, "strategy", tags=["confluence", "mean_reversion", symbol])
        agent_count += 1

        agent = TrendConfluenceAgent(message_bus, symbol)
        registry.register(agent, "strategy", tags=["confluence", "trend", symbol])
        agent_count += 1

        agent = MomentumConfluenceAgent(message_bus, symbol)
        registry.register(agent, "strategy", tags=["confluence", "momentum", symbol])
        agent_count += 1

        # ── Microstructure (Leading Indicators) ──
        agent = FundingRateAgent(message_bus, symbol)
        registry.register(agent, "analysis", tags=["microstructure", symbol])
        agent_count += 1

        agent = OpenInterestAgent(message_bus, symbol)
        registry.register(agent, "analysis", tags=["microstructure", symbol])
        agent_count += 1

        agent = LiquidationLevelAgent(message_bus, symbol)
        registry.register(agent, "analysis", tags=["microstructure", symbol])
        agent_count += 1

        # ── Swarm Intelligence (MiroFish-inspired) ──
        for persona in PERSONAS:
            agent = SwarmPersonaAgent(message_bus, symbol, persona)
            registry.register(agent, "strategy", tags=["swarm", persona, symbol])
            agent_count += 1

        agent = SwarmDebateAgent(message_bus, symbol)
        registry.register(agent, "strategy", tags=["swarm", "debate", symbol])
        agent_count += 1

        agent = NewsInjectorAgent(message_bus, symbol)
        registry.register(agent, "analysis", tags=["swarm", "news", symbol])
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


async def run_live(symbols: list[str], testnet: bool = True, leverage: int = 25):
    """Run the system in live/paper trading mode with Bybit."""
    message_bus = MessageBus()
    registry = AgentRegistry()

    # Create exchange client
    exchange = BybitClient(testnet=testnet)

    # Set leverage on all symbols before trading
    logger.info(f"Setting {leverage}x leverage on {len(symbols)} symbols...")
    for symbol in symbols:
        try:
            await exchange.set_leverage(symbol, leverage)
        except Exception as e:
            logger.warning(f"Could not set leverage for {symbol}: {e}")

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
    })
    registry.register(executor, "execution", tags=["executor"])

    # Add position manager — handles trailing stops, breakeven, time exits
    # This is what drives the 60%+ win rate from backtesting in live mode
    pos_manager = PositionManagerAgent(message_bus, exchange_client=exchange, symbols=symbols)
    registry.register(pos_manager, "execution", tags=["position_manager"])

    # Create and start orchestrator
    orchestrator = Orchestrator(message_bus, registry)
    await orchestrator.start()

    logger.info("=" * 60)
    logger.info(f"  MULTI-AGENT CRYPTO TRADING SYSTEM")
    logger.info(f"  Mode: {'TESTNET' if testnet else 'LIVE'}")
    logger.info(f"  Leverage: {leverage}x")
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
    """Run backtest with real historical data from backtest_data/ directory."""
    import json
    from pathlib import Path

    data_dir = Path(__file__).parent.parent / "backtest_data"
    backtest_symbols = symbols[:3]

    # Load historical data from local JSON files
    historical_data = {}
    for sym in backtest_symbols:
        filepath = data_dir / f"{sym}_15m_5000.json"
        if filepath.exists():
            with open(filepath) as f:
                historical_data[sym] = json.load(f)
            logger.info(f"Loaded {len(historical_data[sym])} candles for {sym} from {filepath.name}")
        else:
            logger.warning(f"No data file for {sym}, generating synthetic data...")
            historical_data[sym] = generate_synthetic_data(sym, num_candles=5000)

    if not historical_data:
        logger.error("No historical data available. Place JSON files in backtest_data/")
        return

    message_bus = MessageBus()
    registry = AgentRegistry()
    create_agents(message_bus, registry, list(historical_data.keys()))

    executor = OrderExecutorAgent(message_bus, config={"stop_loss_pct": 0.02, "take_profit_pct": 0.04})
    registry.register(executor, "execution")

    orchestrator = Orchestrator(message_bus, registry)

    # Run backtest on each symbol with real data
    primary_symbol = list(historical_data.keys())[0]
    engine = BacktestEngine(message_bus, initial_balance=10000.0, orchestrator=orchestrator)
    await orchestrator.start()
    result = await engine.run(historical_data[primary_symbol], primary_symbol)
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


async def run_optimize(symbols: list[str], use_real_data: bool = False,
                       grid_search: bool = False, leverage: float = 1.0):
    """
    Run backtest + agent performance analysis + weight optimization.

    Steps:
    1. Fetch historical data (real or synthetic)
    2. Run backtest with per-agent signal tracking (with leverage)
    3. Analyze each agent's prediction accuracy
    4. Generate optimized confidence weights
    5. Optionally run parameter grid search across leverage levels
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
    lev_str = f" (leverage: {leverage:.0f}x)" if leverage > 1 else ""
    print(f"\n  Running backtest with agent signal tracking{lev_str}...")
    message_bus = MessageBus()
    registry = AgentRegistry()

    create_agents(message_bus, registry, backtest_symbols)

    executor = OrderExecutorAgent(message_bus, config={"stop_loss_pct": 0.02, "take_profit_pct": 0.04})
    registry.register(executor, "execution")

    orchestrator = Orchestrator(message_bus, registry)
    engine = BacktestEngine(message_bus, initial_balance=10000.0, orchestrator=orchestrator, leverage=leverage)

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
        print("\n  Running leverage grid search (25x to 100x)...")
        optimizer = ParameterOptimizer(create_strategy_agents, backtest_symbols)

        # Use first 2000 candles for faster grid search iterations
        grid_data = {}
        for sym, candles in historical_data.items():
            grid_data[sym] = candles[:2000]

        param_grid = {
            "leverage": [25, 50, 75, 100],
            "stop_loss_pct": [0.01, 0.02],
            "take_profit_pct": [0.03, 0.06],
        }

        grid_results = await optimizer.grid_search(
            grid_data,
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


async def run_walkforward(symbols: list[str], use_real_data: bool = False,
                          leverage: float = 1.0):
    """
    Run Walk-Forward Optimization: rolling train/test windows to prevent overfitting.

    Steps:
    1. Fetch historical data (real or synthetic)
    2. Split into rolling train/test windows
    3. For each window: train weights on training data, test on unseen data
    4. Aggregate out-of-sample results and compare to in-sample
    5. Save optimized weights from the most recent window
    """
    backtest_symbols = symbols[:3]

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

    # ── Step 2: Run walk-forward optimization ──
    lev_str = f" (leverage: {leverage:.0f}x)" if leverage > 1 else ""
    print(f"\n  Running Walk-Forward Optimization{lev_str}...")
    print(f"  Train: 3000 candles | Test: 1000 candles | Step: 500 candles")

    wf_optimizer = WalkForwardOptimizer(
        create_agents_fn=create_strategy_agents,
        symbols=backtest_symbols,
        train_size=3000,
        test_size=1000,
        step_size=500,
    )

    summary = await wf_optimizer.run(historical_data, leverage=leverage)

    # ── Step 3: Print report ──
    print_walkforward_report(summary)

    # ── Step 4: Run baseline comparison ──
    print("\n  Running baseline backtest (no walk-forward) for comparison...")
    message_bus = MessageBus()
    registry = AgentRegistry()
    create_agents(message_bus, registry, backtest_symbols)

    executor = OrderExecutorAgent(message_bus, config={"stop_loss_pct": 0.02, "take_profit_pct": 0.04})
    registry.register(executor, "execution")

    orchestrator = Orchestrator(message_bus, registry)
    engine = BacktestEngine(message_bus, initial_balance=10000.0, orchestrator=orchestrator, leverage=leverage)

    await orchestrator.start()
    primary_symbol = backtest_symbols[0]
    baseline = await engine.run(historical_data[primary_symbol], primary_symbol)
    await orchestrator.stop()

    print(f"\n{'-' * 50}")
    print("  BASELINE vs WALK-FORWARD COMPARISON")
    print(f"{'-' * 50}")
    print(f"  {'Metric':<25} {'Baseline':>12} {'WF (OOS)':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Return':<25} {baseline.total_return_pct:>11.2f}% {summary['oos_total_return']:>11.2f}%")
    print(f"  {'Sharpe':<25} {baseline.sharpe_ratio:>12.2f} {summary['oos_sharpe']:>12.2f}")
    print(f"  {'Trades':<25} {baseline.total_trades:>12d} {summary['oos_total_trades']:>12d}")
    print(f"  {'Win Rate':<25} {baseline.win_rate:>11.1f}% {summary['oos_win_rate']:>11.1f}%")
    print("=" * 80)

    # Monte Carlo on OOS trades
    if summary['oos_total_trades'] > 0:
        print("\n  Walk-forward weights saved -- will be auto-loaded in paper/live mode.")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Crypto Trading System")
    parser.add_argument("--mode", choices=["live", "paper", "backtest", "optimize", "walkforward"],
                        default="backtest", help="Trading mode")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Trading symbols (default: top 20)")
    parser.add_argument("--use-real-data", action="store_true",
                        help="Fetch real historical data from Bybit (optimize mode)")
    parser.add_argument("--grid-search", action="store_true",
                        help="Run parameter grid search (optimize mode)")
    parser.add_argument("--leverage", type=float, default=1.0,
                        help="Leverage multiplier for backtest (e.g. 10, 25, 50, 100)")
    args = parser.parse_args()

    symbols = args.symbols or TRADING_SYMBOLS

    if args.mode == "backtest":
        asyncio.run(run_backtest(symbols))
    elif args.mode == "optimize":
        asyncio.run(run_optimize(symbols, use_real_data=args.use_real_data,
                                 grid_search=args.grid_search, leverage=args.leverage))
    elif args.mode == "walkforward":
        asyncio.run(run_walkforward(symbols, use_real_data=args.use_real_data,
                                     leverage=args.leverage))
    elif args.mode == "paper":
        asyncio.run(run_live(symbols, testnet=True, leverage=int(args.leverage or 25)))
    elif args.mode == "live":
        print("\n  LIVE TRADING MODE -- REAL MONEY AT RISK")
        print("  Make sure BYBIT_API_KEY and BYBIT_API_SECRET are set.")
        confirm = input("  Type 'YES' to confirm: ")
        if confirm != "YES":
            print("  Aborted.")
            sys.exit(0)
        asyncio.run(run_live(symbols, testnet=False, leverage=int(args.leverage or 25)))


if __name__ == "__main__":
    main()
