"""
Agent Performance Analyzer & Parameter Optimizer

Analyzes per-agent signal accuracy from backtest results and:
1. Ranks agents by accuracy, profit contribution, and consistency
2. Generates optimized confidence weights for each agent
3. Grid-searches parameter combinations to find best settings
4. Produces a detailed performance report
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ...core.agent_registry import AgentRegistry
from ...core.message_bus import MessageBus
from ...core.orchestrator import Orchestrator
from .backtester import BacktestEngine, BacktestResult, AgentSignalRecord, MonteCarloSimulator

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "optimization_results"


@dataclass
class AgentPerformance:
    """Performance metrics for a single agent."""
    agent_name: str
    strategy: str
    total_signals: int = 0
    correct_10: int = 0
    correct_30: int = 0
    correct_60: int = 0
    accuracy_10: float = 0.0
    accuracy_30: float = 0.0
    accuracy_60: float = 0.0
    avg_strength: float = 0.0
    avg_confidence: float = 0.0
    # Weighted score combining accuracy, consistency, and signal quality
    composite_score: float = 0.0
    recommended_confidence: float = 0.5


@dataclass
class StrategyPerformance:
    """Aggregate performance for a strategy type."""
    strategy: str
    agent_count: int = 0
    total_signals: int = 0
    avg_accuracy_10: float = 0.0
    avg_accuracy_30: float = 0.0
    avg_accuracy_60: float = 0.0
    best_agent: str = ""
    worst_agent: str = ""
    recommended_weight: float = 1.0


@dataclass
class OptimizationResult:
    """Complete optimization output."""
    backtest_result: BacktestResult
    agent_performances: list[AgentPerformance]
    strategy_performances: list[StrategyPerformance]
    optimized_weights: dict[str, float]
    parameter_grid_results: list[dict] = field(default_factory=list)
    best_params: dict = field(default_factory=dict)
    improvement_pct: float = 0.0


class AgentPerformanceAnalyzer:
    """Analyzes agent signal records from a backtest to determine performance."""

    def analyze(self, signals: list[AgentSignalRecord]) -> tuple[list[AgentPerformance], list[StrategyPerformance]]:
        """
        Analyze all agent signals and produce per-agent and per-strategy metrics.

        Returns:
            (agent_performances, strategy_performances)
        """
        # Group signals by agent
        by_agent: dict[str, list[AgentSignalRecord]] = defaultdict(list)
        for sig in signals:
            by_agent[sig.agent_name].append(sig)

        agent_perfs = []
        for agent_name, agent_sigs in by_agent.items():
            perf = self._analyze_agent(agent_name, agent_sigs)
            agent_perfs.append(perf)

        # Sort by composite score descending
        agent_perfs.sort(key=lambda p: p.composite_score, reverse=True)

        # Group by strategy
        by_strategy: dict[str, list[AgentPerformance]] = defaultdict(list)
        for perf in agent_perfs:
            by_strategy[perf.strategy].append(perf)

        strategy_perfs = []
        for strategy, perfs in by_strategy.items():
            sp = self._analyze_strategy(strategy, perfs)
            strategy_perfs.append(sp)

        strategy_perfs.sort(key=lambda s: s.avg_accuracy_30, reverse=True)

        return agent_perfs, strategy_perfs

    def _analyze_agent(self, agent_name: str, signals: list[AgentSignalRecord]) -> AgentPerformance:
        total = len(signals)
        if total == 0:
            return AgentPerformance(agent_name=agent_name, strategy="unknown")

        strategy = signals[0].strategy

        correct_10 = sum(1 for s in signals if s.was_correct_10)
        correct_30 = sum(1 for s in signals if s.was_correct_30)
        correct_60 = sum(1 for s in signals if s.was_correct_60)

        acc_10 = correct_10 / total
        acc_30 = correct_30 / total
        acc_60 = correct_60 / total

        avg_strength = sum(s.strength for s in signals) / total
        avg_confidence = sum(s.confidence for s in signals) / total

        # Composite score: weighted combination
        # Heavier weight on 30-candle accuracy (most relevant for trading)
        composite = (acc_10 * 0.2 + acc_30 * 0.5 + acc_60 * 0.3)

        # Bonus for high-strength signals being correct
        high_strength_sigs = [s for s in signals if s.strength > 0.5]
        if high_strength_sigs:
            hs_correct = sum(1 for s in high_strength_sigs if s.was_correct_30)
            hs_accuracy = hs_correct / len(high_strength_sigs)
            composite = composite * 0.7 + hs_accuracy * 0.3

        # Penalize agents with very few signals (not enough data)
        if total < 10:
            composite *= (total / 10)

        # Recommended confidence: scale composite to 0.1-1.0 range
        recommended = max(0.1, min(1.0, composite * 2))

        return AgentPerformance(
            agent_name=agent_name,
            strategy=strategy,
            total_signals=total,
            correct_10=correct_10,
            correct_30=correct_30,
            correct_60=correct_60,
            accuracy_10=round(acc_10 * 100, 1),
            accuracy_30=round(acc_30 * 100, 1),
            accuracy_60=round(acc_60 * 100, 1),
            avg_strength=round(avg_strength, 3),
            avg_confidence=round(avg_confidence, 3),
            composite_score=round(composite, 4),
            recommended_confidence=round(recommended, 3),
        )

    def _analyze_strategy(self, strategy: str, perfs: list[AgentPerformance]) -> StrategyPerformance:
        active = [p for p in perfs if p.total_signals > 0]
        if not active:
            return StrategyPerformance(strategy=strategy)

        avg_10 = sum(p.accuracy_10 for p in active) / len(active)
        avg_30 = sum(p.accuracy_30 for p in active) / len(active)
        avg_60 = sum(p.accuracy_60 for p in active) / len(active)

        best = max(active, key=lambda p: p.composite_score)
        worst = min(active, key=lambda p: p.composite_score)

        # Recommended strategy weight based on average accuracy
        weight = max(0.1, min(2.0, avg_30 / 50))  # Normalize around 50% accuracy

        return StrategyPerformance(
            strategy=strategy,
            agent_count=len(active),
            total_signals=sum(p.total_signals for p in active),
            avg_accuracy_10=round(avg_10, 1),
            avg_accuracy_30=round(avg_30, 1),
            avg_accuracy_60=round(avg_60, 1),
            best_agent=best.agent_name,
            worst_agent=worst.agent_name,
            recommended_weight=round(weight, 3),
        )


class ParameterOptimizer:
    """
    Grid search over parameter combinations to find the best configuration.
    Runs backtests with different MA periods, thresholds, and risk params.
    """

    def __init__(self, create_agents_fn, symbols: list[str]):
        """
        Args:
            create_agents_fn: Function(message_bus, registry, symbols, **params) -> int
            symbols: Trading symbols to backtest
        """
        self.create_agents_fn = create_agents_fn
        self.symbols = symbols

    async def grid_search(
        self,
        historical_data: dict[str, list[dict]],
        param_grid: dict[str, list],
    ) -> list[dict]:
        """
        Run backtest for each parameter combination in the grid.

        Args:
            historical_data: {symbol: [candles]} data
            param_grid: {"param_name": [value1, value2, ...]}
                Supported params:
                - signal_threshold: min signal strength to act (default 0.3)
                - stop_loss_pct: stop loss percentage
                - take_profit_pct: take profit percentage
                - max_risk_per_trade: max % of balance to risk per trade

        Returns:
            List of result dicts sorted by Sharpe ratio
        """
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        logger.info(f"Running grid search: {len(combinations)} parameter combinations")
        results = []

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            logger.info(f"  [{i+1}/{len(combinations)}] Testing: {params}")

            try:
                result = await self._run_single_backtest(historical_data, params)
                result_entry = {
                    "params": params,
                    "total_return_pct": result.total_return_pct,
                    "sharpe_ratio": result.sharpe_ratio,
                    "sortino_ratio": result.sortino_ratio,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "total_trades": result.total_trades,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "leverage": result.leverage,
                    "liquidations": result.liquidations,
                    "final_balance": result.final_balance,
                }
                results.append(result_entry)
                lev_str = f"Lev: {result.leverage:.0f}x | " if result.leverage > 1 else ""
                logger.info(f"    {lev_str}Return: {result.total_return_pct:.2f}% | "
                           f"Sharpe: {result.sharpe_ratio:.2f} | "
                           f"MaxDD: {result.max_drawdown_pct:.2f}% | "
                           f"Liqs: {result.liquidations}")
            except Exception as e:
                logger.error(f"    Failed: {e}")
                results.append({"params": params, "error": str(e)})

        # Sort by Sharpe ratio (risk-adjusted returns)
        valid = [r for r in results if "error" not in r]
        valid.sort(key=lambda r: r["sharpe_ratio"], reverse=True)
        return valid

    async def _run_single_backtest(
        self, historical_data: dict[str, list[dict]], params: dict
    ) -> BacktestResult:
        """Run a single backtest with specific parameters."""
        from ...agents.execution.order_executor import OrderExecutorAgent
        from ...agents.risk.risk_manager import PositionSizingAgent, DrawdownMonitorAgent, ExposureManagerAgent

        message_bus = MessageBus()
        registry = AgentRegistry()

        # Create strategy agents with default params
        self.create_agents_fn(message_bus, registry, self.symbols)

        # Override risk params if specified
        risk_config = {
            "max_risk_per_trade": params.get("max_risk_per_trade", 0.02),
            "initial_balance": 10000,
        }
        pos_sizer = PositionSizingAgent(message_bus, risk_config)
        registry.register(pos_sizer, "risk", tags=["position_sizing"])

        dd_config = {
            "warning_threshold": params.get("dd_warning", 0.05),
            "reduce_threshold": params.get("dd_reduce", 0.10),
            "emergency_threshold": params.get("dd_emergency", 0.15),
        }
        dd_monitor = DrawdownMonitorAgent(message_bus, dd_config)
        registry.register(dd_monitor, "risk", tags=["drawdown"])

        exposure_config = {
            "max_leverage": params.get("max_leverage", 3.0),
            "max_single_asset_pct": params.get("max_single_asset_pct", 0.20),
        }
        exposure_mgr = ExposureManagerAgent(message_bus, exposure_config)
        registry.register(exposure_mgr, "risk", tags=["exposure"])

        exec_config = {
            "stop_loss_pct": params.get("stop_loss_pct", 0.02),
            "take_profit_pct": params.get("take_profit_pct", 0.04),
        }
        executor = OrderExecutorAgent(message_bus, config=exec_config)
        registry.register(executor, "execution")

        orchestrator = Orchestrator(message_bus, registry)

        # Override signal threshold if specified
        if "signal_threshold" in params:
            # Patch the aggregation loop threshold
            original_loop = orchestrator._aggregation_loop

            async def patched_loop():
                threshold = params["signal_threshold"]
                while orchestrator._running:
                    await asyncio.sleep(orchestrator._signal_window_ms / 1000)
                    for symbol, signals in list(orchestrator._pending_signals.items()):
                        if not signals:
                            continue
                        aggregated = orchestrator._aggregate_signals(symbol, signals)
                        if aggregated and abs(aggregated.strength) > threshold:
                            await orchestrator.message_bus.publish(
                                __import__('crypto_trading_system.core.message_bus', fromlist=['Message']).Message(
                                    type=__import__('crypto_trading_system.core.message_bus', fromlist=['MessageType']).MessageType.ORDER_REQUEST,
                                    channel="risk_check",
                                    payload={
                                        "symbol": aggregated.symbol,
                                        "direction": aggregated.direction,
                                        "strength": aggregated.strength,
                                        "confidence": aggregated.confidence,
                                        "contributing_agents": aggregated.contributing_agents,
                                    },
                                    sender_id="orchestrator",
                                    priority=75,
                                ))
                        orchestrator._pending_signals[symbol] = []

            orchestrator._aggregation_loop = patched_loop

        leverage = params.get("leverage", 1.0)
        engine = BacktestEngine(message_bus, initial_balance=10000.0, orchestrator=orchestrator, leverage=leverage)
        await orchestrator.start()

        # Run on first available symbol
        symbol = self.symbols[0]
        data = historical_data.get(symbol, [])
        if not data:
            data = list(historical_data.values())[0] if historical_data else []

        result = await engine.run(data, symbol)
        await orchestrator.stop()

        return result


def compute_optimized_weights(
    agent_perfs: list[AgentPerformance],
    registry: AgentRegistry | None = None,
) -> dict[str, float]:
    """
    Compute optimized confidence weights for each agent based on backtest performance.
    Returns {agent_name: recommended_confidence}.
    """
    weights = {}
    for perf in agent_perfs:
        weights[perf.agent_name] = perf.recommended_confidence
    return weights


def apply_optimized_weights(
    weights: dict[str, float],
    registry: AgentRegistry,
):
    """Apply optimized confidence weights to agents in the registry."""
    applied = 0
    for agent_name, confidence in weights.items():
        agent = registry.get_by_name(agent_name)
        if agent:
            agent.set_confidence(confidence)
            applied += 1
    logger.info(f"Applied optimized weights to {applied} agents")
    return applied


def print_performance_report(
    result: BacktestResult,
    agent_perfs: list[AgentPerformance],
    strategy_perfs: list[StrategyPerformance],
    grid_results: list[dict] | None = None,
):
    """Print a comprehensive performance report to stdout."""

    print("\n" + "=" * 80)
    print("  BACKTEST PERFORMANCE REPORT")
    print("=" * 80)

    # Overall results
    print(f"\n{'-' * 40}")
    print("  PORTFOLIO PERFORMANCE")
    print(f"{'-' * 40}")
    if result.leverage > 1:
        print(f"  Leverage:          {result.leverage:>10.0f}x")
    print(f"  Total Return:      {result.total_return_pct:>10.2f}%")
    print(f"  Final Balance:    ${result.final_balance:>10.2f}")
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:     {result.sortino_ratio:>10.2f}")
    print(f"  Max Drawdown:      {result.max_drawdown_pct:>10.2f}%")
    print(f"  Total Trades:      {result.total_trades:>10d}")
    if result.liquidations > 0:
        print(f"  Liquidations:      {result.liquidations:>10d}")
    print(f"  Win Rate:          {result.win_rate:>10.1f}%")
    print(f"  Profit Factor:     {result.profit_factor:>10.2f}")
    print(f"  Avg Trade PnL:    ${result.avg_trade_pnl:>10.2f}")
    print(f"  Best Trade:       ${result.best_trade:>10.2f}")
    print(f"  Worst Trade:      ${result.worst_trade:>10.2f}")

    # Strategy performance
    if strategy_perfs:
        print(f"\n{'-' * 80}")
        print("  STRATEGY PERFORMANCE RANKING")
        print(f"{'-' * 80}")
        print(f"  {'Strategy':<25} {'Agents':>7} {'Signals':>8} {'Acc@10':>7} {'Acc@30':>7} {'Acc@60':>7} {'Weight':>7}")
        print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for sp in strategy_perfs:
            print(f"  {sp.strategy:<25} {sp.agent_count:>7d} {sp.total_signals:>8d} "
                  f"{sp.avg_accuracy_10:>6.1f}% {sp.avg_accuracy_30:>6.1f}% "
                  f"{sp.avg_accuracy_60:>6.1f}% {sp.recommended_weight:>7.3f}")

    # Top 20 agents
    if agent_perfs:
        active_agents = [p for p in agent_perfs if p.total_signals > 0]
        print(f"\n{'-' * 80}")
        print(f"  TOP {min(20, len(active_agents))} AGENTS BY COMPOSITE SCORE")
        print(f"{'-' * 80}")
        print(f"  {'Agent Name':<40} {'Sigs':>5} {'Acc@30':>7} {'Score':>7} {'New Wt':>7}")
        print(f"  {'-'*40} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")
        for perf in active_agents[:20]:
            name = perf.agent_name[:40]
            print(f"  {name:<40} {perf.total_signals:>5d} {perf.accuracy_30:>6.1f}% "
                  f"{perf.composite_score:>7.4f} {perf.recommended_confidence:>7.3f}")

        # Bottom 10 agents (underperformers)
        if len(active_agents) > 20:
            print(f"\n{'-' * 80}")
            print(f"  BOTTOM 10 AGENTS (Consider disabling)")
            print(f"{'-' * 80}")
            print(f"  {'Agent Name':<40} {'Sigs':>5} {'Acc@30':>7} {'Score':>7} {'New Wt':>7}")
            print(f"  {'-'*40} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")
            for perf in active_agents[-10:]:
                name = perf.agent_name[:40]
                print(f"  {name:<40} {perf.total_signals:>5d} {perf.accuracy_30:>6.1f}% "
                      f"{perf.composite_score:>7.4f} {perf.recommended_confidence:>7.3f}")

    # Grid search results
    if grid_results:
        print(f"\n{'-' * 80}")
        print(f"  PARAMETER OPTIMIZATION RESULTS (Top 5)")
        print(f"{'-' * 80}")
        for i, gr in enumerate(grid_results[:10]):
            lev = gr.get('leverage', 1)
            liqs = gr.get('liquidations', 0)
            bal = gr.get('final_balance', 0)
            lev_str = f"Lev={lev:.0f}x | " if lev > 1 else ""
            liq_str = f" | Liqs={liqs}" if liqs > 0 else ""
            print(f"\n  #{i+1}: {lev_str}Sharpe={gr['sharpe_ratio']:.2f} | "
                  f"Return={gr['total_return_pct']:.2f}% | "
                  f"Balance=${bal:.2f} | "
                  f"MaxDD={gr['max_drawdown_pct']:.2f}% | "
                  f"WinRate={gr['win_rate']:.1f}%{liq_str}")
            print(f"       Params: {gr['params']}")

        if len(grid_results) > 1:
            best = grid_results[0]
            worst = grid_results[-1]
            print(f"\n  Best Sharpe:  {best['sharpe_ratio']:.2f} ({best['params']})")
            print(f"  Worst Sharpe: {worst['sharpe_ratio']:.2f} ({worst['params']})")
            print(f"  Improvement:  {best['sharpe_ratio'] - worst['sharpe_ratio']:.2f} Sharpe points")

    print("\n" + "=" * 80)


def save_optimization_results(
    result: BacktestResult,
    agent_perfs: list[AgentPerformance],
    strategy_perfs: list[StrategyPerformance],
    weights: dict[str, float],
    grid_results: list[dict] | None = None,
):
    """Save optimization results to JSON for later use."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "portfolio": {
            "total_return_pct": result.total_return_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "max_drawdown_pct": result.max_drawdown_pct,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
        },
        "optimized_weights": weights,
        "agent_performances": [
            {
                "name": p.agent_name,
                "strategy": p.strategy,
                "signals": p.total_signals,
                "accuracy_10": p.accuracy_10,
                "accuracy_30": p.accuracy_30,
                "accuracy_60": p.accuracy_60,
                "composite_score": p.composite_score,
                "recommended_confidence": p.recommended_confidence,
            }
            for p in agent_perfs if p.total_signals > 0
        ],
        "strategy_performances": [
            {
                "strategy": s.strategy,
                "agent_count": s.agent_count,
                "signals": s.total_signals,
                "avg_accuracy_30": s.avg_accuracy_30,
                "recommended_weight": s.recommended_weight,
            }
            for s in strategy_perfs
        ],
    }

    if grid_results:
        output["grid_search"] = grid_results[:10]  # Top 10

    filepath = RESULTS_DIR / f"optimization_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    # Also save a "latest" file for easy loading
    latest = RESULTS_DIR / "latest_weights.json"
    with open(latest, "w") as f:
        json.dump({"weights": weights, "timestamp": timestamp}, f, indent=2)

    logger.info(f"Results saved to {filepath}")
    logger.info(f"Latest weights saved to {latest}")
    return filepath


def load_optimized_weights() -> dict[str, float] | None:
    """Load the most recent optimized weights from disk."""
    latest = RESULTS_DIR / "latest_weights.json"
    if latest.exists():
        with open(latest) as f:
            data = json.load(f)
        logger.info(f"Loaded optimized weights from {data.get('timestamp', 'unknown')}")
        return data.get("weights", {})
    return None


@dataclass
class WalkForwardWindow:
    """Results from a single walk-forward window."""
    window_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_result: BacktestResult | None = None
    test_result: BacktestResult | None = None
    optimized_weights: dict = field(default_factory=dict)


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization: rolling train/test windows to prevent overfitting.

    Instead of training on ALL data and testing on ALL data (in-sample bias),
    this splits data into rolling windows:
      [=== TRAIN 1 ===][= TEST 1 =]
                [=== TRAIN 2 ===][= TEST 2 =]
                          [=== TRAIN 3 ===][= TEST 3 =]

    Weights learned on each train window are applied out-of-sample on the test window.
    Final performance is the aggregate of all out-of-sample test windows.
    """

    def __init__(
        self,
        create_agents_fn,
        symbols: list[str],
        train_size: int = 3000,
        test_size: int = 1000,
        step_size: int = 500,
    ):
        self.create_agents_fn = create_agents_fn
        self.symbols = symbols
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

    async def run(
        self,
        historical_data: dict[str, list[dict]],
        leverage: float = 1.0,
    ) -> dict:
        """
        Run walk-forward optimization across rolling windows.

        Returns a summary dict with per-window and aggregate results.
        """
        symbol = self.symbols[0]
        all_candles = historical_data.get(symbol, [])
        if not all_candles:
            all_candles = list(historical_data.values())[0] if historical_data else []

        total_candles = len(all_candles)
        min_required = self.train_size + self.test_size
        if total_candles < min_required:
            logger.warning(
                f"Not enough data for walk-forward: {total_candles} candles, "
                f"need at least {min_required}. Reducing window sizes."
            )
            self.train_size = int(total_candles * 0.6)
            self.test_size = int(total_candles * 0.2)
            self.step_size = max(self.test_size // 2, 100)

        windows: list[WalkForwardWindow] = []
        start = 0
        window_idx = 0

        while start + self.train_size + self.test_size <= total_candles:
            train_end = start + self.train_size
            test_end = train_end + self.test_size

            wf = WalkForwardWindow(
                window_index=window_idx,
                train_start=start,
                train_end=train_end,
                test_start=train_end,
                test_end=test_end,
            )
            windows.append(wf)
            start += self.step_size
            window_idx += 1

        logger.info(
            f"Walk-forward: {len(windows)} windows, "
            f"train={self.train_size}, test={self.test_size}, step={self.step_size}"
        )

        # Process each window
        all_test_trades = []
        all_test_returns = []
        all_test_equity = []

        for wf in windows:
            logger.info(
                f"  Window {wf.window_index + 1}/{len(windows)}: "
                f"train[{wf.train_start}:{wf.train_end}] "
                f"test[{wf.test_start}:{wf.test_end}]"
            )

            train_data = all_candles[wf.train_start:wf.train_end]
            test_data = all_candles[wf.test_start:wf.test_end]

            # --- TRAIN: run backtest to learn agent weights ---
            train_result, weights = await self._train_window(
                {symbol: train_data}, symbol, leverage
            )
            wf.train_result = train_result
            wf.optimized_weights = weights

            logger.info(
                f"    Train: Return={train_result.total_return_pct:.2f}%, "
                f"Sharpe={train_result.sharpe_ratio:.2f}, "
                f"Trades={train_result.total_trades}"
            )

            # --- TEST: apply learned weights out-of-sample ---
            test_result = await self._test_window(
                {symbol: test_data}, symbol, weights, leverage
            )
            wf.test_result = test_result

            logger.info(
                f"    Test:  Return={test_result.total_return_pct:.2f}%, "
                f"Sharpe={test_result.sharpe_ratio:.2f}, "
                f"Trades={test_result.total_trades}"
            )

            all_test_trades.extend(test_result.trades)
            all_test_returns.append(test_result.total_return_pct)
            if test_result.equity_curve:
                all_test_equity.extend(test_result.equity_curve)

        # --- Aggregate out-of-sample results ---
        summary = self._aggregate_results(windows, all_test_trades, all_test_returns)
        summary["leverage"] = leverage

        # Save walk-forward results
        self._save_results(summary, windows)

        return summary

    async def _train_window(
        self,
        data: dict[str, list[dict]],
        symbol: str,
        leverage: float,
    ) -> tuple[BacktestResult, dict[str, float]]:
        """Run backtest on training data and extract optimized weights."""
        from ...agents.execution.order_executor import OrderExecutorAgent
        from ...agents.risk.risk_manager import (
            PositionSizingAgent, DrawdownMonitorAgent, ExposureManagerAgent,
        )

        message_bus = MessageBus()
        registry = AgentRegistry()

        self.create_agents_fn(message_bus, registry, self.symbols)

        pos_sizer = PositionSizingAgent(
            message_bus, {"max_risk_per_trade": 0.02, "initial_balance": 10000}
        )
        registry.register(pos_sizer, "risk", tags=["position_sizing"])

        dd_monitor = DrawdownMonitorAgent(
            message_bus,
            {"warning_threshold": 0.05, "reduce_threshold": 0.10, "emergency_threshold": 0.15},
        )
        registry.register(dd_monitor, "risk", tags=["drawdown"])

        exposure_mgr = ExposureManagerAgent(
            message_bus, {"max_leverage": 3.0, "max_single_asset_pct": 0.20}
        )
        registry.register(exposure_mgr, "risk", tags=["exposure"])

        executor = OrderExecutorAgent(
            message_bus, config={"stop_loss_pct": 0.02, "take_profit_pct": 0.04}
        )
        registry.register(executor, "execution")

        orchestrator = Orchestrator(message_bus, registry)
        engine = BacktestEngine(
            message_bus, initial_balance=10000.0, orchestrator=orchestrator, leverage=leverage
        )

        await orchestrator.start()
        result = await engine.run(data[symbol], symbol)
        await orchestrator.stop()

        # Analyze signals to get weights
        analyzer = AgentPerformanceAnalyzer()
        agent_perfs, _ = analyzer.analyze(result.agent_signals)
        weights = compute_optimized_weights(agent_perfs, registry)

        return result, weights

    async def _test_window(
        self,
        data: dict[str, list[dict]],
        symbol: str,
        weights: dict[str, float],
        leverage: float,
    ) -> BacktestResult:
        """Run backtest on test data with pre-learned weights applied."""
        from ...agents.execution.order_executor import OrderExecutorAgent
        from ...agents.risk.risk_manager import (
            PositionSizingAgent, DrawdownMonitorAgent, ExposureManagerAgent,
        )

        message_bus = MessageBus()
        registry = AgentRegistry()

        self.create_agents_fn(message_bus, registry, self.symbols)

        # Apply learned weights from training
        apply_optimized_weights(weights, registry)

        pos_sizer = PositionSizingAgent(
            message_bus, {"max_risk_per_trade": 0.02, "initial_balance": 10000}
        )
        registry.register(pos_sizer, "risk", tags=["position_sizing"])

        dd_monitor = DrawdownMonitorAgent(
            message_bus,
            {"warning_threshold": 0.05, "reduce_threshold": 0.10, "emergency_threshold": 0.15},
        )
        registry.register(dd_monitor, "risk", tags=["drawdown"])

        exposure_mgr = ExposureManagerAgent(
            message_bus, {"max_leverage": 3.0, "max_single_asset_pct": 0.20}
        )
        registry.register(exposure_mgr, "risk", tags=["exposure"])

        executor = OrderExecutorAgent(
            message_bus, config={"stop_loss_pct": 0.02, "take_profit_pct": 0.04}
        )
        registry.register(executor, "execution")

        orchestrator = Orchestrator(message_bus, registry)
        engine = BacktestEngine(
            message_bus, initial_balance=10000.0, orchestrator=orchestrator, leverage=leverage
        )

        await orchestrator.start()
        result = await engine.run(data[symbol], symbol)
        await orchestrator.stop()

        return result

    def _aggregate_results(
        self,
        windows: list[WalkForwardWindow],
        all_test_trades: list,
        all_test_returns: list[float],
    ) -> dict:
        """Aggregate out-of-sample test results across all windows."""
        if not all_test_returns:
            return {
                "num_windows": len(windows),
                "oos_mean_return": 0.0,
                "oos_total_return": 0.0,
                "oos_sharpe": 0.0,
                "oos_total_trades": 0,
                "oos_win_rate": 0.0,
                "windows": [],
            }

        # Compound returns across windows
        compound_balance = 10000.0
        for ret in all_test_returns:
            compound_balance *= (1 + ret / 100)
        total_return = (compound_balance / 10000.0 - 1) * 100

        # Win rate across all OOS trades
        winning = sum(1 for t in all_test_trades if t.pnl > 0)
        total_trades = len(all_test_trades)
        win_rate = (winning / total_trades * 100) if total_trades > 0 else 0.0

        # OOS Sharpe from per-window returns
        mean_ret = sum(all_test_returns) / len(all_test_returns)
        if len(all_test_returns) > 1:
            variance = sum((r - mean_ret) ** 2 for r in all_test_returns) / (len(all_test_returns) - 1)
            std_ret = variance ** 0.5
            oos_sharpe = (mean_ret / std_ret) if std_ret > 0 else 0.0
        else:
            oos_sharpe = 0.0

        # Train vs test comparison (overfitting ratio)
        train_returns = [
            w.train_result.total_return_pct
            for w in windows
            if w.train_result is not None
        ]
        avg_train = sum(train_returns) / len(train_returns) if train_returns else 0.0
        avg_test = mean_ret
        overfit_ratio = (avg_test / avg_train) if avg_train != 0 else 1.0

        window_details = []
        for w in windows:
            detail = {
                "window": w.window_index,
                "train_return": w.train_result.total_return_pct if w.train_result else 0.0,
                "test_return": w.test_result.total_return_pct if w.test_result else 0.0,
                "train_sharpe": w.train_result.sharpe_ratio if w.train_result else 0.0,
                "test_sharpe": w.test_result.sharpe_ratio if w.test_result else 0.0,
                "train_trades": w.train_result.total_trades if w.train_result else 0,
                "test_trades": w.test_result.total_trades if w.test_result else 0,
            }
            window_details.append(detail)

        return {
            "num_windows": len(windows),
            "oos_mean_return": round(mean_ret, 4),
            "oos_total_return": round(total_return, 4),
            "oos_compound_balance": round(compound_balance, 2),
            "oos_sharpe": round(oos_sharpe, 4),
            "oos_total_trades": total_trades,
            "oos_win_rate": round(win_rate, 1),
            "avg_train_return": round(avg_train, 4),
            "overfit_ratio": round(overfit_ratio, 4),
            "windows": window_details,
        }

    def _save_results(self, summary: dict, windows: list[WalkForwardWindow]):
        """Save walk-forward results to disk."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        filepath = RESULTS_DIR / f"walkforward_{timestamp}.json"
        output = {"timestamp": timestamp, **summary}
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        # Save the weights from the LAST window (most recent data)
        if windows and windows[-1].optimized_weights:
            latest = RESULTS_DIR / "latest_weights.json"
            with open(latest, "w") as f:
                json.dump(
                    {"weights": windows[-1].optimized_weights, "timestamp": timestamp,
                     "source": "walk_forward"},
                    f, indent=2,
                )
            logger.info(f"Walk-forward weights saved from last window")

        logger.info(f"Walk-forward results saved to {filepath}")
        return filepath


def print_walkforward_report(summary: dict):
    """Print walk-forward optimization results."""
    print("\n" + "=" * 80)
    print("  WALK-FORWARD OPTIMIZATION REPORT")
    print("=" * 80)

    lev = summary.get("leverage", 1)
    lev_str = f"  Leverage:              {lev:.0f}x\n" if lev > 1 else ""

    print(f"\n{'-' * 50}")
    print("  OUT-OF-SAMPLE AGGREGATE RESULTS")
    print(f"{'-' * 50}")
    print(lev_str, end="")
    print(f"  Windows:               {summary['num_windows']}")
    print(f"  OOS Mean Return:       {summary['oos_mean_return']:.2f}%")
    print(f"  OOS Total Return:      {summary['oos_total_return']:.2f}%")
    print(f"  OOS Compound Balance: ${summary.get('oos_compound_balance', 10000):.2f}")
    print(f"  OOS Sharpe Ratio:      {summary['oos_sharpe']:.2f}")
    print(f"  OOS Total Trades:      {summary['oos_total_trades']}")
    print(f"  OOS Win Rate:          {summary['oos_win_rate']:.1f}%")
    print(f"  Avg Train Return:      {summary.get('avg_train_return', 0):.2f}%")

    overfit = summary.get("overfit_ratio", 1.0)
    if overfit > 0.8:
        label = "GOOD (low overfitting)"
    elif overfit > 0.5:
        label = "OK (moderate overfitting)"
    else:
        label = "POOR (significant overfitting)"
    print(f"  Overfit Ratio:         {overfit:.2f} - {label}")

    # Per-window breakdown
    windows = summary.get("windows", [])
    if windows:
        print(f"\n{'-' * 70}")
        print("  PER-WINDOW BREAKDOWN")
        print(f"{'-' * 70}")
        print(f"  {'Window':>6} {'Train Ret':>10} {'Test Ret':>10} "
              f"{'Train Shrp':>11} {'Test Shrp':>10} {'Trades':>7}")
        print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*11} {'-'*10} {'-'*7}")
        for w in windows:
            print(
                f"  {w['window']+1:>6} {w['train_return']:>9.2f}% "
                f"{w['test_return']:>9.2f}% {w['train_sharpe']:>11.2f} "
                f"{w['test_sharpe']:>10.2f} {w['test_trades']:>7}"
            )

    print("\n" + "=" * 80)
