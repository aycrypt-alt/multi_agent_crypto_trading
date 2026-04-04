"""
Position Manager Agent — Live Trade Management

Monitors open positions and manages:
1. ATR-based stop-loss and take-profit (matches backtester logic)
2. Trailing stops that lock in profit as price moves favorably
3. Breakeven stop-loss once trade is slightly profitable
4. Time-based exits for stale trades that aren't moving

This agent bridges the gap between backtester trade management and live execution.
Without it, live trading would use simple fixed-% SL/TP and miss the exit
intelligence that drives the 60%+ win rate in backtesting.
"""

import logging
import time

from ...core.agent_base import Agent, AgentPriority
from ...core.message_bus import Message, MessageBus, MessageType
from ...utils.indicators import atr as compute_atr

logger = logging.getLogger(__name__)


class PositionManagerAgent(Agent):
    """
    Manages open positions with trailing stops, breakeven stops, and time exits.
    Subscribes to market data and monitors all open positions each candle.
    """

    # Must match backtester settings exactly
    ATR_SL_MULT = 1.5
    ATR_TP_MULT = 3.0
    ATR_PERIOD = 14
    TRAIL_ACTIVATION_ATR = 0.3
    TRAIL_DISTANCE_ATR = 0.6
    MAX_HOLD_CANDLES = 12
    FEE_RATE = 0.00075
    DEFAULT_LEVERAGE = 25

    def __init__(self, message_bus: MessageBus, exchange_client=None, symbols: list[str] = None,
                 leverage: int = 25):
        super().__init__(
            name="position_manager",
            message_bus=message_bus,
            priority=AgentPriority.HIGH,
        )
        self.exchange = exchange_client
        self.symbols = symbols or []
        self.leverage = leverage
        # Track positions: {symbol: {direction, entry_price, entry_candle, best_price, ...}}
        self._positions: dict[str, dict] = {}
        # OHLC for ATR
        self._highs: dict[str, list[float]] = {}
        self._lows: dict[str, list[float]] = {}
        self._closes: dict[str, list[float]] = {}
        self._candle_counts: dict[str, int] = {}

    async def on_start(self):
        for symbol in self.symbols:
            await self.subscribe(f"market_data.{symbol}")
            self._highs[symbol] = []
            self._lows[symbol] = []
            self._closes[symbol] = []
            self._candle_counts[symbol] = 0
        # Listen for new fills to track positions
        await self.subscribe("fills")

    async def process(self, message: Message) -> dict | None:
        if message.type == MessageType.ORDER_FILLED:
            return await self._on_fill(message)

        if message.type == MessageType.MARKET_DATA:
            return await self._on_market_data(message)

        return None

    async def _on_fill(self, message: Message) -> dict | None:
        """Track new position from a fill."""
        symbol = message.payload.get("symbol", "")
        side = message.payload.get("side", "")
        fill_price = message.payload.get("fill_price", 0.0)
        size_usd = message.payload.get("size_usd", 0.0)

        if not symbol or not side or fill_price <= 0:
            return None

        direction = "long" if side == "Buy" else "short"

        # Compute ATR for this symbol
        current_atr = self._compute_atr(symbol)

        # Leverage-aware stop: ensure SL is always tighter than liquidation distance
        if self.leverage > 1:
            max_sl_distance = fill_price * (0.95 / self.leverage) * 0.6
            sl_distance = min(current_atr * self.ATR_SL_MULT, max_sl_distance)
            tp_distance = min(current_atr * self.ATR_TP_MULT, max_sl_distance * 2.5)
        else:
            sl_distance = current_atr * self.ATR_SL_MULT
            tp_distance = current_atr * self.ATR_TP_MULT

        # Set ATR-based SL/TP
        if direction == "long":
            stop_loss = fill_price - sl_distance
            take_profit = fill_price + tp_distance
        else:
            stop_loss = fill_price + sl_distance
            take_profit = fill_price - tp_distance

        self._positions[symbol] = {
            "direction": direction,
            "entry_price": fill_price,
            "size_usd": size_usd,
            "best_price": fill_price,
            "trailing_active": False,
            "atr_at_entry": current_atr,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_candle": self._candle_counts.get(symbol, 0),
        }

        # Set initial SL/TP on exchange
        if self.exchange:
            try:
                await self.exchange.set_stop_loss(symbol, round(stop_loss, 2))
                await self.exchange.set_take_profit(symbol, round(take_profit, 2))
                logger.info(
                    f"Position opened: {direction} {symbol} @ {fill_price:.2f} "
                    f"SL={stop_loss:.2f} TP={take_profit:.2f} ATR={current_atr:.2f}"
                )
            except Exception as e:
                logger.error(f"Failed to set SL/TP on exchange: {e}")

        return {"action": "position_tracked", "symbol": symbol}

    async def _on_market_data(self, message: Message) -> dict | None:
        """Check positions on each new candle."""
        symbol = message.payload.get("symbol", "")
        high = message.payload.get("high", 0.0)
        low = message.payload.get("low", 0.0)
        close = message.payload.get("close", 0.0)

        if not symbol or close <= 0:
            return None

        # Update OHLC history
        self._highs.setdefault(symbol, []).append(high)
        self._lows.setdefault(symbol, []).append(low)
        self._closes.setdefault(symbol, []).append(close)
        self._candle_counts[symbol] = self._candle_counts.get(symbol, 0) + 1

        # Trim history
        for hist in (self._highs[symbol], self._lows[symbol], self._closes[symbol]):
            if len(hist) > 200:
                del hist[:-200]

        # Check if we have a position in this symbol
        pos = self._positions.get(symbol)
        if not pos:
            return None

        return await self._manage_position(symbol, pos, close)

    async def _manage_position(self, symbol: str, pos: dict, current_price: float) -> dict | None:
        """Core position management — trailing stop, breakeven, time exit."""
        entry = pos["entry_price"]
        atr_at_entry = pos["atr_at_entry"]
        is_long = pos["direction"] == "long"

        # 1. Update best price
        if is_long:
            pos["best_price"] = max(pos["best_price"], current_price)
        else:
            pos["best_price"] = min(pos["best_price"], current_price)

        # 2. Check trailing stop activation
        profit_in_atr = ((current_price - entry) / atr_at_entry) if is_long else (
            (entry - current_price) / atr_at_entry
        )

        if profit_in_atr >= self.TRAIL_ACTIVATION_ATR and not pos["trailing_active"]:
            pos["trailing_active"] = True

            # Move stop-loss to breakeven + fees
            fee_buffer = entry * self.FEE_RATE * 2.5
            if is_long:
                breakeven_stop = entry + fee_buffer
                if pos["stop_loss"] < breakeven_stop:
                    pos["stop_loss"] = breakeven_stop
            else:
                breakeven_stop = entry - fee_buffer
                if pos["stop_loss"] > breakeven_stop:
                    pos["stop_loss"] = breakeven_stop

            # Update SL on exchange
            if self.exchange:
                try:
                    await self.exchange.set_stop_loss(symbol, round(pos["stop_loss"], 2))
                    logger.info(f"Breakeven stop set for {symbol} @ {pos['stop_loss']:.2f}")
                except Exception as e:
                    logger.error(f"Failed to update SL: {e}")

        # 3. Update trailing stop on exchange
        if pos["trailing_active"]:
            trail_dist = atr_at_entry * self.TRAIL_DISTANCE_ATR
            if is_long:
                new_trail = pos["best_price"] - trail_dist
                if new_trail > pos["stop_loss"]:
                    pos["stop_loss"] = new_trail
                    if self.exchange:
                        try:
                            await self.exchange.set_stop_loss(symbol, round(new_trail, 2))
                            logger.info(f"Trailing stop updated: {symbol} SL={new_trail:.2f}")
                        except Exception as e:
                            logger.error(f"Failed to update trailing SL: {e}")
            else:
                new_trail = pos["best_price"] + trail_dist
                if new_trail < pos["stop_loss"]:
                    pos["stop_loss"] = new_trail
                    if self.exchange:
                        try:
                            await self.exchange.set_stop_loss(symbol, round(new_trail, 2))
                            logger.info(f"Trailing stop updated: {symbol} SL={new_trail:.2f}")
                        except Exception as e:
                            logger.error(f"Failed to update trailing SL: {e}")

        # 4. Time-based exit
        candles_held = self._candle_counts.get(symbol, 0) - pos["entry_candle"]
        if candles_held >= self.MAX_HOLD_CANDLES:
            logger.info(f"Time exit: {symbol} held {candles_held} candles, closing at market")
            await self._close_position(symbol, pos)
            return {"action": "time_exit", "symbol": symbol, "candles_held": candles_held}

        return None

    async def _close_position(self, symbol: str, pos: dict):
        """Close position at market."""
        self._positions.pop(symbol, None)

        if not self.exchange:
            logger.warning(f"Time exit {symbol} — no exchange client (simulated)")
            return

        try:
            # Get current position from exchange to get exact size
            positions = await self.exchange.get_positions()
            for p in positions:
                if p["symbol"] == symbol and float(p.get("size", 0)) > 0:
                    close_side = "Sell" if pos["direction"] == "long" else "Buy"
                    await self.exchange.place_order(
                        symbol=symbol,
                        side=close_side,
                        order_type="Market",
                        qty=float(p["size"]),
                        reduce_only=True,
                    )
                    logger.info(f"Closed {symbol} via time exit")
                    return
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")

    def _compute_atr(self, symbol: str) -> float:
        """Compute current ATR for a symbol."""
        highs = self._highs.get(symbol, [])
        lows = self._lows.get(symbol, [])
        closes = self._closes.get(symbol, [])

        if len(highs) < self.ATR_PERIOD + 1:
            # Fallback: 1% of last close
            return closes[-1] * 0.01 if closes else 100.0

        atr_values = compute_atr(highs, lows, closes, self.ATR_PERIOD)
        return atr_values[-1] if atr_values else closes[-1] * 0.01
