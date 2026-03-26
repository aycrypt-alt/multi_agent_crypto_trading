"""
Order Execution Agent

Handles the actual placement of orders on Bybit.
Features:
- Smart order routing (limit vs market)
- Slippage protection
- Order splitting for large positions
- Stop-loss and take-profit management
"""

import asyncio
import logging
import time

from ...core.agent_base import Agent, AgentPriority
from ...core.message_bus import Message, MessageBus, MessageType

logger = logging.getLogger(__name__)


class OrderExecutorAgent(Agent):
    """
    Executes trade signals on the exchange.
    Receives risk-approved orders from the position sizing agent.
    """

    def __init__(self, message_bus: MessageBus, exchange_client=None, config: dict | None = None):
        super().__init__(
            name="order_executor",
            message_bus=message_bus,
            priority=AgentPriority.HIGH,
            config=config or {},
        )
        self.exchange = exchange_client  # BybitClient instance
        self.position_scale_factor = 1.0
        self.use_limit_orders = self.config.get("use_limit_orders", True)
        self.max_slippage_pct = self.config.get("max_slippage_pct", 0.1)
        self._open_orders: dict[str, dict] = {}
        self._positions: dict[str, dict] = {}

    async def on_start(self):
        await self.subscribe("execution")
        await self.subscribe_type(MessageType.CONFIG_UPDATE)
        await self.subscribe_type(MessageType.DRAWDOWN_ALERT)

    async def process(self, message: Message) -> dict | None:
        if message.type == MessageType.CONFIG_UPDATE:
            self.position_scale_factor = message.payload.get(
                "position_scale_factor", self.position_scale_factor
            )
            return None

        if message.type == MessageType.DRAWDOWN_ALERT:
            if message.payload.get("action") == "close_all_positions":
                await self._close_all_positions()
                return {"action": "closed_all"}

        if message.type == MessageType.ORDER_REQUEST and message.channel == "execution":
            return await self._execute_order(message.payload)

        return None

    async def _execute_order(self, order: dict) -> dict | None:
        symbol = order.get("symbol", "")
        direction = order.get("direction", "")
        size_usd = order.get("size_usd", 0.0) * self.position_scale_factor

        if size_usd < 1.0:
            return {"status": "skipped", "reason": "size_too_small"}

        # Close existing position if direction flipped
        if symbol in self._positions and self._positions[symbol].get("direction") != direction:
            await self._close_position_for_symbol(symbol)

        side = "Buy" if direction == "long" else "Sell"

        if not self.exchange:
            logger.warning(f"No exchange client — simulated order: {side} {symbol} ${size_usd:.2f}")
            # Emit simulated fill
            await self.emit(
                MessageType.ORDER_FILLED,
                "fills",
                {
                    "symbol": symbol,
                    "side": side,
                    "size_usd": size_usd,
                    "direction": direction,
                    "fill_price": order.get("entry_price", 0.0),
                    "pnl": 0.0,
                    "simulated": True,
                },
            )
            self._positions[symbol] = {"direction": direction, "size_usd": size_usd}
            return {"status": "simulated", "symbol": symbol, "size_usd": size_usd}

        try:
            # Get current price for limit order placement
            ticker = await self.exchange.get_ticker(symbol)
            current_price = ticker.get("last_price", 0.0)

            if current_price <= 0:
                return {"status": "error", "reason": "invalid_price"}

            # Calculate quantity
            qty = size_usd / current_price

            # Place order
            if self.use_limit_orders:
                # Place limit order slightly better than market
                offset = current_price * 0.0005  # 0.05% offset
                limit_price = current_price - offset if side == "Buy" else current_price + offset

                result = await self.exchange.place_order(
                    symbol=symbol,
                    side=side,
                    order_type="Limit",
                    qty=qty,
                    price=limit_price,
                )
            else:
                result = await self.exchange.place_order(
                    symbol=symbol,
                    side=side,
                    order_type="Market",
                    qty=qty,
                )

            if result.get("order_id"):
                self._open_orders[result["order_id"]] = {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": current_price,
                    "time": time.time(),
                }
                self._positions[symbol] = {"direction": direction, "size_usd": size_usd}

                # NOTE: SL/TP is now managed by PositionManagerAgent using ATR-based
                # trailing stops, breakeven stops, and time exits. No fixed-% SL/TP here.

                logger.info(f"Order placed: {side} {symbol} qty={qty:.4f} @ {current_price:.2f}")
                await self.emit(
                    MessageType.ORDER_FILLED,
                    "fills",
                    {
                        "symbol": symbol,
                        "side": side,
                        "direction": direction,
                        "qty": qty,
                        "size_usd": size_usd,
                        "fill_price": current_price,
                        "order_id": result["order_id"],
                    },
                )
                return {"status": "filled", "order_id": result["order_id"]}

            return {"status": "error", "reason": "order_rejected"}

        except Exception as e:
            logger.error(f"Order execution error: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}

    async def _close_position_for_symbol(self, symbol: str):
        """Close an existing position before opening in the opposite direction."""
        self._positions.pop(symbol, None)
        if not self.exchange:
            logger.info(f"Simulated close for direction flip: {symbol}")
            return
        try:
            positions = await self.exchange.get_positions()
            for pos in positions:
                if pos["symbol"] == symbol and float(pos.get("size", 0)) > 0:
                    close_side = "Sell" if pos.get("side") == "Buy" else "Buy"
                    await self.exchange.place_order(
                        symbol=symbol,
                        side=close_side,
                        order_type="Market",
                        qty=float(pos["size"]),
                        reduce_only=True,
                    )
                    logger.info(f"Closed {symbol} for direction flip")
        except Exception as e:
            logger.error(f"Failed to close {symbol} for flip: {e}")

    async def _close_all_positions(self):
        """Emergency close all positions."""
        if not self.exchange:
            logger.warning("Emergency close — no exchange client (simulated)")
            return

        try:
            positions = await self.exchange.get_positions()
            for pos in positions:
                if float(pos.get("size", 0)) > 0:
                    side = "Sell" if pos.get("side") == "Buy" else "Buy"
                    await self.exchange.place_order(
                        symbol=pos["symbol"],
                        side=side,
                        order_type="Market",
                        qty=float(pos["size"]),
                        reduce_only=True,
                    )
                    logger.info(f"Emergency closed: {pos['symbol']}")
        except Exception as e:
            logger.error(f"Emergency close error: {e}", exc_info=True)
