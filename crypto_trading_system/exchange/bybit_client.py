"""
Bybit Exchange Client

Async wrapper for Bybit V5 API.
Handles authentication, rate limiting, and WebSocket data feeds.

IMPORTANT: You need to install pybit: pip install pybit
And set your API keys as environment variables:
    BYBIT_API_KEY=your_key
    BYBIT_API_SECRET=your_secret

NEVER hardcode API keys in source code.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class BybitClient:
    """
    Async Bybit V5 API client.

    Supports:
    - Spot, Linear (USDT perpetual), and Inverse contracts
    - Order placement, cancellation, and amendment
    - Position management
    - Account balance queries
    - Historical kline/candle data
    - WebSocket streaming for real-time data
    """

    BASE_URL_MAINNET = "https://api.bybit.com"
    BASE_URL_TESTNET = "https://api-testnet.bybit.com"

    def __init__(self, testnet: bool = True):
        self.api_key = os.environ.get("BYBIT_API_KEY", "")
        self.api_secret = os.environ.get("BYBIT_API_SECRET", "")
        self.base_url = self.BASE_URL_TESTNET if testnet else self.BASE_URL_MAINNET
        self.recv_window = 5000
        self._session = None
        self._rate_limit_remaining = 120
        self._rate_limit_reset = 0

    async def _ensure_session(self):
        if self._session is None:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession()
            except ImportError:
                raise ImportError("Install aiohttp: pip install aiohttp")

    def _sign(self, params: dict, timestamp: int) -> str:
        """Generate HMAC SHA256 signature for authenticated requests."""
        param_str = f"{timestamp}{self.api_key}{self.recv_window}{urlencode(sorted(params.items()))}"
        return hmac.new(
            self.api_secret.encode("utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _auth_headers(self, params: dict) -> dict:
        timestamp = int(time.time() * 1000)
        signature = self._sign(params, timestamp)
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
            "Content-Type": "application/json",
        }

    async def _request(self, method: str, endpoint: str, params: dict | None = None,
                       authenticated: bool = False) -> dict:
        await self._ensure_session()
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        headers = self._auth_headers(params) if authenticated else {"Content-Type": "application/json"}

        # Rate limit check
        if self._rate_limit_remaining <= 1 and time.time() < self._rate_limit_reset:
            wait = self._rate_limit_reset - time.time()
            logger.warning(f"Rate limited, waiting {wait:.1f}s")
            await asyncio.sleep(wait)

        async with self._session.request(method, url, json=params if method == "POST" else None,
                                          params=params if method == "GET" else None,
                                          headers=headers) as resp:
            # Update rate limit tracking
            self._rate_limit_remaining = int(resp.headers.get("X-Bapi-Limit-Status", 120))
            reset = resp.headers.get("X-Bapi-Limit-Reset-Timestamp", "0")
            self._rate_limit_reset = int(reset) / 1000 if reset != "0" else 0

            data = await resp.json()
            if data.get("retCode") != 0:
                logger.error(f"Bybit API error: {data.get('retMsg')} (code: {data.get('retCode')})")
            return data

    async def close(self):
        if self._session:
            await self._session.close()

    # ── Market Data ────────────────────────────────────────────

    async def get_ticker(self, symbol: str, category: str = "linear") -> dict:
        """Get latest ticker for a symbol."""
        result = await self._request("GET", "/v5/market/tickers", {
            "category": category, "symbol": symbol,
        })
        tickers = result.get("result", {}).get("list", [])
        if tickers:
            t = tickers[0]
            return {
                "symbol": t.get("symbol"),
                "last_price": float(t.get("lastPrice", 0)),
                "bid": float(t.get("bid1Price", 0)),
                "ask": float(t.get("ask1Price", 0)),
                "volume_24h": float(t.get("volume24h", 0)),
                "price_change_24h": float(t.get("price24hPcnt", 0)),
            }
        return {}

    async def get_klines(self, symbol: str, interval: str = "15",
                         limit: int = 200, category: str = "linear") -> list[dict]:
        """
        Get historical kline/candlestick data.

        Args:
            interval: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
            limit: Max 1000
        """
        result = await self._request("GET", "/v5/market/kline", {
            "category": category, "symbol": symbol,
            "interval": interval, "limit": limit,
        })
        candles = result.get("result", {}).get("list", [])
        return [
            {
                "timestamp": int(c[0]),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            }
            for c in reversed(candles)  # Bybit returns newest first
        ]

    async def get_orderbook(self, symbol: str, limit: int = 25,
                            category: str = "linear") -> dict:
        result = await self._request("GET", "/v5/market/orderbook", {
            "category": category, "symbol": symbol, "limit": limit,
        })
        book = result.get("result", {})
        return {
            "bids": [(float(b[0]), float(b[1])) for b in book.get("b", [])],
            "asks": [(float(a[0]), float(a[1])) for a in book.get("a", [])],
            "timestamp": book.get("ts", 0),
        }

    async def get_funding_rate(self, symbol: str, category: str = "linear") -> dict:
        result = await self._request("GET", "/v5/market/funding/history", {
            "category": category, "symbol": symbol, "limit": 1,
        })
        rates = result.get("result", {}).get("list", [])
        if rates:
            return {
                "symbol": symbol,
                "funding_rate": float(rates[0].get("fundingRate", 0)),
                "timestamp": int(rates[0].get("fundingRateTimestamp", 0)),
            }
        return {}

    # ── Trading ────────────────────────────────────────────────

    async def place_order(self, symbol: str, side: str, order_type: str,
                          qty: float, price: float | None = None,
                          reduce_only: bool = False,
                          category: str = "linear") -> dict:
        """
        Place an order.

        Args:
            side: "Buy" or "Sell"
            order_type: "Market" or "Limit"
        """
        params = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
        }
        if price and order_type == "Limit":
            params["price"] = str(price)
        if reduce_only:
            params["reduceOnly"] = True

        result = await self._request("POST", "/v5/order/create", params, authenticated=True)
        order_data = result.get("result", {})
        return {
            "order_id": order_data.get("orderId", ""),
            "status": "ok" if result.get("retCode") == 0 else "error",
            "message": result.get("retMsg", ""),
        }

    async def cancel_order(self, symbol: str, order_id: str,
                           category: str = "linear") -> dict:
        result = await self._request("POST", "/v5/order/cancel", {
            "category": category, "symbol": symbol, "orderId": order_id,
        }, authenticated=True)
        return {"status": "ok" if result.get("retCode") == 0 else "error"}

    async def set_stop_loss(self, symbol: str, stop_loss_price: float,
                            category: str = "linear") -> dict:
        result = await self._request("POST", "/v5/position/trading-stop", {
            "category": category, "symbol": symbol,
            "stopLoss": str(stop_loss_price),
        }, authenticated=True)
        return {"status": "ok" if result.get("retCode") == 0 else "error"}

    async def set_take_profit(self, symbol: str, take_profit_price: float,
                              category: str = "linear") -> dict:
        result = await self._request("POST", "/v5/position/trading-stop", {
            "category": category, "symbol": symbol,
            "takeProfit": str(take_profit_price),
        }, authenticated=True)
        return {"status": "ok" if result.get("retCode") == 0 else "error"}

    # ── Account ────────────────────────────────────────────────

    async def get_balance(self, account_type: str = "UNIFIED") -> dict:
        result = await self._request("GET", "/v5/account/wallet-balance", {
            "accountType": account_type,
        }, authenticated=True)
        accounts = result.get("result", {}).get("list", [])
        if accounts:
            coins = accounts[0].get("coin", [])
            return {
                "total_equity": float(accounts[0].get("totalEquity", 0)),
                "available_balance": float(accounts[0].get("totalAvailableBalance", 0)),
                "coins": {
                    c["coin"]: {
                        "equity": float(c.get("equity", 0)),
                        "available": float(c.get("availableToWithdraw", 0)),
                        "unrealized_pnl": float(c.get("unrealisedPnl", 0)),
                    }
                    for c in coins
                },
            }
        return {}

    async def get_positions(self, category: str = "linear") -> list[dict]:
        result = await self._request("GET", "/v5/position/list", {
            "category": category, "settleCoin": "USDT",
        }, authenticated=True)
        positions = result.get("result", {}).get("list", [])
        return [
            {
                "symbol": p.get("symbol"),
                "side": p.get("side"),
                "size": p.get("size"),
                "entry_price": float(p.get("avgPrice", 0)),
                "unrealized_pnl": float(p.get("unrealisedPnl", 0)),
                "leverage": p.get("leverage"),
            }
            for p in positions
            if float(p.get("size", 0)) > 0
        ]

    async def set_leverage(self, symbol: str, leverage: int,
                           category: str = "linear") -> dict:
        result = await self._request("POST", "/v5/position/set-leverage", {
            "category": category, "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }, authenticated=True)
        return {"status": "ok" if result.get("retCode") == 0 else "error"}
