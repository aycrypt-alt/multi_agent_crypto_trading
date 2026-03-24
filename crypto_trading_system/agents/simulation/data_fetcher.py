"""
Historical Data Fetcher — Downloads real kline data from Bybit public API.

No API keys required — uses public market data endpoints.
Supports pagination to fetch months/years of historical data.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent.parent / "backtest_data"


class HistoricalDataFetcher:
    """Fetch historical kline data from Bybit V5 public API."""

    BASE_URL = "https://api.bybit.com"
    MAX_CANDLES_PER_REQUEST = 1000

    # Interval string -> milliseconds per candle
    INTERVAL_MS = {
        "1": 60_000, "3": 180_000, "5": 300_000, "15": 900_000,
        "30": 1_800_000, "60": 3_600_000, "120": 7_200_000,
        "240": 14_400_000, "360": 21_600_000, "720": 43_200_000,
        "D": 86_400_000, "W": 604_800_000,
    }

    def __init__(self):
        self._session = None

    async def _ensure_session(self):
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "15",
        num_candles: int = 5000,
        category: str = "linear",
    ) -> list[dict]:
        """
        Fetch historical kline data, paginating as needed.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT")
            interval: Candle interval (1, 5, 15, 60, 240, D, etc.)
            num_candles: Total number of candles to fetch
            category: "linear" for USDT perpetuals

        Returns:
            List of candle dicts sorted oldest-first:
            [{"timestamp", "open", "high", "low", "close", "volume"}, ...]
        """
        await self._ensure_session()

        all_candles = []
        remaining = num_candles
        end_time = int(time.time() * 1000)  # Start from now, go backwards
        interval_ms = self.INTERVAL_MS.get(interval, 900_000)

        logger.info(f"Fetching {num_candles} candles for {symbol} ({interval}m)...")

        while remaining > 0:
            batch_size = min(remaining, self.MAX_CANDLES_PER_REQUEST)
            params = {
                "category": category,
                "symbol": symbol,
                "interval": interval,
                "limit": batch_size,
                "end": end_time,
            }

            url = f"{self.BASE_URL}/v5/market/kline"
            try:
                async with self._session.get(url, params=params) as resp:
                    data = await resp.json()

                if data.get("retCode") != 0:
                    logger.error(f"Bybit API error: {data.get('retMsg')}")
                    break

                candles_raw = data.get("result", {}).get("list", [])
                if not candles_raw:
                    logger.info(f"No more data available for {symbol}")
                    break

                for c in candles_raw:
                    all_candles.append({
                        "timestamp": int(c[0]),
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[5]),
                    })

                # Move end_time to before the oldest candle in this batch
                oldest_ts = int(candles_raw[-1][0])
                end_time = oldest_ts - 1

                remaining -= len(candles_raw)
                logger.info(f"  Fetched {len(all_candles)}/{num_candles} candles...")

                # Rate limiting — be polite to the API
                await asyncio.sleep(0.15)

            except Exception as e:
                logger.error(f"Fetch error for {symbol}: {e}")
                # Retry with backoff
                await asyncio.sleep(2)
                continue

        # Sort oldest-first
        all_candles.sort(key=lambda c: c["timestamp"])

        # Remove duplicates by timestamp
        seen = set()
        unique = []
        for c in all_candles:
            if c["timestamp"] not in seen:
                seen.add(c["timestamp"])
                unique.append(c)

        logger.info(f"Fetched {len(unique)} unique candles for {symbol}")
        return unique

    async def fetch_and_cache(
        self,
        symbol: str,
        interval: str = "15",
        num_candles: int = 5000,
    ) -> list[dict]:
        """Fetch data and cache to disk. Returns cached data if available."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = DATA_DIR / f"{symbol}_{interval}m_{num_candles}.json"

        if cache_file.exists():
            logger.info(f"Loading cached data: {cache_file}")
            with open(cache_file) as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} cached candles for {symbol}")
            return data

        data = await self.fetch_klines(symbol, interval, num_candles)

        if data:
            with open(cache_file, "w") as f:
                json.dump(data, f)
            logger.info(f"Cached {len(data)} candles to {cache_file}")

        return data

    async def fetch_multi_symbol(
        self,
        symbols: list[str],
        interval: str = "15",
        num_candles: int = 5000,
    ) -> dict[str, list[dict]]:
        """Fetch historical data for multiple symbols concurrently."""
        results = {}
        # Fetch sequentially to avoid rate limits
        for symbol in symbols:
            results[symbol] = await self.fetch_and_cache(symbol, interval, num_candles)
        return results


def generate_synthetic_data(
    symbol: str = "BTCUSDT",
    num_candles: int = 5000,
    start_price: float = 50000.0,
    trend: str = "mixed",
) -> list[dict]:
    """
    Generate realistic synthetic price data for testing without API access.

    Args:
        trend: "bull", "bear", "ranging", or "mixed" (cycles through all)
    """
    import math
    import random

    price = start_price
    historical = []

    for i in range(num_candles):
        # Regime-dependent drift and volatility
        if trend == "bull":
            drift = 0.0003
            vol = 0.015
        elif trend == "bear":
            drift = -0.0002
            vol = 0.02
        elif trend == "ranging":
            drift = 0.0
            vol = 0.01
        else:  # mixed — cycle through regimes
            cycle = (i // 500) % 4
            if cycle == 0:
                drift, vol = 0.0004, 0.015  # Bull
            elif cycle == 1:
                drift, vol = 0.0, 0.01      # Range
            elif cycle == 2:
                drift, vol = -0.0003, 0.025  # Bear
            else:
                drift, vol = 0.0002, 0.02    # Recovery

        # Add volatility clustering (GARCH-like)
        vol *= (1 + 0.5 * math.sin(i / 100))

        # Occasional spikes
        if random.random() < 0.01:
            vol *= 3

        change = random.gauss(drift, vol)
        price *= (1 + change)
        price = max(price, 1.0)  # Floor at $1

        high = price * (1 + abs(random.gauss(0, 0.005)))
        low = price * (1 - abs(random.gauss(0, 0.005)))
        open_price = price * (1 + random.gauss(0, 0.001))
        volume = random.uniform(100, 1000) * (1 + abs(change) * 50)

        historical.append({
            "timestamp": i * 60000,  # 1-minute candles in ms
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(price, 2),
            "volume": round(volume, 2),
        })

    return historical
