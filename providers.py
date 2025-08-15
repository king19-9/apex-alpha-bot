# -*- coding: utf-8 -*-
# External providers: Dexscreener + Scan-based (Etherscan/Polygonscan/BscScan)
# این ماژول برای:
# 1) جستجوی جفت‌ها و فعالیت DEX از Dexscreener
# 2) خواندن تراکنش‌های بزرگ Native از اسکنرهای زنجیره (ETH/MATIC/BNB) در چند بلاک اخیر
# طراحی شده است. خروجی‌ها برای غنی‌سازی گزارش نهنگ‌ها و fallback آن‌چین استفاده می‌شوند.

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp

# Dexscreener API
DEX_SEARCH = "https://api.dexscreener.com/latest/dex/search"

# Scan APIs (proxy module)
SCAN_BASE = {
    "eth": "https://api.etherscan.io/api",
    "matic": "https://api.polygonscan.com/api",
    "bnb": "https://api.bscscan.com/api",
}

# Native coin decimals
DECIMALS = {"eth": 18, "matic": 18, "bnb": 18}


async def dexscreener_search_symbol(symbol: str, topn: int = 5, timeout: int = 6) -> List[Dict[str, Any]]:
    """
    جستجوی جفت‌های مرتبط با نماد در Dexscreener و بازگرداندن لیست مرتب‌شده بر اساس حجم/تراکنش‌ها.
    خروجی هر آیتم:
      {
        "chain": "ethereum|bsc|polygon|...",
        "dex": "uniswap|pancakeswap|...",
        "base": "TOKEN",
        "quote": "USDT/USDC/WETH/...",
        "price_usd": float,
        "vol24h": float,
        "tx_m5_buys": int,
        "tx_m5_sells": int,
        "pair_address": "...",
        "url": "https://dexscreener.com/..."
      }
    """
    q = (symbol or "").upper().strip()
    if not q:
        return []
    params = {"q": q}
    out: List[Dict[str, Any]] = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(DEX_SEARCH, params=params, timeout=timeout) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                pairs = data.get("pairs", []) if isinstance(data, dict) else []
                for p in pairs:
                    base = (p.get("baseToken") or {}).get("symbol")
                    quote = (p.get("quoteToken") or {}).get("symbol")
                    if not base or base.upper() != q:
                        continue
                    tx = p.get("txns") or {}
                    m5 = tx.get("m5") or {}
                    price_usd = p.get("priceUsd")
                    vol24 = p.get("volume") or p.get("volume24h") or p.get("fdv") or 0
                    out.append({
                        "chain": p.get("chainId"),
                        "dex": p.get("dexId"),
                        "base": base,
                        "quote": quote,
                        "price_usd": float(price_usd) if price_usd not in (None, "") else 0.0,
                        "vol24h": float(vol24) if vol24 not in (None, "") else 0.0,
                        "tx_m5_buys": int(m5.get("buys") or 0),
                        "tx_m5_sells": int(m5.get("sells") or 0),
                        "pair_address": p.get("pairAddress"),
                        "url": p.get("url"),
                    })
        # مرتب‌سازی: ابتدا حجم، سپس مجموع تعداد تراکنش‌های 5 دقیقه
        out = sorted(out, key=lambda x: (x["vol24h"], x["tx_m5_buys"] + x["tx_m5_sells"]), reverse=True)
        return out[:max(1, int(topn))]
    except Exception:
        return []


async def scan_get_tx_by_hash(network: str, tx_hash: str, api_key: Optional[str] = None, timeout: int = 10) -> Dict[str, Any]:
    """
    گرفتن جزئیات تراکنش با tx_hash از اسکنر (proxy.eth_getTransactionByHash).
    network: eth | matic | bnb
    """
    base = SCAN_BASE.get((network or "").lower())
    if not base or not tx_hash:
        return {}
    params = {"module": "proxy", "action": "eth_getTransactionByHash", "txhash": tx_hash}
    if api_key:
        params["apikey"] = api_key
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base, params=params, timeout=timeout) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()
                return data.get("result") or {}
    except Exception:
        return {}


async def _scan_get_latest_block(network: str, api_key: Optional[str], timeout: int = 10) -> Optional[int]:
    base = SCAN_BASE.get((network or "").lower())
    if not base:
        return None
    params = {"module": "proxy", "action": "eth_blockNumber"}
    if api_key:
        params["apikey"] = api_key
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base, params=params, timeout=timeout) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                hx = (data.get("result") or "0x0")
                return int(hx, 16)
    except Exception:
        return None


async def _scan_get_block(network: str, block_number: int, api_key: Optional[str], timeout: int = 10) -> Dict[str, Any]:
    base = SCAN_BASE.get((network or "").lower())
    if not base:
        return {}
    tag = hex(int(block_number))
    params = {"module": "proxy", "action": "eth_getBlockByNumber", "tag": tag, "boolean": "true"}
    if api_key:
        params["apikey"] = api_key
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base, params=params, timeout=timeout) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()
                return data.get("result") or {}
    except Exception:
        return {}


async def scan_fetch_recent_native_whales(
    network: str,
    min_usd: float,
    lookback_blocks: int,
    price_usd: float,
    api_keys: Dict[str, Optional[str]],
    timeout: int = 10
) -> List[Dict[str, Any]]:
    """
    اسکن N بلاک اخیر برای تراکنش‌های Native با ارزش USD >= min_usd.
    network: eth | matic | bnb
    price_usd: قیمت فعلی کوین Native برای محاسبه USD Notional
    خروجی هر آیتم: {network, hash, from, to, amount_native, amount_usd, timestamp, direction}
    """
    net = (network or "").lower()
    base = SCAN_BASE.get(net)
    if not base:
        return []
    latest = await _scan_get_latest_block(net, api_keys.get(net), timeout=timeout)
    if latest is None:
        return []
    decimals = DECIMALS.get(net, 18)
    out: List[Dict[str, Any]] = []

    start = max(0, latest - int(lookback_blocks or 1))
    # به‌صورت ترتیبی (برای احترام به محدودیت نرخ). می‌توان به صورت موازی هم کرد.
    for b in range(latest, start, -1):
        blk = await _scan_get_block(net, b, api_keys.get(net), timeout=timeout)
        if not blk:
            continue
        ts_hex = blk.get("timestamp") or "0x0"
        ts = int(ts_hex, 16)
        txs = blk.get("transactions") or []
        for t in txs:
            try:
                val_hex = t.get("value") or "0x0"
                val_native = int(val_hex, 16) / (10 ** decimals)
                usd = float(val_native) * float(price_usd or 0.0)
                if usd >= float(min_usd or 0.0):
                    out.append({
                        "network": net,
                        "hash": t.get("hash"),
                        "from": t.get("from"),
                        "to": t.get("to"),
                        "amount_native": float(val_native),
                        "amount_usd": float(usd),
                        "timestamp": ts,
                        "direction": "MOVE",
                    })
            except Exception:
                continue
        # تأخیر کوچک برای رعایت سهمیه‌ها
        await asyncio.sleep(0.15)

    out = sorted(out, key=lambda x: x["amount_usd"], reverse=True)
    return out