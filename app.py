# -*- coding: utf-8 -*-
# Advanced 24/7 Crypto AI Bot: Telegram + Autoscan + Whales + Research Engine + Deep Backtest + Self-improving
# - PostgreSQL/SQLite (SQLAlchemy)
# - WebSocket (Coinbase) + HTTP Dashboard
# - Multi-timeframe analysis, advanced TA, Elliott, market structure, sessions
# - NLP sentiment (VADER; transformers fallback)
# - Quant ML (RF calibration + stacking) + threshold opt + persistence/learning
# - 24/7 autoscan of universe + deep backtests + strategy optimizer + whales monitor
# - Risk mgmt + leverage suggestion + smart entry/exit + reports
# - Cool Telegram landing + simplified commands (aliases) + menu (inline keyboard)

import os, sys, time, asyncio, statistics, random, logging, json, datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import web
import pandas as pd
import numpy as np

# Optional transformers (fallback to VADER if unavailable)
_USE_TRANSFORMERS = False
try:
    from transformers import pipeline
    _USE_TRANSFORMERS = True
except Exception:
    _USE_TRANSFORMERS = False

# ML
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump, load

# Sentiment fallback
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Exchanges
import ccxt.async_support as ccxt

# WebSocket
import websockets
import json as _json

# Telegram
TELEGRAM_AVAILABLE = False
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
    from telegram.constants import ChatAction
    TELEGRAM_AVAILABLE = True
except Exception:
    TELEGRAM_AVAILABLE = False
    ChatAction = None

# SQLAlchemy (Postgres/SQLite)
from sqlalchemy import create_engine, text

# ---------------- Settings ----------------
def getenv_bool(k: str, default: bool) -> bool:
    v = os.getenv(k)
    if v is None: return default
    return str(v).lower() in ["1","true","yes","y","on"]

def getenv_list(k: str, default: str) -> List[str]:
    return [x.strip() for x in os.getenv(k, default).split(",") if x.strip()]

@dataclass
class Settings:
    # APIs
    COINGECKO_API_KEY: Optional[str] = os.getenv("COINGECKO_API_KEY")
    COINMARKETCAP_API_KEY: Optional[str] = os.getenv("COINMARKETCAP_API_KEY")
    CRYPTOCOMPARE_API_KEY: Optional[str] = os.getenv("CRYPTOCOMPARE_API_KEY")
    CRYPTOPANIC_API_KEY: Optional[str] = os.getenv("CRYPTOPANIC_API_KEY")
    NEWS_API_KEY: Optional[str] = os.getenv("NEWS_API_KEY")

    # Telegram
    TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")

    # Bot runtime
    OFFLINE_MODE: bool = getenv_bool("OFFLINE_MODE", False)
    EXCHANGES: List[str] = field(default_factory=lambda: getenv_list("EXCHANGES", "kraken,kucoin,bybit,bitfinex,coinbase"))
    UNIVERSE_SCOPE: str = os.getenv("UNIVERSE_SCOPE","top200")
    MAX_COINS: int = int(os.getenv("MAX_COINS","200"))
    TIMEFRAMES: List[str] = field(default_factory=lambda: getenv_list("TIMEFRAMES","1h,4h,1d"))
    MODEL_MAX_TRAIN_BARS: int = int(os.getenv("MODEL_MAX_TRAIN_BARS","1500"))
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS","300"))
    CONCURRENT_REQUESTS: int = int(os.getenv("CONCURRENT_REQUESTS","10"))
    BALANCE: float = float(os.getenv("BALANCE","10000"))
    RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE","0.01"))
    ANALYSIS_DEPTH: str = os.getenv("ANALYSIS_DEPTH","deep")
    ENABLE_ADVANCED_TECHNICALS: bool = getenv_bool("ENABLE_ADVANCED_TECHNICALS", True)
    ENABLE_BACKTEST: bool = getenv_bool("ENABLE_BACKTEST", True)

    # Autoschedule / Reports
    ENABLE_AUTOSIGNALS: bool = getenv_bool("ENABLE_AUTOSIGNALS", False)
    AUTO_SIGNAL_INTERVAL_MINUTES: int = int(os.getenv("AUTO_SIGNAL_INTERVAL_MINUTES","60"))
    AUTO_SIGNAL_TOPN: int = int(os.getenv("AUTO_SIGNAL_TOPN","10"))
    ENABLE_DAILY_REPORT: bool = getenv_bool("ENABLE_DAILY_REPORT", False)
    DAILY_REPORT_UTC_HOUR: int = int(os.getenv("DAILY_REPORT_UTC_HOUR","0"))
    PERFORMANCE_HORIZON_HOURS: int = int(os.getenv("PERFORMANCE_HORIZON_HOURS","24"))

    # Realtime Monitor
    ENABLE_MONITOR: bool = getenv_bool("ENABLE_MONITOR", True)
    MONITOR_INTERVAL_SEC: int = int(os.getenv("MONITOR_INTERVAL_SEC","90"))
    MONITOR_TOPN: int = int(os.getenv("MONITOR_TOPN","100"))
    ALERT_COOLDOWN_MIN: int = int(os.getenv("ALERT_COOLDOWN_MIN","30"))

    # Derivatives (Funding/OI)
    FUNDING_ENABLED: bool = getenv_bool("FUNDING_ENABLED", True)

    # DB (Postgres/SQLite)
    POSTGRES_DSN: Optional[str] = os.getenv("POSTGRES_DSN")
    SQLITE_PATH: str = os.getenv("SQLITE_PATH","bot.db")

    # HTTP Dashboard
    ENABLE_HTTP: bool = getenv_bool("ENABLE_HTTP", True)
    PORT: int = int(os.getenv("PORT", "8080"))

    # CLI
    SYMBOL: str = os.getenv("SYMBOL","BTC")
    PRINT_SIGNALS: bool = getenv_bool("PRINT_SIGNALS", False)

    # Fast response + warmup
    FAST_RESPONSE: bool = getenv_bool("FAST_RESPONSE", True)
    FAST_FETCH_OHLCV_LIMIT: int = int(os.getenv("FAST_FETCH_OHLCV_LIMIT","500"))
    FAST_WHALES_TIMEOUT_SEC: int = int(os.getenv("FAST_WHALES_TIMEOUT_SEC","5"))
    FAST_ORDERBOOK_TIMEOUT_SEC: int = int(os.getenv("FAST_ORDERBOOK_TIMEOUT_SEC","5"))
    FAST_DERIV_TIMEOUT_SEC: int = int(os.getenv("FAST_DERIV_TIMEOUT_SEC","4"))
    MODEL_WARMUP_ON_START: bool = getenv_bool("MODEL_WARMUP_ON_START", True)
    MODEL_WARMUP_TOPN: int = int(os.getenv("MODEL_WARMUP_TOPN","30"))

    # Deep research / autoscan 24/7
    ENABLE_AUTOSCAN: bool = getenv_bool("ENABLE_AUTOSCAN", True)
    AUTOSCAN_INTERVAL_MINUTES: int = int(os.getenv("AUTOSCAN_INTERVAL_MINUTES","45"))
    RESEARCH_TOPN: int = int(os.getenv("RESEARCH_TOPN","80"))
    DEEP_HISTORY: bool = getenv_bool("DEEP_HISTORY", True)
    DEEP_HISTORY_LIMIT_1D: int = int(os.getenv("DEEP_HISTORY_LIMIT_1D","1500"))
    DEEP_HISTORY_LIMIT_4H: int = int(os.getenv("DEEP_HISTORY_LIMIT_4H","2000"))
    DEEP_HISTORY_LIMIT_1H: int = int(os.getenv("DEEP_HISTORY_LIMIT_1H","4000"))

    # Whales 24/7
    WHALE_WATCH_TOPN: int = int(os.getenv("WHALE_WATCH_TOPN","120"))
    WHALE_JOB_INTERVAL_SEC: int = int(os.getenv("WHALE_JOB_INTERVAL_SEC","180"))
    WHALE_ALERT_NOTIONAL: float = float(os.getenv("WHALE_ALERT_NOTIONAL","1000000"))
    WHALE_ALERT_BIAS: float = float(os.getenv("WHALE_ALERT_BIAS","0.35"))

    # News digest
    NEWS_DIGEST_INTERVAL_MINUTES: int = int(os.getenv("NEWS_DIGEST_INTERVAL_MINUTES","240"))

S = Settings()

# ---------------- Logger ----------------
logger = logging.getLogger("app")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

# ---------------- SQLAlchemy DB ----------------
def _dsn():
    if S.POSTGRES_DSN:
        return S.POSTGRES_DSN
    return f"sqlite:///{S.SQLITE_PATH}"

engine = create_engine(_dsn(), pool_pre_ping=True, future=True)

def db_init():
    backend = engine.url.get_backend_name()
    pk = "INTEGER PRIMARY KEY AUTOINCREMENT" if backend.startswith("sqlite") else "BIGSERIAL PRIMARY KEY"
    with engine.begin() as conn:
        conn.execute(text("""CREATE TABLE IF NOT EXISTS subscribers (chat_id BIGINT PRIMARY KEY)"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS watchlist (symbol TEXT PRIMARY KEY)"""))
        conn.execute(text(f"""CREATE TABLE IF NOT EXISTS signals (
            id {pk},
            ts BIGINT, symbol TEXT, signal TEXT, confidence DOUBLE PRECISION, price DOUBLE PRECISION,
            source TEXT, evaluated INT DEFAULT 0, ret DOUBLE PRECISION
        )"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS alerts (
            symbol TEXT PRIMARY KEY, last_alert_ts BIGINT
        )"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS thresholds (
            symbol TEXT, timeframe TEXT, buy DOUBLE PRECISION, sell DOUBLE PRECISION,
            PRIMARY KEY(symbol, timeframe)
        )"""))
        conn.execute(text(f"""CREATE TABLE IF NOT EXISTS whale_events (
            id {pk},
            ts BIGINT, exchange TEXT, symbol TEXT, bias DOUBLE PRECISION, notional DOUBLE PRECISION, evaluated INT DEFAULT 0, ret DOUBLE PRECISION
        )"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS whale_perf (
            exchange TEXT, symbol TEXT, count BIGINT, hit BIGINT, avg_ret DOUBLE PRECISION,
            PRIMARY KEY(exchange, symbol)
        )"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS configs (key TEXT PRIMARY KEY, value TEXT)"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS model_registry (
            symbol TEXT, timeframe TEXT, model TEXT, acc DOUBLE PRECISION, updated_ts BIGINT,
            PRIMARY KEY(symbol, timeframe, model)
        )"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS strat_perf (
            symbol TEXT, timeframe TEXT, strategy TEXT,
            hit DOUBLE PRECISION, avg_ret DOUBLE PRECISION, sharpe DOUBLE PRECISION, pf DOUBLE PRECISION, updated_ts BIGINT,
            PRIMARY KEY(symbol, timeframe, strategy)
        )"""))
        conn.execute(text(f"""CREATE TABLE IF NOT EXISTS research_opportunities (
            id {pk},
            ts BIGINT, symbol TEXT, timeframe TEXT, signal TEXT,
            prob DOUBLE PRECISION, rr DOUBLE PRECISION,
            entry DOUBLE PRECISION, sl DOUBLE PRECISION, tp1 DOUBLE PRECISION, tp2 DOUBLE PRECISION,
            leverage DOUBLE PRECISION, notes TEXT
        )"""))

db_init()

# DB helper functions
def db_add_subscriber(chat_id: int):
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO subscribers(chat_id) VALUES(:cid) ON CONFLICT (chat_id) DO NOTHING"""),
                     {"cid": chat_id})

def db_remove_subscriber(chat_id: int):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM subscribers WHERE chat_id=:cid"), {"cid": chat_id})

def db_get_subscribers() -> List[int]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT chat_id FROM subscribers")).mappings().all()
    return [r["chat_id"] for r in rows]

def db_add_watch(symbol: str):
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO watchlist(symbol) VALUES(:s) ON CONFLICT (symbol) DO NOTHING"""), {"s": symbol.upper()})

def db_remove_watch(symbol: str):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM watchlist WHERE symbol=:s"), {"s": symbol.upper()})

def db_get_watchlist() -> List[str]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT symbol FROM watchlist")).mappings().all()
    return [r["symbol"] for r in rows]

def db_log_signal(ts: int, symbol: str, signal: str, confidence: float, price: Optional[float], source: str):
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO signals(ts,symbol,signal,confidence,price,source) VALUES(:ts,:s,:sig,:c,:p,:src)"""),
                     {"ts": ts, "s": symbol.upper(), "sig": signal, "c": confidence, "p": price, "src": source})

def db_get_pending_signals(cutoff_ts: int):
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT * FROM signals WHERE evaluated=0 AND ts<=:t"), {"t": cutoff_ts}).mappings().all()
    return rows

def db_set_signal_evaluated(id_: int, ret: float):
    with engine.begin() as conn:
        conn.execute(text("UPDATE signals SET evaluated=1, ret=:r WHERE id=:i"), {"r": ret, "i": id_})

def db_get_last_alert(symbol: str) -> Optional[int]:
    with engine.begin() as conn:
        row = conn.execute(text("SELECT last_alert_ts FROM alerts WHERE symbol=:s"), {"s": symbol.upper()}).mappings().first()
    return row["last_alert_ts"] if row else None

def db_set_last_alert(symbol: str, ts: int):
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO alerts(symbol,last_alert_ts) VALUES(:s,:t)
                          ON CONFLICT (symbol) DO UPDATE SET last_alert_ts=excluded.last_alert_ts"""),
                     {"s": symbol.upper(), "t": ts})

def db_upsert_thresholds(symbol: str, timeframe: str, buy: float, sell: float):
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO thresholds(symbol,timeframe,buy,sell) VALUES(:s,:tf,:b,:sl)
                          ON CONFLICT (symbol,timeframe) DO UPDATE SET buy=excluded.buy, sell=excluded.sell"""),
                     {"s": symbol.upper(), "tf": timeframe, "b": buy, "sl": sell})

def db_get_thresholds(symbol: str, timeframe: str) -> Tuple[Optional[float], Optional[float]]:
    with engine.begin() as conn:
        row = conn.execute(text("SELECT buy,sell FROM thresholds WHERE symbol=:s AND timeframe=:tf"),
                           {"s": symbol.upper(), "tf": timeframe}).mappings().first()
    return (row["buy"], row["sell"]) if row else (None, None)

def db_log_whale_event(ts: int, exchange: str, symbol: str, bias: float, notional: float):
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO whale_events(ts,exchange,symbol,bias,notional) VALUES(:t,:ex,:s,:b,:n)"""),
                     {"t": ts, "ex": exchange, "s": symbol.upper(), "b": bias, "n": notional})

def db_get_pending_whale_events(cutoff_ts: int):
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT * FROM whale_events WHERE evaluated=0 AND ts<=:t"), {"t": cutoff_ts}).mappings().all()
    return rows

def db_set_whale_event_evaluated(id_: int, ret: float):
    with engine.begin() as conn:
        conn.execute(text("UPDATE whale_events SET evaluated=1, ret=:r WHERE id=:i"), {"r": ret, "i": id_})

def db_update_whale_perf(exchange: str, symbol: str, ret: float):
    with engine.begin() as conn:
        row = conn.execute(text("SELECT count,hit,avg_ret FROM whale_perf WHERE exchange=:ex AND symbol=:s"),
                           {"ex": exchange, "s": symbol.upper()}).mappings().first()
        if not row:
            conn.execute(text("INSERT INTO whale_perf(exchange,symbol,count,hit,avg_ret) VALUES(:ex,:s,1,:h,:a)"),
                         {"ex": exchange, "s": symbol.upper(), "h": 1 if ret>0 else 0, "a": ret})
        else:
            count = row["count"] + 1
            hit = row["hit"] + (1 if ret>0 else 0)
            avg = (row["avg_ret"]*row["count"] + ret)/count
            conn.execute(text("UPDATE whale_perf SET count=:c, hit=:h, avg_ret=:a WHERE exchange=:ex AND symbol=:s"),
                         {"c": count, "h": hit, "a": avg, "ex": exchange, "s": symbol.upper()})

def db_get_best_whales(limit: int = 10):
    with engine.begin() as conn:
        rows = conn.execute(text("""SELECT exchange, symbol, count, hit*1.0/count AS hit_rate, avg_ret
                                    FROM whale_perf WHERE count>=5
                                    ORDER BY hit_rate DESC, avg_ret DESC LIMIT :l"""), {"l": limit}).mappings().all()
    return rows

def db_recent_signals(limit: int = 50):
    with engine.begin() as conn:
        rows = conn.execute(text("""SELECT ts,symbol,signal,confidence,price,source,evaluated,ret
                                    FROM signals ORDER BY ts DESC LIMIT :l"""), {"l": limit}).mappings().all()
    return rows

def db_upsert_model_registry(symbol: str, timeframe: str, model: str, acc: float):
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO model_registry(symbol,timeframe,model,acc,updated_ts)
                             VALUES(:s,:tf,:m,:a,:t)
                             ON CONFLICT(symbol,timeframe,model) DO UPDATE SET acc=excluded.acc, updated_ts=excluded.updated_ts"""),
                     {"s": symbol.upper(), "tf": timeframe, "m": model, "a": float(acc), "t": int(time.time())})

def db_upsert_strat_perf(symbol: str, timeframe: str, strategy: str, metrics: Dict[str, float]):
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO strat_perf(symbol,timeframe,strategy,hit,avg_ret,sharpe,pf,updated_ts)
                             VALUES(:s,:tf,:st,:h,:avg,:sh,:pf,:t)
                             ON CONFLICT(symbol,timeframe,strategy) DO UPDATE SET
                             hit=excluded.hit, avg_ret=excluded.avg_ret, sharpe=excluded.sharpe, pf=excluded.pf, updated_ts=excluded.updated_ts"""),
                     {"s": symbol.upper(), "tf": timeframe, "st": strategy,
                      "h": float(metrics.get("hit",0.5)), "avg": float(metrics.get("avg",0.0)),
                      "sh": float(metrics.get("sharpe",0.0)), "pf": float(metrics.get("pf",1.0)),
                      "t": int(time.time())})

def db_insert_research_opp(ts: int, symbol: str, timeframe: str, signal: str, prob: float, rr: float,
                           entry: float, sl: float, tp1: float, tp2: float, leverage: float, notes: str=""):
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO research_opportunities(ts,symbol,timeframe,signal,prob,rr,entry,sl,tp1,tp2,leverage,notes)
                             VALUES(:ts,:s,:tf,:sig,:p,:rr,:e,:sl,:tp1,:tp2,:lev,:n)"""),
                     {"ts": ts, "s": symbol.upper(), "tf": timeframe, "sig": signal, "p": float(prob),
                      "rr": float(rr), "e": float(entry), "sl": float(sl), "tp1": float(tp1),
                      "tp2": float(tp2), "lev": float(leverage), "n": notes})

def db_recent_opportunities(limit: int = 20):
    with engine.begin() as conn:
        rows = conn.execute(text("""SELECT * FROM research_opportunities ORDER BY ts DESC LIMIT :l"""),
                            {"l": limit}).mappings().all()
    return rows

# ---------------- Cache ----------------
class TTLCache:
    def __init__(self, ttl_seconds: int = S.CACHE_TTL_SECONDS):
        self.ttl = ttl_seconds
        self.store: Dict[str, Tuple[float, Any]] = {}
    def get(self, key: str):
        now = time.time()
        v = self.store.get(key)
        if not v: return None
        ts, data = v
        if now - ts > self.ttl:
            self.store.pop(key, None)
            return None
        return data
    def set(self, key: str, value: Any):
        self.store[key] = (time.time(), value)

cache = TTLCache()

# ---------------- Time Utils / Sessions ----------------
ASIA_START, ASIA_END = 0, 8
EU_START, EU_END = 8, 16
US_START, US_END = 13, 21

def get_session_label_from_ts_ms(ts_ms: int) -> str:
    dt = pd.to_datetime(ts_ms, unit="ms", utc=True)
    hour = dt.hour
    if ASIA_START <= hour < ASIA_END: return "Asia"
    if EU_START <= hour < EU_END: return "Europe"
    if US_START <= hour < US_END: return "US"
    return "Off"

# ---------------- Data Sources (HTTP) ----------------
CG_BASE = "https://api.coingecko.com/api/v3"
async def cg_get_json(session: aiohttp.ClientSession, url: str, params=None):
    headers={}
    if S.COINGECKO_API_KEY:
        headers["x_cg_pro_api_key"] = S.COINGECKO_API_KEY
    async with session.get(url, params=params, headers=headers, timeout=30) as resp:
        resp.raise_for_status()
        return await resp.json()

async def cg_fetch_markets(vs_currency="usd", per_page=250, page=1) -> List[Dict[str,Any]]:
    async with aiohttp.ClientSession() as session:
        url=f"{CG_BASE}/coins/markets"
        params={"vs_currency":vs_currency,"order":"market_cap_desc","per_page":per_page,"page":page,"price_change_percentage":"24h"}
        return await cg_get_json(session,url,params)

async def cg_fetch_market_by_symbol(symbol: str) -> Dict[str,Any]:
    key=f"cg_market_{symbol.lower()}"
    c=cache.get(key)
    if c: return c
    try:
        pages = await asyncio.gather(cg_fetch_markets(page=1), cg_fetch_markets(page=2), cg_fetch_markets(page=3), cg_fetch_markets(page=4))
        flat=[]
        for p in pages:
            if isinstance(p,list): flat+=p
        cands=[x for x in flat if x.get("symbol","").lower()==symbol.lower()]
        if not cands:
            cands=[x for x in flat if x.get("name","").lower()==symbol.lower()]
        if not cands: return {}
        best=sorted(cands,key=lambda x:x.get("market_cap",0), reverse=True)[0]
        data={
            "symbol":best.get("symbol","").upper(),
            "id":best.get("id"),
            "name":best.get("name"),
            "price":best.get("current_price"),
            "price_change_24h":best.get("price_change_percentage_24h"),
            "volume_24h":best.get("total_volume"),
            "market_cap":best.get("market_cap"),
            "sources":["CoinGecko"]
        }
        cache.set(key,data); return data
    except Exception as e:
        logger.error(f"CG market symbol error {symbol}: {e}")
        return {}

CMC_BASE="https://pro-api.coinmarketcap.com/v1"
async def cmc_fetch_quote(symbol: str) -> Dict[str,Any]:
    if not S.COINMARKETCAP_API_KEY: return {}
    try:
        headers={"X-CMC_PRO_API_KEY":S.COINMARKETCAP_API_KEY}
        async with aiohttp.ClientSession(headers=headers) as session:
            params={"symbol":symbol.upper(),"convert":"USD"}
            async with session.get(f"{CMC_BASE}/cryptocurrency/quotes/latest",params=params,timeout=30) as resp:
                resp.raise_for_status()
                data=await resp.json()
                q=data.get("data",{}).get(symbol.upper(),{})
                usd=q.get("quote",{}).get("USD",{})
                if not usd: return {}
                return {"symbol":symbol.upper(),"price":usd.get("price"),"price_change_24h":usd.get("percent_change_24h"),"volume_24h":usd.get("volume_24h"),"market_cap":usd.get("market_cap"),"sources":["CoinMarketCap"]}
    except Exception as e:
        logger.error(f"CMC quote error {symbol}: {e}")
        return {}

CC_BASE="https://min-api.cryptocompare.com/data/v2"
def cc_tf_to_path(tf: str)->str:
    return {"1d":"histoday","4h":"histohour","1h":"histohour","15m":"histominute"}.get(tf,"histohour")
def cc_agg(tf: str)->int:
    return {"4h":4}.get(tf,1)

async def cc_fetch_ohlcv(symbol: str, quote="USD", timeframe="1h", limit=1500, to_ts: Optional[int]=None) -> List[Dict[str,Any]]:
    try:
        headers={}
        if S.CRYPTOCOMPARE_API_KEY:
            headers["authorization"]=f"Apikey {S.CRYPTOCOMPARE_API_KEY}"
        async with aiohttp.ClientSession(headers=headers) as session:
            params={"fsym":symbol.upper(),"tsym":quote.upper(),"limit":limit,"aggregate":cc_agg(timeframe)}
            if to_ts is not None: params["toTs"]=to_ts
            async with session.get(f"{CC_BASE}/{cc_tf_to_path(timeframe)}",params=params,timeout=30) as resp:
                resp.raise_for_status()
                data=await resp.json()
                if data.get("Response")!="Success": return []
                rows=data.get("Data",{}).get("Data",[])
                out=[]
                for r in rows:
                    out.append({"time":r.get("time")*1000,"open":r.get("open"),"high":r.get("high"),"low":r.get("low"),"close":r.get("close"),"volume":r.get("volumefrom")})
                return out
    except Exception as e:
        logger.error(f"CC OHLCV error {symbol} {timeframe}: {e}")
        return []

async def cc_fetch_ohlcv_extended(symbol: str, quote="USD", timeframe="1d", total_limit=1500) -> List[Dict[str,Any]]:
    per_call = min(2000, total_limit)
    out: List[Dict[str,Any]]=[]
    to_ts = None
    while len(out) < total_limit:
        chunk = await cc_fetch_ohlcv(symbol, quote, timeframe, limit=per_call, to_ts=to_ts)
        if not chunk: break
        if out and chunk and chunk[-1]["time"] == out[0]["time"]:
            chunk = chunk[:-1]
        out = chunk + out
        if len(chunk)==0: break
        oldest = chunk[0]["time"]//1000 - 60
        to_ts = oldest
        if len(out)>=total_limit: break
        await asyncio.sleep(0.2)
    return out[-total_limit:]

# Exchanges: price, whales, orderbook walls, derivatives
async def ex_fetch_ticker(symbol: str) -> Dict[str,Any]:
    pairs=[f"{symbol}/USDT", f"{symbol}/USDC", f"{symbol}/USD"]
    prices=[]; vols=[]; sources=[]
    async def run_ex(ex_id):
        try:
            ex=getattr(ccxt, ex_id)()
            ex.enableRateLimit = True
            ex.timeout = 10000
            for p in pairs:
                try:
                    t=await ex.fetch_ticker(p)
                    if t and t.get("last"):
                        prices.append(t["last"]); vols.append(t.get("baseVolume") or 0); sources.append(f"{ex_id}:{p}"); break
                except Exception:
                    continue
            await ex.close()
        except Exception:
            pass
    await asyncio.gather(*[run_ex(e) for e in S.EXCHANGES])
    if not prices: return {}
    return {"symbol":symbol.upper(),"price":sum(prices)/len(prices),"volume_24h":sum([v for v in vols if v]),"sources":[f"EX:{s}" for s in sources]}

async def ex_fetch_trades_whales(symbol: str, lookback_trades: int = 500, usd_threshold: float = 150000.0) -> Dict[str,Any]:
    pairs=[f"{symbol}/USDT", f"{symbol}/USDC", f"{symbol}/USD"]
    stats={"buy_count":0,"sell_count":0,"buy_notional":0.0,"sell_notional":0.0,"examples":[],"flows_by_exchange":{}}
    async def run_ex(ex_id):
        try:
            ex=getattr(ccxt, ex_id)()
            ex.enableRateLimit = True
            ex.timeout = 10000
            for p in pairs:
                try:
                    trades=await ex.fetch_trades(p, limit=lookback_trades)
                    if not trades: continue
                    prices=[t.get("price") for t in trades if t.get("price")]
                    if not prices: continue
                    mid=sum(prices)/len(prices)
                    sizes=[(t.get("amount") or 0)*(t.get("price") or mid) for t in trades]
                    med=float(np.median([s for s in sizes if s])) if sizes else 0.0
                    th=max(usd_threshold, med*5)
                    if ex_id not in stats["flows_by_exchange"]:
                        stats["flows_by_exchange"][ex_id]={"buy":0.0,"sell":0.0,"count_buy":0,"count_sell":0}
                    for t in trades:
                        pr=t.get("price") or mid; amt=t.get("amount") or 0.0
                        notion=pr*amt
                        if notion<th: continue
                        side=t.get("side") or t.get("info",{}).get("side") or ("buy" if pr>=mid else "sell")
                        if side=="buy":
                            stats["buy_count"]+=1; stats["buy_notional"]+=notion
                            stats["flows_by_exchange"][ex_id]["buy"]+=notion; stats["flows_by_exchange"][ex_id]["count_buy"]+=1
                        else:
                            stats["sell_count"]+=1; stats["sell_notional"]+=notion
                            stats["flows_by_exchange"][ex_id]["sell"]+=notion; stats["flows_by_exchange"][ex_id]["count_sell"]+=1
                        if len(stats["examples"])<5:
                            stats["examples"].append({"exchange":ex_id,"pair":p,"price":pr,"amount":amt,"notional":notion,"side":side})
                    break
                except Exception:
                    continue
            await ex.close()
        except Exception:
            pass
    await asyncio.gather(*[run_ex(e) for e in S.EXCHANGES])
    total=stats["buy_notional"]+stats["sell_notional"]; bias=(stats["buy_notional"]-stats["sell_notional"])/total if total>0 else 0.0
    stats["bias"]=float(bias); stats["total_notional"]=float(total)
    return stats

async def ex_fetch_orderbook_walls(symbol: str, depth: int = 50, usd_threshold: float = 300000.0) -> Dict[str,Any]:
    pairs=[f"{symbol}/USDT", f"{symbol}/USDC", f"{symbol}/USD"]
    walls={"bids":[],"asks":[]}
    async def run_ex(ex_id):
        try:
            ex=getattr(ccxt, ex_id)()
            ex.enableRateLimit = True
            ex.timeout = 10000
            for p in pairs:
                try:
                    ob=await ex.fetch_order_book(p, limit=depth)
                    if not ob: continue
                    def scan(levels, side):
                        for lvl in levels[:depth]:
                            price, amount=lvl[0], lvl[1]; notional=price*amount
                            if notional>=usd_threshold:
                                walls[side].append({"exchange":ex_id,"pair":p,"price":price,"amount":amount,"notional":notional})
                    scan(ob.get("bids",[]),"bids"); scan(ob.get("asks",[]),"asks"); break
                except Exception:
                    continue
            await ex.close()
        except Exception:
            pass
    await asyncio.gather(*[run_ex(e) for e in S.EXCHANGES])
    walls["bids"]=sorted(walls["bids"], key=lambda x: x["notional"], reverse=True)[:5]
    walls["asks"]=sorted(walls["asks"], key=lambda x: x["notional"], reverse=True)[:5]
    return walls

async def ex_fetch_derivatives(symbol: str) -> Dict[str, Any]:
    if not S.FUNDING_ENABLED: return {}
    pairs=[f"{symbol}/USDT", f"{symbol}/USDC", f"{symbol}/USD"]
    frs=[]; ois=[]
    async def run_ex(ex_id):
        try:
            ex=getattr(ccxt, ex_id)()
            ex.enableRateLimit = True
            ex.timeout = 10000
            for p in pairs:
                try:
                    if hasattr(ex, 'fetchFundingRate'):
                        fr=await ex.fetchFundingRate(p)
                        if fr and fr.get("fundingRate") is not None:
                            frs.append(fr["fundingRate"]); break
                except Exception:
                    continue
            try:
                if hasattr(ex, 'fetchOpenInterest'):
                    for p in pairs:
                        try:
                            oi=await ex.fetchOpenInterest(p)
                            if oi and oi.get("openInterestAmount") is not None:
                                ois.append(oi["openInterestAmount"]); break
                        except Exception:
                            continue
            except Exception:
                pass
            await ex.close()
        except Exception:
            pass
    await asyncio.gather(*[run_ex(e) for e in S.EXCHANGES])
    return {"funding_rate": float(np.nanmean(frs)) if frs else None, "open_interest": float(np.nansum(ois)) if ois else None}

# ---------------- News ----------------
async def cryptopanic_news(query: Optional[str]=None, filter_: str="hot", limit: int=50) -> List[Dict[str,Any]]:
    if not S.CRYPTOPANIC_API_KEY: return []
    try:
        params={"auth_token":S.CRYPTOPANIC_API_KEY,"filter":filter_,"public":"true","kind":"news","currencies":query.upper() if query else None}
        async with aiohttp.ClientSession() as session:
            async with session.get("https://cryptopanic.com/api/v1/posts/", params={k:v for k,v in params.items() if v is not None}, timeout=30) as resp:
                resp.raise_for_status()
                data=await resp.json()
                res=data.get("results",[])[:limit]
                out=[]
                for r in res:
                    out.append({"title":r.get("title"),"url":r.get("url"),"published_at":r.get("published_at"),"source":r.get("source",{}).get("title"),"currencies":[c.get("code") for c in r.get("currencies",[]) if c.get("code")],"sentiment":r.get("sentiment")})
                return out
    except Exception as e:
        logger.error(f"CryptoPanic error: {e}"); return []

async def newsapi_general(query: str, limit: int=40) -> List[Dict[str,Any]]:
    if not S.NEWS_API_KEY: return []
    try:
        params={"q":query,"language":"en","pageSize":min(limit,100),"sortBy":"publishedAt","apiKey":S.NEWS_API_KEY}
        async with aiohttp.ClientSession() as session:
            async with session.get("https://newsapi.org/v2/everything", params=params, timeout=30) as resp:
                resp.raise_for_status()
                data=await resp.json()
                arts=data.get("articles",[])
                out=[]
                for a in arts:
                    out.append({"title":a.get("title"),"description":a.get("description"),"url":a.get("url"),"published_at":a.get("publishedAt"),"source":a.get("source",{}).get("name")})
                return out
    except Exception as e:
        logger.error(f"NewsAPI error: {e}"); return []

# ---------------- Universe Discovery ----------------
async def discover_universe() -> List[Dict[str,Any]]:
    scope=S.UNIVERSE_SCOPE
    pages={"top50":1,"top100":1,"top200":2,"all":10}
    per_page=250
    count=pages.get(scope,2)
    out=[]
    for p in range(1,count+1):
        data=await cg_fetch_markets(page=p, per_page=per_page)
        if not data: break
        out.extend([{"symbol":d.get("symbol","").upper(),"id":d.get("id"),"name":d.get("name"),"market_cap":d.get("market_cap")} for d in data if d.get("symbol")])
    best={}
    for c in out:
        sym=c["symbol"]
        if sym not in best or (c.get("market_cap",0)>best[sym].get("market_cap",0)):
            best[sym]=c
    universe=list(best.values())
    if S.MAX_COINS and len(universe)>S.MAX_COINS:
        universe=sorted(universe,key=lambda x: x.get("market_cap",0), reverse=True)[:S.MAX_COINS]
    logger.info(f"Universe discovered: {len(universe)} coins (scope={scope})")
    return universe

# ---------------- Technicals / Advanced ----------------
def rsi(series: pd.Series, period=14)->pd.Series:
    delta=series.diff()
    gain=(delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs=gain/(loss+1e-10)
    return 100-(100/(1+rs))

def ema(series: pd.Series, span: int)->pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    line=ema(series,fast)-ema(series,slow)
    sig=ema(line,signal); hist=line-sig
    return line,sig,hist

def bollinger(series: pd.Series, period=20, nstd=2.0):
    ma=series.rolling(period).mean(); sd=series.rolling(period).std(ddof=0)
    up=ma+nstd*sd; lo=ma-nstd*sd; return up, ma, lo

def atr(df: pd.DataFrame, period=14)->pd.Series:
    prev=df["close"].shift(1)
    tr=np.maximum(df["high"]-df["low"], np.maximum((df["high"]-prev).abs(), (df["low"]-prev).abs()))
    return tr.ewm(alpha=1/period, adjust=False).mean()

def compute_technical_summary(df: pd.DataFrame)->Dict[str,Any]:
    c=df["close"]
    r=float(rsi(c).iloc[-1])
    mc,ms,mh=macd(c); bb_u,bb_m,bb_l=bollinger(c)
    em50=float(ema(c,50).iloc[-1]); em200=float(ema(c,200).iloc[-1])
    trend="Bullish" if em50>em200 else "Bearish"
    return {"rsi":r,"macd":{"macd":float(mc.iloc[-1]),"signal":float(ms.iloc[-1]),"hist":float(mh.iloc[-1])},"bollinger":{"upper":float(bb_u.iloc[-1]),"mid":float(bb_m.iloc[-1]),"lower":float(bb_l.iloc[-1])},"atr":float(atr(df).iloc[-1]),"ema":{"ema50":em50,"ema200":em200,"trend":trend}}

def adx(df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
    high, low, close = df['high'], df['low'], df['close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atrv = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / (atrv + 1e-9)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / (atrv + 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    adx_val = float(pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().iloc[-1])
    return {"plus_di": float(plus_di.iloc[-1]), "minus_di": float(minus_di.iloc[-1]), "adx": adx_val}

def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-9)
    d = k.rolling(window=d_period).mean()
    return {"stoch_k": float(k.iloc[-1]), "stoch_d": float(d.iloc[-1])}

def ichimoku(df: pd.DataFrame) -> Dict[str, float | str]:
    high, low, close = df['high'], df['low'], df['close']
    conversion_line = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base_line = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((conversion_line + base_line) / 2)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2)
    price = close.iloc[-1]
    cloud_top = max(span_a.iloc[-1], span_b.iloc[-1])
    cloud_bottom = min(span_a.iloc[-1], span_b.iloc[-1])
    above_cloud = price > cloud_top
    below_cloud = price < cloud_bottom
    cloud_state = "Bullish" if above_cloud else "Bearish" if below_cloud else "Neutral"
    return {
        "tenkan": float(conversion_line.iloc[-1]),
        "kijun": float(base_line.iloc[-1]),
        "senkou_a": float(span_a.iloc[-1]),
        "senkou_b": float(span_b.iloc[-1]),
        "cloud_state": cloud_state
    }

def mfi(df: pd.DataFrame, period: int = 14) -> float:
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    mr = tp.diff()
    pmf = ((mr > 0) * (tp * df['volume'])).rolling(period).sum()
    nmf = ((mr < 0) * (tp * df['volume'])).rolling(period).sum().abs()
    mfr = (pmf / (nmf + 1e-9)).fillna(1.0)
    mfi_val = 100 - (100 / (1 + mfr))
    return float(mfi_val.iloc[-1])

def obv(df: pd.DataFrame) -> float:
    change = np.sign(df['close'].diff().fillna(0.0))
    obv_val = (change * df['volume']).fillna(0.0).cumsum().iloc[-1]
    return float(obv_val)

def compute_advanced_technicals(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        return {
            "adx": adx(df),
            "stochastic": stochastic(df),
            "ichimoku": ichimoku(df),
            "mfi": mfi(df),
            "obv": obv(df),
            "summary": {}
        }
    except Exception:
        return {}

# ---------------- Candlesticks ----------------
def detect_candlestick_patterns(df: pd.DataFrame)->Dict[str,Any]:
    patterns=[]
    if len(df)<3: return {"patterns":[]}
    prev2=df.iloc[-3]; prev=df.iloc[-2]; curr=df.iloc[-1]
    if (prev["close"]<prev["open"]) and (curr["close"]>curr["open"]) and (curr["close"]>=prev["open"]) and (curr["open"]<=prev["close"]):
        patterns.append("Bullish Engulfing")
    if (prev["close"]>prev["open"]) and (curr["close"]<curr["open"]) and (curr["open"]>=prev["close"]) and (curr["close"]<=prev["open"]):
        patterns.append("Bearish Engulfing")
    body=abs(curr["close"]-curr["open"])
    lower=(curr["open"]-curr["low"]) if curr["open"]>curr["close"] else (curr["close"]-curr["low"])
    upper=curr["high"]-max(curr["open"], curr["close"])
    if lower>2*body and upper<body: patterns.append("Hammer")
    lower2=min(curr["open"], curr["close"]) - curr["low"]
    if upper>2*body and lower2<body: patterns.append("Shooting Star")
    rng=curr["high"]-curr["low"]
    if rng>0 and abs(curr["close"]-curr["open"])/rng<0.1: patterns.append("Doji")
    if (prev2['close'] < prev2['open']) and (abs(prev['close']-prev['open'])/max(prev['high']-prev['low'],1e-9) < 0.2) and (curr['close'] > (prev2['open'] + prev2['close'])/2):
        patterns.append("Morning Star")
    if (prev2['close'] > prev2['open']) and (abs(prev['close']-prev['open'])/max(prev['high']-prev['low'],1e-9) < 0.2) and (curr['close'] < (prev2['open'] + prev2['close'])/2):
        patterns.append("Evening Star")
    return {"patterns":patterns}

# ---------------- Elliott (ZigZag + Fibo) ----------------
def zigzag(series: pd.Series, deviation=0.05)->List[Dict[str,Any]]:
    piv=[]; last_p=series.iloc[0]; last_idx=series.index[0]; trend=0
    for i in range(1,len(series)):
        pr=series.iloc[i]; ch=(pr-last_p)/max(last_p,1e-9)
        if trend>=0 and ch<=-deviation:
            piv.append({"idx":last_idx,"price":float(last_p)}); trend=-1; last_p=pr; last_idx=series.index[i]
        elif trend<=0 and ch>=deviation:
            piv.append({"idx":last_idx,"price":float(last_p)}); trend=1; last_p=pr; last_idx=series.index[i]
        else:
            if trend>=0 and pr>last_p: last_p=pr; last_idx=series.index[i]
            elif trend<=0 and pr<last_p: last_p=pr; last_idx=series.index[i]
    piv.append({"idx":last_idx,"price":float(last_p)}); return piv

def _fib_ratio(a: float, b: float) -> float:
    return abs(a)/max(abs(b), 1e-9)

def elliott_validate_and_score(pivots: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(pivots) < 9:
        return {"score": 0.0, "current_wave": "نامشخص", "label": []}
    prices = [p["price"] for p in pivots]
    best_score = 0.0; state="نامشخص"; target=None; invalidation=None
    for start in range(len(prices)-9, len(prices)-5):
        seg = prices[start:start+6]
        if len(seg) < 6: continue
        w1 = seg[1]-seg[0]; w2 = seg[2]-seg[1]; w3 = seg[3]-seg[2]; w4 = seg[4]-seg[3]; w5 = seg[5]-seg[4]
        bull = w1>0 and w3>0 and w5>0 and w2<0 and w4<0
        bear = w1<0 and w3<0 and w5<0 and w2>0 and w4>0
        if not (bull or bear): continue
        score = 0.4
        r2 = _fib_ratio(w2, w1)
        if 0.3 <= r2 <= 0.8: score += 0.1
        r3 = _fib_ratio(w3, w1)
        if r3 >= 1.0: score += 0.15
        r4 = _fib_ratio(w4, w3)
        if 0.2 <= r4 <= 0.6: score += 0.1
        overlap = (min(seg[2], seg[3]) < max(seg[0], seg[1])) if bull else (max(seg[2], seg[3]) > min(seg[0], seg[1]))
        if not overlap: score += 0.1
        if abs(w3) >= max(abs(w1), abs(w5))*0.8: score += 0.05
        state = "موج 5 در جریان" if (bull and w5>0) else ("موج 5 نزولی در جریان" if bear and w5<0 else "اصلاح ABC محتمل")
        if bull:
            invalidation = min(seg[0], seg[1]); target = seg[4] + abs(w1)*0.618
        else:
            invalidation = max(seg[0], seg[1]); target = seg[4] - abs(w1)*0.618
        best_score = max(best_score, min(1.0, score))
    return {"score": float(best_score), "current_wave": state, "target": target, "invalidation": invalidation, "label": ["W1","W2","W3","W4","W5"]}

def analyze_elliott(df: pd.DataFrame, deviation=0.05)->Dict[str,Any]:
    piv=zigzag(df["close"], deviation=deviation); res=elliott_validate_and_score(piv)
    res["pivots_count"]=len(piv); res["last_pivots"]=piv[-10:]; return res

# ---------------- Market Structure / Order Blocks ----------------
def find_swings(df: pd.DataFrame, lookback=5):
    highs=df["high"]; lows=df["low"]; sh=[]; sl=[]
    for i in range(lookback, len(df)-lookback):
        if highs.iloc[i]==highs.iloc[i-lookback:i+lookback+1].max(): sh.append((df.index[i], highs.iloc[i]))
        if lows.iloc[i]==lows.iloc[i-lookback:i+lookback+1].min(): sl.append((df.index[i], lows.iloc[i]))
    return sh, sl

def ms_trend(df: pd.DataFrame)->str:
    sh, sl = find_swings(df)
    if len(sh)<2 or len(sl)<2: return "خنثی"
    hh=[p[1] for p in sh[-3:]]; ll=[p[1] for p in sl[-3:]]
    if hh[-1]>hh[-2] and ll[-1]>ll[-2]: return "صعودی"
    if hh[-1]<hh[-2] and ll[-1]<ll[-2]: return "نزولی"
    return "خنثی"

def ms_bos(df: pd.DataFrame)->Dict[str,Any]:
    sh, sl = find_swings(df); lc=df["close"].iloc[-1]
    return {"bos_up": any(lc>h[1] for h in sh[-3:]), "bos_down": any(lc<l[1] for l in sl[-3:])}

def detect_order_blocks(df: pd.DataFrame, lookback=200)->Dict[str,Any]:
    sub=df.iloc[-lookback:]; tr=ms_trend(sub); bos=ms_bos(sub)
    ob_sup=None; ob_dem=None
    if bos.get("bos_up"):
        for i in range(len(sub)-3,1,-1):
            c=sub.iloc[i]; nxt=sub.iloc[i+1]
            if c["close"]<c["open"] and (nxt["close"]-nxt["open"])/max(c["close"],1e-9)>0.01:
                ob_dem={"low":c["low"],"high":c["open"],"index":sub.index[i]}; break
    if bos.get("bos_down"):
        for i in range(len(sub)-3,1,-1):
            c=sub.iloc[i]; nxt=sub.iloc[i+1]
            if c["close"]>c["open"] and (c["open"]-nxt["close"])/max(c["close"],1e-9)>0.01:
                ob_sup={"low":c["open"],"high":c["high"],"index":sub.index[i]}; break
    return {"trend":tr,"bos":bos,"order_blocks":{"supply":ob_sup,"demand":ob_dem}}

# ---------------- Sessions ----------------
def analyze_sessions(df_1h: pd.DataFrame)->Dict[str,Any]:
    if df_1h.empty: return {"bias":"خنثی","session_stats":{}}
    labels=df_1h.index.map(lambda ts: get_session_label_from_ts_ms(int(ts)))
    rets=df_1h["close"].pct_change().fillna(0.0)
    d=pd.DataFrame({"ret":rets.values,"session":labels.values}, index=df_1h.index)
    g=d.groupby("session")["ret"].agg(["mean","std","count"])
    best=g["mean"].idxmax() if not g.empty else "Off"
    bias="صعودی در سشن "+best if (not g.empty and g["mean"].max()>0) else "خنثی"
    stats={idx:{"mean":float(g.loc[idx,"mean"]), "std":float(g.loc[idx,"std"]), "count":int(g.loc[idx,"count"])} for idx in g.index}
    return {"bias":bias,"session_stats":stats}

# ---------------- Sentiment ----------------
vader=SentimentIntensityAnalyzer()
def sentiment_vader(items: List[Dict[str,Any]])->float:
    if not items: return 0.0
    scores=[]
    for n in items:
        text=" ".join([str(n.get("title") or ""), str(n.get("description") or "")])
        vs=vader.polarity_scores(text); scores.append(vs["compound"])
    return float(np.mean(scores)) if scores else 0.0

def sentiment_transformers(items: List[Dict[str,Any]])->float:
    if not _USE_TRANSFORMERS or not items: return sentiment_vader(items)
    try:
        clf=pipeline("sentiment-analysis", model="ProsusAI/finbert")
        texts=[(n.get("title") or "")+" "+(n.get("description") or "") for n in items[:64] if n.get("title") or n.get("description")]
        if not texts: return 0.0
        outs=clf(texts, truncation=True)
        def map_label(x):
            lab=x["label"].lower()
            if "positive" in lab: return 1.0
            if "negative" in lab: return -1.0
            return 0.0
        vals=[map_label(o)*o.get("score",0.7) for o in outs]
        return float(np.mean(vals))
    except Exception:
        return sentiment_vader(items)

def analyze_sentiment_news(news: List[Dict[str,Any]])->Dict[str,Any]:
    avg=sentiment_transformers(news) if _USE_TRANSFORMERS else sentiment_vader(news)
    topics=list({n.get("source") for n in news if n.get("source")})[:10]
    items=[{"title":n.get("title"),"sentiment":0.0,"source":n.get("source"),"url":n.get("url")} for n in news[:10]]
    return {"average_sentiment":avg,"items":items,"topics":topics}

# ---------------- Quant / ML ----------------
def make_features(df: pd.DataFrame)->pd.DataFrame:
    c=df["close"]; d=c.diff()
    up=d.clip(lower=0).rolling(14).mean(); dn=(-d.clip(upper=0)).rolling(14).mean()
    rsi_val=100-(100/(1+(up/(dn+1e-9))))
    mac=c.ewm(12,adjust=False).mean()-c.ewm(26,adjust=False).mean()
    vol=df["volume"].rolling(20).mean()
    ret1=c.pct_change(); ret5=c.pct_change(5); mom=c.diff()
    f=pd.DataFrame({"ret1":ret1,"ret5":ret5,"vol":vol,"mom":mom,"rsi":rsi_val.fillna(50),"macd":mac.fillna(0.0)}).fillna(0.0)
    return f

def regime_detection(df: pd.DataFrame)->Dict[str,Any]:
    if df is None or df.empty or len(df)<60: return {"regime":"unknown","confidence":0.0}
    X=make_features(df).values[-500:]
    if len(X)<50: return {"regime":"unknown","confidence":0.0}
    gm=GaussianMixture(n_components=2, covariance_type="full", random_state=42)
    gm.fit(X); labels=gm.predict(X)
    rets=make_features(df)["ret1"].values[-len(labels):]
    bull_label=1 if np.nanmean(rets[labels==1])>np.nanmean(rets[labels==0]) else 0
    current=labels[-1]; regime="bull" if current==bull_label else "bear"
    diff=abs(np.nanmean(rets[labels==current])-np.nanmean(rets[labels!=current]))
    conf=float(min(1.0, max(0.0, diff*100)))
    return {"regime":regime,"confidence":conf}

def _optimize_thresholds(probs: np.ndarray, y: np.ndarray) -> Tuple[float,float]:
    best=(0.62,0.38); best_score=-1.0
    for buy in np.linspace(0.55,0.8,6):
        for sell in np.linspace(0.2,0.45,6):
            pred = np.array(["HOLD"]*len(probs), dtype=object)
            pred[probs>=buy]="BUY"; pred[probs<=sell]="SELL"
            mask = pred!="HOLD"
            if mask.sum()<20: continue
            yh = (pred[mask]=="BUY").astype(int)
            yt = y[mask]
            acc = (yh==yt).mean()
            if acc>best_score:
                best_score=acc; best=(buy,sell)
    return best

def rf_train_predict(df: pd.DataFrame, symbol: str, tf: str, force_train: bool = False)->Dict[str,Any]:
    feat=make_features(df)
    y=(df["close"].shift(-1)>df["close"]).astype(int)
    valid=~(feat.isna().any(axis=1) | y.isna())
    X=feat[valid]; yv=y[valid].values
    if len(X)<200: return {"prob_up":0.5,"model_acc":0.5}
    path=f"rfcal_{symbol}_{tf}.joblib"
    if S.FAST_RESPONSE and (not force_train) and (not os.path.exists(path)):
        return {"prob_up":0.55,"model_acc":0.6,"t_buy":None,"t_sell":None}
    clf=None; best_acc=0.0; thresholds=(None,None); trained_now=False
    if os.path.exists(path):
        try:
            obj=load(path); clf=obj["clf"]; thresholds=obj.get("thr",(None,None)); best_acc=obj.get("acc",0.6)
        except Exception:
            clf=None
    if clf is None:
        tscv=TimeSeriesSplit(n_splits=5)
        best=None; ba=0.0; best_thr=(0.62,0.38)
        for depth in [3,5,7]:
            base=RandomForestClassifier(n_estimators=200, max_depth=depth, random_state=42, n_jobs=-1)
            cal=CalibratedClassifierCV(base, method="isotonic", cv=3)
            accs=[]; probs_val=[]; y_val=[]
            for tr, te in tscv.split(X):
                cal.fit(X.iloc[tr], yv[tr])
                pv = cal.predict_proba(X.iloc[te])[:,1]
                accs.append(accuracy_score(yv[te], (pv>0.5).astype(int)))
                probs_val.append(pv); y_val.append(yv[te])
            avg=float(np.mean(accs)); pv=np.concatenate(probs_val); ycat=np.concatenate(y_val)
            thr=_optimize_thresholds(pv, ycat)
            if avg>ba:
                ba=avg; best=cal; best_thr=thr
        clf=best; best_acc=ba; dump({"clf":clf,"thr":best_thr,"acc":best_acc}, path); thresholds=best_thr; trained_now=True
    prob=float(clf.predict_proba([feat.iloc[-1].values])[0][1])
    if thresholds[0] is not None:
        db_upsert_thresholds(symbol, tf, thresholds[0], thresholds[1])
    if trained_now or force_train:
        db_upsert_model_registry(symbol, tf, "RFCalibrated", float(max(best_acc,0.5)))
    return {"prob_up":prob, "model_acc":float(max(best_acc,0.6)), "t_buy": thresholds[0], "t_sell": thresholds[1]}

def strat_signals(df: pd.DataFrame)->Dict[str,np.ndarray]:
    c=df["close"]
    e50=ema(c,50); e200=ema(c,200); s1=np.where(e50>e200,1,-1)
    r=rsi(c); s2=np.zeros(len(r)); s2[r<30]=1; s2[r>70]=-1
    _,_,hist=macd(c); s3=np.where(hist>0,1,-1)
    return {"ema_tf":s1,"rsi_mr":s2,"macd_dir":s3}

def evaluate_strategies(df: pd.DataFrame, horizon: int=3)->Dict[str,Any]:
    if len(df)<200: return {"metrics":{}, "weights":{}, "best":"none"}
    sigs=strat_signals(df); c=df["close"]; fwd_ret=(c.shift(-horizon)/c - 1.0).fillna(0.0)
    metrics={}
    for name, s in sigs.items():
        align=min(len(s), len(fwd_ret))
        s=s[-align:]; fr=fwd_ret.values[-align:]
        mask=s!=0
        if mask.sum()<30:
            metrics[name]={"hit":0.5,"avg":0.0,"pf":1.0,"sharpe":0.0}; continue
        dir_ret=fr[mask]*s[mask]
        hit=float((dir_ret>0).mean()); avg=float(dir_ret.mean())
        pos=dir_ret[dir_ret>0].sum(); neg=-dir_ret[dir_ret<0].sum()
        pf=float(pos/neg) if neg>0 else float("inf")
        sharpe=float(dir_ret.mean()/ (dir_ret.std()+1e-9))
        metrics[name]={"hit":hit,"avg":avg,"pf":pf,"sharpe":sharpe}
    weights={}
    for k,v in metrics.items():
        sc=0.5*v["hit"]+0.3*max(0.0,min(1.0,(v["sharpe"]+2)/4))+0.2*max(0.0,min(1.0,(v["pf"] if v["pf"]!=float("inf") else 3)/3))
        weights[k]=float(sc)
    ssum=sum(weights.values()) or 1.0
    weights={k:v/ssum for k,v in weights.items()}
    best=max(weights, key=weights.get) if weights else "none"
    return {"metrics":metrics,"weights":weights,"best":best}

def stacking_predict(df: pd.DataFrame, symbol: str, timeframe: str, force_train: bool = False) -> Dict[str, Any]:
    path = f"stack_{symbol}_{timeframe}.joblib"
    feat = make_features(df)
    y = (df["close"].shift(-1) > df["close"]).astype(int)
    valid = ~(feat.isna().any(axis=1) | y.isna())
    X = feat[valid]; y = y[valid]
    if len(X) < 250:
        return {"prob_up": 0.5, "model_acc": 0.55}
    if S.FAST_RESPONSE and (not force_train) and (not os.path.exists(path)):
        return {"prob_up": 0.55, "model_acc": 0.6}
    models = None
    best_acc = 0.0; trained_now=False
    if os.path.exists(path):
        try:
            models = load(path)
        except Exception:
            models = None
    if models is None:
        tscv = TimeSeriesSplit(n_splits=5)
        best_models = None
        best = 0.0
        for params in [(150, 3), (250, 5), (350, 7)]:
            rf = RandomForestClassifier(n_estimators=params[0], max_depth=params[1], random_state=42, n_jobs=-1)
            gbc = GradientBoostingClassifier(random_state=42)
            lr = LogisticRegression(max_iter=500, solver="lbfgs")
            accs = []
            for tr, te in tscv.split(X):
                rf.fit(X.iloc[tr], y.iloc[tr])
                gbc.fit(X.iloc[tr], y.iloc[tr])
                base_probs = np.vstack([rf.predict_proba(X.iloc[te])[:,1], gbc.predict_proba(X.iloc[te])[:,1]]).T
                lr.fit(base_probs, y.iloc[te])
                pred = (lr.predict_proba(base_probs)[:,1] > 0.5).astype(int)
                accs.append(accuracy_score(y.iloc[te], pred))
            avg = float(np.mean(accs))
            if avg > best:
                best = avg; best_models = {"rf": rf, "gbc": gbc, "lr": lr}
        models = best_models
        dump(models, path); best_acc = best; trained_now=True
    else:
        rf = models["rf"]; gbc = models["gbc"]; lr = models["lr"]
        test_idx = -min(200, len(X)//5) if len(X) > 300 else -50
        base_probs = np.vstack([rf.predict_proba(X.iloc[test_idx:])[:,1], gbc.predict_proba(X.iloc[test_idx:])[:,1]]).T
        pred = (lr.predict_proba(base_probs)[:,1] > 0.5).astype(int)
        best_acc = float(accuracy_score(y.iloc[test_idx:], pred))
    rf = models["rf"]; gbc = models["gbc"]; lr = models["lr"]
    prob_up = float(lr.predict_proba([[rf.predict_proba([feat.iloc[-1].values])[0][1], gbc.predict_proba([feat.iloc[-1].values])[0][1]]])[0][1])
    if trained_now or force_train:
        db_upsert_model_registry(symbol, timeframe, "Stacking", float(max(0.5, min(0.95, best_acc))))
    return {"prob_up": prob_up, "model_acc": float(max(0.5, min(0.95, best_acc)))}

def auto_ensemble_prob(df: pd.DataFrame, symbol: str, tf: str, sentiment_value: float, regime: str, whales_bias: float, derivatives: Dict[str,Any]|None=None, force_train: bool = False)->Dict[str,Any]:
    rf = rf_train_predict(df, symbol=symbol, tf=tf, force_train=force_train)
    eva=evaluate_strategies(df)
    for name, m in eva.get("metrics",{}).items():
        db_upsert_strat_perf(symbol, tf, name, m)
    w=eva["weights"]; sigs=strat_signals(df); last={k: int(s[-1]) if len(s)>0 else 0 for k,s in sigs.items()}
    vote=0.0
    for k,wt in w.items():
        vote += wt * (0.6 if last.get(k,0)>0 else -0.6 if last.get(k,0)<0 else 0.0)
    prob = 0.6*rf["prob_up"] + 0.4*(0.5 + vote/2)
    if regime=="bull": prob = 0.1 + 0.9*prob
    elif regime=="bear": prob = 0.9*prob
    prob += 0.10 * float(max(-1.0, min(1.0, sentiment_value)))
    prob += 0.08 * float(max(-1.0, min(1.0, whales_bias)))
    if derivatives:
        fr = derivatives.get("funding_rate")
        if fr is not None:
            prob += 0.03 * np.tanh(fr*1000)
    prob=float(max(0.0,min(1.0,prob)))
    strat_quality = float(sum(eva["weights"].get(k,0)*eva["metrics"].get(k,{}).get("hit",0.5) for k in eva["weights"]))
    reliability = float(max(0.5, min(0.95, 0.5*rf["model_acc"] + 0.5*strat_quality)))
    return {"prob_up":prob, "model_acc":reliability, "rf":rf, "strategies":eva, "last_signals":last}

# ---------------- Backtest summary ----------------
def backtest_strategies(df: pd.DataFrame, horizon: int = 3) -> Dict[str, Any]:
    if df is None or df.empty or len(df) < 200:
        return {"cagr": 0.0, "sharpe": 0.0, "max_dd": 0.0}
    sigs = strat_signals(df)
    c = df["close"].values
    ret = (np.roll(c, -horizon)/c - 1.0); ret[-horizon:] = 0.0
    vote = np.zeros_like(ret)
    for _, s in sigs.items():
        vote += s
    vote = np.sign(vote)
    pnl = ret * vote
    equity = (1 + pnl).cumprod()
    cagr = float(equity[-1]**(1/ (len(equity)/365.))*100 - 100) if len(equity)>0 else 0.0
    sharpe = float(np.mean(pnl)/(np.std(pnl)+1e-9) * np.sqrt(365/horizon))
    max_dd = float(((equity - np.maximum.accumulate(equity)) / (np.maximum.accumulate(equity)+1e-9)).min())
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd}

# ---------------- Risk / Entry-Exit + Leverage ----------------
def kelly_fraction(prob_win: float, rr: float) -> float:
    b = max(rr, 0.1); p = max(0.0, min(1.0, prob_win))
    k = (p*(b+1) - 1) / b
    return max(0.0, min(0.25, k))

def suggest_leverage(entry: float, sl: float, direction: str, maintenance_margin: float = 0.004, safety_k: float = 2.5, max_leverage: float = 10.0) -> Tuple[float, float]:
    if not entry or not sl or entry <= 0:
        return 1.0, None
    stop_dist = (entry - sl) if direction=="BUY" else (sl - entry)
    if stop_dist <= 0:
        return 1.0, None
    denom = safety_k * (stop_dist/entry)
    L = min(max_leverage, 1.0 / denom) if denom>0 else 1.0
    L = float(max(1.0, min(L, max_leverage)))
    liq = entry * (1 - 1.0/L) if direction=="BUY" else entry * (1 + 1.0/L)
    return L, float(liq)

def risk_management_calc(df: pd.DataFrame, price: float, balance: float, risk_per_trade: float, prob_up: float = 0.55, rr_hint: float = 1.5)->Dict[str,Any]:
    a=atr(df); atrv=float(a.iloc[-1]) if len(a) else 0.0
    sl=price - 1.5*atrv
    tp1=price + 1.5*atrv; tp2=price + 2.5*atrv; tp3=price + 3.5*atrv
    rr = (tp2 - price) / max(price - sl, 1e-9) if price>sl else rr_hint
    risk_amt=balance*risk_per_trade
    units = risk_amt / max(price - sl, 1e-9) if price>sl else 0.0
    k = kelly_fraction(prob_up, rr if rr>0 else rr_hint)
    k_units = k * (balance / max(price,1e-9))
    units = max(units, k_units)
    rr_out = (tp2 - price) / max(price - sl, 1e-9) if price>sl else 0.0
    lev, liq = suggest_leverage(price, sl, "BUY" if rr_out>=0 else "SELL")
    return {"atr": atrv,"stop_loss": float(sl),"take_profit_1": float(tp1),"take_profit_2": float(tp2),"take_profit_3": float(tp3),"units": float(units),"risk_reward_ratio": float(rr_out),"kelly": float(k),"leverage": float(lev),"liq_price": liq}

def trailing_stop(price: float, atr_value: float, direction: str) -> float:
    if atr_value <= 0:
        return price
    return price - 1.0 * atr_value if direction=="BUY" else price + 1.0 * atr_value

def entry_exit_rules(price: float, df: pd.DataFrame, whales_walls: Dict[str,Any], direction: str, atrv: float)->Dict[str,Any]:
    entry=price; buffer=0.1*atrv
    if direction=="BUY":
        sh, sl = find_swings(df)
        entry = max(price, (sh[-1][1] if sh else price) + buffer)
        asks=whales_walls.get("asks",[])
        tp_hint=min([w["price"] for w in asks], default=None)
        tp_hint=tp_hint if tp_hint and tp_hint>entry else None
        return {"entry":float(entry),"tp_hint":tp_hint,"trailing": trailing_stop(entry, atrv, "BUY")}
    elif direction=="SELL":
        sh, sl = find_swings(df)
        entry = min(price, (sl[-1][1] if sl else price) - buffer)
        bids=whales_walls.get("bids",[])
        tp_hint=max([w["price"] for w in bids], default=None)
        tp_hint=tp_hint if tp_hint and tp_hint<entry else None
        return {"entry":float(entry),"tp_hint":tp_hint,"trailing": trailing_stop(entry, atrv, "SELL")}
    return {"entry":float(price),"tp_hint":None,"trailing": trailing_stop(price, atrv, direction)}

def map_score_to_signal(prob_up: float, reliability: float)->str:
    upper=0.55 + 0.1*(reliability-0.5)
    lower=0.45 - 0.1*(reliability-0.5)
    if prob_up>=upper: return "BUY"
    if prob_up<=lower: return "SELL"
    return "HOLD"

# ---------------- Formatting ----------------
def fmt_num(x, nd=4):
    try:
        if x is None: return "N/A"
        return f"{float(x):,.{nd}f}"
    except Exception:
        return str(x)

def format_analysis(analysis: Dict[str,Any])->str:
    md=analysis.get("market_data",{})
    quant=analysis.get("quant",{})
    whale=analysis.get("whales",{})
    walls=analysis.get("orderbook_walls",{})
    deriv=analysis.get("derivatives",{})
    tfdet=analysis.get("timeframes",{})
    ms=analysis.get("market_structure",{})
    risk=analysis.get("risk_management",{})
    tech=analysis.get("technical",{})
    techx=analysis.get("technical_advanced",{})
    ell=analysis.get("elliott",{})
    sent=analysis.get("sentiment",{})
    signal=analysis.get("signal","HOLD")
    conf=analysis.get("confidence",0.5)
    entry=analysis.get("entry_plan",{})

    out=[]
    out.append(f"📊 تحلیل {analysis.get('symbol','UNKNOWN')}\n")
    if md:
        out.append(f"💰 قیمت: ${fmt_num(md.get('price'))}")
        if md.get("price_change_24h") is not None:
            out.append(f"📈 تغییر 24h: {md.get('price_change_24h'):+.2f}%")
        if md.get("volume_24h") is not None:
            out.append(f"🔄 حجم 24h: ${fmt_num(md.get('volume_24h'),0)}")
        if md.get("market_cap") is not None:
            out.append(f"💎 ارزش بازار: ${fmt_num(md.get('market_cap'),0)}")
        out.append("")
    out.append(f"{'🟢' if signal=='BUY' else '🔴' if signal=='SELL' else '🟡'} سیگنال: {signal}")
    out.append(f"🎯 احتمال موفقیت: {conf:.1%}")
    out.append(f"🧠 مدل: P(up)={quant.get('prob_up',0.5):.2f} | اطمینان: {quant.get('model_acc',0.6):.2f} | رژیم: {quant.get('regime','unknown')} ({quant.get('regime_conf',0.0):.2f})")
    if deriv and (deriv.get("funding_rate") is not None or deriv.get("open_interest") is not None):
        out.append(f"📑 مشتقات: Funding {deriv.get('funding_rate',0):+.6f} | OI {fmt_num(deriv.get('open_interest'),0)}")
    out.append("")
    if tfdet:
        out.append("🧭 تحلیل چندزمانی:")
        for tf, data in tfdet.items():
            t=data.get("technical",{})
            out.append(f"  • {tf}: RSI {t.get('rsi',0):.1f} | MACD Hist {t.get('macd',{}).get('hist',0):.4f} | Trend {t.get('ema',{}).get('trend','?')}")
        out.append("")
    if tech:
        out.append("📈 تکنیکال:")
        out.append(f"  • RSI: {tech.get('rsi',0):.2f}")
        m=tech.get("macd",{})
        out.append(f"  • MACD: {m.get('macd',0):.4f} | Signal: {m.get('signal',0):.4f} | Hist: {m.get('hist',0):.4f}")
        bb=tech.get("bollinger",{})
        out.append(f"  • Bollinger: U:{fmt_num(bb.get('upper'))} M:{fmt_num(bb.get('mid'))} L:{fmt_num(bb.get('lower'))}")
        ema_=tech.get("ema",{})
        out.append(f"  • EMA(50/200): {fmt_num(ema_.get('ema50'),2)} / {fmt_num(ema_.get('ema200'),2)} | Trend: {ema_.get('trend')}")
        out.append("")
    if techx:
        adx_v=techx.get("adx",{}); st=techx.get("stochastic",{}); ichi=techx.get("ichimoku",{})
        out.append("🧪 تکنیکال پیشرفته:")
        out.append(f"  • ADX: {adx_v.get('adx',0):.2f} (DI+:{adx_v.get('plus_di',0):.1f} DI-:{adx_v.get('minus_di',0):.1f})")
        out.append(f"  • Stoch: %K {st.get('stoch_k',0):.1f} | %D {st.get('stoch_d',0):.1f}")
        out.append(f"  • Ichimoku: Tenkan {fmt_num(ichi.get('tenkan'),2)} | Kijun {fmt_num(ichi.get('kijun'),2)} | Cloud {ichi.get('cloud_state')}")
        out.append(f"  • MFI: {techx.get('mfi',0):.1f} | OBV: {fmt_num(techx.get('obv'),0)}")
        out.append("")
    if ell:
        out.append(f"🌀 الیوت: {ell.get('current_wave','نامشخص')} | امتیاز: {ell.get('score',0):.2f}")
        if ell.get("target"): out.append(f"  • هدف فیبوناچی: ${fmt_num(ell.get('target'))}")
        if ell.get("invalidation"): out.append(f"  • ناحیه ابطال: ${fmt_num(ell.get('invalidation'))}")
    if ms:
        b=ms.get("bos",{})
        out.append(f"🏗️ ساختار بازار: روند {ms.get('trend','نامشخص')} | BOS: ↑{b.get('bos_up',False)} ↓{b.get('bos_down',False)}")
        ob=ms.get("order_blocks",{})
        if ob.get("supply"): out.append(f"  • OB عرضه: [{fmt_num(ob['supply']['low'])} - {fmt_num(ob['supply']['high'])}]")
        if ob.get("demand"): out.append(f"  • OB تقاضا: [{fmt_num(ob['demand']['low'])} - {fmt_num(ob['demand']['high'])}]")
        out.append("")
    ses=analysis.get("sessions",{})
    if ses: out.append(f"🕒 بایاس سشن: {ses.get('bias','خنثی')}")
    if sent:
        avg=sent.get("average_sentiment",0.0); emo="😊" if avg>0.2 else "😔" if avg<-0.2 else "😐"
        out.append(f"{emo} احساسات خبر: {avg:.2f} | منابع: {', '.join(sent.get('topics',[])[:5])}")
    out.append("")
    if whale:
        out.append(f"🐋 نهنگ‌ها: Bias={whale.get('bias',0.0):+.2f} | Buy ${fmt_num(whale.get('buy_notional',0),0)} / Sell ${fmt_num(whale.get('sell_notional',0),0)} | نمونه‌ها: {len(whale.get('examples',[]))}")
        flows=whale.get("flows_by_exchange",{})
        if flows:
            parts=[]
            for ex,fv in flows.items():
                parts.append(f"{ex}: Buy ${fmt_num(fv.get('buy',0),0)} ({fv.get('count_buy',0)}) / Sell ${fmt_num(fv.get('sell',0),0)} ({fv.get('count_sell',0)})")
            out.append("   └ جریان نهنگ‌ها: " + " | ".join(parts))
    if walls:
        if walls.get("bids"): out.append("🧱 دیوارهای خرید: " + "; ".join([f"{w['exchange']} {fmt_num(w['price'],2)} (${fmt_num(w['notional'],0)})" for w in walls["bids"]]))
        if walls.get("asks"): out.append("🧱 دیوارهای فروش: " + "; ".join([f"{w['exchange']} {fmt_num(w['price'],2)} (${fmt_num(w['notional'],0)})" for w in walls["asks"]]))
    out.append("")
    if risk:
        out.append("⚠️ مدیریت ریسک:")
        out.append(f"  • ATR: {fmt_num(risk.get('atr'),4)} | Kelly: {risk.get('kelly',0):.2f} | Lev: x{fmt_num(risk.get('leverage',1),2)} | Liq≈ {fmt_num(risk.get('liq_price'))}")
        out.append(f"  • حد ضرر: ${fmt_num(risk.get('stop_loss'))}")
        out.append(f"  • حد سودها: TP1 ${fmt_num(risk.get('take_profit_1'))} | TP2 ${fmt_num(risk.get('take_profit_2'))} | TP3 ${fmt_num(risk.get('take_profit_3'))}")
        out.append(f"  • حجم پیشنهادی: {fmt_num(risk.get('units'),4)} واحد | R/R: {risk.get('risk_reward_ratio',0):.2f}")
    if entry:
        out.append("🎯 برنامه ورود/خروج:")
        out.append(f"  • ورود: ${fmt_num(entry.get('entry'))} | TP Hint: {fmt_num(entry.get('tp_hint'))} | Trailing: ${fmt_num(entry.get('trailing'))}")
    reasons=[]
    if tech and tech.get("ema",{}).get("trend")=="Bullish": reasons.append("روند کلی صعودی")
    if sent and sent.get("average_sentiment",0)>0.2: reasons.append("خبرها نسبتاً مثبت")
    if whale and abs(whale.get("bias",0))>0.3: reasons.append(f"جریان نهنگ‌ها {'خرید' if whale.get('bias',0)>0 else 'فروش'}")
    if quant and quant.get("regime")=="bull": reasons.append("رژیم بازار صعودی")
    if deriv and deriv.get("funding_rate") and deriv.get("funding_rate")>0: reasons.append("فاندینگ مثبت (لانگ‌ها غالب)")
    out.append("\n🧠 خلاصه هوشمند: " + ("، ".join(reasons) if reasons else "سیگنال بر اساس ترکیب مدل‌های تکنیکال/کوانت/احساسات"))
    sources=md.get("sources",[])
    if sources: out.append(f"\n🔗 منابع داده: {', '.join(sources)}")
    out.append("\n⚠️ هشدار: استفاده از اهرم ریسک بالایی دارد. این توصیه مالی نیست.")
    return "\n".join(out)

# ---------------- Data Fetcher ----------------
class DataFetcher:
    def __init__(self): self.offline=S.OFFLINE_MODE
    def offline_bundle(self, symbol: str)->Dict[str,Any]:
        price=100+random.random()*10
        t0=int(time.time())-3600*1000
        def gen(n, step=3600):
            o=[]; p=price
            for i in range(n):
                op=p*(0.99+random.random()*0.02)
                hi=op*(1+random.random()*0.01); lo=op*(1-random.random()*0.01)
                cl=(hi+lo)/2; vol=random.random()*1000
                o.append({"time":(t0+i*step)*1000,"open":op,"high":hi,"low":lo,"close":cl,"volume":vol}); p=cl
            return o
        o1=gen(1200,3600); o4=o1[::4]; od=o1[::24]
        return {"market_data":{"symbol":symbol.upper(),"price":price,"price_change_24h":random.uniform(-5,5),"volume_24h":random.uniform(1e5,1e7),"market_cap":random.uniform(1e7,1e10),"sources":["OFFLINE"]},"ohlcv":{"1h":o1,"4h":o4,"1d":od},"news":[],"whales":{"bias":0.0,"flows_by_exchange":{}},"orderbook_walls":{"bids":[],"asks":[]},"derivatives":{},"sources":["OFFLINE"]}
    async def fetch_bundle(self, symbol: str, fast: Optional[bool]=None)->Dict[str,Any]:
        if self.offline: return self.offline_bundle(symbol)
        fast_flag = S.FAST_RESPONSE if fast is None else fast
        key=f"bundle_{symbol}_{'fast' if fast_flag else 'full'}"
        c=cache.get(key)
        if c: return c
        logger.info(f"Fetching data for {symbol} (fast={fast_flag})")
        m_tasks=[cg_fetch_market_by_symbol(symbol), cmc_fetch_quote(symbol), ex_fetch_ticker(symbol)]
        results=await asyncio.gather(*m_tasks, return_exceptions=True)
        cg, cmc, ex = (r if isinstance(r,dict) else {} for r in results)
        cands=[x for x in [cg, cmc, ex] if x]
        prices=[x.get("price") for x in cands if x.get("price")]
        price=statistics.fmean(prices) if prices else (cg.get("price") if cg else None)
        vols=[x.get("volume_24h") for x in cands if x.get("volume_24h")]
        vol=statistics.fmean(vols) if vols else None
        mcap=None
        for x in cands:
            if x.get("market_cap"): mcap=x.get("market_cap"); break
        src=[]
        for x in cands: src+=x.get("sources",[])
        market_data={"symbol":symbol.upper(),"price":price,"price_change_24h":(cg or cmc or {}).get("price_change_24h"),"volume_24h":vol,"market_cap":mcap,"sources":list(dict.fromkeys(src))}
        ohlcv_limit = S.FAST_FETCH_OHLCV_LIMIT if fast_flag else S.MODEL_MAX_TRAIN_BARS
        o_tasks=[cc_fetch_ohlcv(symbol,"USD",tf,limit=ohlcv_limit) for tf in S.TIMEFRAMES]
        ohlcv_res=await asyncio.gather(*o_tasks, return_exceptions=True)
        ohlcv={tf:(d if isinstance(d,list) else []) for tf,d in zip(S.TIMEFRAMES, ohlcv_res)}
        news=[]
        try:
            x=await asyncio.wait_for(cryptopanic_news(symbol), timeout=5 if fast_flag else 15)
            if isinstance(x,list): news+=x
        except Exception: pass
        try:
            x=await asyncio.wait_for(newsapi_general(f"{symbol} crypto"), timeout=5 if fast_flag else 15)
            if isinstance(x,list): news+=x
        except Exception: pass
        t_whales = asyncio.create_task(ex_fetch_trades_whales(symbol, lookback_trades=(300 if fast_flag else 500)))
        t_walls  = asyncio.create_task(ex_fetch_orderbook_walls(symbol))
        t_deriv  = asyncio.create_task(ex_fetch_derivatives(symbol))
        try:
            whales = await asyncio.wait_for(t_whales, timeout=S.FAST_WHALES_TIMEOUT_SEC if fast_flag else 20)
        except Exception: whales = {}
        try:
            walls = await asyncio.wait_for(t_walls, timeout=S.FAST_ORDERBOOK_TIMEOUT_SEC if fast_flag else 20)
        except Exception: walls = {}
        try:
            deriv = await asyncio.wait_for(t_deriv, timeout=S.FAST_DERIV_TIMEOUT_SEC if fast_flag else 20)
        except Exception: deriv = {}
        bundle={"market_data":market_data,"ohlcv":ohlcv,"news":news,"whales":whales,"orderbook_walls":walls,"derivatives":deriv,"sources":market_data.get("sources",[])}
        if not price and not any(len(v) for v in ohlcv.values()):
            logger.warning(f"No live data for {symbol}, using offline"); bundle=self.offline_bundle(symbol)
        cache.set(key,bundle); return bundle

# ---------------- Core Bot ----------------
def to_df(ohlcv: List[Dict[str,Any]])->pd.DataFrame:
    if not ohlcv: return pd.DataFrame(columns=["time","open","high","low","close","volume"])
    df=pd.DataFrame(ohlcv).dropna()
    if "time" in df.columns:
        df=df.set_index("time", drop=False)
    return df

class CryptoBotAI:
    def __init__(self):
        self.fetcher=DataFetcher()
        self._train_busy=set()
    async def train_models_bg(self, symbol: str, tf: str, df: pd.DataFrame):
        key=(symbol, tf)
        if key in self._train_busy: return
        self._train_busy.add(key)
        try:
            await asyncio.to_thread(lambda: rf_train_predict(df, symbol=symbol, tf=tf, force_train=True))
            await asyncio.to_thread(lambda: stacking_predict(df, symbol=symbol, timeframe=tf, force_train=True))
        except Exception as e:
            logger.error(f"train_models_bg {symbol} {tf} err: {e}")
        finally:
            if key in self._train_busy: self._train_busy.remove(key)

    async def analyze_symbol(self, symbol: str, fast: Optional[bool]=None, force_train: bool=False)->Dict[str,Any]:
        fast_flag = S.FAST_RESPONSE if fast is None else fast
        bundle=await self.fetcher.fetch_bundle(symbol, fast=fast_flag)
        md=bundle.get("market_data",{})
        ohlcv=bundle.get("ohlcv",{})
        news=bundle.get("news",[])
        whales=bundle.get("whales",{})
        walls=bundle.get("orderbook_walls",{})
        derivatives=bundle.get("derivatives",{})

        sentiment=analyze_sentiment_news(news)

        tf_results={}
        for tf, o in ohlcv.items():
            df=to_df(o)
            if df.empty or len(df)<80: continue
            df=df.tail(min(len(df), S.MODEL_MAX_TRAIN_BARS))
            tech=compute_technical_summary(df)
            techx=compute_advanced_technicals(df) if S.ENABLE_ADVANCED_TECHNICALS else {}
            _=detect_candlestick_patterns(df)
            ell=analyze_elliott(df, deviation=0.04 if tf in ["1h","4h"] else 0.06)
            ms=detect_order_blocks(df)
            ses=analyze_sessions(df) if tf=="1h" else {}

            rf_path=f"rfcal_{symbol.upper()}_{tf}.joblib"
            stk_path=f"stack_{symbol.upper()}_{tf}.joblib"
            need_bg = fast_flag and (not os.path.exists(rf_path) or not os.path.exists(stk_path))
            if need_bg:
                asyncio.create_task(self.train_models_bg(symbol.upper(), tf, df))
            reg=regime_detection(df)
            ae=auto_ensemble_prob(df, symbol=symbol.upper(), tf=tf,
                                  sentiment_value=sentiment.get("average_sentiment",0.0),
                                  regime=reg.get("regime","unknown"),
                                  whales_bias=whales.get("bias",0.0),
                                  derivatives=derivatives,
                                  force_train=force_train)
            stk=stacking_predict(df, symbol=symbol.upper(), timeframe=tf, force_train=force_train)
            ae["prob_up"]=float(max(0.0,min(1.0, 0.6*ae["prob_up"]+0.4*stk["prob_up"])))
            ae["model_acc"]=float(max(ae["model_acc"], stk["model_acc"]))

            bt = backtest_strategies(df) if S.ENABLE_BACKTEST else {}
            tf_results[tf]={"technical":tech,"technical_advanced":techx,"elliott":ell,"market_structure":ms,"sessions":ses,"quant":ae,"backtest":bt,"df":df}

        pick=lambda tf,key: tf_results.get(tf,{}).get(key) or tf_results.get("4h",{}).get(key) or tf_results.get("1h",{}).get(key) or {}
        tech_final=pick("1d","technical")
        techx_final=pick("1d","technical_advanced")
        ell_final=pick("1d","elliott")
        ms_final=pick("4h","market_structure")
        ses_final=pick("1h","sessions")

        probs=[]; parts=[]
        for tf in ["1d","4h","1h"]:
            if tf not in tf_results: continue
            q=tf_results[tf]["quant"]; wtf={"1d":0.5,"4h":0.35,"1h":0.15}.get(tf,0.2)
            probs.append((q["prob_up"], q["model_acc"], wtf)); parts.append((tf, q))
        if probs:
            num=sum(p*w*a for p,a,w in probs); den=sum(w*a for _,a,w in probs)+1e-9
            final_prob=num/den; final_acc=sum(a*w for _,a,w in probs)/sum(w for _,_,w in probs)
        else:
            final_prob=0.5; final_acc=0.6

        df_base = tf_results.get("1d",{}).get("df") or tf_results.get("4h",{}).get("df") or pd.DataFrame()
        risk={}
        if not df_base.empty and md.get("price"):
            risk=risk_management_calc(df_base, float(md["price"]), S.BALANCE, S.RISK_PER_TRADE, prob_up=final_prob, rr_hint=1.8)

        direction=map_score_to_signal(final_prob, final_acc)
        entry_plan={}
        if not df_base.empty and md.get("price") and risk.get("atr"):
            entry_plan=entry_exit_rules(float(md["price"]), df_base, walls, direction, risk.get("atr",0.0))

        if whales and whales.get("flows_by_exchange"):
            try:
                total=float(whales.get("total_notional",0)); bias=float(whales.get("bias",0))
                if total>S.WHALE_ALERT_NOTIONAL and abs(bias)>S.WHALE_ALERT_BIAS:
                    flows=whales.get("flows_by_exchange",{})
                    best_ex=None; best_val=0.0
                    for ex, fv in flows.items():
                        val=fv.get("buy",0)-fv.get("sell",0)
                        if abs(val)>abs(best_val): best_ex=ex; best_val=val
                    if best_ex:
                        db_log_whale_event(int(time.time()), best_ex, symbol.upper(), bias, total)
            except Exception as e:
                logger.error(f"log whale event err: {e}")

        quant_final={"prob_up":final_prob,"model_acc":final_acc,"regime":regime_detection(tf_results.get("4h",{}).get("df", pd.DataFrame()) or df_base).get("regime","unknown"),"regime_conf":regime_detection(tf_results.get("4h",{}).get("df", pd.DataFrame()) or df_base).get("confidence",0.0),"parts":{tf:ae for tf,ae in parts},"strategies":(tf_results.get("4h",{}).get("quant",{}).get("strategies",{}) if tf_results else {})}
        analysis={"symbol":symbol.upper(),"market_data":md,"technical":tech_final,"technical_advanced":techx_final,"elliott":ell_final,"market_structure":ms_final,"sentiment":sentiment,"sessions":ses_final,"quant":quant_final,"risk_management":risk,"timeframes":{tf:{k:v for k,v in tf_results[tf].items() if k!='df'} for tf in tf_results},"whales":whales,"orderbook_walls":walls,"derivatives":derivatives}
        analysis["signal"]=direction; analysis["confidence"]=final_prob; analysis["entry_plan"]=entry_plan
        return analysis

    async def get_trading_signals(self, fast: Optional[bool]=None)->List[Dict[str,Any]]:
        fast_flag = S.FAST_RESPONSE if fast is None else fast
        universe=await discover_universe()
        symbols=[c["symbol"] for c in universe]
        if S.MONITOR_TOPN and len(symbols)>S.MONITOR_TOPN:
            symbols=symbols[:S.MONITOR_TOPN]
        sem=asyncio.Semaphore(S.CONCURRENT_REQUESTS)
        out=[]
        async def run(sym):
            async with sem:
                try:
                    a=await self.analyze_symbol(sym, fast=fast_flag, force_train=False)
                    out.append({"symbol":sym,"signal":a.get("signal","HOLD"),"confidence":a.get("confidence",0.5), "price": a.get("market_data",{}).get("price")})
                except Exception as e:
                    logger.error(f"Signal error {sym}: {e}")
                    out.append({"symbol":sym,"signal":"HOLD","confidence":0.5})
        await asyncio.gather(*[run(s) for s in symbols])
        return out

# ---------------- Evaluation & Reports ----------------
async def evaluate_logged_signals_async(horizon_hours: int = None) -> Dict[str, Any]:
    if horizon_hours is None:
        horizon_hours = S.PERFORMANCE_HORIZON_HOURS
    cutoff_ts = int(time.time()) - horizon_hours*3600
    events = db_get_pending_signals(cutoff_ts)
    if not events: return {"evaluated": 0, "hit": 0.0, "avg_ret": 0.0, "pf": 0.0, "count": 0}
    rets=[]
    for ev in events:
        symbol=ev["symbol"]; side=ev["signal"]; entry_price=ev["price"]; id_=ev["id"]
        price_now=None
        try:
            m = await cg_fetch_market_by_symbol(symbol); price_now = m.get("price")
        except Exception: pass
        if not price_now:
            try:
                arr = await cc_fetch_ohlcv(symbol, "USD", "1h", limit=2); price_now = arr[-1]["close"] if arr else entry_price
            except Exception:
                price_now=entry_price
        if entry_price and price_now:
            ret = (price_now - entry_price)/entry_price if side=="BUY" else (entry_price - price_now)/entry_price if side=="SELL" else 0.0
            db_set_signal_evaluated(id_, float(ret)); rets.append(ret)
    if not rets:
        return {"evaluated": 0, "hit": 0.0, "avg_ret": 0.0, "pf": 0.0, "count": 0}
    hit = float((np.array(rets) > 0).mean())
    pos = float(np.sum([r for r in rets if r > 0])); neg = float(-np.sum([r for r in rets if r < 0])); pf = (pos/neg) if neg>0 else float("inf")
    return {"evaluated": len(rets), "hit": hit, "avg_ret": float(np.mean(rets)), "pf": pf, "count": len(rets)}

async def evaluate_whales_async(horizon_hours: int = None):
    if horizon_hours is None:
        horizon_hours = S.PERFORMANCE_HORIZON_HOURS
    cutoff_ts = int(time.time()) - horizon_hours*3600
    events = db_get_pending_whale_events(cutoff_ts)
    if not events: return
    for ev in events:
        id_, sym, ex = ev["id"], ev["symbol"], ev["exchange"]
        price_then=None; price_now=None
        try:
            m = await cg_fetch_market_by_symbol(sym); price_now = m.get("price")
        except Exception: pass
        if not price_now:
            try:
                arr2 = await cc_fetch_ohlcv(sym, "USD", "1h", limit=2); price_now = arr2[-1]["close"] if arr2 else None
            except Exception: pass
        if price_now is None: 
            db_set_whale_event_evaluated(id_, 0.0); continue
        try:
            arrd = await cc_fetch_ohlcv(sym, "USD", "1d", limit=2)
            price_then = arrd[-2]["close"] if arrd and len(arrd)>=2 else price_now
        except Exception:
            price_then = price_now
        ret = (price_now - price_then)/price_then if price_then else 0.0
        db_set_whale_event_evaluated(id_, float(ret))
        db_update_whale_perf(ex, sym, float(ret))

# ---------------- Realtime Monitor ----------------
async def monitor_job(context: ContextTypes.DEFAULT_TYPE):
    app = context.application
    if not app.bot_data.get("monitor_enabled", S.ENABLE_MONITOR):
        return
    bot_instance: CryptoBotAI = app.bot_data.get("bot_instance")
    if not bot_instance: return
    try:
        uni = app.bot_data.get("universe")
        if not uni:
            uni = await discover_universe()
            symbols=[c["symbol"] for c in uni]
            app.bot_data["universe"]=symbols
            app.bot_data["monitor_offset"]=0
        symbols = app.bot_data["universe"]
        offset = int(app.bot_data.get("monitor_offset",0))
        topn_runtime = int(app.bot_data.get("monitor_topn", S.MONITOR_TOPN))
        batch_size = min(topn_runtime, 40)
        batch = symbols[offset:offset+batch_size]
        if not batch: 
            offset = 0; batch = symbols[:batch_size]
        app.bot_data["monitor_offset"]=offset+batch_size
        subs = db_get_subscribers()
        if not subs: return
        now=int(time.time())
        for sym in batch:
            try:
                analysis = await bot_instance.analyze_symbol(sym)
                prob = analysis.get("quant",{}).get("prob_up",0.5)
                tf="4h"; t_buy, t_sell = db_get_thresholds(sym, tf)
                if t_buy is None: t_buy=0.62
                if t_sell is None: t_sell=0.38
                last_alert = db_get_last_alert(sym) or 0
                if now - last_alert < S.ALERT_COOLDOWN_MIN*60:
                    continue
                to_send = None
                if prob>=t_buy and analysis.get("signal")=="BUY":
                    to_send = f"🚀 سیگنال BUY روی {sym}: P={prob:.2f}"
                elif prob<=t_sell and analysis.get("signal")=="SELL":
                    to_send = f"⚠️ سیگنال SELL روی {sym}: P={prob:.2f}"
                if to_send:
                    md=analysis.get("market_data",{}); price=md.get("price")
                    db_log_signal(now, sym, analysis.get("signal","HOLD"), float(prob), float(price) if price else None, "monitor")
                    db_set_last_alert(sym, now)
                    full = format_analysis(analysis)
                    for chat_id in subs:
                        await app.bot.send_message(chat_id=chat_id, text=to_send)
                        for part in split_message(full): await app.bot.send_message(chat_id=chat_id, text=part)
            except Exception as e:
                logger.error(f"monitor {sym} err: {e}")
    except Exception as e:
        logger.error(f"monitor job error: {e}")

# ---------------- WebSocket (Coinbase) ----------------
class WSManager:
    def __init__(self, app):
        self.app = app
        self.ws_task = None
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.running = False
        self.subscribed = set()

    async def _ws_coinbase(self, symbols: List[str]):
        url = "wss://ws-feed.exchange.coinbase.com"
        product_ids = [f"{s}-USD" for s in symbols[:100]]
        msg_sub = {"type":"subscribe","channels":[{"name":"ticker","product_ids": product_ids}]}
        while self.running:
            try:
                async with websockets.connect(url, max_queue=1000, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(_json.dumps(msg_sub))
                    self.subscribed = set(product_ids)
                    while self.running:
                        raw = await ws.recv()
                        data = _json.loads(raw)
                        if data.get("type")=="ticker" and data.get("product_id"):
                            pid = data["product_id"]
                            sym = pid.split("-")[0]
                            await self.queue.put(sym)
            except Exception as e:
                logger.error(f"WS coinbase error: {e}")
                await asyncio.sleep(5)
                continue

    async def consumer(self, bot_instance: 'CryptoBotAI'):
        while self.running:
            subs = db_get_subscribers()
            if not subs:
                await asyncio.sleep(5)
                continue
            sym = await self.queue.get()
            now=int(time.time())
            last_alert = db_get_last_alert(sym) or 0
            if now - last_alert < S.ALERT_COOLDOWN_MIN*60:
                continue
            try:
                analysis = await bot_instance.analyze_symbol(sym)
                prob = analysis.get("quant",{}).get("prob_up",0.5)
                t_buy, t_sell = db_get_thresholds(sym, "4h")
                if t_buy is None: t_buy=0.62
                if t_sell is None: t_sell=0.38
                to_send=None
                if prob>=t_buy and analysis.get("signal")=="BUY":
                    to_send=f"🚀 [WS] BUY {sym}: P={prob:.2f}"
                elif prob<=t_sell and analysis.get("signal")=="SELL":
                    to_send=f"⚠️ [WS] SELL {sym}: P={prob:.2f}"
                if to_send:
                    price=analysis.get("market_data",{}).get("price")
                    db_log_signal(now, sym, analysis.get("signal","HOLD"), float(prob), float(price) if price else None, "ws")
                    db_set_last_alert(sym, now)
                    full=format_analysis(analysis)
                    for chat_id in db_get_subscribers():
                        await self.app.bot.send_message(chat_id=chat_id, text=to_send)
                        for part in split_message(full): await self.app.bot.send_message(chat_id=chat_id, text=part)
            except Exception as e:
                logger.error(f"WS consumer {sym} err: {e}")

    async def start(self, bot_instance: 'CryptoBotAI', symbols: List[str]):
        if self.running: return
        self.running = True
        self.ws_task = asyncio.create_task(self._ws_coinbase(symbols))
        asyncio.create_task(self.consumer(bot_instance))
        logger.info("WS Manager started (Coinbase ticker).")

    async def stop(self):
        self.running=False
        if self.ws_task:
            self.ws_task.cancel()
            self.ws_task=None
        logger.info("WS Manager stopped.")

# ---------------- HTTP Dashboard ----------------
async def http_app_factory():
    async def health(request):
        return web.json_response({"status":"ok","time": int(time.time())})

    async def stats(request):
        subs=len(db_get_subscribers())
        recent=db_recent_signals(200)
        hits=[r["ret"] for r in recent if r["evaluated"]==1]
        hit_rate=float((np.array(hits)>0).mean()) if hits else None
        return web.json_response({"subscribers":subs, "recent_signals":len(recent), "hit_rate":hit_rate})

    async def best_whales(request):
        rows=db_get_best_whales(20)
        data=[{"exchange":r["exchange"],"symbol":r["symbol"],"count":r["count"],"hit_rate":r["hit_rate"],"avg_ret":r["avg_ret"]} for r in rows]
        return web.json_response(data)

    async def recent_signals(request):
        lim=int(request.query.get("limit","50"))
        rows=db_recent_signals(lim)
        data=[dict(r) for r in rows]
        return web.json_response(data)

    async def watchlist(request):
        wl=db_get_watchlist()
        return web.json_response(wl)

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/stats", stats)
    app.router.add_get("/best-whales", best_whales)
    app.router.add_get("/signals/recent", recent_signals)
    app.router.add_get("/watchlist", watchlist)
    return app

async def start_http_server():
    if not S.ENABLE_HTTP: return
    app = await http_app_factory()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", S.PORT)
    await site.start()
    logger.info(f"HTTP Dashboard running on 0.0.0.0:{S.PORT}")

# ---------------- Telegram integration ----------------
def split_message(text: str, limit: int = 3900) -> List[str]:
    if len(text) <= limit:
        return [text]
    parts=[]
    while len(text) > limit:
        idx = text.rfind("\n", 0, limit)
        if idx == -1: idx = limit
        parts.append(text[:idx]); text = text[idx:]
    if text: parts.append(text)
    return parts

def build_help() -> str:
    return (
    "📘 راهنمای ربات:\n"
    "دستورات ساده:\n"
    "• /menu — منوی سریع (دکمه‌ها)\n"
    "• /a SYMBOL — تحلیل (نمونه: /a BTC)\n"
    "• /sig — برترین سیگنال‌ها\n"
    "• /opp [N] — آخرین فرصت‌ها\n"
    "• /auto on [min] [topN] | off — خودکار 24/7\n"
    "• /mon on [sec] [topN] | off — مانیتور لحظه‌ای\n"
    "• /wh — بهترین نهنگ‌ها\n"
    "• /bt SYMBOL [tf] [h] — بک‌تست\n"
    "• /wl — واچ‌لیست | /add SYMBOL | /rm SYMBOL\n"
    "\nدستورات کامل قبلی نیز فعال هستند: /analyze, /signals, /opportunities, /autopilot, /monitor, /whales, /backtest, /watchadd, /watchrm, /watchlist, /report\n"
    )

WELCOME_BANNER = (
    "╔══════════════════════════════════════╗\n"
    "║   ⚡️ Hyper Crypto AI Bot 24/7 ⚡️    ║\n"
    "║  ML • Whales • Derivatives • Alpha  ║\n"
    "╚══════════════════════════════════════╝\n"
)

def make_menu_keyboard() -> InlineKeyboardMarkup:
    kb = [
        [InlineKeyboardButton("🔥 سیگنال‌ها", callback_data="m:signals"),
         InlineKeyboardButton("🧪 فرصت‌ها", callback_data="m:opp")],
        [InlineKeyboardButton("📊 تحلیل BTC", callback_data="m:a:BTC"),
         InlineKeyboardButton("📊 تحلیل ETH", callback_data="m:a:ETH")],
        [InlineKeyboardButton("🤖 Autopilot ON", callback_data="m:auto:on"),
         InlineKeyboardButton("⛔️ Autopilot OFF", callback_data="m:auto:off")],
        [InlineKeyboardButton("📡 Monitor ON", callback_data="m:mon:on"),
         InlineKeyboardButton("🛑 Monitor OFF", callback_data="m:mon:off")],
        [InlineKeyboardButton("🐋 نهنگ‌ها", callback_data="m:whales"),
         InlineKeyboardButton("🆘 راهنما", callback_data="m:help")],
    ]
    return InlineKeyboardMarkup(kb)

async def typing_loop(bot, chat_id: int, stop_event: asyncio.Event, interval: int = 4):
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING if ChatAction else "typing")
        except Exception:
            pass
        await asyncio.sleep(interval)

async def run_cli():
    bot=CryptoBotAI()
    analysis=await bot.analyze_symbol(S.SYMBOL, fast=S.FAST_RESPONSE, force_train=False)
    print(format_analysis(analysis))
    if S.PRINT_SIGNALS:
        sigs=await bot.get_trading_signals(fast=S.FAST_RESPONSE)
        top=sorted(sigs, key=lambda x: x["confidence"], reverse=True)[:10]
        print("\n🔥 برترین سیگنال‌ها:")
        for s in top:
            print(f"  • {s['symbol']}: {s['signal']} ({s['confidence']:.1%})")

# ---------------- Research / Autoscan / Whales / News Jobs ----------------
async def autoscan_job(context: ContextTypes.DEFAULT_TYPE):
    app = context.application
    if not app.bot_data.get("autopilot_enabled", S.ENABLE_AUTOSCAN):
        return
    bot_instance: CryptoBotAI = app.bot_data.get("bot_instance")
    if not bot_instance: return
    try:
        uni = app.bot_data.get("universe")
        if not uni:
            u = await discover_universe(); uni=[c["symbol"] for c in u]
            app.bot_data["universe"]=uni
        topn = int(app.bot_data.get("research_topn", S.RESEARCH_TOPN))
        symbols = uni[:topn]
        sem=asyncio.Semaphore(4)
        results=[]
        async def run(sym):
            async with sem:
                try:
                    a = await bot_instance.analyze_symbol(sym, fast=False, force_train=True)
                    results.append((sym, a))
                except Exception as e:
                    logger.error(f"autoscan {sym} err: {e}")
        await asyncio.gather(*[run(s) for s in symbols])
        if not results: return
        ranked = []
        for sym, a in results:
            prob = a.get("quant",{}).get("prob_up",0.5)
            rr = a.get("risk_management",{}).get("risk_reward_ratio",0.0)
            ranked.append((sym, prob, rr, a))
        ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
        picks = ranked[:min(10, len(ranked))]
        subs = db_get_subscribers()
        ts=int(time.time())
        lines=["🧪 فرصت‌های پژوهش خودکار:"]
        for sym, prob, rr, a in picks:
            r=a.get("risk_management",{})
            md=a.get("market_data",{})
            entry=a.get("entry_plan",{}).get("entry", md.get("price"))
            lev=r.get("leverage",1.0)
            db_insert_research_opp(ts, sym, "mix", a.get("signal","HOLD"), float(prob), float(rr),
                                   float(entry) if entry else 0.0, float(r.get("stop_loss") or 0.0),
                                   float(r.get("take_profit_1") or 0.0), float(r.get("take_profit_2") or 0.0),
                                   float(lev), notes="autoscan")
            lines.append(f"• {sym}: {a.get('signal','HOLD')} | P={prob:.2f} | R/R={rr:.2f} | Lev x{lev:.1f}")
        msg="\n".join(lines)
        for chat_id in subs:
            await app.bot.send_message(chat_id=chat_id, text=msg)
    except Exception as e:
        logger.error(f"autoscan job error: {e}")

async def whale_watch_job(context: ContextTypes.DEFAULT_TYPE):
    app = context.application
    bot_instance: CryptoBotAI = app.bot_data.get("bot_instance")
    if not bot_instance: return
    try:
        uni = app.bot_data.get("universe")
        if not uni:
            u = await discover_universe(); uni=[c["symbol"] for c in u]
            app.bot_data["universe"]=uni
        symbols = uni[:S.WHALE_WATCH_TOPN]
        sem=asyncio.Semaphore(6)
        alerts=[]
        async def run(sym):
            async with sem:
                w = await ex_fetch_trades_whales(sym, lookback_trades=400, usd_threshold=S.WHALE_ALERT_NOTIONAL/10)
                if w and w.get("total_notional",0)>S.WHALE_ALERT_NOTIONAL and abs(w.get("bias",0))>S.WHALE_ALERT_BIAS:
                    side="BUY" if w["bias"]>0 else "SELL"
                    db_log_whale_event(int(time.time()), "multi", sym, float(w["bias"]), float(w["total_notional"]))
                    alerts.append((sym, side, w))
        await asyncio.gather(*[run(s) for s in symbols])
        if alerts:
            subs=db_get_subscribers()
            for sym, side, w in alerts:
                flows=w.get("flows_by_exchange",{})
                flows_line=" | ".join([f"{ex}: B${fmt_num(v['buy'],0)}/S${fmt_num(v['sell'],0)}" for ex,v in list(flows.items())[:4]])
                text=f"🐋 نهنگ‌ها ({side}) روی {sym}: Bias={w['bias']:+.2f} | Notional=${fmt_num(w['total_notional'],0)}\n{flows_line}"
                for chat_id in subs:
                    await app.bot.send_message(chat_id=chat_id, text=text)
    except Exception as e:
        logger.error(f"whale_watch job error: {e}")

async def news_digest_job(context: ContextTypes.DEFAULT_TYPE):
    app = context.application
    try:
        wl = db_get_watchlist()
        if not wl:
            uni = await discover_universe(); wl=[c["symbol"] for c in uni[:10]]
        items=[]
        for s in wl[:10]:
            try:
                n = await cryptopanic_news(s, filter_="hot", limit=10)
                items += [{"symbol": s, "title": i.get("title"), "url": i.get("url"), "src": i.get("source")} for i in n[:3]]
            except Exception: pass
        if items:
            subs=db_get_subscribers()
            lines=["📰 خلاصه خبرهای داغ:"]
            for it in items[:20]:
                lines.append(f"• [{it['symbol']}] {it['title']} ({it['src']})")
            msg="\n".join(lines)
            for chat_id in subs:
                await app.bot.send_message(chat_id=chat_id, text=msg)
    except Exception as e:
        logger.error(f"news_digest job error: {e}")

# ---------------- Telegram Commands (simplified + full) ----------------
def run_telegram():
    if not TELEGRAM_AVAILABLE or not S.TELEGRAM_BOT_TOKEN:
        logger.error("Telegram not available or TELEGRAM_BOT_TOKEN missing. Running CLI + HTTP.")
        async def main():
            if S.ENABLE_HTTP:
                await start_http_server()
            await run_cli()
        asyncio.run(main())
        return

    bot = CryptoBotAI()
    wsman = WSManager(None)  # will set app later

    async def send_welcome(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
        text = WELCOME_BANNER + "به ربات تحلیل هوشمند کریپتو خوش آمدید.\nاز /menu استفاده کنید یا دستور /help را ببینید."
        await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=make_menu_keyboard())

    async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        db_add_subscriber(update.effective_chat.id)
        await send_welcome(update.effective_chat.id, context)

    async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🧭 منوی سریع:", reply_markup=make_menu_keyboard())

    async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(build_help())

    async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("✅ Bot is alive.")

    async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
        db_add_subscriber(update.effective_chat.id)
        await update.message.reply_text("✅ اشتراک ارسال خودکار فعال شد.")

    async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
        db_remove_subscriber(update.effective_chat.id)
        await update.message.reply_text("✅ اشتراک شما لغو شد.")

    async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not context.args:
                await update.message.reply_text("نمونه: /analyze BTC"); return
            symbol = context.args[0].upper()
            ack = await update.message.reply_text(f"🟡 درخواست شما ثبت شد. در حال پردازش {symbol} با حداکثر دقت...")
            chat_id = update.effective_chat.id
            stop_event = asyncio.Event()
            asyncio.create_task(typing_loop(context.bot, chat_id, stop_event))
            async def do_work():
                try:
                    analysis = await bot.analyze_symbol(symbol, fast=False, force_train=True)
                    md=analysis.get("market_data",{}); price=md.get("price")
                    db_log_signal(int(time.time()), symbol, analysis.get("signal","HOLD"), float(analysis.get("confidence",0.5)), float(price) if price else None, "manual")
                    await context.bot.edit_message_text(chat_id=chat_id, message_id=ack.message_id, text=f"✅ نتیجه آماده شد: {symbol}")
                    text = format_analysis(analysis)
                    for part in split_message(text): await context.bot.send_message(chat_id=chat_id, text=part)
                except Exception as e:
                    await context.bot.edit_message_text(chat_id=chat_id, message_id=ack.message_id, text=f"⛔️ خطا در تحلیل {symbol}: {e}")
                finally:
                    stop_event.set()
            asyncio.create_task(do_work())
        except Exception as e:
            logger.error(f"/analyze error: {e}"); await update.message.reply_text(f"خطا در تحلیل: {e}")

    async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            ack = await update.message.reply_text("🟡 درخواست شما ثبت شد. در حال تولید سیگنال‌ها با دقت بالا...")
            chat_id = update.effective_chat.id
            stop_event = asyncio.Event()
            asyncio.create_task(typing_loop(context.bot, chat_id, stop_event))
            async def do_work():
                try:
                    quick = await bot.get_trading_signals(fast=True)
                    if not quick:
                        await context.bot.edit_message_text(chat_id=chat_id, message_id=ack.message_id, text="⛔️ سیگنالی در دسترس نیست."); stop_event.set(); return
                    top = sorted(quick, key=lambda x: x["confidence"], reverse=True)[:20]
                    lines = ["🔥 برترین سیگنال‌ها:"]
                    for s in top:
                        analysis = await bot.analyze_symbol(s["symbol"], fast=False, force_train=True)
                        price = analysis.get("market_data",{}).get("price")
                        price_str = fmt_num(price, 4)
                        conf = analysis.get("confidence", 0.5)
                        lines.append(f"• {s['symbol']}: {analysis.get('signal','HOLD')} ({conf:.1%}) | قیمت: {price_str}")
                        db_log_signal(int(time.time()), s["symbol"], analysis.get("signal","HOLD"), float(conf), float(price) if price else None, "manual")
                    await context.bot.edit_message_text(chat_id=chat_id, message_id=ack.message_id, text="✅ سیگنال‌ها آماده شدند.")
                    for part in split_message("\n".join(lines)): await context.bot.send_message(chat_id=chat_id, text=part)
                    await context.bot.send_message(chat_id=chat_id, text="ℹ️ برای جزئیات هر نماد: /a SYMBOL")
                except Exception as e:
                    await context.bot.edit_message_text(chat_id=chat_id, message_id=ack.message_id, text=f"⛔️ خطا در دریافت سیگنال‌ها: {e}")
                finally:
                    stop_event.set()
            asyncio.create_task(do_work())
        except Exception as e:
            logger.error(f"/signals error: {e}"); await update.message.reply_text(f"خطا در دریافت سیگنال‌ها: {e}")

    async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            hours = int(context.args[0]) if context.args else S.PERFORMANCE_HORIZON_HOURS
            await update.message.reply_text(f"⏳ در حال تهیه گزارش {hours}h ...")
            await evaluate_whales_async(hours)
            stats = await evaluate_logged_signals_async(hours)
            if stats.get("evaluated",0)==0:
                await update.message.reply_text(f"📊 گزارش ({hours}h): داده کافی نیست."); return
            text = (f"📊 گزارش ({hours}h)\n"
                    f"- ارزیابی شده: {stats['evaluated']}\n"
                    f"- Hit: {stats['hit']:.2f}\n"
                    f"- AvgRet: {stats['avg_ret']:+.4f}\n"
                    f"- PF: {stats['pf']:.2f}")
            await update.message.reply_text(text)
        except Exception as e:
            await update.message.reply_text(f"خطا در گزارش: {e}")

    async def cmd_watchadd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args: await update.message.reply_text("نمونه: /watchadd BTC"); return
        db_add_watch(context.args[0].upper()); await update.message.reply_text("✅ به واچ‌لیست اضافه شد.")

    async def cmd_watchrm(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args: await update.message.reply_text("نمونه: /watchrm BTC"); return
        db_remove_watch(context.args[0].upper()); await update.message.reply_text("✅ از واچ‌لیست حذف شد.")

    async def cmd_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
        wl = db_get_watchlist(); await update.message.reply_text("📜 واچ‌لیست:\n" + (", ".join(wl) if wl else "خالی"))

    async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not context.args: await update.message.reply_text("نمونه: /backtest BTC 4h 3"); return
            sym = context.args[0].upper(); tf = context.args[1] if len(context.args) > 1 else "4h"; horizon = int(context.args[2]) if len(context.args) > 2 else 3
            ack = await update.message.reply_text(f"🟡 بک‌تست {sym} ({tf}) ثبت شد؛ با دقت بالا محاسبه می‌شود...")
            chat_id = update.effective_chat.id
            stop_event = asyncio.Event()
            asyncio.create_task(typing_loop(context.bot, chat_id, stop_event))
            async def do_work():
                try:
                    bundle = await bot.fetcher.fetch_bundle(sym, fast=False)
                    df = to_df(bundle.get("ohlcv",{}).get(tf, []))
                    if df.empty:
                        await context.bot.edit_message_text(chat_id=chat_id, message_id=ack.message_id, text="⛔️ داده کافی نیست."); return
                    bt = backtest_strategies(df, horizon=horizon)
                    await context.bot.edit_message_text(chat_id=chat_id, message_id=ack.message_id, text=f"✅ بک‌تست آماده شد.")
                    await context.bot.send_message(chat_id=chat_id, text=f"🧪 نتیجه بک‌تست {sym} ({tf}):\nSharpe {bt['sharpe']:+.2f} | CAGR {bt['cagr']:+.2f}% | MaxDD {bt['max_dd']:.2f}")
                except Exception as e:
                    await context.bot.edit_message_text(chat_id=chat_id, message_id=ack.message_id, text=f"⛔️ خطا: {e}")
                finally:
                    stop_event.set()
            asyncio.create_task(do_work())
        except Exception as e:
            await update.message.reply_text(f"خطا: {e}")

    async def cmd_whales(update: Update, context: ContextTypes.DEFAULT_TYPE):
        rows = db_get_best_whales(limit=10)
        if not rows: await update.message.reply_text("هنوز داده‌ای از عملکرد نهنگ‌ها ثبت نشده."); return
        lines = ["🐋 بهترین نهنگ‌ها (صرافی/نماد):"]
        lines += [f"- {r['exchange']}/{r['symbol']}: Hit {r['hit_rate']:.2f} | AvgRet {r['avg_ret']:+.4f} | n={r['count']}" for r in rows]
        await update.message.reply_text("\n".join(lines))

    async def cmd_opportunities(update: Update, context: ContextTypes.DEFAULT_TYPE):
        topn = int(context.args[0]) if context.args else 10
        rows = db_recent_opportunities(limit=topn)
        if not rows:
            await update.message.reply_text("فرصت پژوهشی ثبت‌شده‌ای وجود ندارد."); return
        lines=["🧪 آخرین فرصت‌های پژوهش:"]
        for r in rows:
            lines.append(f"• {r['symbol']} ({r['timeframe']}): {r['signal']} | P={r['prob']:.2f} | R/R={r['rr']:.2f} | Lev x{r['leverage']:.1f}")
        for part in split_message("\n".join(lines)): await update.message.reply_text(part)

    STATE = {
        "autosignals": S.ENABLE_AUTOSIGNALS,
        "interval": S.AUTO_SIGNAL_INTERVAL_MINUTES,
        "topn": S.AUTO_SIGNAL_TOPN,
        "monitor": S.ENABLE_MONITOR,
        "monitor_interval": S.MONITOR_INTERVAL_SEC,
        "monitor_topn": S.MONITOR_TOPN,
        "autopilot": S.ENABLE_AUTOSCAN,
        "research_topn": S.RESEARCH_TOPN
    }

    async def autosignals_job(context: ContextTypes.DEFAULT_TYPE):
        app = context.application
        bot_instance: CryptoBotAI = app.bot_data.get("bot_instance")
        if not STATE.get("autosignals"):
            return
        signals = await bot_instance.get_trading_signals()
        if signals:
            subs = db_get_subscribers()
            top = sorted(signals, key=lambda x: x["confidence"], reverse=True)[:STATE.get("topn",10)]
            lines=["🔥 برترین سیگنال‌ها (Auto):"]
            for s in top:
                analysis = await bot_instance.analyze_symbol(s["symbol"])
                price = analysis.get("market_data",{}).get("price")
                price_str = fmt_num(price, 4)
                conf = analysis.get("confidence", 0.5)
                lines.append(f"• {s['symbol']}: {analysis.get('signal','HOLD')} ({conf:.1%}) | قیمت: {price_str}")
                db_log_signal(int(time.time()), s["symbol"], analysis.get("signal","HOLD"), float(conf), float(price) if price else None, "auto")
            msg="\n".join(lines)
            for chat_id in subs:
                await app.bot.send_message(chat_id=chat_id, text=msg)

    async def daily_report_job(context: ContextTypes.DEFAULT_TYPE):
        app = context.application
        await evaluate_whales_async(S.PERFORMANCE_HORIZON_HOURS)
        stats = await evaluate_logged_signals_async(S.PERFORMANCE_HORIZON_HOURS)
        subs = db_get_subscribers()
        if stats.get("evaluated",0)==0:
            text = f"📊 گزارش ({S.PERFORMANCE_HORIZON_HOURS}h): داده کافی نیست."
        else:
            text = (f"📊 گزارش ({S.PERFORMANCE_HORIZON_HOURS}h)\n"
                    f"- ارزیابی شده: {stats['evaluated']}\n"
                    f"- Hit: {stats['hit']:.2f}\n"
                    f"- AvgRet: {stats['avg_ret']:+.4f}\n"
                    f"- PF: {stats['pf']:.2f}")
        rows=db_get_best_whales(10)
        if rows:
            text += "\n🐋 بهترین نهنگ‌ها:\n" + "\n".join([f"- {r['exchange']}/{r['symbol']}: Hit {r['hit_rate']:.2f} | AvgRet {r['avg_ret']:+.4f} | n={r['count']}" for r in rows])
        for chat_id in subs: await app.bot.send_message(chat_id=chat_id, text=text)

    async def warmup_job(context: ContextTypes.DEFAULT_TYPE):
        if not S.MODEL_WARMUP_ON_START: return
        app = context.application
        bot_instance: CryptoBotAI = app.bot_data.get("bot_instance")
        try:
            uni = await discover_universe()
            syms = [c["symbol"] for c in uni][:S.MODEL_WARMUP_TOPN]
            sem = asyncio.Semaphore(3)
            async def run(sym):
                async with sem:
                    try:
                        bundle = await bot_instance.fetcher.fetch_bundle(sym, fast=True)
                        for tf in S.TIMEFRAMES:
                            df = to_df(bundle.get("ohlcv",{}).get(tf, []))
                            if not df.empty:
                                await bot_instance.train_models_bg(sym, tf, df)
                    except Exception as e:
                        logger.error(f"warmup {sym} err: {e}")
            await asyncio.gather(*[run(s) for s in syms])
            logger.info("Model warmup done.")
        except Exception as e:
            logger.error(f"warmup job err: {e}")

    async def autoscan_job_wrapper(context: ContextTypes.DEFAULT_TYPE):
        await autoscan_job(context)

    async def whale_watch_job_wrapper(context: ContextTypes.DEFAULT_TYPE):
        await whale_watch_job(context)

    async def news_digest_job_wrapper(context: ContextTypes.DEFAULT_TYPE):
        await news_digest_job(context)

    async def schedule_jobs(app, bot_instance: CryptoBotAI):
        app.bot_data["bot_instance"]=bot_instance
        app.bot_data["monitor_enabled"] = STATE.get("monitor", S.ENABLE_MONITOR)
        app.bot_data["monitor_topn"] = STATE.get("monitor_topn", S.MONITOR_TOPN)
        app.bot_data["autopilot_enabled"] = STATE.get("autopilot", S.ENABLE_AUTOSCAN)
        app.bot_data["research_topn"] = STATE.get("research_topn", S.RESEARCH_TOPN)
        if S.ENABLE_HTTP:
            asyncio.create_task(start_http_server())
        wsman.app = app
        try:
            uni = await discover_universe()
            syms = [c["symbol"] for c in uni][:min(100, len(uni))]
            await wsman.start(bot_instance, syms)
        except Exception as e:
            logger.error(f"WS start error: {e}")
        app.job_queue.run_repeating(autosignals_job, interval=STATE.get("interval",60)*60, first=10, name="autosignals")
        if S.ENABLE_DAILY_REPORT:
            now = datetime.datetime.utcnow()
            target = now.replace(hour=S.DAILY_REPORT_UTC_HOUR, minute=0, second=0, microsecond=0)
            if target <= now: target = target + datetime.timedelta(days=1)
            delay = (target - now).total_seconds()
            app.job_queue.run_repeating(daily_report_job, interval=24*3600, first=delay, name="dailyreport")
        if app.bot_data["monitor_enabled"]:
            app.job_queue.run_repeating(monitor_job, interval=STATE.get("monitor_interval", S.MONITOR_INTERVAL_SEC), first=30, name="monitor")
        app.job_queue.run_repeating(autoscan_job_wrapper, interval=S.AUTOSCAN_INTERVAL_MINUTES*60, first=20, name="autoscan")
        app.job_queue.run_repeating(whale_watch_job_wrapper, interval=S.WHALE_JOB_INTERVAL_SEC, first=25, name="whales")
        app.job_queue.run_repeating(news_digest_job_wrapper, interval=S.NEWS_DIGEST_INTERVAL_MINUTES*60, first=60, name="newsdigest")
        app.job_queue.run_once(warmup_job, when=10, name="warmup")

    async def cmd_autosignals(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not context.args:
                await update.message.reply_text("نمونه: /autosignals on 60 10 یا /autosignals off"); return
            mode=context.args[0].lower()
            app = context.application
            if mode=="on":
                interval=int(context.args[1]) if len(context.args)>1 else STATE["interval"]
                topn=int(context.args[2]) if len(context.args)>2 else STATE["topn"]
                STATE["autosignals"]=True; STATE["interval"]=interval; STATE["topn"]=topn
                for j in app.job_queue.get_jobs_by_name("autosignals"): j.schedule_removal()
                app.job_queue.run_repeating(autosignals_job, interval=interval*60, first=5, name="autosignals")
                await update.message.reply_text(f"✅ ارسال خودکار فعال شد. هر {interval} دقیقه، Top {topn}.")
            elif mode=="off":
                STATE["autosignals"]=False
                for j in app.job_queue.get_jobs_by_name("autosignals"): j.schedule_removal()
                await update.message.reply_text("⛔️ ارسال خودکار غیرفعال شد.")
            else:
                await update.message.reply_text("حالت نامعتبر. از on/off استفاده کنید.")
        except Exception as e:
            await update.message.reply_text(f"خطا: {e}")

    async def cmd_monitor(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not context.args:
                await update.message.reply_text("نمونه: /monitor on 90 100  یا  /monitor off"); return
            mode=context.args[0].lower()
            app = context.application
            if mode=="on":
                interval=int(context.args[1]) if len(context.args)>1 else STATE["monitor_interval"]
                topn=int(context.args[2]) if len(context.args)>2 else STATE.get("monitor_topn", S.MONITOR_TOPN)
                STATE["monitor"]=True; STATE["monitor_interval"]=interval; STATE["monitor_topn"]=topn
                app.bot_data["monitor_enabled"] = True
                app.bot_data["monitor_topn"] = topn
                for j in app.job_queue.get_jobs_by_name("monitor"): j.schedule_removal()
                app.job_queue.run_repeating(monitor_job, interval=interval, first=5, name="monitor")
                await update.message.reply_text(f"✅ مانیتور لحظه‌ای فعال شد. هر {interval}s اسکن. TopN={topn}")
            elif mode=="off":
                STATE["monitor"]=False
                app.bot_data["monitor_enabled"] = False
                for j in app.job_queue.get_jobs_by_name("monitor"): j.schedule_removal()
                await update.message.reply_text("⛔️ مانیتور لحظه‌ای غیرفعال شد.")
            else:
                await update.message.reply_text("حالت نامعتبر. on/off")
        except Exception as e:
            await update.message.reply_text(f"خطا: {e}")

    async def cmd_autopilot(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not context.args:
                await update.message.reply_text("نمونه: /autopilot on 45 80 یا /autopilot off"); return
            mode=context.args[0].lower()
            app=context.application
            if mode=="on":
                interval=int(context.args[1]) if len(context.args)>1 else S.AUTOSCAN_INTERVAL_MINUTES
                topn=int(context.args[2]) if len(context.args)>2 else S.RESEARCH_TOPN
                app.bot_data["autopilot_enabled"]=True; app.bot_data["research_topn"]=topn
                for j in app.job_queue.get_jobs_by_name("autoscan"): j.schedule_removal()
                app.job_queue.run_repeating(autoscan_job, interval=interval*60, first=10, name="autoscan")
                await update.message.reply_text(f"✅ Autopilot فعال شد: هر {interval} دقیقه، TopN={topn}.")
            elif mode=="off":
                app.bot_data["autopilot_enabled"]=False
                for j in app.job_queue.get_jobs_by_name("autoscan"): j.schedule_removal()
                await update.message.reply_text("⛔️ Autopilot غیرفعال شد.")
            else:
                await update.message.reply_text("حالت نامعتبر. on/off")
        except Exception as e:
            await update.message.reply_text(f"خطا: {e}")

    # -------------- Simplified Aliases --------------
    async def cmd_a(update: Update, context: ContextTypes.DEFAULT_TYPE):  # /a BTC
        await cmd_analyze(update, context)
    async def cmd_sig(update: Update, context: ContextTypes.DEFAULT_TYPE):  # /sig
        await cmd_signals(update, context)
    async def cmd_opp(update: Update, context: ContextTypes.DEFAULT_TYPE):  # /opp [N]
        await cmd_opportunities(update, context)
    async def cmd_auto(update: Update, context: ContextTypes.DEFAULT_TYPE):  # /auto on/off ...
        await cmd_autopilot(update, context)
    async def cmd_mon(update: Update, context: ContextTypes.DEFAULT_TYPE):  # /mon on/off ...
        await cmd_monitor(update, context)
    async def cmd_wh(update: Update, context: ContextTypes.DEFAULT_TYPE):  # /wh
        await cmd_whales(update, context)
    async def cmd_bt(update: Update, context: ContextTypes.DEFAULT_TYPE):  # /bt
        await cmd_backtest(update, context)
    async def cmd_wl(update: Update, context: ContextTypes.DEFAULT_TYPE):  # /wl
        await cmd_watchlist(update, context)
    async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE):  # /add SYMBOL
        if not context.args: await update.message.reply_text("نمونه: /add BTC"); return
        db_add_watch(context.args[0].upper()); await update.message.reply_text("✅ به واچ‌لیست اضافه شد.")
    async def cmd_rm(update: Update, context: ContextTypes.DEFAULT_TYPE):  # /rm SYMBOL
        if not context.args: await update.message.reply_text("نمونه: /rm BTC"); return
        db_remove_watch(context.args[0].upper()); await update.message.reply_text("✅ از واچ‌لیست حذف شد.")

    # -------------- Inline Menu Callbacks --------------
    async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data or ""
        chat_id = query.message.chat_id
        # route
        if data == "m:signals":
            fake_update = Update(update.update_id, message=query.message)  # reuse message
            await cmd_signals(fake_update, context)
        elif data == "m:opp":
            rows = db_recent_opportunities(limit=10)
            if not rows:
                await context.bot.send_message(chat_id=chat_id, text="فرصت پژوهشی ثبت‌شده‌ای وجود ندارد.")
            else:
                lines=["🧪 آخرین فرصت‌های پژوهش:"]
                for r in rows:
                    lines.append(f"• {r['symbol']} ({r['timeframe']}): {r['signal']} | P={r['prob']:.2f} | R/R={r['rr']:.2f} | Lev x{r['leverage']:.1f}")
                for part in split_message("\n".join(lines)):
                    await context.bot.send_message(chat_id=chat_id, text=part)
        elif data.startswith("m:a:"):
            symbol = data.split(":")[-1]
            ack = await context.bot.send_message(chat_id=chat_id, text=f"🟡 تحلیل {symbol} در حال انجام...")
            stop_event = asyncio.Event()
            asyncio.create_task(typing_loop(context.bot, chat_id, stop_event))
            async def do_work():
                try:
                    analysis = await bot.analyze_symbol(symbol, fast=False, force_train=True)
                    md=analysis.get("market_data",{}); price=md.get("price")
                    db_log_signal(int(time.time()), symbol, analysis.get("signal","HOLD"), float(analysis.get("confidence",0.5)), float(price) if price else None, "manual")
                    await context.bot.edit_message_text(chat_id=chat_id, message_id=ack.message_id, text=f"✅ نتیجه آماده شد: {symbol}")
                    for part in split_message(format_analysis(analysis)):
                        await context.bot.send_message(chat_id=chat_id, text=part)
                except Exception as e:
                    await context.bot.edit_message_text(chat_id=chat_id, message_id=ack.message_id, text=f"⛔️ خطا در تحلیل {symbol}: {e}")
                finally:
                    stop_event.set()
            asyncio.create_task(do_work())
        elif data == "m:auto:on":
            context.application.bot_data["autopilot_enabled"]=True
            await context.bot.send_message(chat_id=chat_id, text="✅ Autopilot فعال شد.")
        elif data == "m:auto:off":
            context.application.bot_data["autopilot_enabled"]=False
            for j in context.application.job_queue.get_jobs_by_name("autoscan"): j.schedule_removal()
            await context.bot.send_message(chat_id=chat_id, text="⛔️ Autopilot غیرفعال شد.")
        elif data == "m:mon:on":
            context.application.bot_data["monitor_enabled"]=True
            for j in context.application.job_queue.get_jobs_by_name("monitor"): j.schedule_removal()
            context.application.job_queue.run_repeating(monitor_job, interval=S.MONITOR_INTERVAL_SEC, first=5, name="monitor")
            await context.bot.send_message(chat_id=chat_id, text="📡 مانیتور روشن شد.")
        elif data == "m:mon:off":
            context.application.bot_data["monitor_enabled"]=False
            for j in context.application.job_queue.get_jobs_by_name("monitor"): j.schedule_removal()
            await context.bot.send_message(chat_id=chat_id, text="🛑 مانیتور خاموش شد.")
        elif data == "m:whales":
            await cmd_whales(Update(update.update_id, message=query.message), context)
        elif data == "m:help":
            await context.bot.send_message(chat_id=chat_id, text=build_help())
        else:
            await context.bot.send_message(chat_id=chat_id, text="دستور ناشناخته.")

    application = ApplicationBuilder()\
        .token(S.TELEGRAM_BOT_TOKEN)\
        .post_init(lambda app: schedule_jobs(app, bot))\
        .build()

    # Full commands (kept)
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("menu", cmd_menu))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("ping", cmd_ping))
    application.add_handler(CommandHandler("subscribe", cmd_subscribe))
    application.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    application.add_handler(CommandHandler("analyze", cmd_analyze))
    application.add_handler(CommandHandler("signals", cmd_signals))
    application.add_handler(CommandHandler("autosignals", cmd_autosignals))
    application.add_handler(CommandHandler("report", cmd_report))
    application.add_handler(CommandHandler("watchadd", cmd_watchadd))
    application.add_handler(CommandHandler("watchrm", cmd_watchrm))
    application.add_handler(CommandHandler("watchlist", cmd_watchlist))
    application.add_handler(CommandHandler("backtest", cmd_backtest))
    application.add_handler(CommandHandler("whales", cmd_whales))
    application.add_handler(CommandHandler("opportunities", cmd_opportunities))
    application.add_handler(CommandHandler("autopilot", cmd_autopilot))
    application.add_handler(CommandHandler("monitor", cmd_monitor))
    # Aliases (simplified)
    application.add_handler(CommandHandler("a", cmd_a))
    application.add_handler(CommandHandler("sig", cmd_sig))
    application.add_handler(CommandHandler("opp", cmd_opp))
    application.add_handler(CommandHandler("auto", cmd_auto))
    application.add_handler(CommandHandler("mon", cmd_mon))
    application.add_handler(CommandHandler("wh", cmd_wh))
    application.add_handler(CommandHandler("bt", cmd_bt))
    application.add_handler(CommandHandler("wl", cmd_wl))
    application.add_handler(CommandHandler("add", cmd_add))
    application.add_handler(CommandHandler("rm", cmd_rm))
    # Callbacks
    application.add_handler(CallbackQueryHandler(on_cb))

    logger.info("Telegram bot running...")

    if S.TELEGRAM_CHAT_ID:
        try: db_add_subscriber(int(S.TELEGRAM_CHAT_ID))
        except: pass

    application.run_polling()

# ---------------- Entry ----------------
if __name__=="__main__":
    if TELEGRAM_AVAILABLE and S.TELEGRAM_BOT_TOKEN:
        run_telegram()
    else:
        async def main():
            if S.ENABLE_HTTP:
                await start_http_server()
            await run_cli()
        asyncio.run(main())