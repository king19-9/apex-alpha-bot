# bot.py
import os
import logging
import asyncio
import json
import numpy as np
import pandas as pd
import yfinance as yf
import ccxt.async_support as ccxt
import sqlite3
import aiohttp
import requests
import time
from datetime import datetime
from functools import wraps
import redis.asyncio as redis
import re
from scipy.signal import find_peaks

from config import Config

# --- کتابخانه‌های اختیاری ---
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    ta = None

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

# --- لاگینگ ---
logger = logging.getLogger("TradingBot")

# --- دکوراتورهای کمکی ---
def async_retry(attempts=3, delay=2):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == attempts:
                        logger.error(f"Function {func.__name__} failed after {attempts} attempts. Error: {e}")
                        raise
                    logger.warning(f"Attempt {attempt}/{attempts} for {func.__name__} failed. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
        return wrapper
    return decorator

class AdvancedTradingBot:
    def __init__(self):
        logger.info("Initializing AdvancedTradingBot...")
        self.http_session = requests.Session()
        if Config.REQUESTS_PROXY_DICT:
            self.http_session.proxies = Config.REQUESTS_PROXY_DICT
        
        self.setup_database()
        self.setup_exchanges()
        self.setup_redis()

    def setup_database(self):
        self.conn = None
        try:
            if PSYCOPG2_AVAILABLE and Config.DATABASE_URL:
                self.conn = psycopg2.connect(Config.DATABASE_URL)
                logger.info("PostgreSQL connection successful.")
            else:
                raise ConnectionError("Psycopg2 not available or DATABASE_URL not set.")
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}. Falling back to SQLite.")
            self.conn = sqlite3.connect("local_bot.db", check_same_thread=False)

    def setup_exchanges(self):
        self.exchanges = {}
        ccxt_config = {'enableRateLimit': True, 'options': {'defaultType': 'spot'}}
        if Config.AIOHTTP_PROXY_URL:
            # Note: ccxt uses aiohttp_proxy, not a dict like requests
            ccxt_config['aiohttp_proxy'] = Config.AIOHTTP_PROXY_URL
        
        for ex_name in Config.ENABLED_EXCHANGES:
            try:
                self.exchanges[ex_name] = getattr(ccxt, ex_name)(ccxt_config)
            except Exception as e:
                logger.error(f"Failed to initialize exchange {ex_name}: {e}")

    def setup_redis(self):
        self.redis_client = None
        if Config.CACHE_ENABLED:
            try:
                self.redis_client = redis.from_url(Config.REDIS_URL, decode_responses=True)
                # Test connection in an async context
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.redis_client.ping())
                logger.info("Redis connection successful.")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Caching will be disabled.")
                self.redis_client = None

    async def close_connections(self):
        await asyncio.gather(*(ex.close() for ex in self.exchanges.values()))
        if self.redis_client:
            await self.redis_client.close()
        if self.conn:
            self.conn.close()
        logger.info("All connections closed.")
        
    def _get_cache_key(self, func_name, *args):
        return f"{func_name}:{':'.join(map(str, args))}"

    # --- DATA FETCHING METHODS ---
    @async_retry()
    async def get_market_data(self, symbol):
        cache_key = self._get_cache_key("get_market_data", symbol)
        if self.redis_client:
            cached = await self.redis_client.get(cache_key)
            if cached: return json.loads(cached)

        tasks = {
            'coingecko': self.fetch_coingecko(symbol),
            'cryptocompare': self.fetch_cryptocompare(symbol),
            'exchanges': self.fetch_avg_ticker_from_exchanges(symbol)
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        merged_data = {}
        sources = list(tasks.keys())
        for i, res in enumerate(results):
            if not isinstance(res, Exception) and res:
                for key, val in res.items():
                    if key not in merged_data: merged_data[key] = []
                    if val is not None: merged_data[key].append(val)
        
        final_data = {key: np.mean(val) if val else None for key, val in merged_data.items()}
        if not final_data.get('price'): raise ValueError("Could not fetch price from any source.")
        
        if self.redis_client:
            await self.redis_client.setex(cache_key, Config.CACHE_TTL_SECONDS, json.dumps(final_data))
        return final_data

    @async_retry(attempts=2)
    async def get_historical_data(self, symbol, timeframe='1d'):
        cache_key = self._get_cache_key("get_historical_data", symbol, timeframe)
        if self.redis_client:
            cached = await self.redis_client.get(cache_key)
            if cached: return pd.read_json(cached, orient='split')
        
        # Priority: Binance Exchange
        try:
            binance = self.exchanges.get('binance')
            if binance:
                ohlcv = await binance.fetch_ohlcv(f"{symbol.upper()}/USDT", timeframe, limit=1000)
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    if self.redis_client: await self.redis_client.setex(cache_key, 3600, df.to_json(orient='split'))
                    return df
        except Exception as e:
            logger.warning(f"Fetching OHLCV from Binance failed: {e}. Falling back to yfinance.")

        # Fallback: yfinance
        yf_symbol = f"{symbol.upper()}-USD"
        data = await asyncio.to_thread(yf.download, yf_symbol, period="2y", interval=timeframe, progress=False)
        if data.empty:
            raise ValueError(f"No historical data found for {symbol}.")
        data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
        if self.redis_client: await self.redis_client.setex(cache_key, 3600, data.to_json(orient='split'))
        return data

    async def fetch_coingecko(self, symbol):
        # Implementation...
        return {}

    async def fetch_cryptocompare(self, symbol):
        # Implementation...
        return {}
        
    async def fetch_avg_ticker_from_exchanges(self, symbol):
        # Implementation...
        return {}
    
    async def get_news_and_sentiment(self, symbol):
        # Implementation...
        return {"items": [], "sentiment_score": 0.5, "hot_topics": []}
    
    async def get_onchain_data(self, symbol):
        # Implementation using Glassnode or other API...
        return {"nvt_ratio": 0.0, "active_addresses": 0}

    # --- ANALYSIS METHODS ---
    def run_all_sync_analyses(self, df):
        if df.empty: return {}
        
        tech_analyzer = TechnicalAnalyzer(df.copy())
        
        return {
            "technical": tech_analyzer.analyze_all(),
            "wyckoff": tech_analyzer.analyze_wyckoff(),
            "elliott_wave": tech_analyzer.analyze_elliott_wave(),
            "market_structure": tech_analyzer.analyze_market_structure(),
        }

    # --- CORE LOGIC ---
    async def perform_full_analysis(self, symbol, timeframe='1d'):
        logger.info(f"Starting full analysis for {symbol} on {timeframe}...")
        try:
            market_data, historical_data, news_data, onchain_data = await asyncio.gather(
                self.get_market_data(symbol),
                self.get_historical_data(symbol, timeframe),
                self.get_news_and_sentiment(symbol),
                self.get_onchain_data(symbol),
                return_exceptions=True
            )
            
            if isinstance(historical_data, Exception) or historical_data.empty:
                raise ValueError(f"Could not fetch historical data: {historical_data}")
            
            analysis_results = await asyncio.to_thread(self.run_all_sync_analyses, historical_data)
            
            final_signal = self.calculate_final_signal(analysis_results, news_data, onchain_data)

            return {
                "symbol": symbol,
                "market_data": market_data if not isinstance(market_data, Exception) else {},
                "analysis": analysis_results,
                "news_sentiment": news_data if not isinstance(news_data, Exception) else {},
                "onchain": onchain_data if not isinstance(onchain_data, Exception) else {},
                "final_signal": final_signal
            }

        except Exception as e:
            logger.error(f"Full analysis failed for {symbol}: {e}", exc_info=True)
            return {"error": str(e)}

    def calculate_final_signal(self, analysis, news, onchain):
        score = 0.5  # Neutral base
        weights = {"technical": 0.5, "sentiment": 0.2, "onchain": 0.15, "structure": 0.15}
        
        # Technical Score
        tech_score = 0
        tech = analysis.get('technical', {})
        if tech.get('trend_direction') == 'Uptrend': tech_score += 0.2
        if tech.get('rsi', 50) < 35: tech_score += 0.1
        score += tech_score * weights['technical']
        
        # Sentiment Score
        score += (news.get('sentiment_score', 0.5) - 0.5) * weights['sentiment']

        final_score = max(0, min(1, score))
        signal = "BUY" if final_score > 0.6 else "SELL" if final_score < 0.4 else "HOLD"
        return {"signal": signal, "score": round(final_score, 2)}

class TechnicalAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def analyze_all(self):
        return {
            "trend": self.analyze_trend(),
            "oscillators": self.analyze_oscillators(),
            "volatility": self.analyze_volatility(),
            "ichimoku": self.analyze_ichimoku(),
            "pivot_points": self.calculate_pivot_points(),
            "candlestick_patterns": self.find_candlestick_patterns(),
        }

    def analyze_trend(self):
        if not PANDAS_TA_AVAILABLE: return {}
        self.df.ta.ema(length=50, append=True)
        self.df.ta.ema(length=200, append=True)
        last = self.df.iloc[-1]
        direction = "Uptrend" if last['EMA_50'] > last['EMA_200'] else "Downtrend"
        return {"direction": direction}

    def analyze_oscillators(self):
        if not TALIB_AVAILABLE: return {}
        return {
            "rsi": talib.RSI(self.df['close']).iloc[-1],
            "stoch_k": talib.STOCH(self.df['high'], self.df['low'], self.df['close'])[0].iloc[-1]
        }
        
    def analyze_volatility(self):
        if not TALIB_AVAILABLE: return {}
        return {"atr": talib.ATR(self.df['high'], self.df['low'], self.df['close']).iloc[-1]}

    def analyze_ichimoku(self):
        if not PANDAS_TA_AVAILABLE: return {}
        ichimoku_df, _ = self.df.ta.ichimoku()
        self.df = pd.concat([self.df, ichimoku_df], axis=1)
        last = self.df.iloc[-1]
        return {"tenkan_sen": last['ITS_9'], "kijun_sen": last['IKS_26']}

    def calculate_pivot_points(self):
        last_day = self.df.iloc[-1]
        pivot = (last_day['high'] + last_day['low'] + last_day['close']) / 3
        s1 = (2 * pivot) - last_day['high']
        r1 = (2 * pivot) - last_day['low']
        return {"pivot": pivot, "s1": s1, "r1": r1}

    def find_candlestick_patterns(self):
        if not TALIB_AVAILABLE: return {}
        patterns = {}
        for pattern_func in [func for func in dir(talib) if func.startswith('CDL')]:
            result = getattr(talib, pattern_func)(self.df['open'], self.df['high'], self.df['low'], self.df['close'])
            last_signal = result.iloc[-1]
            if last_signal != 0:
                patterns[pattern_func[3:]] = "Bullish" if last_signal > 0 else "Bearish"
        return patterns
    
    def analyze_wyckoff(self): return {"phase": "N/A"}
    def analyze_elliott_wave(self): return {"wave": "N/A"}
    def analyze_market_structure(self): return {"bos": "None", "choch": "None"}