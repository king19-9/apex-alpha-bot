# -*- coding: utf-8 -*-
"""
Advanced 24/7 Crypto AI Bot - Upgraded Version for Railway
- Multi-chain Analysis, Advanced TA, AI Predictions (with optional LSTM/Transformers), Risk Management
- Telegram Bot + Web Dashboard (aiohttp)
- Redis cache with dynamic TTL, Smart retries, Enhanced logging
"""

import os
import sys
import time
import asyncio
import logging
import json
import datetime
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import aiohttp
from aiohttp import web
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
import websockets
import json as _json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from telegram.constants import ChatAction
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, desc, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import redis
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import tweepy
from bs4 import BeautifulSoup
import requests

# Optional heavy deps (Transformers, TF) - safe import
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Settings
@dataclass
class Settings:
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID")

    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///crypto_bot.db")

    COINGECKO_API_KEY: str = os.getenv("COINGECKO_API_KEY")
    COINMARKETCAP_API_KEY: str = os.getenv("COINMARKETCAP_API_KEY")
    CRYPTOCOMPARE_API_KEY: str = os.getenv("CRYPTOCOMPARE_API_KEY")
    CRYPTOPANIC_API_KEY: str = os.getenv("CRYPTOPANIC_API_KEY")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY")
    WHALEALERT_API_KEY: str = os.getenv("WHALEALERT_API_KEY")
    ETHERSCAN_API_KEY: str = os.getenv("ETHERSCAN_API_KEY")
    POLYGONSCAN_API_KEY: str = os.getenv("POLYGONSCAN_API_KEY")
    BSCSCAN_API_KEY: str = os.getenv("BSCSCAN_API_KEY")
    DEXSCREENER_API_KEY: str = os.getenv("DEXSCREENER_API_KEY")
    TWITTER_BEARER_TOKEN: str = os.getenv("TWITTER_BEARER_TOKEN")

    # Feature flags
    OFFLINE_MODE: bool = os.getenv("OFFLINE_MODE", "false").lower() == "true"
    ENABLE_ADVANCED_TA: bool = os.getenv("ENABLE_ADVANCED_TA", "true").lower() == "true"
    ENABLE_ONCHAIN: bool = os.getenv("ENABLE_ONCHAIN", "true").lower() == "true"
    ENABLE_SENTIMENT: bool = os.getenv("ENABLE_SENTIMENT", "true").lower() == "true"
    ENABLE_PREDICTION: bool = os.getenv("ENABLE_PREDICTION", "true").lower() == "true"
    ENABLE_RISK_MANAGEMENT: bool = os.getenv("ENABLE_RISK_MANAGEMENT", "true").lower() == "true"
    ENABLE_WEB_DASHBOARD: bool = os.getenv("ENABLE_WEB_DASHBOARD", "true").lower() == "true"
    ENABLE_DEEP_PREDICTION: bool = os.getenv("ENABLE_DEEP_PREDICTION", "false").lower() == "true"
    ENABLE_TRANSFORMER_SENTIMENT: bool = os.getenv("ENABLE_TRANSFORMER_SENTIMENT", "false").lower() == "true"

    EXCHANGES: List[str] = os.getenv("EXCHANGES", "binance,kucoin,bybit,bitfinex,gateio,bitget").split(",")
    MAX_COINS: int = int(os.getenv("MAX_COINS", "1000"))
    UNIVERSE_MAX_PAGES: int = int(os.getenv("UNIVERSE_MAX_PAGES", "20"))
    TIMEFRAMES: List[str] = os.getenv("TIMEFRAMES", "15m,1h,4h,1d").split(",")

    CONCURRENT_REQUESTS: int = int(os.getenv("CONCURRENT_REQUESTS", "15"))
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))

    DEFAULT_RISK_PER_TRADE: float = float(os.getenv("DEFAULT_RISK_PER_TRADE", "2.0"))  # percent
    DEFAULT_MAX_LEVERAGE: float = float(os.getenv("DEFAULT_MAX_LEVERAGE", "5.0"))
    MIN_LIQUIDITY_THRESHOLD: float = float(os.getenv("MIN_LIQUIDITY_THRESHOLD", "1000000"))

    ENABLE_MONITOR: bool = os.getenv("ENABLE_MONITOR", "true").lower() == "true"
    MONITOR_INTERVAL: int = int(os.getenv("MONITOR_INTERVAL", "120"))
    ALERT_THRESHOLD: float = float(os.getenv("ALERT_THRESHOLD", "0.7"))

    ONCHAIN_MIN_USD: float = float(os.getenv("ONCHAIN_MIN_USD", "500000"))
    ONCHAIN_CHAINS: List[str] = os.getenv("ONCHAIN_CHAINS", "ethereum,polygon,binance-smart-chain,avalanche,arbitrum,fantom").split(",")

    WEB_PORT: int = int(os.getenv("PORT", os.getenv("WEB_PORT", "8080")))  # Railway PORT
S = Settings()

# Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('crypto_bot.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("crypto_ai_bot")

# Redis
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    redis_client.ping()
    redis_available = True
    logger.info("Redis connected")
except Exception as e:
    logging.warning(f"Redis not available: {e}")
    redis_available = False
    redis_client = None

def cache_with_redis(key: str, value: Any, ttl: int = None):
    if not redis_available:
        return
    try:
        ttl = ttl or S.CACHE_TTL_SECONDS
        redis_client.setex(key, ttl, json.dumps(value, default=str))
    except Exception as e:
        logger.error(f"Redis set error: {e}")

def get_from_cache(key: str):
    if not redis_available:
        return None
    try:
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.error(f"Redis get error: {e}")
    return None

# DB
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, unique=True)
    username = Column(String(50))
    preferences = Column(Text)  # JSON
    risk_level = Column(String(20), default='medium')
    max_leverage = Column(Float, default=5.0)
    balance = Column(Float, default=10000.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    signals = relationship("Signal", back_populates="user")
    watchlist = relationship("Watchlist", back_populates="user")
    performance = relationship("Performance", back_populates="user")

class Signal(Base):
    __tablename__ = 'signals'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20))
    signal_type = Column(String(10))  # BUY/SELL/HOLD
    confidence = Column(Float)
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    source = Column(String(50))
    user_id = Column(Integer, ForeignKey('users.id'))
    evaluated = Column(Boolean, default=False)
    result = Column(Float)
    timeframe = Column(String(10), default='1h')
    user = relationship("User", back_populates="signals")

class Watchlist(Base):
    __tablename__ = 'watchlist'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    symbol = Column(String(20))
    added_at = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship("User", back_populates="watchlist")

class Performance(Base):
    __tablename__ = 'performance'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    total_signals = Column(Integer, default=0)
    successful_signals = Column(Integer, default=0)
    avg_return = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship("User", back_populates="performance")

class MarketData(Base):
    __tablename__ = 'market_data'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True)
    price = Column(Float)
    change_24h = Column(Float)
    volume_24h = Column(Float)
    market_cap = Column(Float)
    last_update = Column(DateTime, default=datetime.datetime.utcnow)

class OnChainTransaction(Base):
    __tablename__ = 'onchain_transactions'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20))
    chain = Column(String(20))
    hash = Column(String(66))
    from_address = Column(String(42))
    to_address = Column(String(42))
    value_usd = Column(Float)
    timestamp = Column(DateTime)
    direction = Column(String(10))

# Engine
try:
    engine_kwargs = dict(pool_pre_ping=True)
    if S.DATABASE_URL.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(S.DATABASE_URL, **engine_kwargs)
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    logger.info("Database connected")
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    sys.exit(1)

def get_user_session(chat_id: int):
    session = Session()
    try:
        user = session.query(User).filter_by(chat_id=chat_id).first()
        if not user:
            user = User(chat_id=chat_id)
            session.add(user)
            session.commit()
        return session, user
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error in get_user_session: {e}")
        return None, None

async def retry_request(func, *args, **kwargs):
    max_retries = kwargs.pop("max_retries", 3)
    base_delay = kwargs.pop("base_delay", 1.5)
    for i in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if i == max_retries - 1:
                raise
            await asyncio.sleep(base_delay * (2 ** i))
            logger.warning(f"Retry {i+1} for {getattr(func, '__name__', str(func))}: {e}")

# Technical Analysis
class AdvancedTechnicalAnalyzer:
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(method='bfill')

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min).replace(0, np.nan))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent

    # Smart Money Tools
    def detect_order_blocks(self, df: pd.DataFrame, lookback: int = 50) -> dict:
        bullish_blocks, bearish_blocks = [], []
        for i in range(lookback, len(df)):
            if (df.iloc[i-1]['close'] < df.iloc[i-1]['open'] and 
                df.iloc[i]['close'] > df.iloc[i]['open'] and 
                df.iloc[i]['close'] > df.iloc[i-2]['high']):
                bullish_blocks.append({'index': i-1, 'high': df.iloc[i-1]['high'], 'low': df.iloc[i-1]['low'], 'close': df.iloc[i-1]['close'], 'type': 'bullish'})
            if (df.iloc[i-1]['close'] > df.iloc[i-1]['open'] and 
                df.iloc[i]['close'] < df.iloc[i]['open'] and 
                df.iloc[i]['close'] < df.iloc[i-2]['low']):
                bearish_blocks.append({'index': i-1, 'high': df.iloc[i-1]['high'], 'low': df.iloc[i-1]['low'], 'close': df.iloc[i-1]['close'], 'type': 'bearish'})
        return {'bullish': bullish_blocks, 'bearish': bearish_blocks}

    def detect_liquidity_zones(self, df: pd.DataFrame, window: int = 20) -> dict:
        high_zones, low_zones = [], []
        for i in range(window, len(df)-window):
            if df.iloc[i]['high'] == df.iloc[i-window:i+window]['high'].max():
                strength = len([x for x in df.iloc[i-window:i+window]['high'] if abs(x - df.iloc[i]['high']) < 0.01])
                high_zones.append({'price': float(df.iloc[i]['high']), 'index': i, 'strength': strength})
            if df.iloc[i]['low'] == df.iloc[i-window:i+window]['low'].min():
                strength = len([x for x in df.iloc[i-window:i+window]['low'] if abs(x - df.iloc[i]['low']) < 0.01])
                low_zones.append({'price': float(df.iloc[i]['low']), 'index': i, 'strength': strength})
        return {'high_zones': high_zones, 'low_zones': low_zones}

    def detect_market_structure(self, df: pd.DataFrame) -> dict:
        structure = {'trend': 'ranging', 'higher_highs': [], 'higher_lows': [], 'lower_highs': [], 'lower_lows': []}
        swing_highs, swing_lows = [], []
        for i in range(2, len(df)-2):
            if (df.iloc[i]['high'] > df.iloc[i-1]['high'] and df.iloc[i]['high'] > df.iloc[i+1]['high']):
                swing_highs.append((i, float(df.iloc[i]['high'])))
            if (df.iloc[i]['low'] < df.iloc[i-1]['low'] and df.iloc[i]['low'] < df.iloc[i+1]['low']):
                swing_lows.append((i, float(df.iloc[i]['low'])))
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            if (swing_highs[-1][1] > swing_highs[-2][1] and swing_lows[-1][1] > swing_lows[-2][1]):
                structure['trend'] = 'uptrend'
                structure['higher_highs'] = swing_highs
                structure['higher_lows'] = swing_lows
            elif (swing_highs[-1][1] < swing_highs[-2][1] and swing_lows[-1][1] < swing_lows[-2][1]):
                structure['trend'] = 'downtrend'
                structure['lower_highs'] = swing_highs
                structure['lower_lows'] = swing_lows
        return structure

    def detect_supply_demand_zones(self, df: pd.DataFrame, lookback: int = 50) -> dict:
        supply_zones, demand_zones = [], []
        for i in range(lookback, len(df)):
            if (df.iloc[i]['close'] > df.iloc[i-1]['close'] * 1.02 and df.iloc[i]['volume'] > df.iloc[i-1]['volume'] * 1.5):
                demand_zones.append({'price': float(df.iloc[i-1]['close']), 'strength': float(df.iloc[i]['volume'] / df.iloc[i-1]['volume']), 'index': i-1})
            if (df.iloc[i]['close'] < df.iloc[i-1]['close'] * 0.98 and df.iloc[i]['volume'] > df.iloc[i-1]['volume'] * 1.5):
                supply_zones.append({'price': float(df.iloc[i-1]['close']), 'strength': float(df.iloc[i]['volume'] / df.iloc[i-1]['volume']), 'index': i-1})
        return {'supply': supply_zones, 'demand': demand_zones}

    def detect_inside_bars(self, df: pd.DataFrame) -> list:
        inside_bars = []
        for i in range(1, len(df)):
            if (df.iloc[i]['high'] < df.iloc[i-1]['high'] and df.iloc[i]['low'] > df.iloc[i-1]['low']):
                inside_bars.append({'index': i, 'high': float(df.iloc[i]['high']), 'low': float(df.iloc[i]['low']), 'mother_high': float(df.iloc[i-1]['high']), 'mother_low': float(df.iloc[i-1]['low'])})
        return inside_bars

    def analyze_session_patterns(self, df: pd.DataFrame) -> dict:
        sessions = {'asia': {'start': 0, 'end': 8}, 'europe': {'start': 8, 'end': 16}, 'us': {'start': 13, 'end': 21}}
        session_analysis = {}
        if not isinstance(df.index, pd.DatetimeIndex):
            return session_analysis
        for session_name, times in sessions.items():
            session_df = df[(df.index.hour >= times['start']) & (df.index.hour < times['end'])]
            if not session_df.empty:
                session_analysis[session_name] = {
                    'volatility': float(session_df['high'].max() - session_df['low'].min()),
                    'volume': float(session_df['volume'].mean()),
                    'range': float(session_df['close'].pct_change().std()),
                    'direction': 'bullish' if session_df['close'].iloc[-1] > session_df['close'].iloc[0] else 'bearish'
                }
        return session_analysis

    def detect_decision_zones(self, df: pd.DataFrame) -> list:
        decision_zones = []
        for i in range(10, len(df)-10):
            window = df.iloc[i-10:i+10]
            volatility = (window['high'].max() - window['low'].min()) / (window['close'].mean() or 1)
            avg_volume = window['volume'].mean()
            if volatility < 0.02 and avg_volume > df['volume'].quantile(0.7):
                decision_zones.append({'price': float(window['close'].mean()), 'strength': float(avg_volume / df['volume'].mean()), 'range': float(window['high'].max() - window['low'].min()), 'index': i})
        return decision_zones

    def full_analysis(self, df: pd.DataFrame) -> dict:
        results = {
            'order_blocks': self.detect_order_blocks(df),
            'liquidity_zones': self.detect_liquidity_zones(df),
            'market_structure': self.detect_market_structure(df),
            'supply_demand': self.detect_supply_demand_zones(df),
            'inside_bars': self.detect_inside_bars(df),
            'session_patterns': self.analyze_session_patterns(df),
            'decision_zones': self.detect_decision_zones(df)
        }
        if not df.empty:
            results['indicators'] = {
                'rsi': float(self.calculate_rsi(df['close']).iloc[-1]),
                'macd': float(self.calculate_macd(df['close'])[0].iloc[-1]),
                'bollinger': self.calculate_bollinger_bands(df['close']),
                'atr': float(self.calculate_atr(df).iloc[-1]),
                'stochastic': float(self.calculate_stochastic(df)[0].iloc[-1])
            }
        return results

# Risk Management
class AdvancedRiskManager:
    def __init__(self):
        self.position_sizing_methods = ['fixed_risk', 'kelly', 'volatility_adjusted']
        self.risk_metrics = {}

    def calculate_position_size(self, account_balance: float, risk_percent: float, 
                                entry_price: float, stop_loss: float, method: str = 'fixed_risk',
                                volatility: float = None, liquidity: float = None) -> float:
        # risk_percent is percentage (e.g., 1.0 means 1%)
        risk_amount = account_balance * (risk_percent / 100.0)
        price_risk = max(abs(entry_price - stop_loss), 1e-8)

        if method == 'fixed_risk':
            position_size = risk_amount / price_risk
        elif method == 'kelly':
            win_rate = self.risk_metrics.get('win_rate', 0.5)
            avg_win = self.risk_metrics.get('avg_win', 1.0)
            avg_loss = self.risk_metrics.get('avg_loss', 1.0)
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / max(avg_win, 1e-6)
            kelly_fraction = max(0, min(kelly_fraction, 0.25))
            position_size = (account_balance * kelly_fraction) / max(entry_price, 1e-8)
        elif method == 'volatility_adjusted':
            if volatility is not None and np.isfinite(volatility):
                volatility_factor = 1 / (1 + float(volatility))
                position_size = (risk_amount * volatility_factor) / price_risk
            else:
                position_size = risk_amount / price_risk
        else:
            position_size = risk_amount / price_risk

        if liquidity and liquidity < S.MIN_LIQUIDITY_THRESHOLD:
            liquidity_factor = liquidity / S.MIN_LIQUIDITY_THRESHOLD
            position_size *= liquidity_factor

        return max(0.0, float(position_size))

    def calculate_optimal_leverage(self, volatility: float, account_balance: float, 
                                   position_size: float, max_leverage: float = None) -> float:
        max_leverage = max_leverage or S.DEFAULT_MAX_LEVERAGE
        vol_adj = 1 / (1 + (volatility or 0))
        base_lev = (position_size * S.DEFAULT_MAX_LEVERAGE) / max(account_balance, 1e-8)
        lev = base_lev * vol_adj
        return float(min(max(1.0, lev), max_leverage))

    def dynamic_stop_loss(self, df: pd.DataFrame, entry_price: float, direction: str, atr_period: int = 14) -> float:
        atr = AdvancedTechnicalAnalyzer().calculate_atr(df, atr_period).iloc[-1]
        stop_distance = float(atr) * 2.5
        if direction == 'long':
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        return float(stop_loss)

    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, take_profit: float) -> float:
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        return float(reward / risk) if risk > 0 else 0.0

    def portfolio_heat_check(self, current_positions: List[Dict], new_position_size: float, account_balance: float) -> bool:
        total_risk = sum(pos.get('size', 0) * (pos.get('risk_percent', S.DEFAULT_RISK_PER_TRADE)/100.0) for pos in current_positions)
        new_position_risk = new_position_size * (S.DEFAULT_RISK_PER_TRADE/100.0)
        total_portfolio_risk = (total_risk + new_position_risk) / max(account_balance, 1e-8)
        max_portfolio_risk = 0.20
        return total_portfolio_risk <= max_portfolio_risk

# On-chain
class EnhancedOnChainAnalyzer:
    def __init__(self):
        self.chain_apis = {
            'ethereum': {'url': 'https://api.etherscan.io/api', 'key': S.ETHERSCAN_API_KEY, 'explorer': 'https://etherscan.io'},
            'polygon': {'url': 'https://api.polygonscan.com/api', 'key': S.POLYGONSCAN_API_KEY, 'explorer': 'https://polygonscan.com'},
            'binance-smart-chain': {'url': 'https://api.bscscan.com/api', 'key': S.BSCSCAN_API_KEY, 'explorer': 'https://bscscan.com'},
            'avalanche': {'url': 'https://api.snowtrace.io/api', 'key': os.getenv('AVALANCHE_API_KEY'), 'explorer': 'https://snowtrace.io'},
            'arbitrum': {'url': 'https://api.arbiscan.io/api', 'key': os.getenv('ARBITRUM_API_KEY'), 'explorer': 'https://arbiscan.io'},
            'fantom': {'url': 'https://api.ftmscan.com/api', 'key': os.getenv('FANTOM_API_KEY'), 'explorer': 'https://ftmscan.com'}
        }
        self.token_contracts = {
            'USDT': {
                'ethereum': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
                'polygon': '0xc2132D05D31c914a87C6611C10748AEb04B58e8F',
                'binance-smart-chain': '0x55d398326f99059fF775485246999027B3197955',
                'avalanche': '0x9702230A8Ea53601f5cD2dc00fDBc13d4dF4A8c7',
                'arbitrum': '0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9',
                'fantom': '0x049d68029688eAbF473097a2fC38ef61633A3C7A'
            }
        }

    async def get_large_transactions(self, symbol: str, min_usd: float = None) -> list:
        min_usd = min_usd or S.ONCHAIN_MIN_USD
        all_transactions = []
        for chain, cfg in self.chain_apis.items():
            if chain not in S.ONCHAIN_CHAINS: 
                continue
            contract_address = self.token_contracts.get(symbol, {}).get(chain)
            if not contract_address:
                continue
            try:
                txs = await self._fetch_chain_transactions(chain, contract_address, min_usd)
                all_transactions.extend(txs)
            except Exception as e:
                logger.error(f"Error fetching {chain} transactions: {e}")
        return all_transactions

    async def _fetch_chain_transactions(self, chain: str, contract_address: str, min_usd: float) -> list:
        api_config = self.chain_apis.get(chain)
        if not api_config:
            return []
        url = api_config['url']
        params = {
            'module': 'account', 'action': 'tokentx',
            'contractaddress': contract_address,
            'startblock': 0, 'endblock': 99999999,
            'sort': 'desc', 'apikey': api_config['key']
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=S.REQUEST_TIMEOUT) as response:
                    data = await response.json()
                    if data.get('status') != '1':
                        return []
                    transactions = []
                    for tx in data.get('result', []):
                        value = float(tx.get('value', 0))
                        decimals = int(tx.get('tokenDecimal', 18))
                        usd_value = value / (10 ** decimals)
                        if usd_value >= min_usd:
                            transactions.append({
                                'hash': tx.get('hash'),
                                'from': tx.get('from'),
                                'to': tx.get('to'),
                                'value': usd_value,
                                'timestamp': int(tx.get('timeStamp')),
                                'chain': chain,
                                'explorer_url': f"{api_config['explorer']}/tx/{tx.get('hash')}"
                            })
                    return transactions
        except Exception as e:
            logger.error(f"Error fetching {chain} transactions: {e}")
            return []

    async def analyze_whale_activity(self, symbol: str) -> dict:
        transactions = await self.get_large_transactions(symbol)
        if not transactions:
            return {'activity': 'low', 'bias': 'neutral', 'total_volume': 0, 'transaction_count': 0, 'buy_sell_ratio': 1.0}
        buys = sum(1 for tx in transactions if tx.get('to') in self.token_contracts.get(symbol, {}).values())
        sells = sum(1 for tx in transactions if tx.get('from') in self.token_contracts.get(symbol, {}).values())
        total_volume = sum(tx['value'] for tx in transactions)
        if buys > sells * 1.5:
            bias = 'bullish'
        elif sells > buys * 1.5:
            bias = 'bearish'
        else:
            bias = 'neutral'
        if total_volume > 10000000:
            activity = 'very_high'
        elif total_volume > 5000000:
            activity = 'high'
        elif total_volume > 1000000:
            activity = 'medium'
        else:
            activity = 'low'
        return {'activity': activity, 'bias': bias, 'total_volume': total_volume, 'transaction_count': len(transactions), 'buy_sell_ratio': buys / max(sells, 1)}

# Sentiment
class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        try:
            self.twitter_client = tweepy.Client(bearer_token=S.TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)
            self.twitter_available = True
        except Exception:
            self.twitter_client = None
            self.twitter_available = False
            logger.warning("Twitter client not available")

        self.transformer_pipeline = None
        self.use_transformer = TRANSFORMERS_AVAILABLE and S.ENABLE_TRANSFORMER_SENTIMENT
        if self.use_transformer:
            try:
                # DistilBERT sentiment pipeline (CPU)
                self.transformer_pipeline = pipeline("sentiment-analysis")
                logger.info("Transformer sentiment pipeline loaded")
            except Exception as e:
                self.use_transformer = False
                logger.warning(f"Transformer not usable, fallback to VADER: {e}")

    def _score_text(self, text: str) -> float:
        if self.use_transformer and self.transformer_pipeline:
            r = self.transformer_pipeline(text[:512])[0]
            label, score = r['label'], r['score']
            return float(score if label.upper() == 'POSITIVE' else -score)
        else:
            return float(self.vader.polarity_scores(text).get('compound', 0.0))

    async def analyze_news_sentiment(self, symbol: str) -> dict:
        news = []
        news += await self._fetch_cryptopanic_news(symbol)
        news += await self._fetch_newsapi_news(symbol)
        if not news:
            return {'sentiment': 'neutral', 'score': 0.0, 'sources': 0}
        scores = []
        for n in news:
            text = f"{n.get('title','')} {n.get('description','')}"
            scores.append(self._score_text(text))
        avg_score = float(np.mean(scores)) if scores else 0.0
        sentiment = 'bullish' if avg_score > 0.2 else ('bearish' if avg_score < -0.2 else 'neutral')
        return {'sentiment': sentiment, 'score': avg_score, 'sources': len(news)}

    async def _fetch_cryptopanic_news(self, symbol: str) -> list:
        if not S.CRYPTOPANIC_API_KEY:
            return []
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {'auth_token': S.CRYPTOPANIC_API_KEY, 'currencies': symbol, 'kind': 'news'}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=S.REQUEST_TIMEOUT) as response:
                    data = await response.json()
                    return data.get('results', [])
        except Exception as e:
            logger.error(f"CryptoPanic error: {e}")
            return []

    async def _fetch_newsapi_news(self, symbol: str) -> list:
        if not S.NEWS_API_KEY:
            return []
        url = "https://newsapi.org/v2/everything"
        params = {'q': f'{symbol} cryptocurrency', 'apiKey': S.NEWS_API_KEY, 'language': 'en', 'sortBy': 'publishedAt', 'pageSize': 20}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=S.REQUEST_TIMEOUT) as response:
                    data = await response.json()
                    return data.get('articles', [])
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return []

    async def analyze_twitter_sentiment(self, symbol: str) -> dict:
        if not self.twitter_available or not S.TWITTER_BEARER_TOKEN:
            return {'sentiment': 'neutral', 'score': 0.0, 'tweet_count': 0}
        try:
            query = f"#{symbol} OR ${symbol} -is:retweet lang:en"
            tweets = self.twitter_client.search_recent_tweets(query=query, max_results=100, tweet_fields=['created_at','public_metrics'])
            if not tweets.data:
                return {'sentiment': 'neutral', 'score': 0.0, 'tweet_count': 0}
            scores = [self._score_text(t.text) for t in tweets.data]
            avg_score = float(np.mean(scores)) if scores else 0.0
            sentiment = 'bullish' if avg_score > 0.2 else ('bearish' if avg_score < -0.2 else 'neutral')
            return {'sentiment': sentiment, 'score': avg_score, 'tweet_count': len(tweets.data)}
        except Exception as e:
            logger.error(f"Twitter sentiment error: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'tweet_count': 0}

    async def combined_sentiment_analysis(self, symbol: str) -> dict:
        news_sent = await self.analyze_news_sentiment(symbol)
        twitter_sent = await self.analyze_twitter_sentiment(symbol)
        news_weight, twitter_weight = 0.6, 0.4
        combined_score = (news_sent['score'] * news_weight) + (twitter_sent['score'] * twitter_weight)
        sentiment = 'bullish' if combined_score > 0.2 else ('bearish' if combined_score < -0.2 else 'neutral')
        return {'sentiment': sentiment, 'score': float(combined_score), 'news': news_sent, 'twitter': twitter_sent}

# Prediction
class EnhancedPredictionEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.deep_models = {}

    async def prepare_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        ta = AdvancedTechnicalAnalyzer()
        df = df.copy()
        df['rsi'] = ta.calculate_rsi(df['close'])
        macd_line, macd_signal, macd_hist = ta.calculate_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        bb_upper, bb_mid, bb_lower = ta.calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['atr'] = ta.calculate_atr(df)
        stoch_k, stoch_d = ta.calculate_stochastic(df)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
        else:
            df['hour'] = 0
            df['day_of_week'] = 0
        df = df.dropna()
        return df

    def _features(self) -> List[str]:
        return ['rsi','macd','macd_hist','bb_upper','bb_lower','atr','stoch_k','stoch_d','returns','log_returns','volatility','volume_ratio','hour','day_of_week']

    async def train_model(self, df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        df = await self.prepare_features(df, symbol)
        features = self._features()
        X = df[features]
        y_raw = (df['close'].shift(-1) > df['close']).astype(float)
        y = y_raw.dropna().astype(int)
        X = X.loc[y.index]

        split_idx = int(len(X) * 0.8) if len(X) > 50 else len(X)-1
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) if len(X_test) else X_train_scaled

        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(random_state=42)
        lr = LogisticRegression(max_iter=500)

        rf.fit(X_train_scaled, y_train)
        gb.fit(X_train_scaled, y_train)
        lr.fit(X_train_scaled, y_train)

        def soft_vote_prob(Xs):
            p1 = rf.predict_proba(Xs)[:,1]
            p2 = gb.predict_proba(Xs)[:,1]
            p3 = lr.predict_proba(Xs)[:,1]
            return (0.5*p1 + 0.3*p2 + 0.2*p3)

        train_prob = soft_vote_prob(X_train_scaled)
        test_prob = soft_vote_prob(X_test_scaled)
        train_acc = float(np.mean((train_prob > 0.5) == y_train)) if len(y_train) else 0.0
        test_acc = float(np.mean((test_prob > 0.5) == y_test)) if len(y_test) else 0.0

        model_key = f"{symbol}_{timeframe}"
        self.models[model_key] = (rf, gb, lr)
        self.scalers[model_key] = scaler

        # Optional: deep LSTM
        deep_info = {}
        if S.ENABLE_DEEP_PREDICTION and TF_AVAILABLE and len(df) > 200:
            deep_info = await self._train_lstm(df, symbol, timeframe)

        return {'train_accuracy': train_acc, 'test_accuracy': test_acc, 'deep': deep_info}

    async def _train_lstm(self, df: pd.DataFrame, symbol: str, timeframe: str, seq_len: int = 32):
        try:
            prices = df['close'].values.astype(np.float32)
            returns = np.diff(np.log(prices))
            y = (np.concatenate([[0], returns]) > 0).astype(int)
            X_seq, y_seq = [], []
            for i in range(len(prices)-seq_len-1):
                X_seq.append(prices[i:i+seq_len].reshape(-1,1))
                y_seq.append(y[i+seq_len])
            X_seq, y_seq = np.array(X_seq), np.array(y_seq)
            split = int(len(X_seq)*0.8)
            X_train, X_val = X_seq[:split], X_seq[split:]
            y_train, y_val = y_seq[:split], y_seq[split:]

            model = keras.Sequential([
                layers.Input(shape=(seq_len,1)),
                layers.LSTM(32, return_sequences=True),
                layers.Attention()([layers.Input(shape=(seq_len,1)), layers.Input(shape=(seq_len,1))]) if False else layers.LSTM(16),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            # Note: Simple LSTM (Attention placeholder commented due TF functional need)
            model = keras.Sequential([
                layers.Input(shape=(seq_len,1)),
                layers.LSTM(32, return_sequences=False),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=64, verbose=0)
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            self.deep_models[f"{symbol}_{timeframe}"] = model
            return {'val_accuracy': float(val_acc)}
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}")
            return {}

    async def predict(self, df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        model_key = f"{symbol}_{timeframe}"
        if model_key not in self.models:
            return {}
        df = await self.prepare_features(df, symbol)
        features = self._features()
        X_latest = df[features].iloc[-1:].values
        scaler = self.scalers[model_key]
        X_scaled = scaler.transform(X_latest)
        rf, gb, lr = self.models[model_key]
        p = 0.5*rf.predict_proba(X_scaled)[:,1] + 0.3*gb.predict_proba(X_scaled)[:,1] + 0.2*lr.predict_proba(X_scaled)[:,1]
        probability = float(p[0])
        direction = 'BUY' if probability >= 0.5 else 'SELL'
        conf = float(max(probability, 1 - probability))

        # If deep model exists, blend
        if S.ENABLE_DEEP_PREDICTION and TF_AVAILABLE and model_key in self.deep_models:
            try:
                prices = df['close'].values[-33:].astype(np.float32)
                X_seq = prices[-32:].reshape(1,32,1)
                deep_p = float(self.deep_models[model_key].predict(X_seq, verbose=0)[0][0])
                probability = 0.7*probability + 0.3*deep_p
                direction = 'BUY' if probability >= 0.5 else 'SELL'
                conf = float(max(probability, 1 - probability))
            except Exception as e:
                logger.warning(f"Deep predict failed: {e}")

        return {'direction': direction, 'probability': probability, 'confidence': conf}

# Real-time stream (lightweight)
class BinanceWS:
    def __init__(self):
        self.uri = "wss://stream.binance.com:9443/ws"
        self.tasks = {}

    async def subscribe_trades(self, symbol: str, handler):
        channel = f"{symbol.lower()}usdt@trade"
        url = f"{self.uri}/{channel}"
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                async for message in ws:
                    data = json.loads(message)
                    await handler(data)
        except Exception as e:
            logger.warning(f"WS {symbol} error: {e}")

# Main Bot
class AdvancedCryptoBot:
    def __init__(self):
        self.ta_analyzer = AdvancedTechnicalAnalyzer()
        self.risk_manager = AdvancedRiskManager()
        self.onchain_analyzer = EnhancedOnChainAnalyzer()
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        self.prediction_engine = EnhancedPredictionEngine()
        self.ws = BinanceWS()

    async def fetch_ohlcv(self, exchange, symbol: str, timeframe: str, limit: int = 500):
        return await exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe, limit=limit)

    async def fetch_market_data(self, symbol: str) -> dict:
        cache_key = f"market_data_{symbol}"
        cached = get_from_cache(cache_key)
        if cached:
            # reconstruct DataFrame index
            try:
                df = pd.DataFrame(cached['ohlcv'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                cached['ohlcv'] = df
                return cached
            except Exception:
                pass

        exchange = None
        try:
            exchange = ccxt.binance()
            ticker = await exchange.fetch_ticker(f"{symbol}/USDT")
            ohlcv = await exchange.fetch_ohlcv(f"{symbol}/USDT", "1h", limit=1000)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            data = {
                'symbol': symbol,
                'price': float(ticker.get('last') or df['close'].iloc[-1]),
                'change_24h': float(ticker.get('percentage') or 0),
                'volume_24h': float(ticker.get('quoteVolume') or df['volume'].tail(24).sum()),
                'high_24h': float(ticker.get('high') or df['high'].tail(24).max()),
                'low_24h': float(ticker.get('low') or df['low'].tail(24).min()),
                'ohlcv': df.reset_index().to_dict(orient='list')  # cache-safe
            }
            cache_with_redis(cache_key, data, S.CACHE_TTL_SECONDS)
            # restore df
            data['ohlcv'] = df
            return data
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
        finally:
            if exchange:
                try:
                    await exchange.close()
                except Exception:
                    pass

    def _risk_percent_from_level(self, level: str) -> float:
        mapping = {'low': 0.5, 'medium': 1.0, 'high': 2.0}
        try:
            return float(level)
        except Exception:
            return mapping.get((level or '').lower(), 1.0)

    def _calculate_risk_parameters(self, df: pd.DataFrame, current_price: float, user: User) -> dict:
        stop_loss = self.risk_manager.dynamic_stop_loss(df, current_price, 'long')
        risk_percent = self._risk_percent_from_level(user.risk_level)
        position_size = self.risk_manager.calculate_position_size(
            account_balance=user.balance,
            risk_percent=risk_percent,
            entry_price=current_price,
            stop_loss=stop_loss,
            method='volatility_adjusted',
            volatility=float(df['close'].pct_change().std())
        )
        leverage = self.risk_manager.calculate_optimal_leverage(
            volatility=float(df['close'].pct_change().std()),
            account_balance=user.balance,
            position_size=position_size,
            max_leverage=user.max_leverage
        )
        take_profit_1 = current_price + (current_price - stop_loss) * 2
        take_profit_2 = current_price + (current_price - stop_loss) * 3
        return {
            'stop_loss': float(stop_loss),
            'take_profit_1': float(take_profit_1),
            'take_profit_2': float(take_profit_2),
            'position_size': float(position_size),
            'leverage': float(leverage),
            'risk_reward_ratio': float(self.risk_manager.calculate_risk_reward_ratio(current_price, stop_loss, take_profit_1))
        }

    async def analyze_multiple_timeframes(self, symbol: str) -> dict:
        try:
            exchange = ccxt.binance()
            tf_results = {}
            for tf in S.TIMEFRAMES:
                try:
                    ohlcv = await exchange.fetch_ohlcv(f"{symbol}/USDT", tf, limit=500)
                    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    tf_results[tf] = self.ta_analyzer.full_analysis(df)
                except Exception as e:
                    logger.warning(f"TF {tf} fetch/analyze failed: {e}")
            await exchange.close()
            # Confirmation count
            confirmations = sum(1 for tf, res in tf_results.items() if res.get('market_structure',{}).get('trend')=='uptrend') - \
                            sum(1 for tf, res in tf_results.items() if res.get('market_structure',{}).get('trend')=='downtrend')
            return {'timeframes': tf_results, 'confirmations': confirmations}
        except Exception as e:
            logger.error(f"MTF analysis error: {e}")
            return {'timeframes': {}, 'confirmations': 0}

    def _generate_signal(self, ta_results, onchain_results, sentiment_results, prediction, risk_params, mtf_res=None) -> dict:
        signal_score = 0.0
        # Dynamic weights
        w_ta, w_onchain, w_sent, w_pred, w_mtf = 0.4, 0.3, 0.2, 0.1, 0.2

        # TA
        trend = ta_results.get('market_structure', {}).get('trend')
        if trend == 'uptrend':
            signal_score += w_ta
        elif trend == 'downtrend':
            signal_score -= w_ta

        # On-chain
        bias = onchain_results.get('bias')
        if bias == 'bullish':
            signal_score += w_onchain
        elif bias == 'bearish':
            signal_score -= w_onchain

        # Sentiment
        sent = sentiment_results.get('sentiment')
        if sent == 'bullish':
            signal_score += w_sent
        elif sent == 'bearish':
            signal_score -= w_sent

        # Prediction
        if prediction:
            if prediction.get('direction') == 'BUY':
                signal_score += w_pred * prediction.get('confidence', 0.5)
            else:
                signal_score -= w_pred * prediction.get('confidence', 0.5)

        # Multi-timeframe confirmations
        if mtf_res:
            confirmations = mtf_res.get('confirmations', 0)
            if confirmations > 0:
                signal_score += w_mtf * min(1.0, confirmations/len(S.TIMEFRAMES))
            elif confirmations < 0:
                signal_score -= w_mtf * min(1.0, abs(confirmations)/len(S.TIMEFRAMES))

        if signal_score > 0.6:
            signal = 'BUY'
        elif signal_score < -0.6:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        return {'signal': signal, 'score': float(signal_score), 'confidence': float(abs(signal_score))}

    def _save_analysis(self, user_id, symbol, signal, prediction, market_data, ta_results, onchain_results):
        try:
            session = Session()
            db_signal = Signal(
                user_id=user_id, symbol=symbol,
                signal_type=signal['signal'], confidence=float(signal['confidence']),
                price=float(market_data['price']), source='bot_analysis'
            )
            session.add(db_signal)
            perf = session.query(Performance).filter_by(user_id=user_id).first()
            if not perf:
                perf = Performance(user_id=user_id)
                session.add(perf)
            perf.total_signals = (perf.total_signals or 0) + 1
            perf.updated_at = datetime.datetime.utcnow()
            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")

    async def analyze_symbol(self, symbol: str, user_id: int = None) -> dict:
        try:
            session, user = get_user_session(user_id or 0)
            if not user:
                return None
            prefs = json.loads(user.preferences or '{}')
            market_data = await self.fetch_market_data(symbol)
            if not market_data:
                return None
            df = market_data['ohlcv']
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            ta_results = self.ta_analyzer.full_analysis(df)
            onchain_results = await self.onchain_analyzer.analyze_whale_activity(symbol) if S.ENABLE_ONCHAIN else {}
            sentiment_results = await self.sentiment_analyzer.combined_sentiment_analysis(symbol) if S.ENABLE_SENTIMENT else {}
            # Train model on first call per symbol/timeframe (simple)
            pred = {}
            if S.ENABLE_PREDICTION:
                model_key = f"{symbol}_1h"
                if model_key not in self.prediction_engine.models:
                    await self.prediction_engine.train_model(df, symbol, '1h')
                pred = await self.prediction_engine.predict(df, symbol, '1h')
            risk_params = self._calculate_risk_parameters(df, float(market_data['price']), user)
            mtf = await self.analyze_multiple_timeframes(symbol)
            signal = self._generate_signal(ta_results, onchain_results, sentiment_results, pred, risk_params, mtf_res=mtf)
            self._save_analysis(user.id, symbol, signal, pred, market_data, ta_results, onchain_results)
            result = {
                'symbol': symbol,
                'market_data': market_data,
                'technical_analysis': ta_results,
                'onchain_analysis': onchain_results,
                'sentiment_analysis': sentiment_results,
                'prediction': pred,
                'risk_management': risk_params,
                'multi_timeframe': mtf,
                'signal': signal,
                'timestamp': datetime.datetime.utcnow()
            }
            return result
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

    async def generate_chart(self, symbol: str, analysis_results: dict) -> Optional[InputFile]:
        try:
            df = analysis_results['market_data']['ohlcv']
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.6, 0.2, 0.2])
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)

            indicators = analysis_results['technical_analysis'].get('indicators', {})
            if 'bollinger' in indicators:
                bb_upper, bb_mid, bb_lower = indicators['bollinger']
                fig.add_trace(go.Scatter(x=df.index, y=bb_upper, mode='lines', name='BB Upper', line=dict(color='red', width=1), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=bb_lower, mode='lines', name='BB Lower', line=dict(color='green', width=1), showlegend=False), row=1, col=1)

            for block in analysis_results['technical_analysis']['order_blocks'].get('bullish', []):
                idx = block['index']
                if idx < len(df)-1:
                    fig.add_shape(type="rect", x0=df.index[idx], x1=df.index[min(idx+5, len(df)-1)], y0=block['low'], y1=block['high'], fillcolor="green", opacity=0.15, layer="below", row=1, col=1)

            fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='blue', opacity=0.5), row=2, col=1)

            # RSI quick compute
            rsi_series = AdvancedTechnicalAnalyzer().calculate_rsi(df['close'])
            fig.add_trace(go.Scatter(x=df.index, y=rsi_series, mode='lines', name='RSI', line=dict(color='purple')), row=3, col=1)

            fig.update_layout(title=f'{symbol} Technical Analysis', template='plotly_dark', height=800, showlegend=True)
            img_bytes = fig.to_image(format="png", width=1200, height=800)
            return InputFile(io.BytesIO(img_bytes), filename=f'{symbol}_analysis.png')
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return None

# Web Dashboard
class WebDashboard:
    def __init__(self, bot: AdvancedCryptoBot):
        self.bot = bot
        self.app = web.Application()
        self.setup_routes()

    def setup_routes(self):
        self.app.router.add_get('/', self.index)
        self.app.router.add_get('/api/stats', self.get_stats)
        self.app.router.add_get('/api/signals', self.get_signals)
        self.app.router.add_get('/api/performance', self.get_performance)
        self.app.router.add_get('/api/watchlist', self.get_watchlist)
        self.app.router.add_get('/chart/{symbol}', self.get_chart)

    async def index(self, request):
        return web.Response(text=self._index_html(), content_type='text/html')

    def _index_html(self) -> str:
        return """
<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Crypto AI Bot Dashboard</title>
<style>
body{font-family:Segoe UI,Tahoma,Geneva,Verdana,sans-serif;background:#0a0e27;color:#fff;line-height:1.6;margin:0}
.container{max-width:1200px;margin:0 auto;padding:20px}
header{text-align:center;margin-bottom:40px;padding:20px 0;border-bottom:2px solid #1a237e}
h1{font-size:2.2em;margin-bottom:10px;background:linear-gradient(45deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px;margin-bottom:40px}
.stat-card{background:linear-gradient(135deg,#1a237e,#0f3460);padding:25px;border-radius:15px;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,.3)}
.stat-card h3{font-size:.9em;color:#94a3b8;margin-bottom:10px;text-transform:uppercase;letter-spacing:1px}
.stat-card .value{font-size:2.2em;font-weight:bold}
.content-grid{display:grid;grid-template-columns:2fr 1fr;gap:30px;margin-bottom:40px}
.panel{background:#151b3d;border-radius:15px;padding:25px;box-shadow:0 8px 32px rgba(0,0,0,.3)}
.panel h2{margin-bottom:20px;color:#667eea;font-size:1.3em}
table{width:100%;border-collapse:collapse;margin-top:15px}
th,td{padding:10px;text-align:left;border-bottom:1px solid #2a3f5f}
th{background:#1a237e;color:#667eea;font-weight:600;text-transform:uppercase;font-size:.85em;letter-spacing:.5px}
tr:hover{background:#1a237e}
.signal-buy{color:#4ade80;font-weight:bold}
.signal-sell{color:#f87171;font-weight:bold}
.signal-hold{color:#fbbf24;font-weight:bold}
.refresh-btn{background:linear-gradient(45deg,#667eea,#764ba2);color:#fff;border:none;padding:10px 20px;border-radius:25px;cursor:pointer;font-size:1em;margin:20px 0}
footer{text-align:center;margin-top:40px;padding:20px;border-top:2px solid #1a237e;color:#94a3b8}
@media (max-width:768px){.content-grid{grid-template-columns:1fr}.stats-grid{grid-template-columns:1fr}}
</style></head>
<body><div class="container">
<header><h1>🤖 Advanced Crypto AI Bot</h1><p>Real-time cryptocurrency analysis and trading signals</p></header>
<div class="stats-grid" id="stats">
<div class="stat-card"><h3>Total Users</h3><div class="value" id="totalUsers">-</div></div>
<div class="stat-card"><h3>Total Signals</h3><div class="value" id="totalSignals">-</div></div>
<div class="stat-card"><h3>Success Rate</h3><div class="value" id="successRate">-</div></div>
<div class="stat-card"><h3>Uptime</h3><div class="value" id="uptime">-</div></div>
</div>
<div class="content-grid">
<div class="panel">
<h2>📊 Recent Signals</h2>
<table id="signalsTable"><thead><tr><th>Symbol</th><th>Signal</th><th>Confidence</th><th>Price</th><th>Time</th></tr></thead>
<tbody id="signalsBody"></tbody></table>
<button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
</div>
<div class="panel">
<h2>📈 Performance Metrics</h2>
<div id="performanceMetrics"></div>
</div>
</div>
<div class="panel">
<h2>📋 Watchlist</h2>
<table id="watchlistTable"><thead><tr><th>Symbol</th><th>Added</th><th>Actions</th></tr></thead><tbody id="watchlistBody"></tbody></table>
</div>
<footer><p>&copy; 2024 Advanced Crypto AI Bot. All rights reserved.</p></footer>
</div>
<script>
async function loadStats(){try{const r=await fetch('/api/stats');const s=await r.json();document.getElementById('totalUsers').textContent=s.total_users||0;document.getElementById('totalSignals').textContent=s.total_signals||0;document.getElementById('successRate').textContent=s.win_rate? (s.win_rate*100).toFixed(1)+'%':'-';document.getElementById('uptime').textContent=s.uptime? Math.floor(s.uptime/3600)+'h':'-';}catch(e){console.error(e);}}
async function loadSignals(){try{const r=await fetch('/api/signals');const arr=await r.json();const tbody=document.getElementById('signalsBody');tbody.innerHTML=arr.map(s=>`<tr><td>${s.symbol}</td><td class="signal-${s.signal_type.toLowerCase()}">${s.signal_type}</td><td>${(s.confidence*100).toFixed(1)}%</td><td>$${Number(s.price).toFixed(4)}</td><td>${new Date(s.timestamp).toLocaleString()}</td></tr>`).join('');}catch(e){console.error(e);}}
async function loadWatchlist(){try{const r=await fetch('/api/watchlist');const arr=await r.json();const tbody=document.getElementById('watchlistBody');tbody.innerHTML=arr.map(w=>`<tr><td>${w.symbol}</td><td>${new Date(w.added_at).toLocaleString()}</td><td>-</td></tr>`).join('');}catch(e){console.error(e);}}
async function loadPerformance(){try{const r=await fetch('/api/performance');const p=await r.json();document.getElementById('performanceMetrics').innerHTML=`<ul>
<li>Total Signals: <b>${p.total_signals||0}</b></li>
<li>Win Rate: <b>${((p.win_rate||0)*100).toFixed(1)}%</b></li>
<li>Avg Return: <b>${(p.avg_return||0).toFixed(3)}</b></li>
<li>Sharpe: <b>${(p.sharpe_ratio||0).toFixed(3)}</b></li>
<li>Max DD: <b>${(p.max_drawdown||0).toFixed(3)}</b></li>
<li>Profit Factor: <b>${(p.profit_factor||0).toFixed(3)}</b></li>
</ul>`;}catch(e){console.error(e);}}
async function refreshData(){await Promise.all([loadStats(), loadSignals(), loadWatchlist(), loadPerformance()]);}
refreshData();setInterval(refreshData, 15000);
</script>
</body></html>
        """

    async def get_stats(self, request):
        try:
            session = Session()
            total_users = session.query(func.count(User.id)).scalar() or 0
            total_signals = session.query(func.count(Signal.id)).scalar() or 0
            # naive estimate
            perf = session.query(Performance).first()
            win_rate = float(perf.win_rate) if perf and perf.win_rate else 0.0
            uptime = float(time.time() - START_TIME)
            session.close()
            return web.json_response({'total_users': total_users, 'total_signals': total_signals, 'win_rate': win_rate, 'uptime': uptime})
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def get_signals(self, request):
        try:
            session = Session()
            rows = session.query(Signal).order_by(desc(Signal.timestamp)).limit(50).all()
            data = [{'symbol': r.symbol, 'signal_type': r.signal_type, 'confidence': float(r.confidence or 0), 'price': float(r.price or 0), 'timestamp': r.timestamp.isoformat()} for r in rows]
            session.close()
            return web.json_response(data)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def get_performance(self, request):
        try:
            session = Session()
            perf = session.query(Performance).order_by(desc(Performance.updated_at)).first()
            data = {
                'total_signals': int(perf.total_signals) if perf else 0,
                'successful_signals': int(perf.successful_signals) if perf else 0,
                'avg_return': float(perf.avg_return) if perf else 0.0,
                'sharpe_ratio': float(perf.sharpe_ratio) if perf else 0.0,
                'max_drawdown': float(perf.max_drawdown) if perf else 0.0,
                'win_rate': float(perf.win_rate) if perf else 0.0,
                'profit_factor': float(perf.profit_factor) if perf else 0.0,
                'updated_at': perf.updated_at.isoformat() if perf else None
            }
            session.close()
            return web.json_response(data)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def get_watchlist(self, request):
        try:
            session = Session()
            rows = session.query(Watchlist).order_by(desc(Watchlist.added_at)).limit(100).all()
            data = [{'symbol': r.symbol, 'added_at': r.added_at.isoformat()} for r in rows]
            session.close()
            return web.json_response(data)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def get_chart(self, request):
        symbol = request.match_info.get('symbol', 'BTC')
        try:
            res = await AdvancedCryptoBot().analyze_symbol(symbol, 0)
            if not res:
                return web.Response(status=404, text="No data")
            df = res['market_data']['ohlcv']
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='blue', opacity=0.5), row=2, col=1)
            fig.update_layout(template='plotly_dark', height=600, showlegend=False)
            png = fig.to_image(format="png", width=1200, height=700)
            return web.Response(body=png, content_type='image/png')
        except Exception as e:
            logger.error(f"Chart error: {e}")
            return web.Response(status=500, text=str(e))

# Telegram Handlers
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("سلام! من ربات تحلیل کریپتو هستم. دستورها:\n/signal BTC\n/chart BTC")

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        symbol = (context.args[0] if context.args else "BTC").upper()
        bot = context.bot_data.get('bot_instance')
        await update.message.chat.send_action(ChatAction.TYPING)
        res = await bot.analyze_symbol(symbol, update.effective_chat.id)
        if not res:
            await update.message.reply_text("خطا در تحلیل. دوباره تلاش کن.")
            return
        s = res['signal']
        sent = res['sentiment_analysis'].get('sentiment','-')
        bias = res['onchain_analysis'].get('bias','-')
        price = res['market_data']['price']
        rr = res['risk_management']['risk_reward_ratio']
        msg = f"Symbol: {symbol}\nSignal: {s['signal']} ({s['confidence']:.2f})\nPrice: {price:.4f}\nSentiment: {sent} | OnChain: {bias}\nR/R: {rr:.2f}"
        await update.message.reply_text(msg)
    except Exception as e:
        logger.error(e)
        await update.message.reply_text("یه مشکلی پیش اومد.")

async def cmd_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        symbol = (context.args[0] if context.args else "BTC").upper()
        bot = context.bot_data.get('bot_instance')
        res = await bot.analyze_symbol(symbol, update.effective_chat.id)
        if not res:
            await update.message.reply_text("تحلیل در دسترس نیست.")
            return
        file = await bot.generate_chart(symbol, res)
        if file:
            await update.message.reply_photo(file)
        else:
            await update.message.reply_text("خطا در تولید نمودار.")
    except Exception as e:
        logger.error(e)
        await update.message.reply_text("خطا هنگام ساخت نمودار.")

# App Runner
START_TIME = time.time()

async def start_web_dashboard(bot):
    if not S.ENABLE_WEB_DASHBOARD:
        logger.info("Web dashboard disabled")
        return
    dashboard = WebDashboard(bot)
    runner = web.AppRunner(dashboard.app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=S.WEB_PORT)
    await site.start()
    logger.info(f"Web dashboard running on port {S.WEB_PORT}")

async def start_telegram_bot(bot_instance: AdvancedCryptoBot):
    if not S.TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set; Telegram bot won't start.")
        return
    application = ApplicationBuilder().token(S.TELEGRAM_BOT_TOKEN).build()
    application.bot_data['bot_instance'] = bot_instance
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("signal", cmd_signal))
    application.add_handler(CommandHandler("chart", cmd_chart))
    await application.initialize()
    await application.start()
    logger.info("Telegram bot started")
    await application.updater.start_polling()
    # keep running
    while True:
        await asyncio.sleep(3600)

async def main():
    bot = AdvancedCryptoBot()
    await asyncio.gather(
        start_web_dashboard(bot),
        start_telegram_bot(bot)
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass