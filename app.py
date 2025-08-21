# -*- coding: utf-8 -*-
"""
Advanced 24/7 Crypto AI Bot - Pro Version (Railway/Docker Ready)
- Advanced TA, On-chain (USD-true), Sentiment (Transformers+VADER), Prediction (RF/GB/LR + LSTM TF)
- Risk: SL long/short, portfolio heat, depth-aware sizing
- Web: aiohttp, /api secured with SECRET_TOKEN, /health, /metrics, /api/analyze
- Telegram: PTB v20, inline keyboards, notifications
"""

import os
import sys
import time
import asyncio
import logging
import json
import datetime
import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import web
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
import websockets
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
from telegram.constants import ChatAction
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, desc, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import redis
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import tweepy
import requests

# Optional heavy deps (Transformers, TF) - enabled by default, safe fallback
TRANSFORMERS_AVAILABLE = False
TF_AVAILABLE = False
try:
    from transformers import pipeline as hf_pipeline
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

# ------------------- Settings -------------------

@dataclass
class Settings:
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID")
    SECRET_TOKEN: str = os.getenv("SECRET_TOKEN", "")  # For web API protection

    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///crypto_bot.db")

    COINGECKO_API_KEY: str = os.getenv("COINGECKO_API_KEY")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY")
    CRYPTOPANIC_API_KEY: str = os.getenv("CRYPTOPANIC_API_KEY")
    ETHERSCAN_API_KEY: str = os.getenv("ETHERSCAN_API_KEY")
    POLYGONSCAN_API_KEY: str = os.getenv("POLYGONSCAN_API_KEY")
    BSCSCAN_API_KEY: str = os.getenv("BSCSCAN_API_KEY")
    TWITTER_BEARER_TOKEN: str = os.getenv("TWITTER_BEARER_TOKEN")

    OFFLINE_MODE: bool = os.getenv("OFFLINE_MODE", "false").lower() == "true"
    ENABLE_ADVANCED_TA: bool = True
    ENABLE_ONCHAIN: bool = True
    ENABLE_SENTIMENT: bool = True
    ENABLE_PREDICTION: bool = True
    ENABLE_RISK_MANAGEMENT: bool = True
    ENABLE_WEB_DASHBOARD: bool = True
    ENABLE_DEEP_PREDICTION: bool = True  # LSTM default ON (fallback if no TF)
    ENABLE_TRANSFORMER_SENTIMENT: bool = True  # default ON (fallback to VADER)

    EXCHANGES: List[str] = os.getenv("EXCHANGES", "binance").split(",")
    TIMEFRAMES: List[str] = os.getenv("TIMEFRAMES", "15m,1h,4h,1d").split(",")

    CONCURRENT_REQUESTS: int = int(os.getenv("CONCURRENT_REQUESTS", "10"))
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))

    DEFAULT_RISK_PER_TRADE: float = float(os.getenv("DEFAULT_RISK_PER_TRADE", "2.0"))  # percent
    DEFAULT_MAX_LEVERAGE: float = float(os.getenv("DEFAULT_MAX_LEVERAGE", "5.0"))
    MIN_LIQUIDITY_THRESHOLD: float = float(os.getenv("MIN_LIQUIDITY_THRESHOLD", "1000000"))
    SL_ATR_MULTIPLIER: float = float(os.getenv("SL_ATR_MULTIPLIER", "2.5"))
    MAX_SLIPPAGE_PCT: float = float(os.getenv("MAX_SLIPPAGE_PCT", "0.2"))  # % depth window

    ENABLE_MONITOR: bool = True
    MONITOR_INTERVAL: int = int(os.getenv("MONITOR_INTERVAL", "120"))
    ALERT_THRESHOLD: float = float(os.getenv("ALERT_THRESHOLD", "0.65"))

    ONCHAIN_MIN_USD: float = float(os.getenv("ONCHAIN_MIN_USD", "500000"))
    ONCHAIN_CHAINS: List[str] = os.getenv("ONCHAIN_CHAINS", "ethereum,polygon,binance-smart-chain,avalanche,arbitrum,fantom").split(",")

    WEB_PORT: int = int(os.getenv("PORT", os.getenv("WEB_PORT", "8080")))
    MODELS_DIR: str = os.getenv("MODELS_DIR", "models")

S = Settings()

# ------------------- Logger & Globals -------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("crypto_ai_bot")

START_TIME = time.time()
REQUEST_SEM = asyncio.Semaphore(S.CONCURRENT_REQUESTS)

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
    logger.warning(f"Redis not available: {e}")
    redis_available = False
    redis_client = None

def cache_set(key: str, value: Any, ttl: Optional[int] = None):
    if not redis_available:
        return
    try:
        data = json.dumps(value, default=str)
        redis_client.setex(key, int(ttl or S.CACHE_TTL_SECONDS), data)
    except Exception as e:
        logger.error(f"Redis set error: {e}")

def cache_get(key: str):
    if not redis_available:
        return None
    try:
        data = redis_client.get(key)
        return json.loads(data) if data else None
    except Exception as e:
        logger.error(f"Redis get error: {e}")
        return None

def dynamic_ttl_from_df(df: pd.DataFrame, base: int = None) -> int:
    base = base or S.CACHE_TTL_SECONDS
    if df is None or df.empty:
        return base
    vol = float(df['close'].pct_change().std() or 0)
    # higher vol => lower TTL (min 30s, max base*2)
    ttl = int(max(30, min(base * 2, base / (1 + 10 * vol))))
    return ttl

# ------------------- DB Models -------------------

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
    from_address = Column(String(64))
    to_address = Column(String(64))
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

# ------------------- Utilities -------------------

async def with_limit(coro):
    async with REQUEST_SEM:
        return await coro

async def safe_get_json(session: aiohttp.ClientSession, url: str, params: dict = None, timeout: int = None):
    try:
        async with REQUEST_SEM:
            async with session.get(url, params=params, timeout=timeout or S.REQUEST_TIMEOUT) as r:
                return await r.json()
    except Exception as e:
        logger.warning(f"GET {url} failed: {e}")
        return None

def risk_percent_from_level(level: str) -> float:
    mapping = {'low': 0.5, 'medium': 1.0, 'high': 2.0}
    try:
        return float(level)
    except Exception:
        return mapping.get((level or '').lower(), 1.0)

# ------------------- Technical Analysis -------------------

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

    def detect_order_blocks(self, df: pd.DataFrame, lookback: int = 50) -> dict:
        bullish_blocks, bearish_blocks = [], []
        for i in range(lookback, len(df)):
            if (df.iloc[i-1]['close'] < df.iloc[i-1]['open'] and 
                df.iloc[i]['close'] > df.iloc[i]['open'] and 
                df.iloc[i]['close'] > df.iloc[i-2]['high']):
                bullish_blocks.append({'index': i-1, 'high': float(df.iloc[i-1]['high']), 'low': float(df.iloc[i-1]['low'])})
            if (df.iloc[i-1]['close'] > df.iloc[i-1]['open'] and 
                df.iloc[i]['close'] < df.iloc[i]['open'] and 
                df.iloc[i]['close'] < df.iloc[i-2]['low']):
                bearish_blocks.append({'index': i-1, 'high': float(df.iloc[i-1]['high']), 'low': float(df.iloc[i-1]['low'])})
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
        structure = {'trend': 'ranging'}
        swing_highs, swing_lows = [], []
        for i in range(2, len(df)-2):
            if (df.iloc[i]['high'] > df.iloc[i-1]['high'] and df.iloc[i]['high'] > df.iloc[i+1]['high']):
                swing_highs.append((i, float(df.iloc[i]['high'])))
            if (df.iloc[i]['low'] < df.iloc[i-1]['low'] and df.iloc[i]['low'] < df.iloc[i+1]['low']):
                swing_lows.append((i, float(df.iloc[i]['low'])))
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            if (swing_highs[-1][1] > swing_highs[-2][1] and swing_lows[-1][1] > swing_lows[-2][1]):
                structure['trend'] = 'uptrend'
            elif (swing_highs[-1][1] < swing_highs[-2][1] and swing_lows[-1][1] < swing_lows[-2][1]):
                structure['trend'] = 'downtrend'
        structure['higher_highs'] = swing_highs
        structure['higher_lows'] = swing_lows
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
                inside_bars.append({'index': i, 'high': float(df.iloc[i]['high']), 'low': float(df.iloc[i]['low'])})
        return inside_bars

    def analyze_session_patterns(self, df: pd.DataFrame) -> dict:
        sessions = {'asia': {'start': 0, 'end': 8}, 'europe': {'start': 8, 'end': 16}, 'us': {'start': 13, 'end': 21}}
        session_analysis = {}
        if not isinstance(df.index, pd.DatetimeIndex):
            return session_analysis
        for name, t in sessions.items():
            s = df[(df.index.hour >= t['start']) & (df.index.hour < t['end'])]
            if not s.empty:
                session_analysis[name] = {
                    'volatility': float(s['high'].max() - s['low'].min()),
                    'volume': float(s['volume'].mean()),
                    'range': float(s['close'].pct_change().std()),
                    'direction': 'bullish' if s['close'].iloc[-1] > s['close'].iloc[0] else 'bearish'
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

# ------------------- Risk Management -------------------

class AdvancedRiskManager:
    def calculate_position_size(self, account_balance: float, risk_percent: float, 
                                entry_price: float, stop_loss: float, method: str = 'volatility_adjusted',
                                volatility: float = None, liquidity: float = None) -> float:
        risk_amount = account_balance * (risk_percent / 100.0)
        price_risk = max(abs(entry_price - stop_loss), 1e-8)
        if method == 'volatility_adjusted' and volatility is not None and np.isfinite(volatility):
            volatility_factor = 1 / (1 + float(volatility))
            position_size = (risk_amount * volatility_factor) / price_risk
        else:
            position_size = risk_amount / price_risk
        if liquidity and liquidity < S.MIN_LIQUIDITY_THRESHOLD:
            position_size *= (liquidity / S.MIN_LIQUIDITY_THRESHOLD)
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
        stop_distance = float(atr) * S.SL_ATR_MULTIPLIER
        if direction.lower() == 'long':
            return float(entry_price - stop_distance)
        else:
            return float(entry_price + stop_distance)

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

# ------------------- On-chain Analyzer (USD-correct) -------------------

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
            },
            'USDC': {
                'ethereum': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
                'polygon': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
                'binance-smart-chain': '0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d',
                'avalanche': '0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E',
                'arbitrum': '0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8',
                'fantom': '0x04068DA6C83AFCFA0e13ba15A6696662335D5B75'
            },
            'BTC': {
                'ethereum': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',
                'polygon': '0x1BFD67037B42Cf70AcE5DFE484DdC4D164933822',
                'binance-smart-chain': '0x7130d2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c',
                'avalanche': '0x152b9d0FdC40C096757F570A51E49Fbd288FE6dE',
                'arbitrum': '0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f',
                'fantom': '0x321162Cd933E2Be498Cd2267a90534A804051b11'
            },
            'ETH': {
                'ethereum': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
                'polygon': '0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619',
                'binance-smart-chain': '0x2170Ed0880ac9A755fd29B2688956BD959F933F8',
                'avalanche': '0x49D5c2BdFfac6CE2BFdB6640F4F80f226bc10bAB',
                'arbitrum': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
                'fantom': '0x74b23882a30290451A17c44f4F05243b6b58C76d'
            }
        }

    async def _price_usd(self, symbol: str) -> float:
        # Try Binance spot SYMBOL/USDT
        try:
            ex = ccxt.binance()
            ticker = await ex.fetch_ticker(f"{symbol}/USDT")
            price = float(ticker.get('last') or ticker.get('close') or 0)
            await ex.close()
            if price > 0:
                return price
        except Exception:
            try:
                await ex.close()
            except Exception:
                pass
        # CoinGecko fallback
        try:
            cg_url = "https://api.coingecko.com/api/v3/simple/price"
            params = {'ids': symbol.lower(), 'vs_currencies': 'usd', 'x_cg_pro_api_key': S.COINGECKO_API_KEY or ''}
            async with aiohttp.ClientSession() as session:
                data = await safe_get_json(session, cg_url, params=params, timeout=15)
                if data and symbol.lower() in data:
                    return float(data[symbol.lower()]['usd'])
        except Exception:
            pass
        # stable fallback
        if symbol.upper() in ('USDT', 'USDC'):
            return 1.0
        return 0.0

    async def get_large_transactions(self, symbol: str, min_usd: float = None) -> list:
        min_usd = min_usd or S.ONCHAIN_MIN_USD
        all_transactions = []
        token_map = self.token_contracts.get(symbol.upper(), {})
        if not token_map:
            return []
        price = await self._price_usd(symbol.upper()) or 0.0
        for chain in S.ONCHAIN_CHAINS:
            cfg = self.chain_apis.get(chain)
            if not cfg or not token_map.get(chain):
                continue
            try:
                txs = await self._fetch_chain_transactions(chain, token_map[chain], min_usd, price)
                all_transactions.extend(txs)
            except Exception as e:
                logger.error(f"Error fetching {chain} transactions: {e}")
        return all_transactions

    async def _fetch_chain_transactions(self, chain: str, contract_address: str, min_usd: float, price_usd: float) -> list:
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
                data = await safe_get_json(session, url, params=params, timeout=S.REQUEST_TIMEOUT)
                if not data or data.get('status') != '1':
                    return []
                transactions = []
                for tx in data.get('result', []):
                    value = float(tx.get('value', 0))
                    decimals = int(tx.get('tokenDecimal', 18))
                    amount = value / (10 ** decimals)
                    usd_value = amount * (price_usd or 1.0)
                    if usd_value >= min_usd:
                        transactions.append({
                            'hash': tx.get('hash'),
                            'from': tx.get('from'),
                            'to': tx.get('to'),
                            'value': usd_value,
                            'amount': amount,
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
        buys = sum(1 for tx in transactions if tx.get('to'))
        sells = sum(1 for tx in transactions if tx.get('from'))
        total_volume = sum(tx['value'] for tx in transactions)
        if buys > sells * 1.5:
            bias = 'bullish'
        elif sells > buys * 1.5:
            bias = 'bearish'
        else:
            bias = 'neutral'
        if total_volume > 20000000:
            activity = 'very_high'
        elif total_volume > 8000000:
            activity = 'high'
        elif total_volume > 1500000:
            activity = 'medium'
        else:
            activity = 'low'
        return {'activity': activity, 'bias': bias, 'total_volume': total_volume, 'transaction_count': len(transactions), 'buy_sell_ratio': buys / max(sells, 1)}

# ------------------- Sentiment Analyzer (Transformers + VADER) -------------------

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

        self.use_transformer = TRANSFORMERS_AVAILABLE and S.ENABLE_TRANSFORMER_SENTIMENT
        self.transformer_pipeline = None
        if self.use_transformer:
            try:
                self.transformer_pipeline = hf_pipeline("sentiment-analysis")
                logger.info("Transformer sentiment pipeline loaded")
            except Exception as e:
                self.use_transformer = False
                logger.warning(f"Transformer not usable, fallback to VADER: {e}")

    def _score_text(self, text: str) -> float:
        if self.use_transformer and self.transformer_pipeline:
            r = self.transformer_pipeline(text[:512])[0]
            label, score = r['label'], r['score']
            return float(score if label.upper().startswith('POS') else -score)
        return float(self.vader.polarity_scores(text).get('compound', 0.0))

    async def analyze_news_sentiment(self, symbol: str) -> dict:
        all_news = []
        # CryptoPanic
        if S.CRYPTOPANIC_API_KEY:
            try:
                async with aiohttp.ClientSession() as session:
                    url = "https://cryptopanic.com/api/v1/posts/"
                    params = {'auth_token': S.CRYPTOPANIC_API_KEY, 'currencies': symbol, 'kind': 'news'}
                    data = await safe_get_json(session, url, params=params, timeout=S.REQUEST_TIMEOUT)
                    if data and 'results' in data:
                        all_news += data['results']
            except Exception as e:
                logger.warning(f"CryptoPanic error: {e}")
        # NewsAPI
        if S.NEWS_API_KEY:
            try:
                async with aiohttp.ClientSession() as session:
                    url = "https://newsapi.org/v2/everything"
                    params = {'q': f'{symbol} cryptocurrency', 'apiKey': S.NEWS_API_KEY, 'language': 'en', 'sortBy': 'publishedAt', 'pageSize': 20}
                    data = await safe_get_json(session, url, params=params, timeout=S.REQUEST_TIMEOUT)
                    if data and 'articles' in data:
                        all_news += data['articles']
            except Exception as e:
                logger.warning(f"NewsAPI error: {e}")
        if not all_news:
            return {'sentiment': 'neutral', 'score': 0.0, 'sources': 0}
        scores = []
        for n in all_news:
            text = f"{n.get('title','')} {n.get('description','')}"
            scores.append(self._score_text(text))
        avg = float(np.mean(scores)) if scores else 0.0
        sentiment = 'bullish' if avg > 0.2 else ('bearish' if avg < -0.2 else 'neutral')
        return {'sentiment': sentiment, 'score': avg, 'sources': len(all_news)}

    async def analyze_twitter_sentiment(self, symbol: str) -> dict:
        if not self.twitter_available:
            return {'sentiment': 'neutral', 'score': 0.0, 'tweet_count': 0}
        try:
            tweets = self.twitter_client.search_recent_tweets(
                query=f"#{symbol} OR ${symbol} -is:retweet lang:en",
                max_results=100, tweet_fields=['created_at', 'public_metrics']
            )
            if not tweets.data:
                return {'sentiment': 'neutral', 'score': 0.0, 'tweet_count': 0}
            scores = [self._score_text(t.text) for t in tweets.data]
            avg = float(np.mean(scores)) if scores else 0.0
            sentiment = 'bullish' if avg > 0.2 else ('bearish' if avg < -0.2 else 'neutral')
            return {'sentiment': sentiment, 'score': avg, 'tweet_count': len(tweets.data)}
        except Exception as e:
            logger.warning(f"Twitter error: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'tweet_count': 0}

    async def combined_sentiment_analysis(self, symbol: str) -> dict:
        news = await self.analyze_news_sentiment(symbol)
        tw = await self.analyze_twitter_sentiment(symbol)
        score = 0.6*news['score'] + 0.4*tw['score']
        sentiment = 'bullish' if score > 0.2 else ('bearish' if score < -0.2 else 'neutral')
        return {'sentiment': sentiment, 'score': score, 'news': news, 'twitter': tw}

# ------------------- Prediction Engine (with persistence) -------------------

class EnhancedPredictionEngine:
    def __init__(self):
        self.models = {}  # key -> (rf, gb, lr)
        self.scalers = {}
        self.deep_models = {}
        os.makedirs(S.MODELS_DIR, exist_ok=True)

    def _model_path(self, symbol: str, timeframe: str) -> str:
        return os.path.join(S.MODELS_DIR, f"{symbol}_{timeframe}_cls.joblib")

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

    def _save_model(self, key: str, rf, gb, lr, scaler):
        dump({'rf': rf, 'gb': gb, 'lr': lr, 'scaler': scaler}, self._model_path(*key.split('_', 1)))

    def _load_model(self, symbol: str, timeframe: str) -> bool:
        path = self._model_path(symbol, timeframe)
        if os.path.exists(path):
            data = load(path)
            self.models[f"{symbol}_{timeframe}"] = (data['rf'], data['gb'], data['lr'])
            self.scalers[f"{symbol}_{timeframe}"] = data['scaler']
            return True
        return False

    async def ensure_model(self, df: pd.DataFrame, symbol: str, timeframe: str):
        key = f"{symbol}_{timeframe}"
        if key in self.models and key in self.scalers:
            return
        if self._load_model(symbol, timeframe):
            return
        await self.train_model(df, symbol, timeframe)

    async def train_model(self, df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        df = await self.prepare_features(df, symbol)
        feats = self._features()
        X = df[feats]
        y_raw = (df['close'].shift(-1) > df['close']).astype(float)
        y = y_raw.dropna().astype(int)
        X = X.loc[y.index]
        if len(X) < 50:
            return {'train_accuracy': 0.0, 'test_accuracy': 0.0}

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) if len(X_test) else X_train_scaled

        rf = RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(random_state=42)
        lr = LogisticRegression(max_iter=1000)

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

        key = f"{symbol}_{timeframe}"
        self.models[key] = (rf, gb, lr)
        self.scalers[key] = scaler
        # persist
        try:
            dump({'rf': rf, 'gb': gb, 'lr': lr, 'scaler': scaler}, self._model_path(symbol, timeframe))
        except Exception as e:
            logger.warning(f"Model save failed: {e}")

        # train deep model if available
        deep_info = {}
        if S.ENABLE_DEEP_PREDICTION and TF_AVAILABLE and len(df) > 250:
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
                layers.LSTM(48, return_sequences=False),
                layers.Dense(24, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=4, batch_size=64, verbose=0)
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            self.deep_models[f"{symbol}_{timeframe}"] = model
            return {'val_accuracy': float(val_acc)}
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}")
            return {}

    async def predict(self, df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        key = f"{symbol}_{timeframe}"
        if key not in self.models:
            return {}
        df = await self.prepare_features(df, symbol)
        feats = self._features()
        X_latest = df[feats].iloc[-1:].values
        scaler = self.scalers[key]
        X_scaled = scaler.transform(X_latest)
        rf, gb, lr = self.models[key]
        p = 0.5*rf.predict_proba(X_scaled)[:,1] + 0.3*gb.predict_proba(X_scaled)[:,1] + 0.2*lr.predict_proba(X_scaled)[:,1]
        probability = float(p[0])
        direction = 'BUY' if probability >= 0.5 else 'SELL'
        conf = float(max(probability, 1 - probability))

        if S.ENABLE_DEEP_PREDICTION and TF_AVAILABLE and key in self.deep_models:
            try:
                prices = df['close'].values[-33:].astype(np.float32)
                X_seq = prices[-32:].reshape(1,32,1)
                deep_p = float(self.deep_models[key].predict(X_seq, verbose=0)[0][0])
                probability = 0.7*probability + 0.3*deep_p
                direction = 'BUY' if probability >= 0.5 else 'SELL'
                conf = float(max(probability, 1 - probability))
            except Exception as e:
                logger.warning(f"Deep predict failed: {e}")

        return {'direction': direction, 'probability': probability, 'confidence': conf}

# ------------------- Binance Helpers -------------------

async def fetch_order_book_notional(symbol: str, slippage_pct: float = S.MAX_SLIPPAGE_PCT) -> float:
    # approximate available notional within slippage_pct from mid price
    ex = ccxt.binance()
    try:
        ob = await ex.fetch_order_book(f"{symbol}/USDT", limit=50)
        ticker = await ex.fetch_ticker(f"{symbol}/USDT")
        mid = float(ticker.get('last') or ticker.get('close') or 0)
        await ex.close()
        if mid <= 0:
            return 0.0
        max_dev = mid * (slippage_pct/100.0)
        # sum bids up to mid - max_dev and asks up to mid + max_dev => use min side
        bid_notional = 0.0
        for price, amount in ob['bids']:
            if price >= mid - max_dev:
                bid_notional += price * amount
            else:
                break
        ask_notional = 0.0
        for price, amount in ob['asks']:
            if price <= mid + max_dev:
                ask_notional += price * amount
            else:
                break
        return float(min(bid_notional, ask_notional))
    except Exception as e:
        try:
            await ex.close()
        except Exception:
            pass
        logger.warning(f"Order book fetch failed for {symbol}: {e}")
        return 0.0

# ------------------- Main Bot -------------------

class AdvancedCryptoBot:
    def __init__(self):
        self.ta = AdvancedTechnicalAnalyzer()
        self.risk = AdvancedRiskManager()
        self.onchain = EnhancedOnChainAnalyzer()
        self.sentiment = EnhancedSentimentAnalyzer()
        self.pred = EnhancedPredictionEngine()

    async def fetch_market_data(self, symbol: str) -> Optional[dict]:
        cache_key = f"market_data_{symbol}"
        cached = cache_get(cache_key)
        if cached:
            # restore df
            df = pd.DataFrame(cached['ohlcv'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            cached['ohlcv'] = df
            return cached

        ex = None
        try:
            ex = ccxt.binance()
            ticker = await ex.fetch_ticker(f"{symbol}/USDT")
            ohlcv = await ex.fetch_ohlcv(f"{symbol}/USDT", "1h", limit=1000)
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
                'ohlcv': df.reset_index().to_dict(orient='list')
            }
            cache_set(cache_key, data, ttl=dynamic_ttl_from_df(df))
            data['ohlcv'] = df
            return data
        except Exception as e:
            logger.error(f"Market data error {symbol}: {e}")
            return None
        finally:
            if ex:
                try: await ex.close()
                except Exception: pass

    async def analyze_multiple_timeframes(self, symbol: str) -> dict:
        async def fetch_tf(tf):
            ex = ccxt.binance()
            try:
                o = await ex.fetch_ohlcv(f"{symbol}/USDT", tf, limit=500)
                df = pd.DataFrame(o, columns=['timestamp','open','high','low','close','volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return tf, self.ta.full_analysis(df)
            except Exception as e:
                logger.warning(f"TF {tf} error: {e}")
                return tf, {}
            finally:
                try: await ex.close()
                except Exception: pass

        tasks = [with_limit(fetch_tf(tf)) for tf in S.TIMEFRAMES]
        res = await asyncio.gather(*tasks)
        tf_results = {tf: r for tf, r in res}
        confirmations = sum(1 for tf, r in tf_results.items() if r.get('market_structure',{}).get('trend')=='uptrend') - \
                        sum(1 for tf, r in tf_results.items() if r.get('market_structure',{}).get('trend')=='downtrend')
        return {'timeframes': tf_results, 'confirmations': confirmations}

    async def fetch_liquidity_notional(self, symbol: str) -> float:
        return await fetch_order_book_notional(symbol, slippage_pct=S.MAX_SLIPPAGE_PCT)

    def _generate_signal(self, ta_results, onchain_results, sentiment_results, prediction, mtf_res) -> dict:
        w_ta, w_onchain, w_sent, w_pred, w_mtf = 0.35, 0.25, 0.2, 0.1, 0.1
        score = 0.0

        trend = ta_results.get('market_structure', {}).get('trend')
        if trend == 'uptrend': score += w_ta
        elif trend == 'downtrend': score -= w_ta

        bias = onchain_results.get('bias')
        if bias == 'bullish': score += w_onchain
        elif bias == 'bearish': score -= w_onchain

        sent = sentiment_results.get('sentiment')
        if sent == 'bullish': score += w_sent
        elif sent == 'bearish': score -= w_sent

        if prediction:
            c = prediction.get('confidence', 0.5)
            if prediction.get('direction') == 'BUY': score += w_pred * c
            else: score -= w_pred * c

        confirmations = mtf_res.get('confirmations', 0) if mtf_res else 0
        if confirmations > 0: score += w_mtf * min(1.0, confirmations/len(S.TIMEFRAMES))
        elif confirmations < 0: score -= w_mtf * min(1.0, abs(confirmations)/len(S.TIMEFRAMES))

        if score > 0.6: signal = 'BUY'
        elif score < -0.6: signal = 'SELL'
        else: signal = 'HOLD'
        return {'signal': signal, 'score': float(score), 'confidence': float(abs(score))}

    async def analyze_symbol(self, symbol: str, user_id: int = 0, timeframe: str = '1h') -> Optional[dict]:
        try:
            # ensure user
            session, user = get_user_session(user_id or 0)
            if not user:
                return None
            # market data
            market = await self.fetch_market_data(symbol)
            if not market: return None
            df = market['ohlcv'] if isinstance(market['ohlcv'], pd.DataFrame) else pd.DataFrame(market['ohlcv'])
            if not isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            # analyses
            ta_results = self.ta.full_analysis(df)
            onchain_results = await self.onchain.analyze_whale_activity(symbol) if S.ENABLE_ONCHAIN else {}
            sentiment_results = await self.sentiment.combined_sentiment_analysis(symbol) if S.ENABLE_SENTIMENT else {}

            # prediction (load or train persisted)
            pred = {}
            if S.ENABLE_PREDICTION:
                await self.pred.ensure_model(df, symbol, timeframe)
                pred = await self.pred.predict(df, symbol, timeframe)

            # liquidity/depth
            liq_notional = await self.fetch_liquidity_notional(symbol)

            # MTF
            mtf = await self.analyze_multiple_timeframes(symbol)

            # signal
            signal = self._generate_signal(ta_results, onchain_results, sentiment_results, pred, mtf)

            # risk adjusted (respect BUY/SELL)
            direction = 'long' if signal['signal'] == 'BUY' else ('short' if signal['signal']=='SELL' else 'long')
            stop_loss = self.risk.dynamic_stop_loss(df, market['price'], direction)
            risk_percent = risk_percent_from_level(user.risk_level)
            position_size = self.risk.calculate_position_size(
                account_balance=user.balance, risk_percent=risk_percent,
                entry_price=market['price'], stop_loss=stop_loss,
                method='volatility_adjusted', volatility=float(df['close'].pct_change().std()),
                liquidity=liq_notional
            )
            leverage = self.risk.calculate_optimal_leverage(
                volatility=float(df['close'].pct_change().std()),
                account_balance=user.balance, position_size=position_size, max_leverage=user.max_leverage
            )
            tp1 = market['price'] + (market['price'] - stop_loss) * (2 if direction=='long' else -2)
            tp2 = market['price'] + (market['price'] - stop_loss) * (3 if direction=='long' else -3)
            rr = self.risk.calculate_risk_reward_ratio(market['price'], stop_loss, tp1)

            # portfolio heat gate (example; current_positions empty in demo)
            ok_heat = self.risk.portfolio_heat_check([], position_size, user.balance)
            if not ok_heat:
                # reduce size by half if heat exceeded
                position_size *= 0.5

            risk_params = {
                'direction': direction,
                'stop_loss': float(stop_loss),
                'take_profit_1': float(tp1),
                'take_profit_2': float(tp2),
                'position_size': float(position_size),
                'leverage': float(leverage),
                'risk_reward_ratio': float(rr),
                'liquidity_notional': float(liq_notional)
            }

            # save signal
            try:
                db = Session()
                db_signal = Signal(
                    user_id=user.id, symbol=symbol, signal_type=signal['signal'],
                    confidence=float(signal['confidence']), price=float(market['price']),
                    source='bot_analysis', timeframe=timeframe
                )
                db.add(db_signal)
                perf = db.query(Performance).filter_by(user_id=user.id).first()
                if not perf:
                    perf = Performance(user_id=user.id)
                    db.add(perf)
                perf.total_signals = (perf.total_signals or 0) + 1
                perf.updated_at = datetime.datetime.utcnow()
                db.commit()
                db.close()
            except Exception as e:
                logger.warning(f"Save signal failed: {e}")

            return {
                'symbol': symbol,
                'market_data': market,
                'technical_analysis': ta_results,
                'onchain_analysis': onchain_results,
                'sentiment_analysis': sentiment_results,
                'prediction': pred,
                'risk_management': risk_params,
                'multi_timeframe': mtf,
                'signal': signal,
                'timestamp': datetime.datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Analyze error {symbol}: {e}")
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
                if 0 <= idx < len(df)-1:
                    fig.add_shape(type="rect", x0=df.index[idx], x1=df.index[min(idx+5, len(df)-1)], y0=block['low'], y1=block['high'], fillcolor="green", opacity=0.15, layer="below", row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='blue', opacity=0.5), row=2, col=1)
            rsi = AdvancedTechnicalAnalyzer().calculate_rsi(df['close'])
            fig.add_trace(go.Scatter(x=df.index, y=rsi, mode='lines', name='RSI', line=dict(color='purple')), row=3, col=1)
            fig.update_layout(title=f'{symbol} Technical Analysis', template='plotly_dark', height=800, showlegend=True)
            img = fig.to_image(format="png", width=1200, height=800)
            return InputFile(io.BytesIO(img), filename=f'{symbol}_analysis.png')
        except Exception as e:
            logger.error(f"Chart error: {e}")
            return None

# ------------------- Web Dashboard -------------------

class WebDashboard:
    def __init__(self, bot: AdvancedCryptoBot):
        self.bot = bot
        self.app = web.Application()
        self.setup_routes()

    def authorized(self, request: web.Request) -> bool:
        if not S.SECRET_TOKEN:
            return True  # allow if no secret set (dev mode)
        key = request.headers.get("X-API-KEY") or request.query.get("key") or ""
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            key = auth.split(" ", 1)[1]
        return key == S.SECRET_TOKEN

    def setup_routes(self):
        self.app.router.add_get('/health', self.health)
        self.app.router.add_get('/metrics', self.metrics)
        self.app.router.add_get('/', self.index)
        self.app.router.add_get('/api/stats', self.get_stats)
        self.app.router.add_get('/api/signals', self.get_signals)
        self.app.router.add_get('/api/performance', self.get_performance)
        self.app.router.add_get('/api/watchlist', self.get_watchlist)
        self.app.router.add_get('/api/analyze', self.api_analyze)
        self.app.router.add_get('/chart/{symbol}', self.get_chart)

    async def health(self, request):
        return web.json_response({'status': 'ok', 'uptime': time.time() - START_TIME})

    async def metrics(self, request):
        # basic metrics
        try:
            session = Session()
            total_users = session.query(func.count(User.id)).scalar() or 0
            total_signals = session.query(func.count(Signal.id)).scalar() or 0
            session.close()
        except Exception:
            total_users = total_signals = 0
        return web.json_response({'users': total_users, 'signals': total_signals, 'uptime_sec': int(time.time()-START_TIME)})

    async def index(self, request):
        return web.Response(text=self._html(), content_type='text/html')

    def _html(self) -> str:
        return """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Crypto AI Bot Dashboard</title>
<style>body{font-family:Segoe UI,Tahoma,Geneva,Verdana,sans-serif;background:#0a0e27;color:#fff;margin:0}
.container{max-width:1200px;margin:0 auto;padding:20px}header{text-align:center;margin-bottom:20px}
h1{font-size:2em;background:linear-gradient(45deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.panel{background:#151b3d;border-radius:12px;padding:20px;margin:12px 0}
table{width:100%;border-collapse:collapse}th,td{padding:8px;border-bottom:1px solid #2a3f5f}th{background:#1a237e;color:#9aa3f0}
.signal-buy{color:#4ade80;font-weight:bold}.signal-sell{color:#f87171;font-weight:bold}.signal-hold{color:#fbbf24;font-weight:bold}
.btn{background:linear-gradient(45deg,#667eea,#764ba2);border:none;color:#fff;padding:8px 14px;border-radius:18px;cursor:pointer}
</style></head><body><div class="container">
<header><h1>🤖 Advanced Crypto AI Bot</h1><p>Real-time analysis and trading signals</p></header>
<div class="panel"><h2>Stats</h2><div id="stats"></div></div>
<div class="panel"><h2>Recent Signals</h2><table><thead><tr><th>Symbol</th><th>Signal</th><th>Conf</th><th>Price</th><th>Time</th></tr></thead><tbody id="signals"></tbody></table></div>
<div class="panel"><h2>Watchlist</h2><table><thead><tr><th>Symbol</th><th>Added</th></tr></thead><tbody id="watchlist"></tbody></table></div>
<script>
async function loadStats(){const r=await fetch('/api/stats');const s=await r.json();document.getElementById('stats').innerHTML=`Users: ${s.total_users||0} | Signals: ${s.total_signals||0} | Uptime: ${Math.floor((s.uptime||0)/3600)}h`; }
async function loadSignals(){const r=await fetch('/api/signals');const arr=await r.json();document.getElementById('signals').innerHTML=arr.map(s=>`<tr><td>${s.symbol}</td><td class="signal-${s.signal_type.toLowerCase()}">${s.signal_type}</td><td>${(s.confidence*100).toFixed(1)}%</td><td>$${Number(s.price).toFixed(4)}</td><td>${new Date(s.timestamp).toLocaleString()}</td></tr>`).join('');}
async function loadWatch(){const r=await fetch('/api/watchlist');const arr=await r.json();document.getElementById('watchlist').innerHTML=arr.map(w=>`<tr><td>${w.symbol}</td><td>${new Date(w.added_at).toLocaleString()}</td></tr>`).join('');}
async function refresh(){try{await Promise.all([loadStats(),loadSignals(),loadWatch()]);}catch(e){}}
refresh();setInterval(refresh,15000);
</script></div></body></html>"""

    async def get_stats(self, request):
        if not self.authorized(request):
            return web.json_response({'error': 'unauthorized'}, status=401)
        try:
            session = Session()
            total_users = session.query(func.count(User.id)).scalar() or 0
            total_signals = session.query(func.count(Signal.id)).scalar() or 0
            perf = session.query(Performance).first()
            win_rate = float(perf.win_rate) if perf and perf.win_rate else 0.0
            uptime = float(time.time() - START_TIME)
            session.close()
            return web.json_response({'total_users': total_users, 'total_signals': total_signals, 'win_rate': win_rate, 'uptime': uptime})
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def get_signals(self, request):
        if not self.authorized(request):
            return web.json_response({'error': 'unauthorized'}, status=401)
        try:
            session = Session()
            rows = session.query(Signal).order_by(desc(Signal.timestamp)).limit(50).all()
            data = [{'symbol': r.symbol, 'signal_type': r.signal_type, 'confidence': float(r.confidence or 0), 'price': float(r.price or 0), 'timestamp': r.timestamp.isoformat()} for r in rows]
            session.close()
            return web.json_response(data)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def get_performance(self, request):
        if not self.authorized(request):
            return web.json_response({'error': 'unauthorized'}, status=401)
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
        if not self.authorized(request):
            return web.json_response({'error': 'unauthorized'}, status=401)
        try:
            session = Session()
            rows = session.query(Watchlist).order_by(desc(Watchlist.added_at)).limit(200).all()
            data = [{'symbol': r.symbol, 'added_at': r.added_at.isoformat()} for r in rows]
            session.close()
            return web.json_response(data)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def api_analyze(self, request):
        if not self.authorized(request):
            return web.json_response({'error': 'unauthorized'}, status=401)
        symbol = request.query.get('symbol', 'BTC').upper()
        res = await self.bot.analyze_symbol(symbol, 0)
        if not res:
            return web.json_response({'error': 'analysis_failed'}, status=500)
        # sanitize heavy fields
        out = dict(res)
        out['market_data'] = {k: v for k, v in res['market_data'].items() if k != 'ohlcv'}
        return web.json_response(out)

    async def get_chart(self, request):
        if not self.authorized(request):
            return web.Response(status=401, text="unauthorized")
        symbol = request.match_info.get('symbol', 'BTC').upper()
        try:
            res = await self.bot.analyze_symbol(symbol, 0)
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

# ------------------- Telegram Bot -------------------

# Inline callbacks
CALLBACK_ADD_WATCH = "add_watch"
CALLBACK_REM_WATCH = "rem_watch"
CALLBACK_TF_PREFIX = "tf_"
CALLBACK_NOTIFY_TOGGLE = "notify_toggle"

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("سلام! 👋\nدستورها:\n/signal BTC\n/chart BTC\n/watchlist\n/notify on|off")

def _signal_keyboard(symbol: str):
    tfs = ['15m','1h','4h','1d']
    buttons = [
        [InlineKeyboardButton("➕ واچ‌لیست", callback_data=f"{CALLBACK_ADD_WATCH}:{symbol}"), InlineKeyboardButton("➖ حذف", callback_data=f"{CALLBACK_REM_WATCH}:{symbol}")],
        [InlineKeyboardButton("15m", callback_data=f"{CALLBACK_TF_PREFIX}{symbol}:15m"),
         InlineKeyboardButton("1h", callback_data=f"{CALLBACK_TF_PREFIX}{symbol}:1h"),
         InlineKeyboardButton("4h", callback_data=f"{CALLBACK_TF_PREFIX}{symbol}:4h"),
         InlineKeyboardButton("1d", callback_data=f"{CALLBACK_TF_PREFIX}{symbol}:1d")],
        [InlineKeyboardButton("🔔 نوتیف", callback_data=f"{CALLBACK_NOTIFY_TOGGLE}:{symbol}")]
    ]
    return InlineKeyboardMarkup(buttons)

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = (context.args[0] if context.args else "BTC").upper()
    bot: AdvancedCryptoBot = context.application.bot_data['bot_instance']
    await update.message.chat.send_action(ChatAction.TYPING)
    res = await bot.analyze_symbol(symbol, update.effective_chat.id)
    if not res:
        await update.message.reply_text("خطا در تحلیل. دوباره تلاش کن.")
        return
    s = res['signal']; sent = res['sentiment_analysis'].get('sentiment','-'); bias = res['onchain_analysis'].get('bias','-')
    price = res['market_data']['price']; rr = res['risk_management']['risk_reward_ratio']
    liq = res['risk_management']['liquidity_notional']
    msg = f"Symbol: {symbol}\nSignal: {s['signal']} ({s['confidence']:.2f})\nPrice: {price:.4f}\nSentiment: {sent} | OnChain: {bias}\nR/R: {rr:.2f}\nLiquidity(~{S.MAX_SLIPPAGE_PCT}%): ${liq:,.0f}"
    await update.message.reply_text(msg, reply_markup=_signal_keyboard(symbol))

async def cmd_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = (context.args[0] if context.args else "BTC").upper()
    bot: AdvancedCryptoBot = context.application.bot_data['bot_instance']
    res = await bot.analyze_symbol(symbol, update.effective_chat.id)
    if not res:
        await update.message.reply_text("تحلیل در دسترس نیست.")
        return
    file = await bot.generate_chart(symbol, res)
    if file:
        await update.message.reply_photo(file)
    else:
        await update.message.reply_text("خطا در تولید نمودار.")

async def cmd_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = Session()
    user = session.query(User).filter_by(chat_id=update.effective_chat.id).first()
    if not user:
        user = User(chat_id=update.effective_chat.id)
        session.add(user); session.commit()
    rows = session.query(Watchlist).filter_by(user_id=user.id).order_by(desc(Watchlist.added_at)).all()
    session.close()
    if not rows:
        await update.message.reply_text("واچ‌لیست خالی است.")
        return
    text = "واچ‌لیست شما:\n" + "\n".join([f"- {r.symbol}" for r in rows])
    await update.message.reply_text(text)

async def cmd_notify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # simple toggle in user.preferences
    opt = (context.args[0] if context.args else "").lower()
    session, user = get_user_session(update.effective_chat.id)
    prefs = json.loads(user.preferences or '{}')
    if opt in ('on','off'):
        prefs['notify'] = (opt == 'on')
        user.preferences = json.dumps(prefs)
        session.commit()
        await update.message.reply_text(f"نوتیف {'فعال' if prefs['notify'] else 'غیرفعال'} شد.")
    else:
        await update.message.reply_text("استفاده: /notify on یا /notify off")

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    session, user = get_user_session(update.effective_chat.id)
    if data.startswith(CALLBACK_ADD_WATCH):
        _, symbol = data.split(":")
        session.add(Watchlist(user_id=user.id, symbol=symbol))
        session.commit()
        await query.edit_message_text(f"{symbol} به واچ‌لیست اضافه شد ✅")
    elif data.startswith(CALLBACK_REM_WATCH):
        _, symbol = data.split(":")
        session.query(Watchlist).filter_by(user_id=user.id, symbol=symbol).delete()
        session.commit()
        await query.edit_message_text(f"{symbol} از واچ‌لیست حذف شد ✅")
    elif data.startswith(CALLBACK_TF_PREFIX):
        payload = data[len(CALLBACK_TF_PREFIX):]
        symbol, tf = payload.split(":")
        bot: AdvancedCryptoBot = context.application.bot_data['bot_instance']
        res = await bot.analyze_symbol(symbol.upper(), update.effective_chat.id, timeframe=tf)
        if not res:
            await query.edit_message_text("خطا در تحلیل تایم‌فریم.")
            return
        s = res['signal']; price = res['market_data']['price']
        await query.edit_message_text(f"[{tf}] {symbol}: {s['signal']} ({s['confidence']:.2f}) @ {price:.4f}")
    elif data.startswith(CALLBACK_NOTIFY_TOGGLE):
        prefs = json.loads(user.preferences or '{}')
        prefs['notify'] = not prefs.get('notify', True)
        user.preferences = json.dumps(prefs)
        session.commit()
        await query.edit_message_text(f"نوتیف: {'فعال' if prefs['notify'] else 'غیرفعال'}")
    session.close()

# ------------------- Monitor & Notifications -------------------

async def monitor_task(app, bot: AdvancedCryptoBot):
    last_alert_key = "last_alerts"  # in redis: map of user:symbol -> score
    inmem_last = {}
    while True:
        try:
            session = Session()
            users = session.query(User).all()
            for u in users:
                prefs = json.loads(u.preferences or '{}')
                if prefs.get('notify', True) is False:
                    continue
                wl = session.query(Watchlist).filter_by(user_id=u.id).all()
                for w in wl:
                    res = await bot.analyze_symbol(w.symbol, u.chat_id)
                    if not res:
                        continue
                    sig = res['signal']
                    if sig['signal'] == 'HOLD' or sig['confidence'] < S.ALERT_THRESHOLD:
                        continue
                    key = f"{u.id}:{w.symbol}"
                    last = cache_get(key) if redis_available else inmem_last.get(key)
                    # avoid spamming identical signal
                    if last and abs(last.get('score', 0) - sig['score']) < 0.05 and last.get('signal') == sig['signal']:
                        continue
                    text = f"هشدار {w.symbol}: {sig['signal']} ({sig['confidence']:.2f})\nقیمت: {res['market_data']['price']:.4f}"
                    try:
                        await app.bot.send_message(chat_id=u.chat_id, text=text)
                    except Exception as e:
                        logger.warning(f"Notify send failed: {e}")
                    store = {'signal': sig['signal'], 'score': sig['score'], 'ts': time.time()}
                    if redis_available: cache_set(key, store, ttl=600)
                    else: inmem_last[key] = store
            session.close()
        except Exception as e:
            logger.warning(f"Monitor loop error: {e}")
        await asyncio.sleep(S.MONITOR_INTERVAL)

# ------------------- Runner -------------------

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

async def main():
    bot = AdvancedCryptoBot()
    application = ApplicationBuilder().token(S.TELEGRAM_BOT_TOKEN).build()
    application.bot_data['bot_instance'] = bot
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("signal", cmd_signal))
    application.add_handler(CommandHandler("chart", cmd_chart))
    application.add_handler(CommandHandler("watchlist", cmd_watchlist))
    application.add_handler(CommandHandler("notify", cmd_notify))
    application.add_handler(CallbackQueryHandler(on_callback))

    # Start tasks concurrently: web, monitor, telegram
    await asyncio.gather(
        start_web_dashboard(bot),
        monitor_task(application, bot),
        application.run_polling()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass