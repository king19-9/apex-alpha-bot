# -*- coding: utf-8 -*-
"""
Advanced 24/7 Crypto AI Bot - Enhanced Version with Dynamic Dependency Loading
"""

import os
import sys
import time
import asyncio
import logging
import json
import datetime
import subprocess
import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports (lightweight)
import aiohttp
from aiohttp import web
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
import websockets
import json as _json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
from telegram.constants import ChatAction
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import redis
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tweepy
from bs4 import BeautifulSoup
import requests
import prometheus_client
from prometheus_client import Counter, Histogram

# Global variables for heavy dependencies
transformers_available = False
torch_available = False
psycopg2_available = False

# Initialize Prometheus metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

# Initialize Redis
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    redis_available = True
except Exception as e:
    logging.warning(f"Redis not available: {e}")
    redis_available = False
    redis_client = None

# Database Setup with fallback
try:
    # Try to use PostgreSQL if available
    if os.getenv("DATABASE_URL", "").startswith("postgresql"):
        try:
            import psycopg2
            psycopg2_available = True
            logger.info("psycopg2 imported successfully")
        except ImportError:
            logger.warning("psycopg2 not available, falling back to SQLite")
            psycopg2_available = False
            # Fallback to SQLite
            os.environ["DATABASE_URL"] = "sqlite:///crypto_bot.db"
    
    # Create database engine
    engine = create_engine(
        os.getenv("DATABASE_URL", "sqlite:///crypto_bot.db"), 
        pool_pre_ping=True, 
        pool_size=20, 
        max_overflow=30
    )
    Session = sessionmaker(bind=engine)
    
    # Create tables
    Base = declarative_base()
    
    # Define models
    class User(Base):
        __tablename__ = 'users'
        id = Column(Integer, primary_key=True)
        chat_id = Column(Integer, unique=True)
        username = Column(String(50))
        preferences = Column(Text)
        risk_level = Column(String(20), default='medium')
        max_leverage = Column(Float, default=5.0)
        balance = Column(Float, default=10000.0)
        created_at = Column(DateTime, default=datetime.datetime.utcnow)
        language = Column(String(10), default='en')
        
        signals = relationship("Signal", back_populates="user")
        watchlist = relationship("Watchlist", back_populates="user")
        performance = relationship("Performance", back_populates="user")

    class Signal(Base):
        __tablename__ = 'signals'
        id = Column(Integer, primary_key=True)
        symbol = Column(String(20))
        signal_type = Column(String(10))
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

    class ArbitrageOpportunity(Base):
        __tablename__ = 'arbitrage_opportunities'
        id = Column(Integer, primary_key=True)
        symbol = Column(String(20))
        buy_exchange = Column(String(50))
        sell_exchange = Column(String(50))
        buy_price = Column(Float)
        sell_price = Column(Float)
        profit_percentage = Column(Float)
        timestamp = Column(DateTime, default=datetime.datetime.utcnow)
        executed = Column(Boolean, default=False)
    
    # Create tables
    Base.metadata.create_all(engine)
    logger.info("Database connected successfully")
    
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    # Create a minimal fallback for the app to run
    Base = declarative_base()
    engine = None
    Session = None
    sys.exit(1)

# Settings Class
@dataclass
class Settings:
    # Bot Configuration
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///crypto_bot.db")
    
    # API Keys
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
    
    # Exchange API Keys
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET")
    KUCOIN_API_KEY: str = os.getenv("KUCOIN_API_KEY")
    KUCOIN_API_SECRET: str = os.getenv("KUCOIN_API_SECRET")
    
    # Bot Settings
    OFFLINE_MODE: bool = os.getenv("OFFLINE_MODE", "false").lower() == "true"
    EXCHANGES: List[str] = field(default_factory=lambda: os.getenv("EXCHANGES", "binance,kucoin,bybit,bitfinex,gateio,bitget").split(","))
    MAX_COINS: int = int(os.getenv("MAX_COINS", "1000"))
    UNIVERSE_MAX_PAGES: int = int(os.getenv("UNIVERSE_MAX_PAGES", "20"))
    TIMEFRAMES: List[str] = field(default_factory=lambda: os.getenv("TIMEFRAMES", "1m,5m,15m,1h,4h,1d").split(","))
    
    # Performance Settings
    CONCURRENT_REQUESTS: int = int(os.getenv("CONCURRENT_REQUESTS", "15"))
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    
    # Analysis Settings
    ENABLE_ADVANCED_TA: bool = os.getenv("ENABLE_ADVANCED_TA", "true").lower() == "true"
    ENABLE_ONCHAIN: bool = os.getenv("ENABLE_ONCHAIN", "true").lower() == "true"
    ENABLE_SENTIMENT: bool = os.getenv("ENABLE_SENTIMENT", "true").lower() == "true"
    ENABLE_PREDICTION: bool = os.getenv("ENABLE_PREDICTION", "true").lower() == "true"
    ENABLE_RISK_MANAGEMENT: bool = os.getenv("ENABLE_RISK_MANAGEMENT", "true").lower() == "true"
    ENABLE_ARBITRAGE: bool = os.getenv("ENABLE_ARBITRAGE", "true").lower() == "true"
    ENABLE_TRANSFORMERS: bool = os.getenv("ENABLE_TRANSFORMERS", "true").lower() == "true"
    
    # Risk Management
    DEFAULT_RISK_PER_TRADE: float = float(os.getenv("DEFAULT_RISK_PER_TRADE", "0.02"))
    DEFAULT_MAX_LEVERAGE: float = float(os.getenv("DEFAULT_MAX_LEVERAGE", "5.0"))
    MIN_LIQUIDITY_THRESHOLD: float = float(os.getenv("MIN_LIQUIDITY_THRESHOLD", "1000000"))
    
    # Monitoring
    ENABLE_MONITOR: bool = os.getenv("ENABLE_MONITOR", "true").lower() == "true"
    MONITOR_INTERVAL: int = int(os.getenv("MONITOR_INTERVAL", "120"))
    ALERT_THRESHOLD: float = float(os.getenv("ALERT_THRESHOLD", "0.7"))
    
    # On-chain Settings
    ONCHAIN_MIN_USD: float = float(os.getenv("ONCHAIN_MIN_USD", "500000"))
    ONCHAIN_CHAINS: List[str] = field(default_factory=lambda: os.getenv("ONCHAIN_CHAINS", "ethereum,polygon,binance-smart-chain,avalanche,arbitrum,fantom").split(","))
    
    # Web Dashboard
    ENABLE_WEB_DASHBOARD: bool = os.getenv("ENABLE_WEB_DASHBOARD", "true").lower() == "true"
    WEB_PORT: int = int(os.getenv("WEB_PORT", "8080"))
    
    # Backup Settings
    ENABLE_BACKUP: bool = os.getenv("ENABLE_BACKUP", "true").lower() == "true"
    BACKUP_INTERVAL_HOURS: int = int(os.getenv("BACKUP_INTERVAL_HOURS", "24"))
    CLOUD_STORAGE_BUCKET: str = os.getenv("CLOUD_STORAGE_BUCKET", "")

S = Settings()

# Logger Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Dynamic dependency loader
async def install_heavy_dependencies():
    global transformers_available, torch_available, psycopg2_available
    
    if not S.ENABLE_TRANSFORMERS:
        return
    
    try:
        # Check if already installed
        import transformers
        import torch
        transformers_available = True
        torch_available = True
        logger.info("Heavy dependencies already available")
        return
    except ImportError:
        logger.info("Installing heavy dependencies...")
    
    try:
        # Install heavy dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-heavy.txt"])
        
        # Import after installation
        import transformers
        import torch
        transformers_available = True
        torch_available = True
        logger.info("Heavy dependencies installed successfully")
    except Exception as e:
        logger.error(f"Failed to install heavy dependencies: {e}")
        transformers_available = False
        torch_available = False

# Helper Functions
def get_user_session(chat_id: int):
    if not Session:
        return None, None
        
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

def cache_with_redis(key: str, value: Any, ttl: int = 300):
    if not redis_available:
        return
    try:
        redis_client.setex(key, ttl, json.dumps(value, default=str))
    except Exception as e:
        logger.error(f"Redis error: {e}")

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

async def retry_request(func, *args, **kwargs):
    max_retries = 3
    for i in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if i == max_retries - 1:
                raise
            await asyncio.sleep(2 ** i)
            logger.warning(f"Retry {i+1} for {func.__name__}: {e}")

# Localization System
class Localization:
    def __init__(self):
        self.translations = {
            'en': {
                'buy_signal': 'Buy signal detected',
                'sell_signal': 'Sell signal detected',
                'hold_signal': 'Hold signal detected',
                'high_confidence': 'High confidence signal',
                'whale_activity': 'Large whale activity detected',
                'market_volatility': 'High market volatility detected',
                'arbitrage_opportunity': 'Arbitrage opportunity found',
                'system_error': 'System error occurred',
                'installing_deps': 'Installing advanced AI models...'
            },
            'fa': {
                'buy_signal': 'سیگنال خرید شناسایی شد',
                'sell_signal': 'سیگنال فروش شناسایی شد',
                'hold_signal': 'سیگنال نگهداری شناسایی شد',
                'high_confidence': 'سیگنال با اعتماد بالا',
                'whale_activity': 'فعالیت بزرگ نهنگ‌ها شناسایی شد',
                'market_volatility': 'نوسانات بالای بازار شناسایی شد',
                'arbitrage_opportunity': 'فرصت آربیتراژ یافت شد',
                'system_error': 'خطای سیستمی رخ داد',
                'installing_deps': 'در حال نصب مدل‌های هوش مصنوعی پیشرفته...'
            }
        }
    
    def get_text(self, key: str, lang: str = 'en') -> str:
        return self.translations.get(lang, {}).get(key, key)

localization = Localization()

# Advanced Technical Analysis Module
class AdvancedTechnicalAnalyzer:
    def __init__(self):
        self.indicators_cache = {}
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
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
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        williams_r = (highest_high - df['close']) / (highest_high - lowest_low) * -100
        return williams_r
    
    def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad)
        return cci
    
    def calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        
        positive_flow = mf.where(tp > tp.shift(1), 0).rolling(window=period).sum()
        negative_flow = mf.where(tp < tp.shift(1), 0).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi
    
    def detect_doji(self, df: pd.DataFrame) -> pd.Series:
        body = np.abs(df['close'] - df['open'])
        range_size = df['high'] - df['low']
        doji = body < (range_size * 0.1)
        return doji
    
    def detect_hammer(self, df: pd.DataFrame) -> pd.Series:
        body = np.abs(df['close'] - df['open'])
        upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
        lower_shadow = np.minimum(df['open'], df['close']) - df['low']
        
        hammer = (body < (upper_shadow + lower_shadow) * 0.3) & (lower_shadow > body * 2)
        return hammer
    
    def calculate_volume_profile(self, df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
        price_bins = pd.cut(df['close'], bins=bins)
        volume_profile = df.groupby(price_bins)['volume'].sum()
        return volume_profile
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    
    def detect_order_blocks(self, df: pd.DataFrame, lookback: int = 50) -> dict:
        bullish_blocks = []
        bearish_blocks = []
        
        for i in range(lookback, len(df)):
            if (df.iloc[i-1]['close'] < df.iloc[i-1]['open'] and 
                df.iloc[i]['close'] > df.iloc[i]['open'] and      
                df.iloc[i]['close'] > df.iloc[i-2]['high']):      
                
                bullish_blocks.append({
                    'index': i-1,
                    'high': df.iloc[i-1]['high'],
                    'low': df.iloc[i-1]['low'],
                    'close': df.iloc[i-1]['close'],
                    'type': 'bullish'
                })
            
            if (df.iloc[i-1]['close'] > df.iloc[i-1]['open'] and  
                df.iloc[i]['close'] < df.iloc[i]['open'] and      
                df.iloc[i]['close'] < df.iloc[i-2]['low']):      
                
                bearish_blocks.append({
                    'index': i-1,
                    'high': df.iloc[i-1]['high'],
                    'low': df.iloc[i-1]['low'],
                    'close': df.iloc[i-1]['close'],
                    'type': 'bearish'
                })
        
        return {'bullish': bullish_blocks, 'bearish': bearish_blocks}
    
    def detect_liquidity_zones(self, df: pd.DataFrame, window: int = 20) -> dict:
        high_zones = []
        low_zones = []
        
        for i in range(window, len(df)-window):
            if df.iloc[i]['high'] == df.iloc[i-window:i+window]['high'].max():
                strength = len([x for x in df.iloc[i-window:i+window]['high'] 
                              if abs(x - df.iloc[i]['high']) < 0.01])
                high_zones.append({
                    'price': df.iloc[i]['high'],
                    'index': i,
                    'strength': strength
                })
            
            if df.iloc[i]['low'] == df.iloc[i-window:i+window]['low'].min():
                strength = len([x for x in df.iloc[i-window:i+window]['low'] 
                              if abs(x - df.iloc[i]['low']) < 0.01])
                low_zones.append({
                    'price': df.iloc[i]['low'],
                    'index': i,
                    'strength': strength
                })
        
        return {'high_zones': high_zones, 'low_zones': low_zones}
    
    def detect_market_structure(self, df: pd.DataFrame) -> dict:
        structure = {
            'trend': 'ranging',
            'higher_highs': [],
            'higher_lows': [],
            'lower_highs': [],
            'lower_lows': []
        }
        
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(df)-2):
            if (df.iloc[i]['high'] > df.iloc[i-1]['high'] and 
                df.iloc[i]['high'] > df.iloc[i+1]['high']):
                swing_highs.append((i, df.iloc[i]['high']))
            
            if (df.iloc[i]['low'] < df.iloc[i-1]['low'] and 
                df.iloc[i]['low'] < df.iloc[i+1]['low']):
                swing_lows.append((i, df.iloc[i]['low']))
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            if (swing_highs[-1][1] > swing_highs[-2][1] and 
                swing_lows[-1][1] > swing_lows[-2][1]):
                structure['trend'] = 'uptrend'
                structure['higher_highs'] = swing_highs
                structure['higher_lows'] = swing_lows
            
            elif (swing_highs[-1][1] < swing_highs[-2][1] and 
                  swing_lows[-1][1] < swing_lows[-2][1]):
                structure['trend'] = 'downtrend'
                structure['lower_highs'] = swing_highs
                structure['lower_lows'] = swing_lows
        
        return structure
    
    def detect_supply_demand_zones(self, df: pd.DataFrame, lookback: int = 50) -> dict:
        supply_zones = []
        demand_zones = []
        
        for i in range(lookback, len(df)):
            if (df.iloc[i]['close'] > df.iloc[i-1]['close'] * 1.02 and  
                df.iloc[i]['volume'] > df.iloc[i-1]['volume'] * 1.5):  
                
                demand_zones.append({
                    'price': df.iloc[i-1]['close'],
                    'strength': df.iloc[i]['volume'] / df.iloc[i-1]['volume'],
                    'index': i-1
                })
            
            if (df.iloc[i]['close'] < df.iloc[i-1]['close'] * 0.98 and  
                df.iloc[i]['volume'] > df.iloc[i-1]['volume'] * 1.5):  
                
                supply_zones.append({
                    'price': df.iloc[i-1]['close'],
                    'strength': df.iloc[i]['volume'] / df.iloc[i-1]['volume'],
                    'index': i-1
                })
        
        return {'supply': supply_zones, 'demand': demand_zones}
    
    def detect_inside_bars(self, df: pd.DataFrame) -> list:
        inside_bars = []
        
        for i in range(1, len(df)):
            if (df.iloc[i]['high'] < df.iloc[i-1]['high'] and 
                df.iloc[i]['low'] > df.iloc[i-1]['low']):
                inside_bars.append({
                    'index': i,
                    'high': df.iloc[i]['high'],
                    'low': df.iloc[i]['low'],
                    'mother_high': df.iloc[i-1]['high'],
                    'mother_low': df.iloc[i-1]['low']
                })
        
        return inside_bars
    
    def analyze_session_patterns(self, df: pd.DataFrame) -> dict:
        sessions = {
            'asia': {'start': 0, 'end': 8},
            'europe': {'start': 8, 'end': 16},
            'us': {'start': 13, 'end': 21}
        }
        
        session_analysis = {}
        
        for session_name, times in sessions.items():
            session_df = df[df.index.hour >= times['start']]
            session_df = session_df[session_df.index.hour < times['end']]
            
            if not session_df.empty:
                session_analysis[session_name] = {
                    'volatility': session_df['high'].max() - session_df['low'].min(),
                    'volume': session_df['volume'].mean(),
                    'range': session_df['close'].pct_change().std(),
                    'direction': 'bullish' if session_df['close'].iloc[-1] > session_df['close'].iloc[0] else 'bearish'
                }
        
        return session_analysis
    
    def detect_decision_zones(self, df: pd.DataFrame) -> list:
        decision_zones = []
        
        for i in range(10, len(df)-10):
            window = df.iloc[i-10:i+10]
            
            volatility = (window['high'].max() - window['low'].min()) / window['close'].mean()
            avg_volume = window['volume'].mean()
            
            if volatility < 0.02 and avg_volume > df['volume'].quantile(0.7):
                decision_zones.append({
                    'price': window['close'].mean(),
                    'strength': avg_volume / df['volume'].mean(),
                    'range': window['high'].max() - window['low'].min(),
                    'index': i
                })
        
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
                'rsi': self.calculate_rsi(df['close']).iloc[-1],
                'macd': self.calculate_macd(df['close'])[0].iloc[-1],
                'bollinger': self.calculate_bollinger_bands(df['close']),
                'atr': self.calculate_atr(df).iloc[-1],
                'stochastic': self.calculate_stochastic(df)[0].iloc[-1],
                'williams_r': self.calculate_williams_r(df).iloc[-1],
                'cci': self.calculate_cci(df).iloc[-1],
                'mfi': self.calculate_mfi(df).iloc[-1],
                'vwap': self.calculate_vwap(df).iloc[-1],
                'doji': self.detect_doji(df).iloc[-1],
                'hammer': self.detect_hammer(df).iloc[-1]
            }
        
        return results

# Enhanced Risk Management Module
class AdvancedRiskManager:
    def __init__(self):
        self.position_sizing_methods = ['fixed_risk', 'kelly', 'volatility_adjusted']
        self.risk_metrics = {}
    
    def calculate_position_size(self, account_balance: float, risk_percent: float, 
                              entry_price: float, stop_loss: float, method: str = 'fixed_risk',
                              volatility: float = None, liquidity: float = None) -> float:
        risk_amount = account_balance * (risk_percent / 100)
        
        if method == 'fixed_risk':
            position_size = risk_amount / abs(entry_price - stop_loss)
        
        elif method == 'kelly':
            win_rate = self.risk_metrics.get('win_rate', 0.5)
            avg_win = self.risk_metrics.get('avg_win', 1.0)
            avg_loss = self.risk_metrics.get('avg_loss', 1.0)
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))
            
            position_size = (account_balance * kelly_fraction) / entry_price
        
        elif method == 'volatility_adjusted':
            if volatility:
                volatility_factor = 1 / (1 + volatility)
                position_size = (risk_amount * volatility_factor) / abs(entry_price - stop_loss)
            else:
                position_size = risk_amount / abs(entry_price - stop_loss)
        
        if liquidity and liquidity < S.MIN_LIQUIDITY_THRESHOLD:
            liquidity_factor = liquidity / S.MIN_LIQUIDITY_THRESHOLD
            position_size *= liquidity_factor
        
        return max(0, position_size)
    
    def calculate_optimal_leverage(self, volatility: float, account_balance: float, 
                                  position_size: float, max_leverage: float = None) -> float:
        if not max_leverage:
            max_leverage = S.DEFAULT_MAX_LEVERAGE
        
        volatility_adjustment = 1 / (1 + volatility)
        base_leverage = (position_size * S.DEFAULT_MAX_LEVERAGE) / account_balance
        optimal_leverage = base_leverage * volatility_adjustment
        
        return min(optimal_leverage, max_leverage)
    
    def dynamic_stop_loss(self, df: pd.DataFrame, entry_price: float, 
                         direction: str, atr_period: int = 14) -> float:
        atr = self.calculate_atr(df, atr_period).iloc[-1]
        
        if direction == 'long':
            stop_distance = atr * 2.5
            stop_loss = entry_price - stop_distance
        else:
            stop_distance = atr * 2.5
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, 
                                  take_profit: float) -> float:
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return 0
        
        return reward / risk
    
    def portfolio_heat_check(self, current_positions: List[Dict], 
                           new_position_size: float, account_balance: float) -> bool:
        total_risk = sum(pos['size'] * pos['risk_percent'] for pos in current_positions)
        new_position_risk = new_position_size * S.DEFAULT_RISK_PER_TRADE
        
        total_portfolio_risk = (total_risk + new_position_risk) / account_balance
        max_portfolio_risk = 0.20
        
        return total_portfolio_risk <= max_portfolio_risk
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()
    
    async def calculate_dynamic_risk(self, market_conditions: dict) -> dict:
        volatility = market_conditions.get('volatility', 0)
        trend_strength = market_conditions.get('trend_strength', 0)
        
        if volatility > 0.3:  
            risk_per_trade = 0.01
            max_leverage = 2.0
        elif trend_strength > 0.7:  
            risk_per_trade = 0.03
            max_leverage = 5.0
        else:
            risk_per_trade = 0.02
            max_leverage = 3.0
            
        return {
            'risk_per_trade': risk_per_trade,
            'max_leverage': max_leverage
        }

# Enhanced On-chain Analysis Module
class EnhancedOnChainAnalyzer:
    def __init__(self):
        self.chain_apis = {
            'ethereum': {
                'url': 'https://api.etherscan.io/api',
                'key': S.ETHERSCAN_API_KEY,
                'explorer': 'https://etherscan.io'
            },
            'polygon': {
                'url': 'https://api.polygonscan.com/api',
                'key': S.POLYGONSCAN_API_KEY,
                'explorer': 'https://polygonscan.com'
            },
            'binance-smart-chain': {
                'url': 'https://api.bscscan.com/api',
                'key': S.BSCSCAN_API_KEY,
                'explorer': 'https://bscscan.com'
            },
            'avalanche': {
                'url': 'https://api.snowtrace.io/api',
                'key': os.getenv('AVALANCHE_API_KEY'),
                'explorer': 'https://snowtrace.io'
            },
            'arbitrum': {
                'url': 'https://api.arbiscan.io/api',
                'key': os.getenv('ARBITRUM_API_KEY'),
                'explorer': 'https://arbiscan.io'
            },
            'fantom': {
                'url': 'https://api.ftmscan.com/api',
                'key': os.getenv('FANTOM_API_KEY'),
                'explorer': 'https://ftmscan.com'
            }
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
                'ethereum': '0xA0b86a33E6417aAb7b6DbCBbe9FD4E89c0778a4B',
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
                'arbitrum': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
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
    
    async def get_large_transactions(self, symbol: str, min_usd: float = 500000) -> list:
        all_transactions = []
        
        for chain in S.ONCHAIN_CHAINS:
            if chain not in self.chain_apis:
                continue
                
            if symbol not in self.token_contracts:
                continue
                
            contract_address = self.token_contracts[symbol].get(chain)
            if not contract_address:
                continue
            
            try:
                transactions = await self._fetch_chain_transactions(chain, contract_address, min_usd)
                all_transactions.extend(transactions)
            except Exception as e:
                logger.error(f"Error fetching {chain} transactions: {e}")
        
        return all_transactions
    
    async def _fetch_chain_transactions(self, chain: str, contract_address: str, min_usd: float) -> list:
        api_config = self.chain_apis.get(chain)
        if not api_config:
            return []
        
        url = f"{api_config['url']}/api"
        
        params = {
            'module': 'account',
            'action': 'tokentx',
            'contractaddress': contract_address,
            'startblock': 0,
            'endblock': 99999999,
            'sort': 'desc',
            'apikey': api_config['key']
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
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
            return {'activity': 'low', 'bias': 'neutral'}
        
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
        
        return {
            'activity': activity,
            'bias': bias,
            'total_volume': total_volume,
            'transaction_count': len(transactions),
            'buy_sell_ratio': buys / max(sells, 1)
        }
    
    async def analyze_defi_metrics(self, symbol: str) -> dict:
        return {
            'tvl_trend': 'increasing',
            'liquidity_concentration': 'medium',
            'yield_farming_activity': 'high'
        }

# Enhanced Sentiment Analysis Module
class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.transformers_available = transformers_available
        self.sentiment_pipeline = None
        self.finbert = None
        
        if self.transformers_available:
            try:
                from transformers import pipeline
                self.sentiment_pipeline = pipeline("sentiment-analysis")
                self.finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
                logger.info("Transformers models loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load transformers models: {e}")
                self.transformers_available = False
        
        try:
            self.twitter_client = tweepy.Client(
                bearer_token=S.TWITTER_BEARER_TOKEN,
                wait_on_rate_limit=True
            )
            self.twitter_available = True
        except Exception:
            self.twitter_available = False
            logger.warning("Twitter client not available")
    
    async def analyze_news_sentiment(self, symbol: str) -> dict:
        news_sources = [
            self._fetch_cryptopanic_news,
            self._fetch_newsapi_news
        ]
        
        all_news = []
        for source in news_sources:
            try:
                news = await source(symbol)
                all_news.extend(news)
            except Exception as e:
                logger.error(f"Error fetching news: {e}")
        
        if not all_news:
            return {'sentiment': 'neutral', 'score': 0.0}
        
        scores = []
        for news in all_news:
            text = f"{news.get('title', '')} {news.get('description', '')}"
            score = self.vader.polarity_scores(text)['compound']
            scores.append(score)
        
        avg_score = np.mean(scores)
        
        if avg_score > 0.2:
            sentiment = 'bullish'
        elif avg_score < -0.2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': avg_score,
            'sources': len(all_news)
        }
    
    async def _fetch_cryptopanic_news(self, symbol: str) -> list:
        if not S.CRYPTOPANIC_API_KEY:
            return []
        
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            'auth_token': S.CRYPTOPANIC_API_KEY,
            'currencies': symbol,
            'kind': 'news'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    data = await response.json()
                    return data.get('results', [])
        except Exception as e:
            logger.error(f"CryptoPanic error: {e}")
            return []
    
    async def _fetch_newsapi_news(self, symbol: str) -> list:
        if not S.NEWS_API_KEY:
            return []
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f'{symbol} cryptocurrency',
            'apiKey': S.NEWS_API_KEY,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 20
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    data = await response.json()
                    return data.get('articles', [])
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return []
    
    async def analyze_twitter_sentiment(self, symbol: str) -> dict:
        if not self.twitter_available:
            return {'sentiment': 'neutral', 'score': 0.0}
        
        try:
            query = f"#{symbol} OR ${symbol} -is:retweet lang:en"
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=100,
                tweet_fields=['created_at', 'public_metrics']
            )
            
            if not tweets.data:
                return {'sentiment': 'neutral', 'score': 0.0}
            
            scores = []
            for tweet in tweets.data:
                score = self.vader.polarity_scores(tweet.text)['compound']
                scores.append(score)
            
            avg_score = np.mean(scores)
            
            if avg_score > 0.2:
                sentiment = 'bullish'
            elif avg_score < -0.2:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'score': avg_score,
                'tweet_count': len(tweets.data)
            }
        except Exception as e:
            logger.error(f"Twitter sentiment error: {e}")
            return {'sentiment': 'neutral', 'score': 0.0}
    
    async def analyze_with_transformers(self, text: str) -> dict:
        if not self.transformers_available or not self.finbert:
            return {'label': 'neutral', 'score': 0.0}
        
        try:
            result = self.finbert(text)
            return {
                'label': result[0]['label'],
                'score': result[0]['score']
            }
        except Exception as e:
            logger.error(f"Transformers analysis error: {e}")
            return {'label': 'neutral', 'score': 0.0}
    
    async def combined_sentiment_analysis(self, symbol: str) -> dict:
        news_sentiment = await self.analyze_news_sentiment(symbol)
        twitter_sentiment = await self.analyze_twitter_sentiment(symbol)
        
        news_weight = 0.6
        twitter_weight = 0.4
        
        combined_score = (
            news_sentiment['score'] * news_weight +
            twitter_sentiment['score'] * twitter_weight
        )
        
        if combined_score > 0.2:
            sentiment = 'bullish'
        elif combined_score < -0.2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': combined_score,
            'news': news_sentiment,
            'twitter': twitter_sentiment,
            'transformers_available': self.transformers_available
        }

# Enhanced Prediction Engine
class EnhancedPredictionEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    async def prepare_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        ta_analyzer = AdvancedTechnicalAnalyzer()
        df['rsi'] = ta_analyzer.calculate_rsi(df['close'])
        df['macd'], df['macd_signal'], df['macd_hist'] = ta_analyzer.calculate_macd(df['close'])
        df['bb_upper'], df['bb_mid'], df['bb_lower'] = ta_analyzer.calculate_bollinger_bands(df['close'])
        df['atr'] = ta_analyzer.calculate_atr(df)
        df['stoch_k'], df['stoch_d'] = ta_analyzer.calculate_stochastic(df)
        df['williams_r'] = ta_analyzer.calculate_williams_r(df)
        df['cci'] = ta_analyzer.calculate_cci(df)
        df['mfi'] = ta_analyzer.calculate_mfi(df)
        df['vwap'] = ta_analyzer.calculate_vwap(df)
        
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        df['doji'] = ta_analyzer.detect_doji(df)
        df['hammer'] = ta_analyzer.detect_hammer(df)
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        df = df.dropna()
        
        return df
    
    async def train_model(self, df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        df = await self.prepare_features(df, symbol)
        
        features = ['rsi', 'macd', 'macd_hist', 'bb_upper', 'bb_lower', 'atr', 
                   'stoch_k', 'stoch_d', 'returns', 'log_returns', 'volatility', 'volume_ratio',
                   'hour', 'day_of_week', 'williams_r', 'cci', 'mfi', 'vwap', 'doji', 'hammer']
        
        X = df[features]
        y = (df['close'].shift(-1) > df['close']).astype(int)
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        model_key = f"{symbol}_{timeframe}"
        self.models[model_key] = model
        self.scalers[model_key] = scaler
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': dict(zip(features, model.feature_importances_))
        }
    
    async def predict(self, df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        model_key = f"{symbol}_{timeframe}"
        
        if model_key not in self.models:
            return None
        
        df = await self.prepare_features(df, symbol)
        
        features = ['rsi', 'macd', 'macd_hist', 'bb_upper', 'bb_lower', 'atr', 
                   'stoch_k', 'stoch_d', 'returns', 'log_returns', 'volatility', 'volume_ratio',
                   'hour', 'day_of_week', 'williams_r', 'cci', 'mfi', 'vwap', 'doji', 'hammer']
        
        X = df[features].iloc[-1:].values
        
        scaler = self.scalers[model_key]
        X_scaled = scaler.transform(X)
        
        model = self.models[model_key]
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        return {
            'direction': 'BUY' if prediction == 1 else 'SELL',
            'probability': probability,
            'confidence': max(probability, 1 - probability)
        }

# Exchange Connector
class ExchangeConnector:
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.exchange = getattr(ccxt, exchange_name)({
            'apiKey': os.getenv(f'{exchange_name.upper()}_API_KEY'),
            'secret': os.getenv(f'{exchange_name.upper()}_API_SECRET'),
            'enableRateLimit': True
        })
    
    async def get_ticker(self, symbol: str) -> dict:
        try:
            return await self.exchange.fetch_ticker(f"{symbol}/USDT")
        except Exception as e:
            logger.error(f"Error fetching ticker from {self.exchange_name}: {e}")
            return None
    
    async def get_orderbook(self, symbol: str) -> dict:
        try:
            return await self.exchange.fetch_order_book(f"{symbol}/USDT")
        except Exception as e:
            logger.error(f"Error fetching orderbook from {self.exchange_name}: {e}")
            return None
    
    async def place_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None):
        try:
            if order_type == 'market':
                order = await self.exchange.create_market_order(
                    symbol=f"{symbol}/USDT",
                    side=side,
                    amount=amount
                )
            else:
                order = await self.exchange.create_limit_order(
                    symbol=f"{symbol}/USDT",
                    side=side,
                    amount=amount,
                    price=price
                )
            return order
        except Exception as e:
            logger.error(f"Order failed on {self.exchange_name}: {e}")
            return None

# Arbitrage Strategy
class ArbitrageStrategy:
    def __init__(self):
        self.exchanges = [
            ExchangeConnector('binance'),
            ExchangeConnector('kucoin'),
            ExchangeConnector('bybit')
        ]
    
    async def get_prices_from_exchanges(self, symbols: List[str]) -> dict:
        prices = {}
        
        tasks = []
        for exchange in self.exchanges:
            for symbol in symbols:
                tasks.append(self._get_price(exchange, symbol))
        
        results = await asyncio.gather(*tasks)
        
        for result in results:
            if result:
                exchange_name, symbol, price = result
                if symbol not in prices:
                    prices[symbol] = {}
                prices[symbol][exchange_name] = price
        
        return prices
    
    async def _get_price(self, exchange: ExchangeConnector, symbol: str) -> tuple:
        ticker = await exchange.get_ticker(symbol)
        if ticker:
            return (exchange.exchange_name, symbol, ticker['last'])
        return None
    
    async def find_arbitrage_opportunities(self) -> List[dict]:
        symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'DOT']
        prices = await self.get_prices_from_exchanges(symbols)
        
        opportunities = []
        for symbol in prices:
            if len(prices[symbol]) < 2:
                continue
                
            min_price = min(prices[symbol].values())
            max_price = max(prices[symbol].values())
            
            if (max_price - min_price) / min_price > 0.002:  # 0.2% arbitrage
                buy_exchange = min(prices[symbol], key=prices[symbol].get)
                sell_exchange = max(prices[symbol], key=prices[symbol].get)
                
                opportunities.append({
                    'symbol': symbol,
                    'buy_exchange': buy_exchange,
                    'sell_exchange': sell_exchange,
                    'buy_price': min_price,
                    'sell_price': max_price,
                    'profit_percentage': (max_price - min_price) / min_price * 100
                })
        
        return opportunities
    
    async def execute_arbitrage(self, opportunity: dict, amount: float) -> bool:
        try:
            buy_exchange = next(ex for ex in self.exchanges if ex.exchange_name == opportunity['buy_exchange'])
            buy_order = await buy_exchange.place_order(
                symbol=opportunity['symbol'],
                order_type='market',
                side='buy',
                amount=amount
            )
            
            if not buy_order:
                return False
            
            sell_exchange = next(ex for ex in self.exchanges if ex.exchange_name == opportunity['sell_exchange'])
            sell_order = await sell_exchange.place_order(
                symbol=opportunity['symbol'],
                order_type='market',
                side='sell',
                amount=amount
            )
            
            return sell_order is not None
        except Exception as e:
            logger.error(f"Arbitrage execution failed: {e}")
            return False

# Alert System
class AlertSystem:
    def __init__(self):
        self.alert_history = []
    
    async def check_alerts(self, analysis: dict, user: User) -> List[dict]:
        alerts = []
        
        if analysis['signal']['confidence'] > 0.8:
            alert = {
                'type': 'high_confidence',
                'message': localization.get_text('high_confidence', user.language),
                'data': analysis['signal']
            }
            alerts.append(alert)
        
        if analysis.get('onchain_analysis', {}).get('activity') == 'very_high':
            alert = {
                'type': 'whale_activity',
                'message': localization.get_text('whale_activity', user.language),
                'data': analysis['onchain_analysis']
            }
            alerts.append(alert)
        
        volatility = analysis.get('market_data', {}).get('ohlcv', pd.DataFrame()).get('close', pd.Series()).pct_change().std()
        if volatility and volatility > 0.05:
            alert = {
                'type': 'market_volatility',
                'message': localization.get_text('market_volatility', user.language),
                'data': {'volatility': volatility}
            }
            alerts.append(alert)
        
        if analysis.get('arbitrage_opportunity'):
            alert = {
                'type': 'arbitrage_opportunity',
                'message': localization.get_text('arbitrage_opportunity', user.language),
                'data': analysis['arbitrage_opportunity']
            }
            alerts.append(alert)
        
        for alert in alerts:
            self.alert_history.append({
                'user_id': user.id,
                'timestamp': datetime.datetime.utcnow(),
                'alert': alert
            })
        
        return alerts
    
    async def send_telegram_alert(self, alert: dict, user: User):
        try:
            message = f"🚨 {alert['message']}\n\n"
            message += f"Data: {json.dumps(alert['data'], indent=2)}"
            
            logger.info(f"Alert sent to user {user.id}: {alert['type']}")
        except Exception as e:
            logger.error(f"Error sending alert: {e}")

# Backup System
class BackupSystem:
    def __init__(self):
        self.last_backup = datetime.datetime.utcnow()
    
    async def backup_database(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"backup_{timestamp}.sql"
        
        try:
            if S.DATABASE_URL.startswith('sqlite'):
                import shutil
                shutil.copy2(S.DATABASE_URL.replace('sqlite:///', ''), backup_file)
            elif S.DATABASE_URL.startswith('postgresql') and psycopg2_available:
                os.system(f"pg_dump {S.DATABASE_URL} > {backup_file}")
            
            if S.CLOUD_STORAGE_BUCKET:
                await self.upload_to_cloud(backup_file)
            
            self.last_backup = datetime.datetime.utcnow()
            logger.info(f"Database backup completed: {backup_file}")
        except Exception as e:
            logger.error(f"Backup failed: {e}")
    
    async def upload_to_cloud(self, file_path: str):
        logger.info(f"Uploading {file_path} to cloud storage")
    
    async def should_backup(self) -> bool:
        if not S.ENABLE_BACKUP:
            return False
        
        hours_since_backup = (datetime.datetime.utcnow() - self.last_backup).total_seconds() / 3600
        return hours_since_backup >= S.BACKUP_INTERVAL_HOURS

# Multi-Timeframe Strategy
class MultiTimeframeStrategy:
    async def analyze_multiple_timeframes(self, symbol: str) -> dict:
        timeframes = ['1h', '4h', '1d']
        signals = {}
        
        for tf in timeframes:
            exchange = ccxt.binance()
            ohlcv = await exchange.fetch_ohlcv(f"{symbol}/USDT", tf, limit=500)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            ta_analyzer = AdvancedTechnicalAnalyzer()
            ta_results = ta_analyzer.full_analysis(df)
            
            if ta_results['market_structure']['trend'] == 'uptrend':
                signals[tf] = 'BUY'
            elif ta_results['market_structure']['trend'] == 'downtrend':
                signals[tf] = 'SELL'
            else:
                signals[tf] = 'HOLD'
        
        buy_count = sum(1 for s in signals.values() if s == 'BUY')
        sell_count = sum(1 for s in signals.values() if s == 'SELL')
        
        if buy_count > sell_count:
            combined_signal = 'BUY'
        elif sell_count > buy_count:
            combined_signal = 'SELL'
        else:
            combined_signal = 'HOLD'
        
        return {
            'signals': signals,
            'combined_signal': combined_signal,
            'confidence': max(buy_count, sell_count) / len(timeframes)
        }

# Main Bot Class
class AdvancedCryptoBot:
    def __init__(self):
        self.ta_analyzer = AdvancedTechnicalAnalyzer()
        self.risk_manager = AdvancedRiskManager()
        self.onchain_analyzer = EnhancedOnChainAnalyzer()
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        self.prediction_engine = EnhancedPredictionEngine()
        self.arbitrage_strategy = ArbitrageStrategy()
        self.alert_system = AlertSystem()
        self.backup_system = BackupSystem()
        self.multitimeframe_strategy = MultiTimeframeStrategy()
        
    async def fetch_market_data(self, symbol: str) -> dict:
        cache_key = f"market_data_{symbol}"
        cached_data = get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            exchange = ccxt.binance()
            ticker = await exchange.fetch_ticker(f"{symbol}/USDT")
            ohlcv = await exchange.fetch_ohlcv(f"{symbol}/USDT", "1h", limit=1000)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            data = {
                'symbol': symbol,
                'price': ticker['last'],
                'change_24h': ticker['percentage'],
                'volume_24h': ticker['quoteVolume'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low'],
                'ohlcv': df
            }
            
            cache_with_redis(cache_key, data, S.CACHE_TTL_SECONDS)
            
            return data
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    async def analyze_symbol(self, symbol: str, user_id: int = None) -> dict:
        try:
            session, user = get_user_session(user_id)
            if not user or not session:
                return None
            
            user_prefs = json.loads(user.preferences or '{}')
            
            market_data = await self.fetch_market_data(symbol)
            if not market_data:
                return None
            
            ta_results = self.ta_analyzer.full_analysis(market_data['ohlcv'])
            
            onchain_results = await self.onchain_analyzer.analyze_whale_activity(symbol) if S.ENABLE_ONCHAIN else {}
            
            defi_metrics = await self.onchain_analyzer.analyze_defi_metrics(symbol) if S.ENABLE_ONCHAIN else {}
            
            sentiment_results = await self.sentiment_analyzer.combined_sentiment_analysis(symbol) if S.ENABLE_SENTIMENT else {}
            
            multitimeframe_results = await self.multitimeframe_strategy.analyze_multiple_timeframes(symbol)
            
            prediction = await self.prediction_engine.predict(
                market_data['ohlcv'], symbol, '1h'
            ) if S.ENABLE_PREDICTION else {}
            
            arbitrage_opportunity = None
            if S.ENABLE_ARBITRAGE:
                opportunities = await self.arbitrage_strategy.find_arbitrage_opportunities()
                arbitrage_opportunity = next((opp for opp in opportunities if opp['symbol'] == symbol), None)
            
            current_price = market_data['price']
            risk_params = await self._calculate_risk_parameters(
                market_data['ohlcv'], current_price, user
            )
            
            signal = self._generate_signal(
                ta_results, onchain_results, sentiment_results, 
                prediction, risk_params, multitimeframe_results
            )
            
            analysis_data = {
                'symbol': symbol,
                'market_data': market_data,
                'technical_analysis': ta_results,
                'onchain_analysis': onchain_results,
                'sentiment_analysis': sentiment_results,
                'prediction': prediction,
                'risk_management': risk_params,
                'signal': signal,
                'arbitrage_opportunity': arbitrage_opportunity,
                'timestamp': datetime.datetime.utcnow()
            }
            
            alerts = await self.alert_system.check_alerts(analysis_data, user)
            
            for alert in alerts:
                await self.alert_system.send_telegram_alert(alert, user)
            
            self._save_analysis(
                user_id, symbol, signal, prediction, 
                market_data, ta_results, onchain_results
            )
            
            if arbitrage_opportunity:
                analysis_data['arbitrage_opportunity'] = arbitrage_opportunity
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    async def _calculate_risk_parameters(self, df: pd.DataFrame, current_price: float, user: User) -> dict:
        volatility = df['close'].pct_change().std()
        trend_strength = abs(df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
        
        market_conditions = {
            'volatility': volatility,
            'trend_strength': trend_strength
        }
        
        dynamic_risk = await self.risk_manager.calculate_dynamic_risk(market_conditions)
        
        stop_loss = self.risk_manager.dynamic_stop_loss(df, current_price, 'long')
        
        position_size = self.risk_manager.calculate_position_size(
            account_balance=user.balance,
            risk_percent=dynamic_risk['risk_per_trade'],
            entry_price=current_price,
            stop_loss=stop_loss,
            method='volatility_adjusted',
            volatility=volatility
        )
        
        leverage = self.risk_manager.calculate_optimal_leverage(
            volatility=volatility,
            account_balance=user.balance,
            position_size=position_size,
            max_leverage=dynamic_risk['max_leverage']
        )
        
        take_profit_1 = current_price + (current_price - stop_loss) * 2
        take_profit_2 = current_price + (current_price - stop_loss) * 3
        
        return {
            'stop_loss': stop_loss,
            'take_profit_1': take_profit_1,
            'take_profit_2': take_profit_2,
            'position_size': position_size,
            'leverage': leverage,
            'risk_reward_ratio': self.risk_manager.calculate_risk_reward_ratio(
                current_price, stop_loss, take_profit_1
            ),
            'dynamic_risk': dynamic_risk
        }
    
    def _generate_signal(self, ta_results, onchain_results, sentiment_results, 
                        prediction, risk_params, multitimeframe_results) -> dict:
        signal_score = 0
        
        if ta_results.get('market_structure', {}).get('trend') == 'uptrend':
            signal_score += 0.3
        elif ta_results.get('market_structure', {}).get('trend') == 'downtrend':
            signal_score -= 0.3
        
        if onchain_results.get('bias') == 'bullish':
            signal_score += 0.2
        elif onchain_results.get('bias') == 'bearish':
            signal_score -= 0.2
        
        if sentiment_results.get('sentiment') == 'bullish':
            signal_score += 0.15
        elif sentiment_results.get('sentiment') == 'bearish':
            signal_score -= 0.15
        
        if prediction and prediction.get('direction') == 'BUY':
            signal_score += 0.1
        elif prediction and prediction.get('direction') == 'SELL':
            signal_score -= 0.1
        
        if multitimeframe_results.get('combined_signal') == 'BUY':
            signal_score += 0.25 * multitimeframe_results.get('confidence', 1)
        elif multitimeframe_results.get('combined_signal') == 'SELL':
            signal_score -= 0.25 * multitimeframe_results.get('confidence', 1)
        
        if signal_score > 0.6:
            signal = 'BUY'
        elif signal_score < -0.6:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'score': signal_score,
            'confidence': abs(signal_score),
            'multitimeframe': multitimeframe_results
        }
    
    def _save_analysis(self, user_id, symbol, signal, prediction, market_data, ta_results, onchain_results):
        try:
            session, _ = get_user_session(user_id)
            if not session:
                return
                
            new_signal = Signal(
                user_id=user_id,
                symbol=symbol,
                signal_type=signal['signal'],
                confidence=signal['confidence'],
                price=market_data['price'],
                source='AI_BOT',
                timeframe='1h'
            )
            session.add(new_signal)
            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
    
    async def monitor_and_analyze(self):
        symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'DOT', 'SOL', 'AVAX', 'MATIC']
        
        while True:
            try:
                if await self.backup_system.should_backup():
                    await self.backup_system.backup_database()
                
                for symbol in symbols:
                    analysis = await self.analyze_symbol(symbol)
                    if analysis:
                        logger.info(f"Analysis completed for {symbol}: {analysis['signal']['signal']}")
                
                await asyncio.sleep(S.MONITOR_INTERVAL)
            except Exception as e:
                logger.error(f"Error in monitoring task: {e}")
                await asyncio.sleep(60)

# Web Dashboard
async def handle_dashboard(request):
    try:
        installing = not transformers_available and S.ENABLE_TRANSFORMERS
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Crypto AI Bot Dashboard</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
                <div class="container">
                    <a class="navbar-brand" href="#">Crypto AI Bot Dashboard</a>
                    <div class="ms-auto">
                        <span class="navbar-text">
                            Status: <span id="status" class="badge {'bg-warning' if installing else 'bg-success'}">{'Installing AI Models...' if installing else 'Online'}</span>
                        </span>
                    </div>
                </div>
            </nav>
            
            <div class="container mt-4">
                <div class="row">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <h5>Market Overview</h5>
                                <button class="btn btn-sm btn-primary" onclick="refreshData()">Refresh</button>
                            </div>
                            <div class="card-body">
                                <div id="market-data">
                                    <div class="text-center">
                                        <div class="spinner-border" role="status">
                                            <span class="visually-hidden">Loading...</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>Technical Analysis</h5>
                            </div>
                            <div class="card-body">
                                <div id="ta-chart" style="height: 400px;"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>Sentiment Analysis</h5>
                            </div>
                            <div class="card-body">
                                <div id="sentiment-chart" style="height: 400px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">
                                <h5>Recent Signals</h5>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table class="table table-striped">
                                        <thead>
                                            <tr>
                                                <th>Symbol</th>
                                                <th>Signal</th>
                                                <th>Confidence</th>
                                                <th>Price</th>
                                                <th>Timestamp</th>
                                            </tr>
                                        </thead>
                                        <tbody id="signals-table">
                                            <tr>
                                                <td colspan="5" class="text-center">Loading signals...</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">
                                <h5>Arbitrage Opportunities</h5>
                            </div>
                            <div class="card-body">
                                <div id="arbitrage-data">
                                    <p>No arbitrage opportunities found</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                let socket;
                
                function connectWebSocket() {{
                    socket = new WebSocket('ws://' + window.location.host + '/ws');
                    
                    socket.onmessage = function(event) {{
                        const data = JSON.parse(event.data);
                        updateDashboard(data);
                    }};
                    
                    socket.onclose = function() {{
                        setTimeout(connectWebSocket, 1000);
                    }};
                }}
                
                function updateDashboard(data) {{
                    if (data.market_data) {{
                        updateMarketData(data.market_data);
                    }}
                    
                    if (data.technical_analysis) {{
                        updateTechnicalAnalysis(data.technical_analysis);
                    }}
                    
                    if (data.sentiment_analysis) {{
                        updateSentimentAnalysis(data.sentiment_analysis);
                    }}
                    
                    if (data.signals) {{
                        updateSignals(data.signals);
                    }}
                    
                    if (data.arbitrage) {{
                        updateArbitrage(data.arbitrage);
                    }}
                }}
                
                function updateMarketData(marketData) {{
                    let html = '<div class="row">';
                    for (const [symbol, data] of Object.entries(marketData)) {{
                        html += `
                            <div class="col-md-3 mb-3">
                                <div class="card">
                                    <div class="card-body">
                                        <h6>${{symbol}}/USDT</h6>
                                        <h3>${{data.price.toFixed(2)}}</h3>
                                        <p class="mb-0">
                                            <span class="badge ${{data.change_24h >= 0 ? 'bg-success' : 'bg-danger'}}">
                                                ${{data.change_24h >= 0 ? '+' : ''}}${{data.change_24h.toFixed(2)}}%
                                            </span>
                                        </p>
                                    </div>
                                </div>
                            </div>
                        `;
                    }}
                    html += '</div>';
                    document.getElementById('market-data').innerHTML = html;
                }}
                
                function updateTechnicalAnalysis(taData) {{
                    // Implementation for technical analysis charts
                }}
                
                function updateSentimentAnalysis(sentimentData) {{
                    // Implementation for sentiment analysis charts
                }}
                
                function updateSignals(signals) {{
                    let html = '';
                    signals.forEach(signal => {{
                        html += `
                            <tr>
                                <td>${{signal.symbol}}</td>
                                <td><span class="badge bg-${{signal.signal_type === 'BUY' ? 'success' : signal.signal_type === 'SELL' ? 'danger' : 'secondary'}}">${{signal.signal_type}}</span></td>
                                <td>${{(signal.confidence * 100).toFixed(1)}}%</td>
                                <td>${{signal.price.toFixed(2)}}</td>
                                <td>${{new Date(signal.timestamp).toLocaleString()}}</td>
                            </tr>
                        `;
                    }});
                    document.getElementById('signals-table').innerHTML = html;
                }}
                
                function updateArbitrage(arbitrageData) {{
                    if (arbitrageData.length > 0) {{
                        let html = '<div class="table-responsive"><table class="table table-striped">';
                        html += '<thead><tr><th>Symbol</th><th>Buy Exchange</th><th>Sell Exchange</th><th>Profit</th></tr></thead><tbody>';
                        
                        arbitrageData.forEach(opp => {{
                            html += `
                                <tr>
                                    <td>${{opp.symbol}}</td>
                                    <td>${{opp.buy_exchange}}</td>
                                    <td>${{opp.sell_exchange}}</td>
                                    <td><span class="badge bg-success">${{opp.profit_percentage.toFixed(2)}}%</span></td>
                                </tr>
                            `;
                        }});
                        
                        html += '</tbody></table></div>';
                        document.getElementById('arbitrage-data').innerHTML = html;
                    }} else {{
                        document.getElementById('arbitrage-data').innerHTML = '<p>No arbitrage opportunities found</p>';
                    }}
                }}
                
                function refreshData() {{
                    fetch('/api/dashboard')
                        .then(response => response.json())
                        .then(data => updateDashboard(data))
                        .catch(error => console.error('Error refreshing data:', error));
                }}
                
                document.addEventListener('DOMContentLoaded', function() {{
                    connectWebSocket();
                    refreshData();
                    
                    setInterval(refreshData, 30000);
                }});
            </script>
        </body>
        </html>
        """
        return web.Response(text=html_content, content_type='text/html')
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return web.Response(text=f"Error: {str(e)}", status=500)

async def handle_api_dashboard(request):
    """API endpoint for dashboard data"""
    try:
        session, _ = get_user_session(1)  # Use a default user ID
        if not session:
            return web.json_response({'error': 'Database not available'}, status=500)
            
        signals = session.query(Signal).order_by(Signal.timestamp.desc()).limit(10).all()
        
        symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'DOT']
        market_data = {}
        
        for symbol in symbols:
            cache_key = f"market_data_{symbol}"
            cached_data = get_from_cache(cache_key)
            if cached_data:
                market_data[symbol] = cached_data
        
        bot = AdvancedCryptoBot()
        arbitrage_opportunities = await bot.arbitrage_strategy.find_arbitrage_opportunities()
        
        data = {
            'market_data': market_data,
            'signals': [
                {
                    'symbol': s.symbol,
                    'signal_type': s.signal_type,
                    'confidence': s.confidence,
                    'price': s.price,
                    'timestamp': s.timestamp.isoformat()
                } for s in signals
            ],
            'arbitrage': arbitrage_opportunities
        }
        
        session.close()
        return web.json_response(data)
    except Exception as e:
        logger.error(f"API dashboard error: {e}")
        return web.json_response({'error': str(e)}, status=500)

async def handle_health(request):
    """Health check endpoint"""
    try:
        return web.json_response({
            "status": "healthy", 
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "transformers_available": transformers_available,
            "psycopg2_available": psycopg2_available,
            "version": "2.0.0"
        })
    except Exception as e:
        return web.json_response({"status": "unhealthy", "error": str(e)}, status=500)

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    try:
        while True:
            session, _ = get_user_session(1)  # Use a default user ID
            if not session:
                await ws.send_json({"error": "Database not available"})
                await asyncio.sleep(5)
                continue
                
            signals = session.query(Signal).order_by(Signal.timestamp.desc()).limit(5).all()
            
            data = {
                'signals': [
                    {
                        'symbol': s.symbol,
                        'signal_type': s.signal_type,
                        'confidence': s.confidence,
                        'price': s.price,
                        'timestamp': s.timestamp.isoformat()
                    } for s in signals
                ]
            }
            
            await ws.send_json(data)
            session.close()
            
            await asyncio.sleep(10)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await ws.close()
    
    return ws

# Main execution
if __name__ == "__main__":
    # Initialize bot
    bot = AdvancedCryptoBot()
    
    # Setup web app
    app = web.Application()
    app.router.add_get('/', handle_dashboard)
    app.router.add_get('/health', handle_health)
    app.router.add_get('/api/dashboard', handle_api_dashboard)
    app.router.add_get('/ws', websocket_handler)
    
    # Error handler
    async def handle_error(request):
        return web.json_response({"error": "Not Found"}, status=404)
    
    app.router.add_route('*', '/{tail:.*}', handle_error)
    
    # Start background task for heavy dependencies
    if S.ENABLE_TRANSFORMERS:
        loop = asyncio.get_event_loop()
        loop.create_task(install_heavy_dependencies())
    
    # Start background monitoring task
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(bot.monitor_and_analyze())
    except Exception as e:
        logger.error(f"Error creating background task: {e}")
    
    # Start web server
    if S.ENABLE_WEB_DASHBOARD:
        try:
            logger.info(f"Starting server on port {S.WEB_PORT}")
            web.run_app(app, host='0.0.0.0', port=S.WEB_PORT)
        except Exception as e:
            logger.error(f"Error starting web server: {e}")
    else:
        logger.info("Web dashboard disabled")