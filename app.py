# -*- coding: utf-8 -*-
"""
Advanced Crypto AI Trading Bot - Iranian Version
بهینه‌شده برای استفاده در ایران با APIهای قابل دسترس
"""

import os
import sys
import time
import asyncio
import logging
import json
import datetime
import io
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import asynccontextmanager

import aiohttp
from aiohttp import web
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
from telegram.constants import ChatAction
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, desc, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from joblib import dump, load
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import redis
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import talib
from scipy import stats
import quantstats as qs

# Optional imports with fallback
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Filter warnings
warnings.filterwarnings('ignore')

# ------------------- Settings -------------------

@dataclass
class Settings:
    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")
    SECRET_TOKEN: str = os.getenv("SECRET_TOKEN", "iran_crypto_bot_secret_2024")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///iran_crypto_bot.db")
    
    # API Keys (optional - for enhanced features)
    COINGECKO_API_KEY: str = os.getenv("COINGECKO_API_KEY", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    CRYPTOPANIC_API_KEY: str = os.getenv("CRYPTOPANIC_API_KEY", "")
    
    # Feature flags
    ENABLE_ADVANCED_TA: bool = True
    ENABLE_ONCHAIN: bool = True
    ENABLE_SENTIMENT: bool = True
    ENABLE_PREDICTION: bool = True
    ENABLE_RISK_MANAGEMENT: bool = True
    ENABLE_WEB_DASHBOARD: bool = True
    ENABLE_DEEP_LEARNING: bool = True
    ENABLE_TRANSFORMER_SENTIMENT: bool = True
    ENABLE_MONITOR: bool = True
    
    # Lists
    EXCHANGES: List[str] = field(default_factory=lambda: ["binance", "bybit", "okx"])
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["15m", "1h", "4h", "1d", "1w"])
    DEFAULT_MONITOR_SYMBOLS: List[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL", "BNB", "ADA", "XRP", "DOT", "DOGE", "AVAX", "MATIC"])
    IRAN_FRIENDLY_EXCHANGES: List[str] = field(default_factory=lambda: ["binance", "bybit", "okx"])
    
    # Parameters
    CONCURRENT_REQUESTS: int = 5
    CACHE_TTL_SECONDS: int = 600  # 10 minutes cache
    REQUEST_TIMEOUT: int = 45  # Longer timeout for Iran
    DEFAULT_RISK_PER_TRADE: float = 2.0
    DEFAULT_MAX_LEVERAGE: float = 3.0  # Conservative for Iran
    MIN_LIQUIDITY_THRESHOLD: float = 500000
    SL_ATR_MULTIPLIER: float = 2.5
    MAX_SLIPPAGE_PCT: float = 1.0  # Higher slippage tolerance
    MONITOR_INTERVAL: int = 600  # 10 minutes
    ALERT_THRESHOLD: float = 0.7
    WEB_PORT: int = int(os.getenv("PORT", 8080))
    MODELS_DIR: str = "models"
    DATA_DIR: str = "data"
    
    # Advanced parameters
    ANOMALY_DETECTION_THRESHOLD: float = 0.95
    MAX_DRAWDOWN_LIMIT: float = 0.15  # 15% max drawdown

S = Settings()

# ------------------- Logger & Globals -------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("iran_crypto_bot")

START_TIME = time.time()
REQUEST_SEM = asyncio.Semaphore(S.CONCURRENT_REQUESTS)

# Redis (optional)
redis_client = None
redis_available = False
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=int(os.getenv('REDIS_DB', 0)),
        password=os.getenv('REDIS_PASSWORD', None),
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    redis_client.ping()
    redis_available = True
    logger.info("Redis connected successfully")
except Exception as e:
    logger.warning(f"Redis not available: {e}")
    redis_available = False

def cache_set(key: str, value: Any, ttl: Optional[int] = None):
    if not redis_available or not redis_client:
        return
    try:
        data = json.dumps(value, default=str)
        redis_client.setex(key, int(ttl or S.CACHE_TTL_SECONDS), data)
    except Exception as e:
        logger.error(f"Redis set error: {e}")

def cache_get(key: str):
    if not redis_available or not redis_client:
        return None
    try:
        data = redis_client.get(key)
        return json.loads(data) if data else None
    except Exception as e:
        logger.error(f"Redis get error: {e}")
        return None

# ------------------- Database Models -------------------

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, unique=True)
    username = Column(String(50))
    preferences = Column(Text, default='{}')
    risk_level = Column(String(20), default='medium')
    max_leverage = Column(Float, default=3.0)
    balance = Column(Float, default=10000.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    signals = relationship("Signal", back_populates="user")
    watchlist = relationship("Watchlist", back_populates="user")
    performance = relationship("Performance", back_populates="user")
    portfolio = relationship("Portfolio", back_populates="user")

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
    sortino_ratio = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    calmar_ratio = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship("User", back_populates="performance")

class Portfolio(Base):
    __tablename__ = 'portfolio'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    symbol = Column(String(20))
    amount = Column(Float, default=0.0)
    entry_price = Column(Float, default=0.0)
    current_value = Column(Float, default=0.0)
    pnl = Column(Float, default=0.0)
    pnl_percent = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship("User", back_populates="portfolio")

class MarketData(Base):
    __tablename__ = 'market_data'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True)
    price = Column(Float)
    change_24h = Column(Float)
    volume_24h = Column(Float)
    market_cap = Column(Float)
    last_update = Column(DateTime, default=datetime.datetime.utcnow)

# Database setup
engine = None
SessionLocal = None

try:
    engine_kwargs = {}
    if S.DATABASE_URL.startswith("postgres"):
        engine_kwargs = {"pool_size": 5, "max_overflow": 10, "pool_pre_ping": True}
    elif S.DATABASE_URL.startswith("sqlite"):
        engine_kwargs = {"connect_args": {"check_same_thread": False}}
    
    engine = create_engine(S.DATABASE_URL, **engine_kwargs)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)
    logger.info("Database connected successfully")
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    try:
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(engine)
        logger.info("Fallback to in-memory SQLite database")
    except Exception as inner_e:
        logger.error(f"Fallback database also failed: {inner_e}")
        sys.exit(1)

@asynccontextmanager
async def get_db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()

def get_user_session(chat_id: int):
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(chat_id=chat_id).first()
        if not user:
            user = User(chat_id=chat_id, username=f"user_{chat_id}")
            session.add(user)
            session.commit()
            session.refresh(user)
        return session, user
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error in get_user_session: {e}")
        session.close()
        return None, None

# ------------------- Advanced Technical Analysis -------------------

class AdvancedTechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}
    
    def calculate_ichimoku(self, df: pd.DataFrame) -> dict:
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Tenkan-sen (Conversion Line)
        tenkan_period = 9
        tenkan_high = high.rolling(window=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_period = 26
        kijun_high = high.rolling(window=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
        
        # Senkou Span B (Leading Span B)
        senkou_period = 52
        senkou_high = high.rolling(window=senkou_period).max()
        senkou_low = low.rolling(window=senkou_period).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(kijun_period)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-kijun_period)
        
        return {
            'tenkan_sen': tenkan_sen.iloc[-1] if len(tenkan_sen) > 0 else 0,
            'kijun_sen': kijun_sen.iloc[-1] if len(kijun_sen) > 0 else 0,
            'senkou_span_a': senkou_span_a.iloc[-1] if len(senkou_span_a) > 0 else 0,
            'senkou_span_b': senkou_span_b.iloc[-1] if len(senkou_span_b) > 0 else 0,
            'chikou_span': chikou_span.iloc[-1] if len(chikou_span) > 0 else 0,
            'cloud_green': senkou_span_a.iloc[-1] > senkou_span_b.iloc[-1] if len(senkou_span_a) > 0 and len(senkou_span_b) > 0 else False
        }
    
    def calculate_fibonacci(self, df: pd.DataFrame) -> dict:
        high = df['high'].max()
        low = df['low'].min()
        diff = high - low
        
        levels = {
            '0.0': low,
            '0.236': low + diff * 0.236,
            '0.382': low + diff * 0.382,
            '0.5': low + diff * 0.5,
            '0.618': low + diff * 0.618,
            '0.786': low + diff * 0.786,
            '1.0': high
        }
        
        return levels
    
    def calculate_market_profile(self, df: pd.DataFrame, bins: int = 20) -> dict:
        # Market Profile analysis
        prices = df['close']
        volume = df['volume']
        
        if len(prices) == 0:
            return {'poc': 0, 'value_area_low': 0, 'value_area_high': 0}
        
        # Calculate Value Area
        hist, bin_edges = np.histogram(prices, bins=bins, weights=volume)
        if len(hist) == 0:
            return {'poc': 0, 'value_area_low': 0, 'value_area_high': 0}
        
        poc_index = np.argmax(hist)
        poc = (bin_edges[poc_index] + bin_edges[poc_index + 1]) / 2
        
        # Value Area (70% of volume)
        total_volume = np.sum(hist)
        if total_volume == 0:
            return {'poc': poc, 'value_area_low': 0, 'value_area_high': 0}
        
        sorted_indices = np.argsort(hist)[::-1]
        cumulative_volume = 0
        value_area_indices = []
        
        for idx in sorted_indices:
            cumulative_volume += hist[idx]
            value_area_indices.append(idx)
            if cumulative_volume >= total_volume * 0.7:
                break
        
        value_area_low = bin_edges[min(value_area_indices)] if value_area_indices else 0
        value_area_high = bin_edges[max(value_area_indices) + 1] if value_area_indices else 0
        
        return {
            'poc': poc,
            'value_area_low': value_area_low,
            'value_area_high': value_area_high,
            'value_area_width': value_area_high - value_area_low
        }
    
    def detect_anomalies(self, df: pd.DataFrame) -> dict:
        # Use statistical methods for anomaly detection (no ML)
        if len(df) < 10:
            return {'anomalies': [], 'anomaly_score': 0.0}
        
        returns = df['close'].pct_change().dropna()
        if len(returns) < 5:
            return {'anomalies': [], 'anomaly_score': 0.0}
        
        # Use Z-score for anomaly detection
        mean = returns.mean()
        std = returns.std()
        z_scores = (returns - mean) / std
        
        anomalies = np.abs(z_scores) > 2.5  # 2.5 standard deviations
        anomaly_score = float(np.mean(np.abs(z_scores)))
        
        return {
            'anomalies': anomalies.astype(int).tolist(),
            'anomaly_score': anomaly_score,
            'anomaly_dates': returns.index[anomalies].tolist() if hasattr(returns.index, 'tolist') else []
        }
    
    def full_analysis(self, df: pd.DataFrame) -> dict:
        try:
            results = {
                'ichimoku': self.calculate_ichimoku(df),
                'fibonacci': self.calculate_fibonacci(df),
                'market_profile': self.calculate_market_profile(df),
                'anomalies': self.detect_anomalies(df),
                'traditional_indicators': self.calculate_traditional_indicators(df)
            }
            return results
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return {}
    
    def calculate_traditional_indicators(self, df: pd.DataFrame) -> dict:
        if len(df) < 20:
            return {}
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Calculate indicators manually if TA-Lib not available
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            # Bollinger Bands
            sma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            upper_band = sma20 + (std20 * 2)
            lower_band = sma20 - (std20 * 2)
            
            # ATR
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            
            indicators = {
                'rsi': rsi.iloc[-1] if not rsi.empty else 50,
                'macd': macd.iloc[-1] if not macd.empty else 0,
                'macd_signal': signal.iloc[-1] if not signal.empty else 0,
                'macd_histogram': histogram.iloc[-1] if not histogram.empty else 0,
                'bollinger_upper': upper_band.iloc[-1] if not upper_band.empty else 0,
                'bollinger_middle': sma20.iloc[-1] if not sma20.empty else 0,
                'bollinger_lower': lower_band.iloc[-1] if not lower_band.empty else 0,
                'atr': atr.iloc[-1] if not atr.empty else 0
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return {}

# ------------------- Iran-Friendly Data Sources -------------------

class IranFriendlyData:
    def __init__(self):
        self.exchanges = {}
        self.init_exchanges()
    
    def init_exchanges(self):
        """Initialize exchanges that work in Iran"""
        for exchange_id in S.IRAN_FRIENDLY_EXCHANGES:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                self.exchanges[exchange_id] = exchange_class({
                    'timeout': S.REQUEST_TIMEOUT,
                    'enableRateLimit': True,
                    'proxies': {
                        'http': os.getenv('HTTP_PROXY', ''),
                        'https': os.getenv('HTTPS_PROXY', '')
                    }
                })
                logger.info(f"Initialized {exchange_id} exchange for Iran")
            except Exception as e:
                logger.warning(f"Failed to initialize {exchange_id}: {e}")
    
    async def fetch_iran_market_data(self, symbol: str, exchange_id: str = "binance") -> Optional[dict]:
        """Fetch market data with Iran-friendly settings"""
        cache_key = f"iran_market_{symbol}_{exchange_id}"
        cached = cache_get(cache_key)
        
        if cached:
            return cached
        
        exchange = self.exchanges.get(exchange_id)
        if not exchange:
            return None
        
        try:
            # Fetch data with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ticker = await exchange.fetch_ticker(f"{symbol}/USDT")
                    ohlcv = await exchange.fetch_ohlcv(f"{symbol}/USDT", "1h", limit=100)
                    
                    # Process data
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    market_data = {
                        'symbol': symbol,
                        'exchange': exchange_id,
                        'price': float(ticker['last']),
                        'change_24h': float(ticker['percentage']),
                        'volume_24h': float(ticker['quoteVolume']),
                        'high_24h': float(ticker['high']),
                        'low_24h': float(ticker['low']),
                        'timestamp': datetime.datetime.utcnow(),
                        'ohlcv': df.reset_index().to_dict(orient='list')
                    }
                    
                    cache_set(cache_key, market_data, S.CACHE_TTL_SECONDS)
                    return market_data
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(2)  # Wait before retry
                    
        except Exception as e:
            logger.error(f"Error fetching Iran market data for {symbol}: {e}")
            return None
        finally:
            await exchange.close()
    
    async def get_iran_news_sentiment(self, symbol: str) -> dict:
        """Get news sentiment from Iran-friendly sources"""
        # Use cached news if available
        cache_key = f"news_{symbol}"
        cached_news = cache_get(cache_key)
        
        if cached_news:
            return cached_news
        
        # Simple sentiment analysis based on price action
        # In a real implementation, you would use Iranian news sources
        try:
            market_data = await self.fetch_iran_market_data(symbol)
            if not market_data:
                return {'sentiment': 'neutral', 'score': 0.5, 'sources': 0}
            
            # Simple sentiment based on price change
            price_change = market_data['change_24h']
            if price_change > 3:
                sentiment = 'bullish'
                score = min(0.5 + (price_change / 20), 0.9)
            elif price_change < -3:
                sentiment = 'bearish'
                score = min(0.5 + (abs(price_change) / 20), 0.9)
            else:
                sentiment = 'neutral'
                score = 0.5
                
            result = {
                'sentiment': sentiment,
                'score': score,
                'sources': 1  # Using price action as a source
            }
            
            cache_set(cache_key, result, 3600)  # Cache for 1 hour
            return result
            
        except Exception as e:
            logger.error(f"Error getting Iran news sentiment: {e}")
            return {'sentiment': 'neutral', 'score': 0.5, 'sources': 0}

# ------------------- Advanced Sentiment Analysis -------------------

class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.iran_data = IranFriendlyData()
    
    def analyze_text(self, text: str) -> dict:
        """Analyze text sentiment using VADER (works offline)"""
        if not text:
            return {'sentiment': 'neutral', 'score': 0.0}
        
        try:
            scores = self.vader.polarity_scores(text)
            compound = scores['compound']
            
            if compound >= 0.05:
                return {'sentiment': 'bullish', 'score': compound}
            elif compound <= -0.05:
                return {'sentiment': 'bearish', 'score': abs(compound)}
            else:
                return {'sentiment': 'neutral', 'score': 0.0}
        except Exception:
            return {'sentiment': 'neutral', 'score': 0.0}
    
    async def analyze_iran_sentiment(self, symbol: str) -> dict:
        """Analyze sentiment using Iran-friendly methods"""
        # Get news sentiment
        news_sentiment = await self.iran_data.get_iran_news_sentiment(symbol)
        
        # Get price-based sentiment
        market_data = await self.iran_data.fetch_iran_market_data(symbol)
        if not market_data:
            return news_sentiment
        
        # Calculate price momentum sentiment
        price = market_data['price']
        change = market_data['change_24h']
        volume = market_data['volume_24h']
        
        # Simple momentum calculation
        if change > 5 and volume > 1000000:
            price_sentiment = 'bullish'
            price_score = 0.7
        elif change < -5 and volume > 1000000:
            price_sentiment = 'bearish'
            price_score = 0.7
        else:
            price_sentiment = 'neutral'
            price_score = 0.5
        
        # Combine news and price sentiment
        combined_score = (news_sentiment['score'] * 0.6) + (price_score * 0.4)
        
        if combined_score > 0.6:
            overall_sentiment = 'bullish'
        elif combined_score < 0.4:
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'overall_score': combined_score,
            'news_sentiment': news_sentiment,
            'price_sentiment': {'sentiment': price_sentiment, 'score': price_score}
        }

# ------------------- Advanced Prediction Engine -------------------

class AdvancedPredictionEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        os.makedirs(S.MODELS_DIR, exist_ok=True)
    
    async def prepare_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Prepare features for prediction"""
        if len(df) < 50:
            return pd.DataFrame()
        
        df = df.copy()
        
        # Basic features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Target variable (next period return)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        return df.dropna()
    
    async def train_model(self, df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        """Train a Random Forest model"""
        try:
            features = await self.prepare_features(df, symbol)
            if len(features) < 30:
                return {'success': False, 'error': 'Insufficient data'}
            
            X = features.drop('target', axis=1)
            y = features['target']
            
            # Remove non-numeric columns
            X = X.select_dtypes(include=[np.number])
            
            if len(X) == 0:
                return {'success': False, 'error': 'No features available'}
            
            # Train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test) if len(X_test) > 0 else 0
            
            # Save model and scaler
            model_path = os.path.join(S.MODELS_DIR, f"{symbol}_{timeframe}_model.joblib")
            scaler_path = os.path.join(S.MODELS_DIR, f"{symbol}_{timeframe}_scaler.joblib")
            
            dump(model, model_path)
            dump(scaler, scaler_path)
            
            return {
                'success': True,
                'train_score': train_score,
                'test_score': test_score,
                'model_path': model_path,
                'scaler_path': scaler_path
            }
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def predict(self, df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        """Make prediction using trained model"""
        try:
            model_path = os.path.join(S.MODELS_DIR, f"{symbol}_{timeframe}_model.joblib")
            scaler_path = os.path.join(S.MODELS_DIR, f"{symbol}_{timeframe}_scaler.joblib")
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                # Train model if not exists
                train_result = await self.train_model(df, symbol, timeframe)
                if not train_result['success']:
                    return {'direction': 'HOLD', 'confidence': 0.5}
            
            # Load model and scaler
            model = load(model_path)
            scaler = load(scaler_path)
            
            # Prepare features for prediction
            features = await self.prepare_features(df, symbol)
            if len(features) == 0:
                return {'direction': 'HOLD', 'confidence': 0.5}
            
            X_latest = features.iloc[-1:].drop('target', axis=1, errors='ignore')
            X_latest = X_latest.select_dtypes(include=[np.number])
            
            if len(X_latest) == 0:
                return {'direction': 'HOLD', 'confidence': 0.5}
            
            # Scale and predict
            X_scaled = scaler.transform(X_latest)
            prediction = model.predict_proba(X_scaled)[0]
            
            # Get probability of positive class
            positive_prob = prediction[1] if len(prediction) > 1 else 0.5
            
            direction = 'BUY' if positive_prob > 0.5 else 'SELL'
            confidence = max(positive_prob, 1 - positive_prob)
            
            return {
                'direction': direction,
                'confidence': confidence,
                'probability': positive_prob
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return {'direction': 'HOLD', 'confidence': 0.5}

# ------------------- Advanced Risk Management -------------------

class AdvancedRiskManager:
    def calculate_position_size(self, account_balance: float, risk_percent: float, 
                                entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        risk_amount = account_balance * (risk_percent / 100.0)
        price_risk = max(abs(entry_price - stop_loss), 1e-8)
        position_size = risk_amount / price_risk
        return max(0.0, float(position_size))
    
    def dynamic_stop_loss(self, df: pd.DataFrame, entry_price: float, direction: str) -> float:
        """Calculate dynamic stop loss based on ATR"""
        if len(df) < 15:
            return entry_price * 0.95 if direction == 'long' else entry_price * 1.05
        
        # Calculate ATR manually
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        
        stop_distance = atr * S.SL_ATR_MULTIPLIER
        
        if direction == 'long':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, take_profit: float) -> float:
        """Calculate risk/reward ratio"""
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        return reward / risk if risk > 0 else 0.0

# ------------------- Main Bot Class -------------------

class IranCryptoBot:
    def __init__(self):
        self.ta = AdvancedTechnicalAnalyzer()
        self.sentiment = EnhancedSentimentAnalyzer()
        self.prediction = AdvancedPredictionEngine()
        self.risk = AdvancedRiskManager()
        self.iran_data = IranFriendlyData()
        
        # In-memory cache for frequently accessed data
        self.cache = {}
    
    async def analyze_symbol(self, symbol: str, user_id: int = 0, timeframe: str = '1h') -> Optional[dict]:
        """Comprehensive analysis for a symbol"""
        try:
            # Fetch market data
            market_data = await self.iran_data.fetch_iran_market_data(symbol)
            if not market_data:
                return None
            
            # Convert to DataFrame
            ohlcv = market_data.get('ohlcv', {})
            df = pd.DataFrame(ohlcv)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            # Run analyses in parallel
            ta_results = self.ta.full_analysis(df)
            sentiment_results = await self.sentiment.analyze_iran_sentiment(symbol)
            prediction_results = await self.prediction.predict(df, symbol, timeframe)
            
            # Risk management calculations
            direction = 'long' if prediction_results['direction'] == 'BUY' else 'short'
            stop_loss = self.risk.dynamic_stop_loss(df, market_data['price'], direction)
            
            # Get user risk profile
            session, user = get_user_session(user_id)
            if not user:
                risk_percent = S.DEFAULT_RISK_PER_TRADE
                balance = 10000.0
            else:
                risk_percent = risk_percent_from_level(user.risk_level)
                balance = user.balance
                session.close()
            
            position_size = self.risk.calculate_position_size(
                balance, risk_percent, market_data['price'], stop_loss
            )
            
            # Calculate take profit levels
            if direction == 'long':
                take_profit_1 = market_data['price'] + (market_data['price'] - stop_loss)
                take_profit_2 = market_data['price'] + (market_data['price'] - stop_loss) * 2
            else:
                take_profit_1 = market_data['price'] - (stop_loss - market_data['price'])
                take_profit_2 = market_data['price'] - (stop_loss - market_data['price']) * 2
            
            risk_reward = self.risk.calculate_risk_reward_ratio(
                market_data['price'], stop_loss, take_profit_1
            )
            
            # Generate signal
            signal_score = (
                prediction_results['confidence'] * 0.4 +
                (1 if sentiment_results['overall_sentiment'] == prediction_results['direction'].lower() else 0) * 0.3 +
                (1 if ta_results.get('ichimoku', {}).get('cloud_green', False) and direction == 'long' or
                not ta_results.get('ichimoku', {}).get('cloud_green', False) and direction == 'short' else 0) * 0.3
            )
            
            if signal_score > 0.6:
                signal = prediction_results['direction']
            else:
                signal = 'HOLD'
                signal_score = 0.5
            
            # Prepare response
            result = {
                'symbol': symbol,
                'market_data': {
                    'price': market_data['price'],
                    'change_24h': market_data['change_24h'],
                    'volume_24h': market_data['volume_24h'],
                    'high_24h': market_data['high_24h'],
                    'low_24h': market_data['low_24h']
                },
                'technical_analysis': ta_results,
                'sentiment_analysis': sentiment_results,
                'prediction': prediction_results,
                'risk_management': {
                    'direction': direction,
                    'stop_loss': stop_loss,
                    'take_profit_1': take_profit_1,
                    'take_profit_2': take_profit_2,
                    'position_size': position_size,
                    'risk_reward_ratio': risk_reward,
                    'risk_level': 'high' if risk_reward < 1 else 'medium' if risk_reward < 2 else 'low'
                },
                'signal': {
                    'signal': signal,
                    'confidence': signal_score,
                    'timestamp': datetime.datetime.utcnow()
                }
            }
            
            # Save to database
            try:
                session, user = get_user_session(user_id)
                if user:
                    db_signal = Signal(
                        user_id=user.id,
                        symbol=symbol,
                        signal_type=signal,
                        confidence=signal_score,
                        price=market_data['price'],
                        source='iran_bot',
                        timeframe=timeframe
                    )
                    session.add(db_signal)
                    session.commit()
                    session.close()
            except Exception as e:
                logger.error(f"Error saving signal to database: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    async def generate_chart(self, symbol: str, analysis_results: dict) -> Optional[InputFile]:
        """Generate technical analysis chart"""
        try:
            market_data = await self.iran_data.fetch_iran_market_data(symbol)
            if not market_data or 'ohlcv' not in market_data:
                return None
            
            df = pd.DataFrame(market_data['ohlcv'])
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'{symbol} Price', 'Volume'),
                row_width=[0.7, 0.3]
            )
            
            # Price chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Volume chart
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume',
                    marker_color='blue',
                    opacity=0.5
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Technical Analysis',
                template='plotly_dark',
                height=600,
                showlegend=False,
                xaxis_rangeslider_visible=False
            )
            
            # Convert to image
            img = fig.to_image(format="png", width=1000, height=600)
            return InputFile(io.BytesIO(img), filename=f'{symbol}_chart.png')
            
        except Exception as e:
            logger.error(f"Error generating chart for {symbol}: {e}")
            return None

# ------------------- Telegram Bot -------------------

def risk_percent_from_level(level: str) -> float:
    """Convert risk level to percentage"""
    mapping = {'low': 1.0, 'medium': 2.0, 'high': 3.0}
    return mapping.get(level.lower(), 2.0)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    welcome_text = """
    🤖 ربات تحلیلگر پیشرفته ارز دیجیتال (نسخه ایران)
    
    دستورات موجود:
    /signal [نماد] - دریافت سیگنال برای یک ارز دیجیتال
    /chart [نماد] - دریافت نمودار تحلیل تکنیکال
    /watchlist - مدیریت واچ لیست
    /portfolio - مشاهده پرتفولیو
    /help - راهنمایی
    
    مثال: /signal BTC
    """
    await update.message.reply_text(welcome_text)

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Signal command handler"""
    if not context.args:
        await update.message.reply_text("لطفاً یک نماد ارز مشخص کنید. مثال: /signal BTC")
        return
    
    symbol = context.args[0].upper()
    await update.message.chat.send_action(ChatAction.TYPING)
    
    try:
        bot = context.application.bot_data.get('bot_instance')
        if not bot:
            bot = IranCryptoBot()
            context.application.bot_data['bot_instance'] = bot
        
        analysis = await bot.analyze_symbol(symbol, update.effective_chat.id)
        if not analysis:
            await update.message.reply_text(f"خطا در تحلیل {symbol}")
            return
        
        signal = analysis['signal']
        market = analysis['market_data']
        risk = analysis['risk_management']
        
        # Prepare message
        message = f"""
        📊 تحلیل {symbol}
        
        🎯 سیگنال: {signal['signal']} (اعتماد: {signal['confidence']:.0%})
        💰 قیمت: ${market['price']:,.2f}
        📈 تغییر 24h: {market['change_24h']:.2f}%
        📊 حجم: ${market['volume_24h']:,.0f}
        
        📈 تکنیکال: {analysis['technical_analysis'].get('ichimoku', {}).get('cloud_green', False) and 'صعودی' or 'نزولی'}
        📰 سنتیمنت: {analysis['sentiment_analysis'].get('overall_sentiment', 'خنثی')}
        
        ⚠️ مدیریت ریسک:
        - حد ضرر: ${risk['stop_loss']:,.2f}
        - حد سود 1: ${risk['take_profit_1']:,.2f}
        - نسبت سود/ضرر: {risk['risk_reward_ratio']:.2f}
        - اندازه پوزیشن: {risk['position_size']:,.4f}
        
        🕒 زمان تحلیل: {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        await update.message.reply_text(message)
        
        # Send chart
        chart = await bot.generate_chart(symbol, analysis)
        if chart:
            await update.message.reply_photo(chart, caption=f"نمودار تحلیل تکنیکال {symbol}")
        
    except Exception as e:
        logger.error(f"Error in signal command: {e}")
        await update.message.reply_text("خطا در تحلیل نماد. لطفاً دوباره尝试 کنید.")

async def cmd_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Chart command handler"""
    if not context.args:
        await update.message.reply_text("لطفاً یک نماد ارز مشخص کنید. مثال: /chart BTC")
        return
    
    symbol = context.args[0].upper()
    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    
    try:
        bot = context.application.bot_data.get('bot_instance')
        if not bot:
            bot = IranCryptoBot()
            context.application.bot_data['bot_instance'] = bot
        
        analysis = await bot.analyze_symbol(symbol, update.effective_chat.id)
        if not analysis:
            await update.message.reply_text(f"خطا در تحلیل {symbol}")
            return
        
        chart = await bot.generate_chart(symbol, analysis)
        if chart:
            await update.message.reply_photo(chart, caption=f"نمودار تحلیل تکنیکال {symbol}")
        else:
            await update.message.reply_text("خطا در تولید نمودار.")
            
    except Exception as e:
        logger.error(f"Error in chart command: {e}")
        await update.message.reply_text("خطا در تولید نمودار. لطفاً دوباره尝试 کنید.")

async def cmd_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Watchlist command handler"""
    try:
        session, user = get_user_session(update.effective_chat.id)
        if not user:
            await update.message.reply_text("خطا در دریافت اطلاعات کاربر.")
            return
        
        watchlist = session.query(Watchlist).filter_by(user_id=user.id).all()
        session.close()
        
        if not watchlist:
            await update.message.reply_text("واچ لیست شما خالی است.")
            return
        
        message = "📋 واچ لیست شما:\n\n"
        for item in watchlist:
            message += f"• {item.symbol} (اضافه شده در {item.added_at.strftime('%Y-%m-%d')})\n"
        
        await update.message.reply_text(message)
        
    except Exception as e:
        logger.error(f"Error in watchlist command: {e}")
        await update.message.reply_text("خطا در دریافت واچ لیست.")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command handler"""
    help_text = """
    📖 راهنمای ربات تحلیلگر ارز دیجیتال
    
    این ربات مخصوص کاربران ایرانی طراحی شده و با محدودیت‌های دسترسی در ایران سازگار است.
    
    دستورات اصلی:
    • /signal [نماد] - دریافت سیگنال معاملاتی
    • /chart [نماد] - دریافت نمودار تحلیل تکنیکال
    • /watchlist - مدیریت لیست نظارت
    • /portfolio - مشاهده پرتفولیو
    • /help - نمایش این راهنما
    
    مثال‌ها:
    /signal BTC - دریافت سیگنال بیت‌کوین
    /chart ETH - دریافت نمودار اتریوم
    
    ⚠️ توجه: این ربات برای کمک به تحلیل طراحی شده و مسئولیت معاملات بر عهده کاربر است.
    """
    await update.message.reply_text(help_text)

# ------------------- Web Dashboard -------------------

class IranWebDashboard:
    def __init__(self, bot: IranCryptoBot):
        self.bot = bot
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup web routes"""
        self.app.router.add_get('/', self.index)
        self.app.router.add_get('/health', self.health)
        self.app.router.add_get('/api/market/{symbol}', self.get_market_data)
    
    async def index(self, request):
        """Main dashboard page"""
        html_content = """
        <!DOCTYPE html>
        <html lang="fa">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ربات تحلیلگر ارز دیجیتال - نسخه ایران</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    direction: rtl;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #2c3e50;
                    text-align: center;
                }
                .card {
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                    background: #f9f9f9;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 ربات تحلیلگر ارز دیجیتال - نسخه ایران</h1>
                
                <div class="card">
                    <h2>درباره ربات</h2>
                    <p>این ربات مخصوص کاربران ایرانی طراحی شده و با محدودیت‌های دسترسی در ایران سازگار است.</p>
                </div>
                
                <div class="card">
                    <h2>امکانات اصلی</h2>
                    <ul>
                        <li>تحلیل تکنیکال پیشرفته</li>
                        <li>تحلیل سنتیمنت بازار</li>
                        <li>سیگنال‌های معاملاتی</li>
                        <li>مدیریت ریسک هوشمند</li>
                        <li>نمودارهای تحلیلی</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h2>نحوه استفاده</h2>
                    <p>برای استفاده از ربات، از دستورات زیر در تلگرام استفاده کنید:</p>
                    <ul>
                        <li><code>/signal BTC</code> - دریافت سیگنال بیت‌کوین</li>
                        <li><code>/chart ETH</code> - دریافت نمودار اتریوم</li>
                        <li><code>/watchlist</code> - مدیریت واچ لیست</li>
                        <li><code>/help</code> - راهنمایی</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        return web.Response(text=html_content, content_type='text/html')
    
    async def health(self, request):
        """Health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'version': 'iran-1.0',
            'timestamp': datetime.datetime.utcnow().isoformat()
        })
    
    async def get_market_data(self, request):
        """Market data endpoint"""
        symbol = request.match_info.get('symbol', 'BTC').upper()
        
        try:
            market_data = await self.bot.iran_data.fetch_iran_market_data(symbol)
            if not market_data:
                return web.json_response({'error': 'Data not available'}, status=404)
            
            return web.json_response({
                'symbol': symbol,
                'price': market_data['price'],
                'change_24h': market_data['change_24h'],
                'volume_24h': market_data['volume_24h'],
                'timestamp': market_data['timestamp'].isoformat() if hasattr(market_data['timestamp'], 'isoformat') else str(market_data['timestamp'])
            })
            
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

# ------------------- Main Application -------------------

async def main():
    """Main application entry point"""
    logger.info("Starting Iran Crypto Bot...")
    
    # Initialize bot
    bot = IranCryptoBot()
    
    # Create tasks
    tasks = []
    
    # Start web dashboard if enabled
    if S.ENABLE_WEB_DASHBOARD:
        web_dashboard = IranWebDashboard(bot)
        runner = web.AppRunner(web_dashboard.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', S.WEB_PORT)
        await site.start()
        logger.info(f"Web dashboard running on port {S.WEB_PORT}")
    
    # Start Telegram bot if token is provided
    if S.TELEGRAM_BOT_TOKEN and S.TELEGRAM_BOT_TOKEN != "YOUR_TELEGRAM_BOT_TOKEN":
        try:
            application = ApplicationBuilder().token(S.TELEGRAM_BOT_TOKEN).build()
            application.bot_data['bot_instance'] = bot
            
            # Add handlers
            application.add_handler(CommandHandler("start", cmd_start))
            application.add_handler(CommandHandler("signal", cmd_signal))
            application.add_handler(CommandHandler("chart", cmd_chart))
            application.add_handler(CommandHandler("watchlist", cmd_watchlist))
            application.add_handler(CommandHandler("help", cmd_help))
            
            await application.initialize()
            await application.start()
            logger.info("Telegram bot started")
            
            # Start polling
            await application.updater.start_polling()
            
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
    else:
        logger.warning("Telegram bot token not provided, Telegram bot disabled")
    
    # Keep the application running
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("Application stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)