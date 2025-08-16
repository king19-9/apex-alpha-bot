# -*- coding: utf-8 -*-
"""
Advanced 24/7 Crypto AI Bot - Complete Version
Features: Multi-chain Analysis, Advanced TA, AI Predictions, Risk Management, Telegram Bot, Web Dashboard
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
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, InputMediaPhoto
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler, ConversationHandler
from telegram.constants import ChatAction
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import redis
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tweepy
from bs4 import BeautifulSoup
import requests

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

# Database Setup
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, unique=True)
    username = Column(String(50))
    preferences = Column(Text)  # JSON string
    risk_level = Column(String(20), default='medium')
    max_leverage = Column(Float, default=5.0)
    balance = Column(Float, default=10000.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
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
    
    # Relationships
    user = relationship("User", back_populates="signals")

class Watchlist(Base):
    __tablename__ = 'watchlist'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    symbol = Column(String(20))
    added_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
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
    
    # Relationships
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
    direction = Column(String(10))  # BUY/SELL/MOVE

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
    
    # Bot Settings
    OFFLINE_MODE: bool = os.getenv("OFFLINE_MODE", "false").lower() == "true"
    EXCHANGES: List[str] = os.getenv("EXCHANGES", "binance,kucoin,bybit,bitfinex,gateio,bitget").split(",")
    MAX_COINS: int = int(os.getenv("MAX_COINS", "1000"))
    UNIVERSE_MAX_PAGES: int = int(os.getenv("UNIVERSE_MAX_PAGES", "20"))
    TIMEFRAMES: List[str] = os.getenv("TIMEFRAMES", "1m,5m,15m,1h,4h,1d").split(",")
    
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
    ONCHAIN_CHAINS: List[str] = os.getenv("ONCHAIN_CHAINS", "ethereum,polygon,binance-smart-chain,avalanche,arbitrum,fantom").split(",")
    
    # Web Dashboard
    ENABLE_WEB_DASHBOARD: bool = os.getenv("ENABLE_WEB_DASHBOARD", "true").lower() == "true"
    WEB_PORT: int = int(os.getenv("WEB_PORT", "8080"))

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

# Database Setup
try:
    engine = create_engine(S.DATABASE_URL, pool_pre_ping=True, pool_size=20, max_overflow=30)
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    logger.info("Database connected successfully")
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    sys.exit(1)

# Helper Functions
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
    
    def detect_order_blocks(self, df: pd.DataFrame, lookback: int = 50) -> dict:
        bullish_blocks = []
        bearish_blocks = []
        
        for i in range(lookback, len(df)):
            # Bullish Order Block
            if (df.iloc[i-1]['close'] < df.iloc[i-1]['open'] and  # Bearish candle
                df.iloc[i]['close'] > df.iloc[i]['open'] and      # Bullish candle
                df.iloc[i]['close'] > df.iloc[i-2]['high']):      # Breaks previous high
                
                bullish_blocks.append({
                    'index': i-1,
                    'high': df.iloc[i-1]['high'],
                    'low': df.iloc[i-1]['low'],
                    'close': df.iloc[i-1]['close'],
                    'type': 'bullish'
                })
            
            # Bearish Order Block
            if (df.iloc[i-1]['close'] > df.iloc[i-1]['open'] and  # Bullish candle
                df.iloc[i]['close'] < df.iloc[i]['open'] and      # Bearish candle
                df.iloc[i]['close'] < df.iloc[i-2]['low']):      # Breaks previous low
                
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
            # Swing High
            if df.iloc[i]['high'] == df.iloc[i-window:i+window]['high'].max():
                strength = len([x for x in df.iloc[i-window:i+window]['high'] 
                              if abs(x - df.iloc[i]['high']) < 0.01])
                high_zones.append({
                    'price': df.iloc[i]['high'],
                    'index': i,
                    'strength': strength
                })
            
            # Swing Low
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
        
        # Find swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(df)-2):
            # Swing High
            if (df.iloc[i]['high'] > df.iloc[i-1]['high'] and 
                df.iloc[i]['high'] > df.iloc[i+1]['high']):
                swing_highs.append((i, df.iloc[i]['high']))
            
            # Swing Low
            if (df.iloc[i]['low'] < df.iloc[i-1]['low'] and 
                df.iloc[i]['low'] < df.iloc[i+1]['low']):
                swing_lows.append((i, df.iloc[i]['low']))
        
        # Analyze structure
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Uptrend
            if (swing_highs[-1][1] > swing_highs[-2][1] and 
                swing_lows[-1][1] > swing_lows[-2][1]):
                structure['trend'] = 'uptrend'
                structure['higher_highs'] = swing_highs
                structure['higher_lows'] = swing_lows
            
            # Downtrend
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
            # Demand Zone
            if (df.iloc[i]['close'] > df.iloc[i-1]['close'] * 1.02 and  # 2% up move
                df.iloc[i]['volume'] > df.iloc[i-1]['volume'] * 1.5):  # High volume
                
                demand_zones.append({
                    'price': df.iloc[i-1]['close'],
                    'strength': df.iloc[i]['volume'] / df.iloc[i-1]['volume'],
                    'index': i-1
                })
            
            # Supply Zone
            if (df.iloc[i]['close'] < df.iloc[i-1]['close'] * 0.98 and  # 2% down move
                df.iloc[i]['volume'] > df.iloc[i-1]['volume'] * 1.5):  # High volume
                
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
            
            # Check for consolidation
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
        
        # Calculate indicators
        if not df.empty:
            results['indicators'] = {
                'rsi': self.calculate_rsi(df['close']).iloc[-1],
                'macd': self.calculate_macd(df['close'])[0].iloc[-1],
                'bollinger': self.calculate_bollinger_bands(df['close']),
                'atr': self.calculate_atr(df).iloc[-1],
                'stochastic': self.calculate_stochastic(df)[0].iloc[-1]
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
        
        # Adjust for liquidity
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
        
        # Calculate buy/sell ratio
        buys = sum(1 for tx in transactions if tx.get('to') in self.token_contracts.get(symbol, {}).values())
        sells = sum(1 for tx in transactions if tx.get('from') in self.token_contracts.get(symbol, {}).values())
        
        total_volume = sum(tx['value'] for tx in transactions)
        
        # Determine bias
        if buys > sells * 1.5:
            bias = 'bullish'
        elif sells > buys * 1.5:
            bias = 'bearish'
        else:
            bias = 'neutral'
        
        # Determine activity level
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

# Enhanced Sentiment Analysis Module
class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
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
        
        # Analyze sentiment
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
    
    async def combined_sentiment_analysis(self, symbol: str) -> dict:
        news_sentiment = await self.analyze_news_sentiment(symbol)
        twitter_sentiment = await self.analyze_twitter_sentiment(symbol)
        
        # Combine scores with weights
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
            'twitter': twitter_sentiment
        }

# Enhanced Prediction Engine
class EnhancedPredictionEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    async def prepare_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        # Technical indicators
        ta_analyzer = AdvancedTechnicalAnalyzer()
        df['rsi'] = ta_analyzer.calculate_rsi(df['close'])
        df['macd'], df['macd_signal'], df['macd_hist'] = ta_analyzer.calculate_macd(df['close'])
        df['bb_upper'], df['bb_mid'], df['bb_lower'] = ta_analyzer.calculate_bollinger_bands(df['close'])
        df['atr'] = ta_analyzer.calculate_atr(df)
        df['stoch_k'], df['stoch_d'] = ta_analyzer.calculate_stochastic(df)
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    async def train_model(self, df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        # Prepare features
        df = await self.prepare_features(df, symbol)
        
        # Define features and target
        features = ['rsi', 'macd', 'macd_hist', 'bb_upper', 'bb_lower', 'atr', 
                   'stoch_k', 'stoch_d', 'returns', 'log_returns', 'volatility', 'volume_ratio',
                   'hour', 'day_of_week']
        
        X = df[features]
        y = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Save model and scaler
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
        
        # Prepare features
        df = await self.prepare_features(df, symbol)
        
        # Get latest features
        features = ['rsi', 'macd', 'macd_hist', 'bb_upper', 'bb_lower', 'atr', 
                   'stoch_k', 'stoch_d', 'returns', 'log_returns', 'volatility', 'volume_ratio',
                   'hour', 'day_of_week']
        
        X = df[features].iloc[-1:].values
        
        # Scale features
        scaler = self.scalers[model_key]
        X_scaled = scaler.transform(X)
        
        # Make prediction
        model = self.models[model_key]
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        return {
            'direction': 'BUY' if prediction == 1 else 'SELL',
            'probability': probability,
            'confidence': max(probability, 1 - probability)
        }

# Main Bot Class
class AdvancedCryptoBot:
    def __init__(self):
        self.ta_analyzer = AdvancedTechnicalAnalyzer()
        self.risk_manager = AdvancedRiskManager()
        self.onchain_analyzer = EnhancedOnChainAnalyzer()
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        self.prediction_engine = EnhancedPredictionEngine()
        
    async def fetch_market_data(self, symbol: str) -> dict:
        cache_key = f"market_data_{symbol}"
        cached_data = get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Try to get data from primary exchange first
            exchange = ccxt.binance()
            ticker = await exchange.fetch_ticker(f"{symbol}/USDT")
            ohlcv = await exchange.fetch_ohlcv(f"{symbol}/USDT", "1h", limit=1000)
            
            # Convert to DataFrame
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
            
            # Cache result
            cache_with_redis(cache_key, data, S.CACHE_TTL_SECONDS)
            
            return data
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    async def analyze_symbol(self, symbol: str, user_id: int = None) -> dict:
        try:
            # Get user preferences
            session, user = get_user_session(user_id)
            if not user:
                return None
            
            user_prefs = json.loads(user.preferences or '{}')
            
            # Fetch market data
            market_data = await self.fetch_market_data(symbol)
            if not market_data:
                return None
            
            # Technical Analysis
            ta_results = self.ta_analyzer.full_analysis(market_data['ohlcv'])
            
            # On-chain Analysis
            onchain_results = await self.onchain_analyzer.analyze_whale_activity(symbol) if S.ENABLE_ONCHAIN else {}
            
            # Sentiment Analysis
            sentiment_results = await self.sentiment_analyzer.combined_sentiment_analysis(symbol) if S.ENABLE_SENTIMENT else {}
            
            # Prediction
            prediction = await self.prediction_engine.predict(
                market_data['ohlcv'], symbol, '1h'
            ) if S.ENABLE_PREDICTION else {}
            
            # Risk Management
            current_price = market_data['price']
            risk_params = self._calculate_risk_parameters(
                market_data['ohlcv'], current_price, user
            )
            
            # Generate signal
            signal = self._generate_signal(
                ta_results, onchain_results, sentiment_results, 
                prediction, risk_params
            )
            
            # Save to database
            self._save_analysis(
                user_id, symbol, signal, prediction, 
                market_data, ta_results, onchain_results
            )
            
            return {
                'symbol': symbol,
                'market_data': market_data,
                'technical_analysis': ta_results,
                'onchain_analysis': onchain_results,
                'sentiment_analysis': sentiment_results,
                'prediction': prediction,
                'risk_management': risk_params,
                'signal': signal,
                'timestamp': datetime.datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _calculate_risk_parameters(self, df: pd.DataFrame, current_price: float, user: User) -> dict:
        # Dynamic stop loss
        stop_loss = self.risk_manager.dynamic_stop_loss(df, current_price, 'long')
        
        # Position size
        position_size = self.risk_manager.calculate_position_size(
            account_balance=user.balance,
            risk_percent=user.risk_level,
            entry_price=current_price,
            stop_loss=stop_loss,
            method='volatility_adjusted',
            volatility=df['close'].pct_change().std()
        )
        
        # Optimal leverage
        leverage = self.risk_manager.calculate_optimal_leverage(
            volatility=df['close'].pct_change().std(),
            account_balance=user.balance,
            position_size=position_size,
            max_leverage=user.max_leverage
        )
        
        # Take profit levels
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
            )
        }
    
    def _generate_signal(self, ta_results, onchain_results, sentiment_results, 
                        prediction, risk_params) -> dict:
        signal_score = 0
        
        # Technical Analysis Weight (40%)
        if ta_results.get('market_structure', {}).get('trend') == 'uptrend':
            signal_score += 0.4
        elif ta_results.get('market_structure', {}).get('trend') == 'downtrend':
            signal_score -= 0.4
        
        # On-chain Weight (30%)
        if onchain_results.get('bias') == 'bullish':
            signal_score += 0.3
        elif onchain_results.get('bias') == 'bearish':
            signal_score -= 0.3
        
        # Sentiment Weight (20%)
        if sentiment_results.get('sentiment') == 'bullish':
            signal_score += 0.2
        elif sentiment_results.get('sentiment') == 'bearish':
            signal_score -= 0.2
        
        # Prediction Weight (10%)
        if prediction and prediction.get('direction') == 'BUY':
            signal_score += 0.1
        elif prediction and prediction.get('direction') == 'SELL':
            signal_score -= 0.1
        
        # Generate final signal
        if signal_score > 0.6:
            signal = 'BUY'
        elif signal_score < -0.6:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'score': signal_score,
            'confidence': abs(signal_score)
        }
    
    def _save_analysis(self, user_id, symbol, signal, prediction, 
                     market_data, ta_results, onchain_results):
        try:
            session = Session()
            
            # Save signal
            db_signal = Signal(
                user_id=user_id,
                symbol=symbol,
                signal_type=signal['signal'],
                confidence=signal['confidence'],
                price=market_data['price'],
                source='bot_analysis'
            )
            session.add(db_signal)
            
            # Update user performance
            performance = session.query(Performance).filter_by(user_id=user_id).first()
            if not performance:
                performance = Performance(user_id=user_id)
                session.add(performance)
            
            performance.total_signals += 1
            performance.updated_at = datetime.datetime.utcnow()
            
            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
    
    async def generate_chart(self, symbol: str, analysis_results: dict) -> InputFile:
        try:
            # Get OHLCV data
            df = analysis_results['market_data']['ohlcv']
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Candlestick chart
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
            
            # Add technical indicators
            if 'indicators' in analysis_results['technical_analysis']:
                indicators = analysis_results['technical_analysis']['indicators']
                
                # Bollinger Bands
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicators['bb_upper'][0],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='red', width=1),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicators['bb_lower'][0],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='green', width=1),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Add order blocks
            for block in analysis_results['technical_analysis']['order_blocks']['bullish']:
                fig.add_shape(
                    type="rect",
                    x0=df.index[block['index']],
                    x1=df.index[block['index']+5],
                    y0=block['low'],
                    y1=block['high'],
                    fillcolor="green",
                    opacity=0.2,
                    layer="below",
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
            
            # RSI chart
            if 'indicators' in analysis_results['technical_analysis']:
                rsi_data = []
                for i in range(len(df)):
                    if i >= 13:  # RSI period
                        rsi = self.ta_analyzer.calculate_rsi(df['close'].iloc[:i+1]).iloc[-1]
                        rsi_data.append(rsi)
                    else:
                        rsi_data.append(None)
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=rsi_data,
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple'),
                        yaxis='y2'
                    ),
                    row=3, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Technical Analysis',
                yaxis=dict(title='Price (USDT)'),
                yaxis2=dict(title='RSI', overlaying='y', side='right'),
                template='plotly_dark',
                height=800,
                showlegend=True
            )
            
            # Convert to image
            img_bytes = fig.to_image(format="png", width=1200, height=800)
            
            return InputFile(io.BytesIO(img_bytes), filename=f'{symbol}_analysis.png')
            
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return None

# Web Dashboard
class WebDashboard:
    def __init__(self, bot):
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
        return web.Response(text="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crypto AI Bot Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #0a0e27;
            color: #ffffff;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            margin-bottom: 40px;
            padding: 20px 0;
            border-bottom: 2px solid #1a237e;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #1a237e, #0f3460);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-card h3 {
            font-size: 0.9em;
            color: #94a3b8;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stat-card .value {
            font-size: 2.2em;
            font-weight: bold;
            color: #ffffff;
        }
        
        .content-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }
        
        .panel {
            background: #151b3d;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .panel h2 {
            margin-bottom: 20px;
            color: #667eea;
            font-size: 1.5em;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #2a3f5f;
        }
        
        th {
            background: #1a237e;
            color: #667eea;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }
        
        tr:hover {
            background: #1a237e;
        }
        
        .signal-buy {
            color: #4ade80;
            font-weight: bold;
        }
        
        .signal-sell {
            color: #f87171;
            font-weight: bold;
        }
        
        .signal-hold {
            color: #fbbf24;
            font-weight: bold;
        }
        
        .refresh-btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s ease;
            margin: 20px 0;
        }
        
        .refresh-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .chart-container {
            margin-top: 20px;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            border-top: 2px solid #1a237e;
            color: #94a3b8;
        }
        
        @media (max-width: 768px) {
            .content-grid {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 Advanced Crypto AI Bot</h1>
            <p>Real-time cryptocurrency analysis and trading signals</p>
        </header>
        
        <div class="stats-grid" id="stats">
            <div class="stat-card">
                <h3>Total Users</h3>
                <div class="value" id="totalUsers">-</div>
            </div>
            <div class="stat-card">
                <h3>Total Signals</h3>
                <div class="value" id="totalSignals">-</div>
            </div>
            <div class="stat-card">
                <h3>Success Rate</h3>
                <div class="value" id="successRate">-</div>
            </div>
            <div class="stat-card">
                <h3>Uptime</h3>
                <div class="value" id="uptime">-</div>
            </div>
        </div>
        
        <div class="content-grid">
            <div class="panel">
                <h2>📊 Recent Signals</h2>
                <table id="signalsTable">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Signal</th>
                            <th>Confidence</th>
                            <th>Price</th>
                            <th>Time</th>
                        </tr>
                    </thead>
                    <tbody id="signalsBody">
                        <!-- Signals will be populated here -->
                    </tbody>
                </table>
                <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
            </div>
            
            <div class="panel">
                <h2>📈 Performance Metrics</h2>
                <div id="performanceMetrics">
                    <!-- Performance metrics will be populated here -->
                </div>
            </div>
        </div>
        
        <div class="panel">
            <h2>📋 Watchlist</h2>
            <table id="watchlistTable">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Added</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="watchlistBody">
                    <!-- Watchlist will be populated here -->
                </tbody>
            </table>
        </div>
        
        <footer>
            <p>&copy; 2024 Advanced Crypto AI Bot. All rights reserved.</p>
        </footer>
    </div>
    
    <script>
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                document.getElementById('totalUsers').textContent = stats.total_users || 0;
                document.getElementById('totalSignals').textContent = stats.total_signals || 0;
                document.getElementById('successRate').textContent = 
                    stats.success_rate ? (stats.success_rate * 100).toFixed(1) + '%' : '0%';
                document.getElementById('uptime').textContent = 
                    stats.uptime ? Math.floor(stats.uptime / 3600) + 'h' : '0h';
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        }
        
        async function loadSignals() {
            try {
                const response = await fetch('/api/signals');
                const signals = await response.json();
                const tbody = document.getElementById('signalsBody');
                
                tbody.innerHTML = signals.map(s => `
                    <tr>
                        <td>${s.symbol}</td>
                        <td class="signal-${s.signal_type.toLowerCase()}">${s.signal_type}</td>
                        <td>${(s.confidence * 100).toFixed(1)}%</td>
                        <td>$${s.price.toFixed(2)}</td>
                        <td>${new Date(s.timestamp).toLocaleString()}</td>
                    </tr>
                `).join('');
            } catch (error) {
                console.error('Error loading signals:',