# -*- coding: utf-8 -*-
# Advanced 24/7 Crypto AI Bot: Multi-chain + Advanced TA + AI + Risk Management
# Complete version with all improvements and new features

import os, sys, time, asyncio, statistics, random, logging, json, datetime, re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import aiohttp
from aiohttp import web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import redis
from datetime import timedelta
import ccxt.async_support as ccxt
import websockets
import json as _json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler, ConversationHandler
from telegram.constants import ChatAction
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import tweepy

# Initialize Redis
redis_client = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), 
                          port=int(os.getenv('REDIS_PORT', 6379)), 
                          db=0, decode_responses=True)

# Base for SQLAlchemy models
Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, unique=True)
    username = Column(String(50))
    preferences = Column(Text)  # JSON string
    risk_level = Column(String(20), default='medium')
    max_leverage = Column(Float, default=5.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Signal(Base):
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20))
    signal_type = Column(String(10))  # BUY/SELL/HOLD
    confidence = Column(Float)
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    source = Column(String(50))
    user_id = Column(Integer)
    evaluated = Column(Boolean, default=False)
    result = Column(Float)

class Watchlist(Base):
    __tablename__ = 'watchlist'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    symbol = Column(String(20))
    added_at = Column(DateTime, default=datetime.datetime.utcnow)

class Performance(Base):
    __tablename__ = 'performance'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    total_signals = Column(Integer, default=0)
    successful_signals = Column(Integer, default=0)
    avg_return = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

# Settings Class
@dataclass
class Settings:
    # API Keys
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID")
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
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Bot Settings
    OFFLINE_MODE: bool = os.getenv("OFFLINE_MODE", "false").lower() == "true"
    EXCHANGES: List[str] = os.getenv("EXCHANGES", "binance,kucoin,bybit,bitfinex,gateio,bitget").split(",")
    MAX_COINS: int = int(os.getenv("MAX_COINS", "500"))
    UNIVERSE_MAX_PAGES: int = int(os.getenv("UNIVERSE_MAX_PAGES", "10"))
    TIMEFRAMES: List[str] = os.getenv("TIMEFRAMES", "1m,5m,15m,1h,4h,1d").split(",")
    
    # Performance Settings
    CONCURRENT_REQUESTS: int = int(os.getenv("CONCURRENT_REQUESTS", "15"))
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    
    # Analysis Settings
    ENABLE_ADVANCED_TA: bool = os.getenv("ENABLE_ADVANCED_TA", "true").lower() == "true"
    ENABLE_ONCHAIN: bool = os.getenv("ENABLE_ONCHAIN", "true").lower() == "true"
    ENABLE_SENTIMENT: bool = os.getenv("ENABLE_SENTIMENT", "true").lower() == "true"
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
    ONCHAIN_CHAINS: List[str] = os.getenv("ONCHAIN_CHAINS", "ethereum,polygon,binance-smart-chain,avalanche,arbitrum").split(",")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///bot.db")
    
    # Web Dashboard
    ENABLE_WEB_DASHBOARD: bool = os.getenv("ENABLE_WEB_DASHBOARD", "true").lower() == "true"
    WEB_PORT: int = int(os.getenv("WEB_PORT", "8080"))

S = Settings()

# Logger Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Database Setup
engine = create_engine(S.DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

# Helper Functions
def get_user_session(chat_id: int):
    session = Session()
    user = session.query(User).filter_by(chat_id=chat_id).first()
    if not user:
        user = User(chat_id=chat_id)
        session.add(user)
        session.commit()
    return session, user

def cache_with_redis(key: str, value: Any, ttl: int = 300):
    try:
        redis_client.setex(key, ttl, json.dumps(value))
    except Exception as e:
        logger.error(f"Redis error: {e}")

def get_from_cache(key: str):
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
        self.support_resistance_levels = {}
        self.order_blocks = {}
        self.liquidity_zones = {}
        self.market_structure = {}
        
    def detect_order_blocks(self, df: pd.DataFrame, lookback: int = 50):
        """Detect order blocks in price action"""
        bullish_blocks = []
        bearish_blocks = []
        
        for i in range(lookback, len(df)):
            # Bullish Order Block: Strong down candle followed by strong up move
            if (df.iloc[i-1]['close'] < df.iloc[i-1]['open'] and  # Bearish candle
                df.iloc[i]['close'] > df.iloc[i]['open'] and      # Bullish candle
                df.iloc[i]['close'] > df.iloc[i-2]['high']):      # Breaks previous high
                
                bullish_blocks.append({
                    'index': i-1,
                    'high': df.iloc[i-1]['high'],
                    'low': df.iloc[i-1]['low'],
                    'type': 'bullish'
                })
            
            # Bearish Order Block: Strong up candle followed by strong down move
            if (df.iloc[i-1]['close'] > df.iloc[i-1]['open'] and  # Bullish candle
                df.iloc[i]['close'] < df.iloc[i]['open'] and      # Bearish candle
                df.iloc[i]['close'] < df.iloc[i-2]['low']):      # Breaks previous low
                
                bearish_blocks.append({
                    'index': i-1,
                    'high': df.iloc[i-1]['high'],
                    'low': df.iloc[i-1]['low'],
                    'type': 'bearish'
                })
        
        return {'bullish': bullish_blocks, 'bearish': bearish_blocks}
    
    def detect_liquidity_zones(self, df: pd.DataFrame, window: int = 20):
        """Detect liquidity zones where stop hunts might occur"""
        high_zones = []
        low_zones = []
        
        # Find swing highs and lows
        for i in range(window, len(df)-window):
            # Swing High
            if df.iloc[i]['high'] == df.iloc[i-window:i+window]['high'].max():
                high_zones.append({
                    'price': df.iloc[i]['high'],
                    'index': i,
                    'strength': len([x for x in df.iloc[i-window:i+window]['high'] if abs(x - df.iloc[i]['high']) < 0.01])
                })
            
            # Swing Low
            if df.iloc[i]['low'] == df.iloc[i-window:i+window]['low'].min():
                low_zones.append({
                    'price': df.iloc[i]['low'],
                    'index': i,
                    'strength': len([x for x in df.iloc[i-window:i+window]['low'] if abs(x - df.iloc[i]['low']) < 0.01])
                })
        
        return {'high_zones': high_zones, 'low_zones': low_zones}
    
    def detect_market_structure(self, df: pd.DataFrame):
        """Analyze market structure for higher highs/higher lows or lower highs/lower lows"""
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
            # Check for uptrend
            if (swing_highs[-1][1] > swing_highs[-2][1] and 
                swing_lows[-1][1] > swing_lows[-2][1]):
                structure['trend'] = 'uptrend'
                structure['higher_highs'] = swing_highs
                structure['higher_lows'] = swing_lows
            
            # Check for downtrend
            elif (swing_highs[-1][1] < swing_highs[-2][1] and 
                  swing_lows[-1][1] < swing_lows[-2][1]):
                structure['trend'] = 'downtrend'
                structure['lower_highs'] = swing_highs
                structure['lower_lows'] = swing_lows
        
        return structure
    
    def detect_supply_demand_zones(self, df: pd.DataFrame, lookback: int = 50):
        """Detect supply and demand zones"""
        supply_zones = []
        demand_zones = []
        
        for i in range(lookback, len(df)):
            # Demand Zone: Strong rally from a consolidation area
            if (df.iloc[i]['close'] > df.iloc[i-1]['close'] * 1.02 and  # 2% up move
                df.iloc[i]['volume'] > df.iloc[i-1]['volume'] * 1.5):  # High volume
                
                demand_zones.append({
                    'price': df.iloc[i-1]['close'],
                    'strength': df.iloc[i]['volume'] / df.iloc[i-1]['volume'],
                    'index': i-1
                })
            
            # Supply Zone: Strong drop from a consolidation area
            if (df.iloc[i]['close'] < df.iloc[i-1]['close'] * 0.98 and  # 2% down move
                df.iloc[i]['volume'] > df.iloc[i-1]['volume'] * 1.5):  # High volume
                
                supply_zones.append({
                    'price': df.iloc[i-1]['close'],
                    'strength': df.iloc[i]['volume'] / df.iloc[i-1]['volume'],
                    'index': i-1
                })
        
        return {'supply': supply_zones, 'demand': demand_zones}
    
    def detect_inside_bars(self, df: pd.DataFrame):
        """Detect inside bar patterns"""
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
    
    def analyze_session_patterns(self, df: pd.DataFrame):
        """Analyze price behavior during different trading sessions"""
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
    
    def detect_decision_zones(self, df: pd.DataFrame):
        """Detect areas where price decisions are made"""
        decision_zones = []
        
        # Find areas of high volume and price consolidation
        for i in range(10, len(df)-10):
            window = df.iloc[i-10:i+10]
            
            # Check for consolidation (low volatility)
            volatility = (window['high'].max() - window['low'].min()) / window['close'].mean()
            
            # Check for high volume
            avg_volume = window['volume'].mean()
            
            if volatility < 0.02 and avg_volume > df['volume'].quantile(0.7):
                decision_zones.append({
                    'price': window['close'].mean(),
                    'strength': avg_volume / df['volume'].mean(),
                    'range': window['high'].max() - window['low'].min(),
                    'index': i
                })
        
        return decision_zones
    
    def full_analysis(self, df: pd.DataFrame):
        """Perform complete technical analysis"""
        results = {
            'order_blocks': self.detect_order_blocks(df),
            'liquidity_zones': self.detect_liquidity_zones(df),
            'market_structure': self.detect_market_structure(df),
            'supply_demand': self.detect_supply_demand_zones(df),
            'inside_bars': self.detect_inside_bars(df),
            'session_patterns': self.analyze_session_patterns(df),
            'decision_zones': self.detect_decision_zones(df)
        }
        
        return results

# Enhanced Risk Management Module
class AdvancedRiskManager:
    def __init__(self):
        self.position_sizing_methods = ['fixed_risk', 'kelly', 'volatility_adjusted']
        self.risk_metrics = {}
        
    def calculate_position_size(self, account_balance: float, risk_percent: float, 
                              entry_price: float, stop_loss: float, method: str = 'fixed_risk',
                              volatility: float = None, liquidity: float = None):
        """Calculate optimal position size based on multiple factors"""
        risk_amount = account_balance * (risk_percent / 100)
        
        if method == 'fixed_risk':
            # Fixed fractional risk
            position_size = risk_amount / abs(entry_price - stop_loss)
        
        elif method == 'kelly':
            # Kelly Criterion
            win_rate = self.risk_metrics.get('win_rate', 0.5)
            avg_win = self.risk_metrics.get('avg_win', 1.0)
            avg_loss = self.risk_metrics.get('avg_loss', 1.0)
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            position_size = (account_balance * kelly_fraction) / entry_price
        
        elif method == 'volatility_adjusted':
            # Adjust position size based on market volatility
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
                                  position_size: float, max_leverage: float = None):
        """Calculate optimal leverage based on volatility"""
        if not max_leverage:
            max_leverage = S.DEFAULT_MAX_LEVERAGE
        
        # Lower leverage in high volatility environments
        volatility_adjustment = 1 / (1 + volatility)
        
        # Calculate leverage based on position size relative to account
        base_leverage = (position_size * S.DEFAULT_MAX_LEVERAGE) / account_balance
        
        # Apply volatility adjustment
        optimal_leverage = base_leverage * volatility_adjustment
        
        return min(optimal_leverage, max_leverage)
    
    def dynamic_stop_loss(self, df: pd.DataFrame, entry_price: float, 
                         direction: str, atr_period: int = 14):
        """Calculate dynamic stop loss based on market conditions"""
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=atr_period).mean().iloc[-1]
        
        # Calculate stop distance based on ATR
        if direction == 'long':
            stop_distance = atr * 2.5  # 2.5 ATR for long positions
            stop_loss = entry_price - stop_distance
        else:
            stop_distance = atr * 2.5  # 2.5 ATR for short positions
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, 
                                  take_profit: float):
        """Calculate risk/reward ratio"""
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return 0
        
        return reward / risk
    
    def portfolio_heat_check(self, current_positions: List[Dict], 
                           new_position_size: float, account_balance: float):
        """Check if adding new position exceeds portfolio heat limits"""
        total_risk = sum(pos['size'] * pos['risk_percent'] for pos in current_positions)
        new_position_risk = new_position_size * S.DEFAULT_RISK_PER_TRADE
        
        total_portfolio_risk = (total_risk + new_position_risk) / account_balance
        
        # Maximum 20% portfolio risk at any time
        max_portfolio_risk = 0.20
        
        return total_portfolio_risk <= max_portfolio_risk

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
            }
        }
        
        self.token_contracts = {
            'USDT': {
                'ethereum': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
                'polygon': '0xc2132D05D31c914a87C6611C10748AEb04B58e8F',
                'binance-smart-chain': '0x55d398326f99059fF775485246999027B3197955'
            },
            'USDC': {
                'ethereum': '0xA0b86a33E6417aAb7b6DbCBbe9FD4E89c0778a4B',
                'polygon': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
                'binance-smart-chain': '0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d'
            }
        }
    
    async def get_large_transactions(self, symbol: str, min_usd: float = 500000):
        """Get large transactions from multiple chains"""
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
    
    async def _fetch_chain_transactions(self, chain: str, contract_address: str, min_usd: float):
        """Fetch transactions from a specific chain"""
        api_config = self.chain_apis[chain]
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
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get('status') != '1':
                    return []
                
                transactions = []
                for tx in data.get('result', []):
                    value = float(tx.get('value', 0))
                    # Convert token value to USD (simplified)
                    usd_value = value * 1.0  # Should use actual price
                    
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
    
    async def analyze_whale_activity(self, symbol: str):
        """Analyze whale activity patterns"""
        transactions = await self.get_large_transactions(symbol)
        
        if not transactions:
            return {'activity': 'low', 'bias': 'neutral'}
        
        # Calculate buy/sell ratio
        buys = sum(1 for tx in transactions if tx.get('to') == self.token_contracts[symbol].get('ethereum'))
        sells = sum(1 for tx in transactions if tx.get('from') == self.token_contracts[symbol].get('ethereum'))
        
        total_volume = sum(tx['value'] for tx in transactions)
        
        # Determine bias
        if buys > sells * 1.5:
            bias = 'bullish'
        elif sells > buys * 1.5:
            bias = 'bearish'
        else:
            bias = 'neutral'
        
        # Determine activity level
        if total_volume > 10000000:  # $10M
            activity = 'very_high'
        elif total_volume > 5000000:  # $5M
            activity = 'high'
        elif total_volume > 1000000:  # $1M
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
        self.twitter_client = tweepy.Client(
            bearer_token=S.TWITTER_BEARER_TOKEN,
            wait_on_rate_limit=True
        )
        
    async def analyze_news_sentiment(self, symbol: str):
        """Analyze sentiment from news sources"""
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
    
    async def _fetch_cryptopanic_news(self, symbol: str):
        """Fetch news from CryptoPanic"""
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            'auth_token': S.CRYPTOPANIC_API_KEY,
            'currencies': symbol,
            'kind': 'news'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data.get('results', [])
    
    async def _fetch_newsapi_news(self, symbol: str):
        """Fetch news from NewsAPI"""
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f'{symbol} cryptocurrency',
            'apiKey': S.NEWS_API_KEY,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 20
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data.get('articles', [])
    
    async def analyze_twitter_sentiment(self, symbol: str):
        """Analyze sentiment from Twitter"""
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
    
    async def combined_sentiment_analysis(self, symbol: str):
        """Combine news and Twitter sentiment"""
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

# Enhanced Prediction Module
class EnhancedPredictionEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    async def prepare_features(self, df: pd.DataFrame, symbol: str):
        """Prepare features for ML models"""
        # Technical indicators
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'] = self._calculate_macd(df['close'])
        df['bollinger_upper'] = self._calculate_bollinger_bands(df['close'])[0]
        df['bollinger_lower'] = self._calculate_bollinger_bands(df['close'])[1]
        df['atr'] = self._calculate_atr(df)
        
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
    
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd - signal_line
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def _calculate_atr(self, df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()
    
    async def train_model(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Train ML model for prediction"""
        # Prepare features
        df = await self.prepare_features(df, symbol)
        
        # Define features and target
        features = ['rsi', 'macd', 'bollinger_upper', 'bollinger_lower', 'atr', 
                   'returns', 'log_returns', 'volatility', 'volume_ratio',
                   'hour', 'day_of_week']
        
        X = df[features]
        y = (df['close'].shift(-1) > df['close']).astype(int)  # Next period direction
        
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
    
    async def predict(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Make prediction using trained model"""
        model_key = f"{symbol}_{timeframe}"
        
        if model_key not in self.models:
            return None
        
        # Prepare features
        df = await self.prepare_features(df, symbol)
        
        # Get latest features
        features = ['rsi', 'macd', 'bollinger_upper', 'bollinger_lower', 'atr', 
                   'returns', 'log_returns', 'volatility', 'volume_ratio',
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
        
    async def analyze_symbol(self, symbol: str, user_id: int = None):
        """Complete symbol analysis"""
        try:
            # Get user preferences
            session, user = get_user_session(user_id)
            user_prefs = json.loads(user.preferences or '{}')
            
            # Fetch market data
            market_data = await self._fetch_market_data(symbol)
            
            # Technical Analysis
            ta_results = self.ta_analyzer.full_analysis(market_data['ohlcv']['1h'])
            
            # On-chain Analysis
            onchain_results = await self.onchain_analyzer.analyze_whale_activity(symbol)
            
            # Sentiment Analysis
            sentiment_results = await self.sentiment_analyzer.combined_sentiment_analysis(symbol)
            
            # Prediction
            prediction = await self.prediction_engine.predict(
                market_data['ohlcv']['1h'], symbol, '1h'
            )
            
            # Risk Management
            current_price = market_data['market_data']['price']
            risk_params = self._calculate_risk_parameters(
                market_data['ohlcv']['1h'], current_price, user
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
                'market_data': market_data['market_data'],
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
    
    async def _fetch_market_data(self, symbol: str):
        """Fetch market data from multiple sources"""
        # Check cache first
        cache_key = f"market_data_{symbol}"
        cached_data = get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # Fetch from exchanges
        ohlcv = {}
        for tf in S.TIMEFRAMES:
            ohlcv[tf] = await self._fetch_ohlcv(symbol, tf)
        
        # Get market info
        market_info = await self._fetch_market_info(symbol)
        
        data = {
            'market_data': market_info,
            'ohlcv': ohlcv
        }
        
        # Cache result
        cache_with_redis(cache_key, data, S.CACHE_TTL_SECONDS)
        
        return data
    
    async def _fetch_ohlcv(self, symbol: str, timeframe: str):
        """Fetch OHLCV data"""
        try:
            exchange = ccxt.binance()
            limit = 1000
            
            ohlcv = await exchange.fetch_ohlcv(
                f"{symbol}/USDT", 
                timeframe, 
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _fetch_market_info(self, symbol: str):
        """Fetch market information"""
        try:
            exchange = ccxt.binance()
            ticker = await exchange.fetch_ticker(f"{symbol}/USDT")
            
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'change_24h': ticker['percentage'],
                'volume_24h': ticker['quoteVolume'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low']
            }
        except Exception as e:
            logger.error(f"Error fetching market info for {symbol}: {e}")
            return {}
    
    def _calculate_risk_parameters(self, df: pd.DataFrame, current_price: float, user: User):
        """Calculate risk management parameters"""
        # Dynamic stop loss
        stop_loss = self.risk_manager.dynamic_stop_loss(df, current_price, 'long')
        
        # Position size
        position_size = self.risk_manager.calculate_position_size(
            account_balance=10000,  # Should get from user
            risk_percent=user.risk_level,
            entry_price=current_price,
            stop_loss=stop_loss,
            method='volatility_adjusted',
            volatility=df['close'].pct_change().std()
        )
        
        # Optimal leverage
        leverage = self.risk_manager.calculate_optimal_leverage(
            volatility=df['close'].pct_change().std(),
            account_balance=10000,
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
                        prediction, risk_params):
        """Generate trading signal"""
        signal_score = 0
        
        # Technical Analysis Weight (40%)
        if ta_results['market_structure']['trend'] == 'uptrend':
            signal_score += 0.4
        elif ta_results['market_structure']['trend'] == 'downtrend':
            signal_score -= 0.4
        
        # On-chain Weight (30%)
        if onchain_results['bias'] == 'bullish':
            signal_score += 0.3
        elif onchain_results['bias'] == 'bearish':
            signal_score -= 0.3
        
        # Sentiment Weight (20%)
        if sentiment_results['sentiment'] == 'bullish':
            signal_score += 0.2
        elif sentiment_results['sentiment'] == 'bearish':
            signal_score -= 0.2
        
        # Prediction Weight (10%)
        if prediction and prediction['direction'] == 'BUY':
            signal_score += 0.1
        elif prediction and prediction['direction'] == 'SELL':
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
        """Save analysis to database"""
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
    
    async def generate_chart(self, symbol: str, analysis_results):
        """Generate interactive chart with analysis"""
        try:
            # Get OHLCV data
            df = await self._fetch_ohlcv(symbol, '1h')
            
            # Create candlestick chart
            fig = go.Figure(data=[
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                )
            ])
            
            # Add technical indicators
            if 'bollinger_upper' in analysis_results['technical_analysis']:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=analysis_results['technical_analysis']['bollinger_upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='red', width=1)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=analysis_results['technical_analysis']['bollinger_lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='green', width=1)
                ))
            
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
                    layer="below"
                )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Technical Analysis',
                yaxis_title='Price (USDT)',
                template='plotly_dark',
                height=600
            )
            
            # Convert to image
            img_bytes = fig.to_image(format="png")
            
            return InputFile(io.BytesIO(img_bytes), filename=f'{symbol}_chart.png')
            
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
        self.app.router.add_get('/chart/{symbol}', self.get_chart)
    
    async def index(self, request):
        return web.FileResponse('dashboard/index.html')
    
    async def get_stats(self, request):
        session = Session()
        users = session.query(User).count()
        signals = session.query(Signal).count()
        session.close()
        
        return web.json_response({
            'total_users': users,
            'total_signals': signals,
            'uptime': time.time() - start_time
        })
    
    async def get_signals(self, request):
        session = Session()
        signals = session.query(Signal).order_by(Signal.timestamp.desc()).limit(50).all()
        session.close()
        
        return web.json_response([{
            'symbol': s.symbol,
            'signal': s.signal_type,
            'confidence': s.confidence,
            'price': s.price,
            'timestamp': s.timestamp.isoformat()
        } for s in signals])
    
    async def get_performance(self, request):
        session = Session()
        performance = session.query(Performance).all()
        session.close()
        
        return web.json_response([{
            'user_id': p.user_id,
            'total_signals': p.total_signals,
            'successful_signals': p.successful_signals,
            'avg_return': p.avg_return,
            'sharpe_ratio': p.sharpe_ratio,
            'max_drawdown': p.max_drawdown
        } for p in performance])
    
    async def get_chart(self, request):
        symbol = request.match_info['symbol']
        analysis = await self.bot.analyze_symbol(symbol)
        chart = await self.bot.generate_chart(symbol, analysis)
        
        if chart:
            return web.Response(
                body=chart.read(),
                content_type='image/png'
            )
        else:
            return web.Response(status=404)

# Telegram Bot Handlers
class TelegramBotHandlers:
    def __init__(self, bot):
        self.bot = bot
        self.dashboard = WebDashboard(bot)
    
    def setup_handlers(self, app):
        # Command handlers
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("help", self.help))
        app.add_handler(CommandHandler("analyze", self.analyze))
        app.add_handler(CommandHandler("settings", self.settings))
        app.add_handler(CommandHandler("performance", self.performance))
        app.add_handler(CommandHandler("watchlist", self.watchlist))
        app.add_handler(CommandHandler("add", self.add_to_watchlist))
        app.add_handler(CommandHandler("remove", self.remove_from_watchlist))
        
        # Callback handlers
        app.add_handler(CallbackQueryHandler(self.button_handler))
        
        # Conversation handler for settings
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('settings', self.settings)],
            states={
                'RISK_LEVEL': [CallbackQueryHandler(self.set_risk_level)],
                'MAX_LEVERAGE': [CallbackQueryHandler(self.set_max_leverage)],
            },
            fallbacks=[CommandHandler('cancel', self.cancel_settings)],
            per_message=False
        )
        app.add_handler(conv_handler)
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        session, db_user = get_user_session(user.id)
        
        welcome_text = f"""
🤖 Welcome to Advanced Crypto AI Bot, {user.first_name}!

I'm your intelligent cryptocurrency trading assistant with:
🔍 Advanced Technical Analysis
🐋 On-chain Whale Monitoring
📊 Sentiment Analysis
🧠 AI-Powered Predictions
⚖️ Smart Risk Management

Use /help to see all available commands.
        """
        
        keyboard = [
            [InlineKeyboardButton("📊 Analyze Symbol", callback_data="analyze_menu"),
             InlineKeyboardButton("⚙️ Settings", callback_data="settings_menu")],
            [InlineKeyboardButton("📈 Performance", callback_data="performance_menu"),
             InlineKeyboardButton("📋 Watchlist", callback_data="watchlist_menu")]
        ]
        
        await update.message.reply_text(
            welcome_text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
📚 Available Commands:

🔍 Analysis:
• /analyze <symbol> - Complete analysis of a symbol
• /watchlist - Show your watchlist
• /add <symbol> - Add symbol to watchlist
• /remove <symbol> - Remove from watchlist

⚙️ Settings:
• /settings - Configure your preferences
• /performance - View your performance stats

📊 Dashboard:
• Web dashboard available for detailed analysis

🤖 Features:
• Advanced Technical Analysis
• On-chain Whale Monitoring
• Sentiment Analysis
• AI Predictions
• Risk Management
• Real-time Alerts
        """
        
        await update.message.reply_text(help_text)
    
    async def analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /analyze command"""
        if not context.args:
            await update.message.reply_text("Please provide a symbol. Example: /analyze BTC")
            return
        
        symbol = context.args[0].upper()
        
        # Send typing action
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING
        )
        
        # Perform analysis
        analysis = await self.bot.analyze_symbol(symbol, update.effective_user.id)
        
        if not analysis:
            await update.message.reply_text("❌ Error analyzing symbol. Please try again.")
            return
        
        # Generate chart
        chart = await self.bot.generate_chart(symbol, analysis)
        
        # Send results
        await self.send_analysis_results(update, analysis, chart)
    
    async def send_analysis_results(self, update: Update, analysis, chart):
        """Send analysis results to user"""
        # Format message
        message = self.format_analysis_message(analysis)
        
        # Send chart if available
        if chart:
            await update.message.reply_photo(
                photo=chart,
                caption=message
            )
        else:
            await update.message.reply_text(message)
    
    def format_analysis_message(self, analysis):
        """Format analysis results into readable message"""
        symbol = analysis['symbol']
        signal = analysis['signal']
        market_data = analysis['market_data']
        risk = analysis['risk_management']
        
        message = f"""
📊 Analysis for {symbol}

💰 Current Price: ${market_data['price']:.2f}
📈 24h Change: {market_data['change_24h']:+.2f}%
📊 24h Volume: ${market_data['volume_24h']:,.0f}

🎯 Signal: {signal['signal']}
📊 Confidence: {signal['confidence']:.2f}

⚖️ Risk Management:
• Stop Loss: ${risk['stop_loss']:.2f}
• Take Profit 1: ${risk['take_profit_1']:.2f}
• Take Profit 2: ${risk['take_profit_2']:.2f}
• Position Size: {risk['position_size']:.4f}
• Leverage: {risk['leverage']:.1f}x
• Risk/Reward: {risk['risk_reward_ratio']:.2f}

🔍 Technical Analysis:
• Market Structure: {analysis['technical_analysis']['market_structure']['trend']}
• Order Blocks: {len(analysis['technical_analysis']['order_blocks']['bullish'])} Bullish / {len(analysis['technical_analysis']['order_blocks']['bearish'])} Bearish
• Liquidity Zones: {len(analysis['technical_analysis']['liquidity_zones']['high_zones'])} High / {len(analysis['technical_analysis']['liquidity_zones']['low_zones'])} Low

🐋 On-chain Analysis:
• Activity: {analysis['onchain_analysis']['activity']}
• Bias: {analysis['onchain_analysis']['bias']}
• Volume: ${analysis['onchain_analysis']['total_volume']:,.0f}

💬 Sentiment Analysis:
• Overall: {analysis['sentiment_analysis']['sentiment']}
• Score: {analysis['sentiment_analysis']['score']:.2f}

🧠 AI Prediction:
• Direction: {analysis['prediction']['direction']}
• Probability: {analysis['prediction']['probability']:.2f}

⚠️ This is not financial advice. Trade at your own risk.
        """
        
        return message
    
    async def settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command"""
        keyboard = [
            [InlineKeyboardButton("Risk Level", callback_data="set_risk_level")],
            [InlineKeyboardButton("Max Leverage", callback_data="set_max_leverage")],
            [InlineKeyboardButton("Back", callback_data="main_menu")]
        ]
        
        await update.message.reply_text(
            "⚙️ Settings Menu:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        return 'RISK_LEVEL'
    
    async def set_risk_level(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set user risk level"""
        query = update.callback_query
        await query.answer()
        
        keyboard = [
            [InlineKeyboardButton("Low (1%)", callback_data="risk_low")],
            [InlineKeyboardButton("Medium (2%)", callback_data="risk_medium")],
            [InlineKeyboardButton("High (3%)", callback_data="risk_high")],
            [InlineKeyboardButton("Back", callback_data="settings_menu")]
        ]
        
        await query.edit_message_text(
            "Select your risk level:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        return 'RISK_LEVEL'
    
    async def set_max_leverage(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set user max leverage"""
        query = update.callback_query
        await query.answer()
        
        keyboard = [
            [InlineKeyboardButton("5x", callback_data="leverage_5")],
            [InlineKeyboardButton("10x", callback_data="leverage_10")],
            [InlineKeyboardButton("20x", callback_data="leverage_20")],
            [InlineKeyboardButton("Back", callback_data="settings_menu")]
        ]
        
        await query.edit_message_text(
            "Select maximum leverage:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        return 'MAX_LEVERAGE'
    
    async def cancel_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Cancel settings conversation"""
        await update.message.reply_text("Settings cancelled.")
        return ConversationHandler.END
    
    async def performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command"""
        session = Session()
        user_id = update.effective_user.id
        
        performance = session.query(Performance).filter_by(user_id=user_id).first()
        signals = session.query(Signal).filter_by(user_id=user_id).all()
        
        if not performance:
            await update.message.reply_text("No performance data available yet.")
            return
        
        # Calculate metrics
        total_signals = performance.total_signals
        success_rate = (performance.successful_signals / total_signals * 100) if total_signals > 0 else 0
        
        message = f"""
📊 Your Performance:

📈 Total Signals: {total_signals}
✅ Successful Signals: {performance.successful_signals}
📊 Success Rate: {success_rate:.1f}%
💰 Average Return: {performance.avg_return:+.2f}%
📊 Sharpe Ratio: {performance.sharpe_ratio:.2f}
📉 Max Drawdown: {performance.max_drawdown:.2f}%

Recent Signals:
        """
        
        # Add recent signals
        for signal in signals[-5:]:
            message += f"\n• {signal.symbol}: {signal.signal_type} at ${signal.price:.2f}"
        
        await update.message.reply_text(message)
        session.close()
    
    async def watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /watchlist command"""
        session = Session()
        user_id = update.effective_user.id
        
        watchlist = session.query(Watchlist).filter_by(user_id=user_id).all()
        
        if not watchlist:
            await update.message.reply_text("Your watchlist is empty.")
            return
        
        message = "📋 Your Watchlist:\n"
        for item in watchlist:
            message += f"• {item.symbol}\n"
        
        await update.message.reply_text(message)
        session.close()
    
    async def add_to_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Add symbol to watchlist"""
        if not context.args:
            await update.message.reply_text("Please provide a symbol. Example: /add BTC")
            return
        
        symbol = context.args[0].upper()
        user_id = update.effective_user.id
        
        session = Session()
        
        # Check if already in watchlist
        existing = session.query(Watchlist).filter_by(
            user_id=user_id, symbol=symbol
        ).first()
        
        if existing:
            await update.message.reply_text(f"{symbol} is already in your watchlist.")
            return
        
        # Add to watchlist
        watchlist_item = Watchlist(user_id=user_id, symbol=symbol)
        session.add(watchlist_item)
        session.commit()
        
        await update.message.reply_text(f"✅ {symbol} added to your watchlist.")
        session.close()
    
    async def remove_from_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Remove symbol from watchlist"""
        if not context.args:
            await update.message.reply_text("Please provide a symbol. Example: /remove BTC")
            return
        
        symbol = context.args[0].upper()
        user_id = update.effective_user.id
        
        session = Session()
        
        # Remove from watchlist
        item = session.query(Watchlist).filter_by(
            user_id=user_id, symbol=symbol
        ).first()
        
        if not item:
            await update.message.reply_text(f"{symbol} is not in your watchlist.")
            return
        
        session.delete(item)
        session.commit()
        
        await update.message.reply_text(f"✅ {symbol} removed from your watchlist.")
        session.close()
    
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard buttons"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data == "analyze_menu":
            keyboard = [
                [InlineKeyboardButton("BTC", callback_data="analyze_BTC"),
                 InlineKeyboardButton("ETH", callback_data="analyze_ETH")],
                [InlineKeyboardButton("BNB", callback_data="analyze_BNB"),
                 InlineKeyboardButton("ADA", callback_data="analyze_ADA")],
                [InlineKeyboardButton("Custom", callback_data="analyze_custom"),
                 InlineKeyboardButton("Back", callback_data="main_menu")]
            ]
            
            await query.edit_message_text(
                "📊 Select symbol to analyze:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        
        elif data.startswith("analyze_"):
            symbol = data.split("_")[1]
            if symbol == "custom":
                await query.edit_message_text("Please use /analyze <symbol> command.")
            else:
                # Send typing action
                await context.bot.send_chat_action(
                    chat_id=update.effective_chat.id,
                    action=ChatAction.TYPING
                )
                
                # Perform analysis
                analysis = await self.bot.analyze_symbol(symbol, update.effective_user.id)
                
                if analysis:
                    chart = await self.bot.generate_chart(symbol, analysis)
                    await self.send_analysis_results(update, analysis, chart)
                else:
                    await query.edit_message_text("❌ Error analyzing symbol.")
        
        elif data == "settings_menu":
            await self.settings(update, context)
        
        elif data == "performance_menu":
            await self.performance(update, context)
        
        elif data == "watchlist_menu":
            await self.watchlist(update, context)
        
        elif data == "main_menu":
            keyboard = [
                [InlineKeyboardButton("📊 Analyze Symbol", callback_data="analyze_menu"),
                 InlineKeyboardButton("⚙️ Settings", callback_data="settings_menu")],
                [InlineKeyboardButton("📈 Performance", callback_data="performance_menu"),
                 InlineKeyboardButton("📋 Watchlist", callback_data="watchlist_menu")]
            ]
            
            await query.edit_message_text(
                "🤖 Main Menu:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        
        elif data.startswith("risk_"):
            risk_level = data.split("_")[1]
            risk_percent = {"low": 1, "medium": 2, "high": 3}[risk_level]
            
            # Update user settings
            session, user = get_user_session(update.effective_user.id)
            user.risk_level = risk_level
            session.commit()
            
            await query.edit_message_text(
                f"✅ Risk level set to {risk_level} ({risk_percent}%)"
            )
        
        elif data.startswith("leverage_"):
            leverage = int(data.split("_")[1])
            
            # Update user settings
            session, user = get_user_session(update.effective_user.id)
            user.max_leverage = leverage
            session.commit()
            
            await query.edit_message_text(
                f"✅ Max leverage set to {leverage}x"
            )

# Main Application
class CryptoAIApp:
    def __init__(self):
        self.bot = AdvancedCryptoBot()
        self.handlers = TelegramBotHandlers(self.bot)
        self.web_dashboard = WebDashboard(self.bot)
        
    async def start_web_server(self):
        """Start web dashboard"""
        if S.ENABLE_WEB_DASHBOARD:
            runner = web.AppRunner(self.web_dashboard.app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', S.WEB_PORT)
            await site.start()
            logger.info(f"Web dashboard started on port {S.WEB_PORT}")
    
    async def periodic_tasks(self):
        """Run periodic background tasks"""
        while True:
            try:
                # Monitor watchlist symbols
                await self.monitor_watchlist()
                
                # Update prediction models
                await self.update_models()
                
                # Clean up old data
                await self.cleanup_old_data()
                
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Error in periodic tasks: {e}")
                await asyncio.sleep(60)
    
    async def monitor_watchlist(self):
        """Monitor symbols in user watchlists"""
        session = Session()
        watchlist_items = session.query(Watchlist).all()
        
        for item in watchlist_items:
            try:
                analysis = await self.bot.analyze_symbol(item.symbol, item.user_id)
                
                # Check if strong signal
                if analysis and analysis['signal']['confidence'] > 0.8:
                    # Send alert
                    await self.send_signal_alert(item.user_id, item.symbol, analysis)
            except Exception as e:
                logger.error(f"Error monitoring {item.symbol}: {e}")
        
        session.close()
    
    async def send_signal_alert(self, user_id, symbol, analysis):
        """Send signal alert to user"""
        try:
            message = f"""
🚨 Strong Signal Alert for {symbol}!

Signal: {analysis['signal']['signal']}
Confidence: {analysis['signal']['confidence']:.2f}
Price: ${analysis['market_data']['price']:.2f}

Use /analyze {symbol} for full analysis.
            """
            
            # Send via Telegram
            app = context.application
            await app.bot.send_message(
                chat_id=user_id,
                text=message
            )
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    async def update_models(self):
        """Update prediction models"""
        # Get top symbols
        symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'DOT', 'SOL', 'MATIC', 'AVAX']
        
        for symbol in symbols:
            try:
                # Fetch data
                market_data = await self.bot._fetch_market_data(symbol)
                df = market_data['ohlcv']['1h']
                
                # Train model
                await self.bot.prediction_engine.train_model(df, symbol, '1h')
                
                logger.info(f"Updated model for {symbol}")
            except Exception as e:
                logger.error(f"Error updating model for {symbol}: {e}")
    
    async def cleanup_old_data(self):
        """Clean up old data from database"""
        session = Session()
        
        # Delete signals older than 30 days
        cutoff_date = datetime.datetime.utcnow() - timedelta(days=30)
        session.query(Signal).filter(Signal.timestamp < cutoff_date).delete()
        
        session.commit()
        session.close()
    
    async def run(self):
        """Run the application"""
        # Start web server
        await self.start_web_server()
        
        # Start periodic tasks
        asyncio.create_task(self.periodic_tasks())
        
        # Setup Telegram bot
        app = ApplicationBuilder().token(S.TELEGRAM_BOT_TOKEN).build()
        self.handlers.setup_handlers(app)
        
        # Start bot
        logger.info("Starting Telegram bot...")
        await app.run_polling()

# Global variables
start_time = time.time()

# Main entry point
async def main():
    app = CryptoAIApp()
    await app.run()

if __name__ == "__main__":
    asyncio.run(main())