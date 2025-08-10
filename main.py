import os
import logging
import asyncio
import json
import numpy as np
import pandas as pd
import yfinance as yf
import ccxt
import sqlite3
import aiohttp
import requests
import time
from datetime import datetime, timedelta
import pytz
import io
import base64
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.request import HTTPXRequest
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from asyncio_throttle import Throttler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
from collections import Counter
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import talib
import pandas_ta as ta

# بارگذاری متغیرهای محیطی
load_dotenv()

# تنظیمات لاگینگ پیشرفته
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# تنظیمات پروکسی برای کاربران ایرانی
PROXY_SETTINGS = {
    'proxy': {
        'url': os.getenv('PROXY_URL'),
        'username': os.getenv('PROXY_USERNAME'),
        'password': os.getenv('PROXY_PASSWORD')
    }
} if os.getenv('PROXY_URL') else {}

# وارد کردن کتابخانه‌ها به صورت شرطی
LIBRARIES = {
    'tensorflow': False,
    'talib': False,
    'pandas_ta': False,
    'pywt': False,
    'lightgbm': False,
    'xgboost': False,
    'prophet': False,
    'statsmodels': False,
    'seaborn': False,
    'psycopg2': False,
    'plotly': False
}

for lib in LIBRARIES:
    try:
        globals()[lib] = __import__(lib)
        LIBRARIES[lib] = True
        logger.info(f"{lib} loaded successfully")
    except ImportError as e:
        logger.warning(f"{lib} not available: {e}")

class AdvancedTradingBot:
    def __init__(self):
        """مقداردهی اولیه ربات"""
        logger.info("Initializing AdvancedTradingBot...")
        
        # تنظیمات پروکسی
        self.session = requests.Session()
        if PROXY_SETTINGS:
            self.session.proxies = {
                'http': PROXY_SETTINGS['proxy']['url'],
                'https': PROXY_SETTINGS['proxy']['url']
            }
            if PROXY_SETTINGS['proxy'].get('username'):
                self.session.auth = (PROXY_SETTINGS['proxy']['username'], PROXY_SETTINGS['proxy']['password'])
        
        # تنظیمات rate limiting
        self.throttler = Throttler(rate_limit=5, period=1.0)
        
        # پایگاه داده
        self.setup_database()
        
        # مدل‌های تحلیل
        self.models = self.initialize_models()
        
        # تنظیمات صرافی‌ها
        self.exchanges = self.setup_exchanges()
        
        # تنظیمات APIها
        self.api_keys = {
            'coingecko': os.getenv('COINGECKO_API_KEY'),
            'news': os.getenv('NEWS_API_KEY'),
            'cryptopanic': os.getenv('CRYPTOPANIC_API_KEY'),
            'coinmarketcap': os.getenv('COINMARKETCAP_API_KEY'),
            'cryptocompare': os.getenv('CRYPTOCOMPARE_API_KEY'),
            'binance': os.getenv('BINANCE_API_KEY'),
            'coinalyze': os.getenv('COINANALYZE_API_KEY')
        }
        
        # تست دسترسی به اینترنت
        self.internet_available = self.test_internet_connection()
        logger.info(f"Internet available: {self.internet_available}")
        
        # اگر اینترنت در دسترس نیست، از حالت آفلاین استفاده کن
        self.offline_mode = not self.internet_available
        if self.offline_mode:
            logger.warning("Internet connection not available. Using offline mode.")
        
        # راه‌اندازی سیستم تحلیل متن پیشرفته
        self.setup_text_analysis()
        
        # راه‌اندازی سیستم تحلیل تکنیکال پیشرفته
        self.setup_advanced_analysis()
        
        logger.info("AdvancedTradingBot initialized successfully")
    
    def setup_advanced_analysis(self):
        """راه‌اندازی سیستم تحلیل تکنیکال پیشرفته"""
        # تنظیمات روش‌های تحلیلی
        self.analysis_methods = {
            'wyckoff': self.wyckoff_analysis,
            'volume_profile': self.volume_profile_analysis,
            'market_profile': self.market_profile_analysis,
            'fibonacci': self.fibonacci_analysis,
            'harmonic_patterns': self.harmonic_patterns_analysis,
            'ichimoku': self.ichimoku_analysis,
            'support_resistance': self.support_resistance_analysis,
            'trend_lines': self.trend_lines_analysis,
            'order_flow': self.order_flow_analysis,
            'vwap': self.vwap_analysis,
            'pivot_points': self.pivot_points_analysis,
            'candlestick_patterns': self.advanced_candlestick_patterns,
            'elliott_wave': self.advanced_elliott_wave,
            'market_structure': self.market_structure_analysis
        }
        
        # الگوهای هارمونیک
        self.harmonic_patterns = {
            'gartley': {'XA': 0.618, 'AB': 0.618, 'BC': 0.382, 'CD': 1.27},
            'butterfly': {'XA': 0.786, 'AB': 0.786, 'BC': 0.382, 'CD': 1.618},
            'bat': {'XA': 0.382, 'AB': 0.382, 'BC': 0.886, 'CD': 2.618},
            'crab': {'XA': 0.886, 'AB': 0.382, 'BC': 0.618, 'CD': 3.14}
        }
        
        # الگوهای شمعی پیشرفته
        self.advanced_candlesticks = {
            'three_white_soldiers': 'سه سرباز سفید - سیگنال خرید قوی',
            'three_black_crows': 'سه کلاغ سیاه - سیگنال فروش قوی',
            'morning_star': 'ستاره صبحگاهی - سیگنال خرید',
            'evening_star': 'ستاره عصرگاهی - سیگنال فروش',
            'abandoned_baby': 'نوزاد رها شده - سیگنال معکوس قوی',
            'kicking': 'ضربه - سیگنال معکوس قوی',
            'matching_low': 'کف هم‌تراز - سیگنال خرید',
            'unique_three_river': 'سه رودخانه منحصر به فرد - سیگنال خرید',
            'concealing_baby_swallow': 'پنهان کردن جوجه قوق - سیگنال خرید'
        }
    
    def setup_database(self):
        """راه‌اندازی پایگاه داده"""
        try:
            if LIBRARIES['psycopg2']:
                import psycopg2
                self.conn = psycopg2.connect(os.getenv("DATABASE_URL"))
                logger.info("PostgreSQL connection established")
            else:
                self.conn = sqlite3.connect('trading_bot.db', check_same_thread=False)
                logger.warning("Using SQLite as fallback")
            
            self.create_tables()
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            # فallback به SQLite
            self.conn = sqlite3.connect('trading_bot.db', check_same_thread=False)
            self.create_tables()
    
    def setup_exchanges(self):
        """راه‌اندازی صرافی‌ها"""
        exchanges = {
            'binance': ccxt.binance(PROXY_SETTINGS),
            'coinbase': ccxt.coinbase(PROXY_SETTINGS),
            'kucoin': ccxt.kucoin(PROXY_SETTINGS),
            'bybit': ccxt.bybit(PROXY_SETTINGS),
            'gate': ccxt.gateio(PROXY_SETTINGS),
            'huobi': ccxt.huobi(PROXY_SETTINGS),
            'okx': ccxt.okx(PROXY_SETTINGS)
        }
        return exchanges
    
    def test_internet_connection(self):
        """تست دسترسی به اینترنت"""
        try:
            response = self.session.get('https://www.google.com', timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def setup_text_analysis(self):
        """راه‌اندازی سیستم تحلیل متن پیشرفته"""
        # کلمات کلیدی برای تحلیل احساسات
        self.positive_keywords = [
            'صعود', 'رشد', 'افزایش', 'موفق', 'بالا', 'خرید', 'bullish', 'growth', 'increase', 
            'success', 'high', 'buy', 'profit', 'gain', 'positive', 'optimistic', 'bull',
            'rally', 'surge', 'boom', 'breakthrough', 'upgrade', 'adoption', 'partnership',
            'الگو', 'سیگنال', 'تحلیل', 'پیش‌بینی', 'فرصت', 'پتانسیل', 'بهبود', 'بهینه'
        ]
        
        self.negative_keywords = [
            'نزول', 'کاهش', 'افت', 'ضرر', 'پایین', 'فروش', 'bearish', 'decrease', 'drop', 
            'loss', 'low', 'sell', 'negative', 'pessimistic', 'bear', 'crash', 'dump', 
            'decline', 'fall', 'slump', 'recession', 'risk', 'warning', 'fraud', 'hack',
            'ریسک', 'خطر', 'مشکل', 'کاهش', 'ضرر', 'فروش', 'فشار', 'نزولی'
        ]
        
        # الگوهای تحلیل تکنیکال
        self.technical_patterns = {
            'bullish_engulfing': ['الگوی پوشاننده صعودی', 'سیگنال خرید قوی'],
            'bearish_engulfing': ['الگوی پوشاننده نزولی', 'سیگنال فروش قوی'],
            'head_and_shoulders': ['الگوی سر و شانه', 'سیگنال فروش'],
            'inverse_head_and_shoulders': ['الگوی سر و شانه معکوس', 'سیگنال خرید'],
            'double_top': ['الگوی دو قله', 'سیگنال فروش'],
            'double_bottom': ['الگوی دو کف', 'سیگنال خرید'],
            'ascending_triangle': ['مثلث صعودی', 'ادامه روند صعودی'],
            'descending_triangle': ['مثلث نزولی', 'ادامه روند نزولی'],
            'cup_and_handle': ['الگوی فنجان و دسته', 'سیگنال خرید'],
            'rising_wedge': ['گوه صعودی', 'سیگنال فروش'],
            'falling_wedge': ['گوه نزولی', 'سیگنال خرید']
        ]
    
    def create_tables(self):
        """ایجاد جداول پایگاه داده"""
        cursor = self.conn.cursor()
        
        # جدول کاربران
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            language TEXT DEFAULT 'fa',
            preferences TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # جدول تحلیل‌ها
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symbol TEXT,
            analysis_type TEXT,
            result TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # جدول سیگنال‌ها
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symbol TEXT,
            signal_type TEXT,
            signal_value TEXT,
            confidence REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # جدول واچ‌لیست
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symbol TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # جدول عملکرد
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symbol TEXT,
            strategy TEXT,
            entry_price REAL,
            exit_price REAL,
            profit_loss REAL,
            duration INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # جدول داده‌های بازار
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            source TEXT,
            price REAL,
            volume_24h REAL,
            market_cap REAL,
            price_change_24h REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # جدول اخبار
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            source TEXT,
            url TEXT,
            published_at TIMESTAMP,
            sentiment_score REAL,
            symbols TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # جدول تحلیل‌های هوش مصنوعی
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            analysis_type TEXT,
            result TEXT,
            confidence REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # جدول داده‌های اقتصادی
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS economic_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            event_name TEXT,
            event_date TIMESTAMP,
            actual_value REAL,
            forecast_value REAL,
            previous_value REAL,
            impact TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # جدول داده‌های جلسه معاملاتی
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trading_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            session_type TEXT,
            session_start TIMESTAMP,
            session_end TIMESTAMP,
            high_price REAL,
            low_price REAL,
            volume REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        self.conn.commit()
        logger.info("Database tables created successfully")
    
    def initialize_models(self):
        """مقداردهی اولیه مدل‌های تحلیل"""
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'svm': SVR(kernel='rbf', C=50, gamma=0.1),
            'knn': KNeighborsRegressor(n_neighbors=5),
            'linear_regression': LinearRegression(),
        }
        
        # اضافه کردن مدل‌های موجود
        if LIBRARIES['xgboost']:
            models['xgboost'] = xgboost.XGBRegressor(n_estimators=100, random_state=42)
        
        if LIBRARIES['lightgbm']:
            models['lightgbm'] = lightgbm.LGBMRegressor(n_estimators=100, random_state=42)
        
        if LIBRARIES['prophet']:
            models['prophet'] = prophet.Prophet()
        
        # اضافه کردن مدل‌های عمیق در صورت وجود
        if LIBRARIES['tensorflow']:
            models['lstm'] = self.build_lstm_model()
            models['gru'] = self.build_gru_model()
        
        logger.info("Machine learning models initialized")
        return models
    
    def build_lstm_model(self):
        """ساخت مدل LSTM"""
        if not LIBRARIES['tensorflow']:
            return None
            
        model = tensorflow.keras.Sequential([
            tensorflow.keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 5)),
            tensorflow.keras.layers.Dropout(0.2),
            tensorflow.keras.layers.LSTM(50),
            tensorflow.keras.layers.Dropout(0.2),
            tensorflow.keras.layers.Dense(25),
            tensorflow.keras.layers.Dense(1)
        ])
        model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model
    
    def build_gru_model(self):
        """ساخت مدل GRU"""
        if not LIBRARIES['tensorflow']:
            return None
            
        model = tensorflow.keras.Sequential([
            tensorflow.keras.layers.GRU(50, return_sequences=True, input_shape=(60, 5)),
            tensorflow.keras.layers.Dropout(0.2),
            tensorflow.keras.layers.GRU(50),
            tensorflow.keras.layers.Dropout(0.2),
            tensorflow.keras.layers.Dense(25),
            tensorflow.keras.layers.Dense(1)
        ])
        model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model
    
    async def fetch_data_from_multiple_sources(self, symbol):
        """دریافت داده‌ها از چندین منبع با مدیریت خطا"""
        data = {}
        
        # اگر در حالت آفلاین هستیم، داده‌های ساختگی برگردان
        if self.offline_mode:
            return self.generate_offline_data(symbol)
        
        # تلاش برای دریافت داده از CoinGecko
        try:
            data['coingecko'] = await self.fetch_coingecko_data(symbol)
        except Exception as e:
            logger.error(f"Error fetching from CoinGecko: {e}")
            data['coingecko'] = {}
        
        # تلاش برای دریافت داده از CoinMarketCap
        try:
            data['coinmarketcap'] = await self.fetch_coinmarketcap_data(symbol)
        except Exception as e:
            logger.error(f"Error fetching from CoinMarketCap: {e}")
            data['coinmarketcap'] = {}
        
        # تلاش برای دریافت داده از CryptoCompare (جایگزین برای ایران)
        try:
            data['cryptocompare'] = await self.fetch_cryptocompare_data(symbol)
        except Exception as e:
            logger.error(f"Error fetching from CryptoCompare: {e}")
            data['cryptocompare'] = {}
        
        # تلاش برای دریافت داده از CoinLyze (جایگزین برای ایران)
        try:
            data['coinalyze'] = await self.fetch_coinalyze_data(symbol)
        except Exception as e:
            logger.error(f"Error fetching from CoinLyze: {e}")
            data['coinalyze'] = {}
        
        # تلاش برای دریافت داده از صرافی‌ها
        try:
            data['exchanges'] = await self.fetch_exchange_data(symbol)
        except Exception as e:
            logger.error(f"Error fetching from exchanges: {e}")
            data['exchanges'] = {}
        
        # اگر هیچ داده‌ای دریافت نشد، از داده‌های ساختگی استفاده کن
        if not any(data.values()):
            logger.warning(f"No data received for {symbol}. Using offline data.")
            return self.generate_offline_data(symbol)
        
        return data
    
    def generate_offline_data(self, symbol):
        """تولید داده‌های آفلاین برای تست"""
        logger.info(f"Generating offline data for {symbol}")
        
        # قیمت‌های ساختگی بر اساس نماد
        base_prices = {
            'BTC': 43000,
            'ETH': 2200,
            'BNB': 300,
            'SOL': 100,
            'XRP': 0.6,
            'ADA': 0.5,
            'DOT': 7,
            'DOGE': 0.08,
            'AVAX': 35,
            'MATIC': 0.8
        }
        
        base_price = base_prices.get(symbol, 100)
        
        # ایجاد تغییرات تصادفی
        change = np.random.uniform(-0.05, 0.05)
        price = base_price * (1 + change)
        
        return {
            'coingecko': {
                'usd': price,
                'usd_market_cap': price * 20000000,
                'usd_24h_vol': price * 1000000,
                'usd_24h_change': change * 100
            },
            'exchanges': {
                'binance': {
                    'price': price,
                    'volume': price * 1000000,
                    'change': change * 100
                }
            }
        }
    
    async def fetch_coingecko_data(self, symbol):
        """دریافت داده‌ها از CoinGecko"""
        async with aiohttp.ClientSession() as session:
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': symbol.lower(),
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true',
                'include_last_updated_at': 'true'
            }
            
            if self.api_keys['coingecko']:
                params['x_cg_demo_api_key'] = self.api_keys['coingecko']
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get(symbol.lower(), {})
                return {}
    
    async def fetch_coinmarketcap_data(self, symbol):
        """دریافت داده‌ها از CoinMarketCap"""
        if not self.api_keys['coinmarketcap']:
            return {}
        
        async with aiohttp.ClientSession() as session:
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
            headers = {
                'X-CMC_PRO_API_KEY': self.api_keys['coinmarketcap'],
                'Accept': 'application/json'
            }
            params = {'start': '1', 'limit': '100', 'convert': 'USD'}
            
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    for crypto in data['data']:
                        if crypto['symbol'].lower() == symbol.lower():
                            return {
                                'price': crypto['quote']['USD']['price'],
                                'volume_24h': crypto['quote']['USD']['volume_24h'],
                                'market_cap': crypto['quote']['USD']['market_cap'],
                                'percent_change_24h': crypto['quote']['USD']['percent_change_24h']
                            }
                return {}
    
    async def fetch_cryptocompare_data(self, symbol):
        """دریافت داده‌ها از CryptoCompare (جایگزین برای ایران)"""
        async with aiohttp.ClientSession() as session:
            url = f"https://min-api.cryptocompare.com/data/price"
            params = {
                'fsym': symbol,
                'tsyms': 'USD',
                'api_key': self.api_keys['cryptocompare']
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'price': data.get('USD', 0),
                        'volume_24h': data.get('USD', 0) * 1000000,  # تخمین حجم
                        'market_cap': data.get('USD', 0) * 20000000,  # تخمین مارکت کپ
                        'percent_change_24h': np.random.uniform(-5, 5)  # تغییرات تصادفی
                    }
                return {}
    
    async def fetch_coinalyze_data(self, symbol):
        """دریافت داده‌ها از CoinLyze (جایگزین برای ایران)"""
        if not self.api_keys['coinalyze']:
            return {}
        
        async with aiohttp.ClientSession() as session:
            url = f"https://api.coinalyze.net/v1/ticker"
            params = {
                'symbol': f'{symbol}USDT',
                'api_key': self.api_keys['coinalyze']
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'price': data.get('price', 0),
                        'volume_24h': data.get('volume', 0),
                        'market_cap': data.get('market_cap', 0),
                        'percent_change_24h': data.get('change', 0)
                    }
                return {}
    
    async def fetch_exchange_data(self, symbol):
        """دریافت داده‌ها از صرافی‌ها با مدیریت خطا"""
        exchange_data = {}
        
        # اگر در حالت آفلاین هستیم، داده‌های ساختگی برگردان
        if self.offline_mode:
            return self.generate_offline_exchange_data(symbol)
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                # تبدیل نماد به فرمت مناسب برای صرافی
                exchange_symbol = self.convert_symbol_for_exchange(symbol, exchange_name)
                
                # دریافت تیکر
                ticker = exchange.fetch_ticker(exchange_symbol)
                
                exchange_data[exchange_name] = {
                    'price': ticker['last'],
                    'volume': ticker['quoteVolume'],
                    'high': ticker['high'],
                    'low': ticker['low'],
                    'change': ticker['change']
                }
                
                # رعایت محدودیت درخواست
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching from {exchange_name}: {e}")
                # استفاده از داده‌های پیش‌فرض
                exchange_data[exchange_name] = {
                    'price': 0,
                    'volume': 0,
                    'high': 0,
                    'low': 0,
                    'change': 0
                }
        
        return exchange_data
    
    def generate_offline_exchange_data(self, symbol):
        """تولید داده‌های صرافی آفلاین"""
        base_prices = {
            'BTC': 43000,
            'ETH': 2200,
            'BNB': 300,
            'SOL': 100,
            'XRP': 0.6,
            'ADA': 0.5,
            'DOT': 7,
            'DOGE': 0.08,
            'AVAX': 35,
            'MATIC': 0.8
        }
        
        base_price = base_prices.get(symbol, 100)
        change = np.random.uniform(-0.05, 0.05)
        price = base_price * (1 + change)
        
        return {
            'binance': {
                'price': price,
                'volume': price * 1000000,
                'high': price * 1.02,
                'low': price * 0.98,
                'change': change * 100
            }
        }
    
    def convert_symbol_for_exchange(self, symbol, exchange_name):
        """تبدیل نماد به فرمت مناسب برای صرافی"""
        # تبدیل نمادهای رایج
        symbol_map = {
            'BTC': 'BTC/USDT',
            'ETH': 'ETH/USDT',
            'BNB': 'BNB/USDT',
            'SOL': 'SOL/USDT',
            'XRP': 'XRP/USDT',
            'ADA': 'ADA/USDT',
            'DOT': 'DOT/USDT',
            'DOGE': 'DOGE/USDT',
            'AVAX': 'AVAX/USDT',
            'MATIC': 'MATIC/USDT'
        }
        
        # اگر نماد در نقشه وجود داشت، از آن استفاده کن
        if symbol in symbol_map:
            return symbol_map[symbol]
        
        # در غیر این صورت، فرمت پیش‌فرض را برگردان
        return f"{symbol}/USDT"
    
    async def fetch_news_from_multiple_sources(self, symbol=None):
        """دریافت اخبار از چندین منبع"""
        news = []
        
        # دریافت اخبار از CryptoPanic
        try:
            news.extend(await self.fetch_cryptopanic_news(symbol))
        except Exception as e:
            logger.error(f"Error fetching from CryptoPanic: {e}")
        
        # دریافت اخبار از CryptoCompare (جایگزین برای ایران)
        try:
            news.extend(await self.fetch_cryptocompare_news(symbol))
        except Exception as e:
            logger.error(f"Error fetching from CryptoCompare: {e}")
        
        # دریافت اخبار از CoinGecko
        try:
            news.extend(await self.fetch_coingecko_news(symbol))
        except Exception as e:
            logger.error(f"Error fetching from CoinGecko: {e}")
        
        return news
    
    async def fetch_cryptopanic_news(self, symbol=None):
        """دریافت اخبار از CryptoPanic"""
        if not self.api_keys['cryptopanic']:
            return []
        
        async with aiohttp.ClientSession() as session:
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                'auth_token': self.api_keys['cryptopanic'],
                'kind': 'news',
                'filter': 'hot'
            }
            
            if symbol:
                params['currencies'] = symbol.lower()
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        {
                            'title': item['title'],
                            'content': item.get('metadata', {}).get('description', ''),
                            'source': 'CryptoPanic',
                            'url': item['url'],
                            'published_at': datetime.fromtimestamp(item['published_at']),
                            'symbols': item.get('currencies', [])
                        }
                        for item in data['results']
                    ]
        return []
    
    async def fetch_cryptocompare_news(self, symbol=None):
        """دریافت اخبار از CryptoCompare (جایگزین برای ایران)"""
        async with aiohttp.ClientSession() as session:
            url = "https://min-api.cryptocompare.com/data/v2/news/"
            params = {
                'lang': 'EN',
                'api_key': self.api_keys['cryptocompare']
            }
            
            if symbol:
                params['categories'] = symbol.lower()
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        {
                            'title': item['title'],
                            'content': item.get('body', ''),
                            'source': 'CryptoCompare',
                            'url': item['url'],
                            'published_at': datetime.fromtimestamp(item['published_on']),
                            'symbols': item.get('categories', [])
                        }
                        for item in data['Data']
                    ]
        return []
    
    async def fetch_coingecko_news(self, symbol=None):
        """دریافت اخبار از CoinGecko"""
        async with aiohttp.ClientSession() as session:
            url = "https://api.coingecko.com/api/v3/news"
            params = {
                'per_page': 10,
                'page': 1
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        {
                            'title': item['title'],
                            'content': item['description'],
                            'source': 'CoinGecko',
                            'url': item['url'],
                            'published_at': datetime.strptime(item['publishedAt'], '%Y-%m-%dT%H:%M:%S%z'),
                            'symbols': item.get('tags', [])
                        }
                        for item in data['data']
                    ]
        return []
    
    async def fetch_economic_news(self):
        """دریافت اخبار اقتصادی شامل NFP, CPI, FOMC"""
        economic_news = []
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'apiKey': self.api_keys['news'],
                    'q': 'NFP OR CPI OR FOMC OR Federal Reserve OR interest rates OR inflation OR jobs report',
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 10
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        economic_news = [
                            {
                                'title': item['title'],
                                'content': item['description'],
                                'source': item['source']['name'],
                                'url': item['url'],
                                'published_at': datetime.strptime(item['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                            }
                            for item in data['articles']
                        ]
        except Exception as e:
            logger.error(f"Error fetching economic news: {e}")
        
        return economic_news
    
    async def advanced_sentiment_analysis(self, news_items):
        """تحلیل احساسات پیشرفته با هوش مصنوعی داخلی"""
        if not news_items:
            return {
                'average_sentiment': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'topics': [],
                'trends': []
            }
        
        sentiments = []
        topics = []
        all_text = ""
        
        for news in news_items:
            text = f"{news['title']} {news['content']}"
            all_text += text + " "
            
            # تحلیل احساسات پیشرفته
            sentiment_score = self.analyze_text_sentiment(text)
            sentiments.append(sentiment_score)
            
            # استخراج موضوعات
            news_topics = self.extract_topics(text)
            topics.extend(news_topics)
        
        # تحلیل روندها
        trends = self.analyze_trends(all_text)
        
        # تحلیل آماری
        if sentiments:
            return {
                'average_sentiment': np.mean(sentiments),
                'positive_count': len([s for s in sentiments if s > 0.2]),
                'negative_count': len([s for s in sentiments if s < -0.2]),
                'neutral_count': len([s for s in sentiments if -0.2 <= s <= 0.2]),
                'topics': self.get_top_topics(topics),
                'trends': trends
            }
        return {'average_sentiment': 0, 'positive_count': 0, 'negative_count': 0, 'neutral_count': 0, 'topics': [], 'trends': []}
    
    def analyze_text_sentiment(self, text):
        """تحلیل احساسات متن با روش‌های پیشرفته"""
        text_lower = text.lower()
        
        # شمارش کلمات مثبت و منفی
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
        
        # تحلیل وزنی بر اساس موقعیت کلمات
        words = text_lower.split()
        positive_weight = 0
        negative_weight = 0
        
        for i, word in enumerate(words):
            if word in self.positive_keywords:
                # کلمات در ابتدای متن وزن بیشتری دارند
                position_weight = 1.5 if i < len(words) * 0.3 else 1.0
                positive_weight += position_weight
            elif word in self.negative_keywords:
                position_weight = 1.5 if i < len(words) * 0.3 else 1.0
                negative_weight += position_weight
        
        # تحلیل بر اساس شدت احساسات
        intensity_indicators = ['بسیار', 'خیلی', 'کاملا', 'قطعا', 'حتما', 'شدید', 'فوق', 'hyper', 'very', 'extremely']
        intensity_multiplier = 1.0
        
        for indicator in intensity_indicators:
            if indicator in text_lower:
                intensity_multiplier = 1.5
                break
        
        # محاسبه امتیاز نهایی
        total_sentiment_words = positive_weight + negative_weight
        if total_sentiment_words > 0:
            sentiment_score = ((positive_weight - negative_weight) / total_sentiment_words) * intensity_multiplier
        else:
            sentiment_score = 0
        
        # نرمال‌سازی بین -1 و 1
        return max(-1, min(1, sentiment_score))
    
    def extract_topics(self, text):
        """استخراج موضوعات از متن"""
        # کلمات کلیدی موضوعات مختلف
        topic_keywords = {
            'تکنولوژی': ['بلاکچین', 'blockchain', 'فناوری', 'technology', 'نوآوری', 'innovation'],
            'تنظیم': ['قانون', 'regulation', 'مقررات', 'حکومت', 'government', 'سیاست', 'policy'],
            'بازار': ['بازار', 'market', 'معامله', 'trading', 'قیمت', 'price', 'عرضه', 'demand'],
            'امنیت': ['امنیت', 'security', 'هک', 'hack', 'حفاظت', 'protection', 'ریسک', 'risk'],
            'پذیرش': ['پذیرش', 'adoption', 'استفاده', 'usage', 'کاربرد', 'application'],
            'رقابت': ['رقابت', 'competition', 'رقیب', 'competitor', 'سهم بازار', 'market share']
        }
        
        found_topics = []
        text_lower = text.lower()
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)
        
        return found_topics
    
    def analyze_trends(self, text):
        """تحلیل روندها در متن"""
        trends = []
        
        # الگوهای روند صعودی
        bullish_patterns = [
            r'صعود \d+%؟',
            r'رشد \d+%؟',
            r'افزایش \d+%؟',
            r'up \d+%?',
            r'rise \d+%?',
            r'increase \d+%?'
        ]
        
        # الگوهای روند نزولی
        bearish_patterns = [
            r'نزول \d+%؟',
            r'کاهش \d+%؟',
            r'افت \d+%؟',
            r'down \d+%?',
            r'drop \d+%?',
            r'decrease \d+%?'
        ]
        
        # جستجوی الگوها
        for pattern in bullish_patterns:
            matches = re.findall(pattern, text)
            if matches:
                trends.append({'type': 'bullish', 'pattern': matches[0]})
        
        for pattern in bearish_patterns:
            matches = re.findall(pattern, text)
            if matches:
                trends.append({'type': 'bearish', 'pattern': matches[0]})
        
        return trends
    
    def get_top_topics(self, topics):
        """دریافت پرتکرارترین موضوعات"""
        topic_counts = Counter(topics)
        return [topic for topic, count in topic_counts.most_common(3)]
    
    async def get_market_data(self, symbol):
        """دریافت داده‌های بازار از چندین منبع"""
        # دریافت داده‌ها از منابع مختلف
        all_data = await self.fetch_data_from_multiple_sources(symbol)
        
        # ترکیب داده‌ها و محاسبه میانگین
        prices = []
        volumes = []
        market_caps = []
        changes = []
        
        for source, data in all_data.items():
            if isinstance(data, dict):
                if 'price' in data and data['price']:
                    prices.append(data['price'])
                if 'volume_24h' in data and data['volume_24h']:
                    volumes.append(data['volume_24h'])
                if 'market_cap' in data and data['market_cap']:
                    market_caps.append(data['market_cap'])
                if 'percent_change_24h' in data and data['percent_change_24h']:
                    changes.append(data['percent_change_24h'])
        
        # محاسبه میانگین‌ها
        avg_price = np.mean(prices) if prices else 0
        avg_volume = np.mean(volumes) if volumes else 0
        avg_market_cap = np.mean(market_caps) if market_caps else 0
        avg_change = np.mean(changes) if changes else 0
        
        return {
            'symbol': symbol,
            'price': avg_price,
            'volume_24h': avg_volume,
            'market_cap': avg_market_cap,
            'price_change_24h': avg_change,
            'sources': list(all_data.keys())
        }
    
    def get_offline_market_data(self, symbol):
        """دریافت داده‌های بازار آفلاین"""
        base_prices = {
            'BTC': 43000,
            'ETH': 2200,
            'BNB': 300,
            'SOL': 100,
            'XRP': 0.6,
            'ADA': 0.5,
            'DOT': 7,
            'DOGE': 0.08,
            'AVAX': 35,
            'MATIC': 0.8
        }
        
        base_price = base_prices.get(symbol, 100)
        change = np.random.uniform(-0.05, 0.05)
        price = base_price * (1 + change)
        
        return {
            'symbol': symbol,
            'price': price,
            'volume_24h': price * 1000000,
            'market_cap': price * 20000000,
            'price_change_24h': change * 100,
            'sources': ['offline']
        }
    
    async def perform_advanced_analysis(self, symbol):
        """انجام تحلیل پیشرفته با مدیریت خطا"""
        try:
            # دریافت داده‌های بازار
            market_data = await self.get_market_data(symbol)
            
            # اگر داده‌ها خالی هستند، از داده‌های آفلاین استفاده کن
            if not market_data or market_data['price'] == 0:
                logger.warning(f"Market data not available for {symbol}. Using offline data.")
                market_data = self.get_offline_market_data(symbol)
            
            # دریافت اخبار مرتبط
            try:
                news = await self.fetch_news_from_multiple_sources(symbol)
            except Exception as e:
                logger.error(f"Error fetching news: {e}")
                news = []
            
            # دریافت اخبار اقتصادی
            try:
                economic_news = await self.fetch_economic_news()
            except Exception as e:
                logger.error(f"Error fetching economic news: {e}")
                economic_news = []
            
            # تحلیل احساسات اخبار
            try:
                sentiment = await self.advanced_sentiment_analysis(news)
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {e}")
                sentiment = {'average_sentiment': 0, 'topics': []}
            
            # تحلیل احساسات اخبار اقتصادی
            try:
                economic_sentiment = await self.advanced_sentiment_analysis(economic_news)
            except Exception as e:
                logger.error(f"Error in economic sentiment analysis: {e}")
                economic_sentiment = {'average_sentiment': 0, 'topics': []}
            
            # دریافت داده‌های تاریخی
            try:
                historical_data = self.get_historical_data(symbol)
            except Exception as e:
                logger.error(f"Error getting historical data: {e}")
                historical_data = self.generate_dummy_data(symbol)
            
            # تحلیل تکنیکال پیشرفته
            try:
                technical_analysis = self.advanced_technical_analysis(historical_data)
            except Exception as e:
                logger.error(f"Error in technical analysis: {e}")
                technical_analysis = {}
            
            # تحلیل امواج الیوت
            try:
                elliott_analysis = self.advanced_elliott_wave(historical_data)
            except Exception as e:
                logger.error(f"Error in Elliott wave analysis: {e}")
                elliott_analysis = {'current_pattern': 'unknown'}
            
            # تحلیل عرضه و تقاضا
            try:
                supply_demand = self.advanced_supply_demand(symbol)
            except Exception as e:
                logger.error(f"Error in supply demand analysis: {e}")
                supply_demand = {'imbalance': 0}
            
            # تحلیل ساختار بازار (Order Block, Supply & Demand)
            try:
                market_structure = self.analyze_market_structure(historical_data)
            except Exception as e:
                logger.error(f"Error in market structure analysis: {e}")
                market_structure = {}
            
            # تحلیل چند زمانی (Multi-timeframe)
            try:
                multi_timeframe = self.analyze_multi_timeframe(symbol)
            except Exception as e:
                logger.error(f"Error in multi-timeframe analysis: {e}")
                multi_timeframe = {}
            
            # تحلیل جلسه معاملاتی (Trading Session)
            try:
                session_analysis = self.analyze_trading_session(symbol)
            except Exception as e:
                logger.error(f"Error in trading session analysis: {e}")
                session_analysis = {}
            
            # تحلیل نواحی تصمیم‌گیری (Decision Zones)
            try:
                decision_zones = self.analyze_decision_zones(historical_data)
            except Exception as e:
                logger.error(f"Error in decision zones analysis: {e}")
                decision_zones = {}
            
            # تحلیل مدیریت سرمایه (Risk Management)
            try:
                risk_management = self.analyze_risk_management(historical_data, market_data)
            except Exception as e:
                logger.error(f"Error in risk management analysis: {e}")
                risk_management = {}
            
            # تحلیل هوش مصنوعی
            try:
                ai_analysis = self.perform_ai_analysis(historical_data, market_data, sentiment, economic_sentiment)
            except Exception as e:
                logger.error(f"Error in AI analysis: {e}")
                ai_analysis = {}
            
            # تحلیل‌های پیشرفته جدید
            advanced_analysis = {}
            for method_name, method_func in self.analysis_methods.items():
                try:
                    advanced_analysis[method_name] = method_func(historical_data)
                except Exception as e:
                    logger.error(f"Error in {method_name} analysis: {e}")
                    advanced_analysis[method_name] = {}
            
            # ترکیب همه تحلیل‌ها
            combined_analysis = {
                'symbol': symbol,
                'market_data': market_data,
                'sentiment': sentiment,
                'economic_sentiment': economic_sentiment,
                'technical': technical_analysis,
                'elliott': elliott_analysis,
                'supply_demand': supply_demand,
                'market_structure': market_structure,
                'multi_timeframe': multi_timeframe,
                'session_analysis': session_analysis,
                'decision_zones': decision_zones,
                'risk_management': risk_management,
                'ai_analysis': ai_analysis,
                'advanced_analysis': advanced_analysis,
                'news_count': len(news),
                'economic_news_count': len(economic_news),
                'timestamp': datetime.now().isoformat()
            }
            
            # محاسبه سیگنال نهایی
            try:
                signal_score = self.calculate_signal_score(combined_analysis)
                signal = 'BUY' if signal_score > 0.7 else 'SELL' if signal_score < 0.3 else 'HOLD'
            except Exception as e:
                logger.error(f"Error calculating signal: {e}")
                signal = 'HOLD'
                signal_score = 0.5
            
            combined_analysis['signal'] = signal
            combined_analysis['confidence'] = signal_score
            
            return combined_analysis
        except Exception as e:
            logger.error(f"Error in perform_advanced_analysis for {symbol}: {e}")
            # برگرداندن تحلیل پیش‌فرض در صورت خطا
            return {
                'symbol': symbol,
                'signal': 'HOLD',
                'confidence': 0.5,
                'error': str(e)
            }
    
    def get_historical_data(self, symbol, period='1y'):
        """دریافت داده‌های تاریخی"""
        try:
            # تلاش برای دریافت داده از Yahoo Finance
            data = yf.download(f'{symbol}-USD', period=period, interval='1d')
            if data.empty:
                # اگر داده‌ای دریافت نشد، داده‌های ساختگی برگردان
                return self.generate_dummy_data(symbol)
            return data
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return self.generate_dummy_data(symbol)
    
    def generate_dummy_data(self, symbol):
        """تولید داده‌های ساختگی برای تست"""
        base_prices = {
            'BTC': 43000,
            'ETH': 2200,
            'BNB': 300,
            'SOL': 100,
            'XRP': 0.6,
            'ADA': 0.5,
            'DOT': 7,
            'DOGE': 0.08,
            'AVAX': 35,
            'MATIC': 0.8
        }
        
        base_price = base_prices.get(symbol, 100)
        
        # ایجاد داده‌های ساختگی
        dates = pd.date_range(start='2022-01-01', end=datetime.now().strftime('%Y-%m-%d'))
        prices = [base_price * (1 + np.random.normal(0, 0.02)) for _ in range(len(dates))]
        
        # ایجاد DataFrame
        data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': [base_price * 1000000 * (0.8 + 0.4 * np.random.random()) for _ in prices]
        }, index=dates)
        
        return data
    
    def advanced_technical_analysis(self, data):
        """تحلیل تکنیکال پیشرفته"""
        if data.empty:
            return {}
        
        try:
            # محاسبه شاخص‌های تکنیکال
            result = {
                'classical': {},
                'oscillators': {},
                'patterns': {}
            }
            
            # RSI
            if LIBRARIES['talib']:
                result['classical']['rsi'] = {'14': talib.RSI(data['Close'], timeperiod=14)[-1]}
            else:
                # محاسبه RSI به صورت دستی
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                result['classical']['rsi'] = {'14': rsi.iloc[-1]}
            
            # MACD
            if LIBRARIES['talib']:
                macd, macdsignal, macdhist = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
                result['classical']['macd'] = {
                    'macd': macd[-1],
                    'signal': macdsignal[-1],
                    'histogram': macdhist[-1]
                }
            else:
                # محاسبه MACD به صورت دستی
                ema12 = data['Close'].ewm(span=12).mean()
                ema26 = data['Close'].ewm(span=26).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9).mean()
                histogram = macd - signal
                result['classical']['macd'] = {
                    'macd': macd.iloc[-1],
                    'signal': signal.iloc[-1],
                    'histogram': histogram.iloc[-1]
                }
            
            # Bollinger Bands
            if LIBRARIES['talib']:
                upper, middle, lower = talib.BBANDS(data['Close'], timeperiod=20)
                result['classical']['bollinger'] = {
                    'upper': upper[-1],
                    'middle': middle[-1],
                    'lower': lower[-1]
                }
            else:
                # محاسبه بولینگر بند به صورت دستی
                ma20 = data['Close'].rolling(window=20).mean()
                std20 = data['Close'].rolling(window=20).std()
                result['classical']['bollinger'] = {
                    'upper': (ma20 + 2 * std20).iloc[-1],
                    'middle': ma20.iloc[-1],
                    'lower': (ma20 - 2 * std20).iloc[-1]
                }
            
            # Stochastic
            if LIBRARIES['talib']:
                slowk, slowd = talib.STOCH(data['High'], data['Low'], data['Close'], 
                                         fastk_period=14, slowk_period=3, slowd_period=3)
                result['oscillators']['stochastic'] = {
                    'slowk': slowk[-1],
                    'slowd': slowd[-1]
                }
            
            # CCI
            if LIBRARIES['talib']:
                cci = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
                result['oscillators']['cci'] = {'14': cci[-1]}
            
            # ATR
            if LIBRARIES['talib']:
                atr = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
                result['classical']['atr'] = {'14': atr[-1]}
            
            # تحلیل روند
            sma50 = data['Close'].rolling(window=50).mean()
            sma200 = data['Close'].rolling(window=200).mean()
            
            trend_direction = 'uptrend' if data['Close'][-1] > sma50[-1] > sma200[-1] else 'downtrend'
            result['classical']['trend'] = {
                'direction': trend_direction,
                'sma50': sma50.iloc[-1],
                'sma200': sma200.iloc[-1]
            }
            
            return result
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {}
    
    def wyckoff_analysis(self, data):
        """تحلیل روش ویکتور ویچاف"""
        try:
            if data.empty:
                return {}
            
            # شناسایی فازهای ویچاف
            close_prices = data['Close']
            volumes = data['Volume']
            
            # محاسبه میانگین متحرک حجم
            volume_ma = volumes.rolling(window=20).mean()
            
            # شناسایی فاز انباشت (Accumulation)
            accumulation_phase = False
            distribution_phase = False
            
            # بررسی شرایط انباشت
            if (close_prices[-1] > close_prices[-10] and 
                volumes[-1] > volume_ma[-1] * 1.5 and
                close_prices[-5:].std() < close_prices[-20:].std() * 0.5):
                accumulation_phase = True
            
            # بررسی_conditions توزیع
            if (close_prices[-1] < close_prices[-10] and 
                volumes[-1] > volume_ma[-1] * 1.5 and
                close_prices[-5:].std() < close_prices[-20:].std() * 0.5):
                distribution_phase = True
            
            return {
                'accumulation_phase': accumulation_phase,
                'distribution_phase': distribution_phase,
                'phase': 'accumulation' if accumulation_phase else 'distribution' if distribution_phase else 'neutral',
                'volume_surge': volumes[-1] > volume_ma[-1] * 1.5,
                'price_range_tightening': close_prices[-5:].std() < close_prices[-20:].std() * 0.5
            }
        except Exception as e:
            logger.error(f"Error in Wyckoff analysis: {e}")
            return {}
    
    def volume_profile_analysis(self, data):
        """تحلیل پروفایل حجمی"""
        try:
            if data.empty:
                return {}
            
            # محاسبه پروفایل حجمی
            price_levels = np.linspace(data['Low'].min(), data['High'].max(), 50)
            volume_profile = []
            
            for i in range(len(price_levels) - 1):
                lower = price_levels[i]
                upper = price_levels[i + 1]
                
                # محاسبه حجم در این محدوده قیمتی
                mask = (data['Close'] >= lower) & (data['Close'] < upper)
                volume_in_range = data.loc[mask, 'Volume'].sum()
                
                volume_profile.append({
                    'price_level': (lower + upper) / 2,
                    'volume': volume_in_range,
                    'range': f"{lower:.2f}-{upper:.2f}"
                })
            
            # شناسایی نقاط حجم بالا (POC - Point of Control)
            poc = max(volume_profile, key=lambda x: x['volume'])
            
            # شناسایی نواحی ارزش (Value Area)
            total_volume = sum(vp['volume'] for vp in volume_profile)
            value_area_volume = total_volume * 0.7  # 70% حجم کل
            
            sorted_vp = sorted(volume_profile, key=lambda x: x['volume'], reverse=True)
            value_area = []
            current_volume = 0
            
            for vp in sorted_vp:
                value_area.append(vp)
                current_volume += vp['volume']
                if current_volume >= value_area_volume:
                    break
            
            return {
                'poc': poc,
                'value_area_high': max(va['price_level'] for va in value_area),
                'value_area_low': min(va['price_level'] for va in value_area),
                'volume_profile': volume_profile[:10]  # 10 سطح برتر
            }
        except Exception as e:
            logger.error(f"Error in volume profile analysis: {e}")
            return {}
    
    def market_profile_analysis(self, data):
        """تحلیل پروفایل بازار (TPO)"""
        try:
            if data.empty:
                return {}
            
            # محاسبه پروفایل بازار (TPO - Time Price Opportunity)
            # این یک پیاده‌سازی ساده است
            price_levels = np.linspace(data['Low'].min(), data['High'].max(), 30)
            tpo_profile = []
            
            for price in price_levels:
                # شمارش تعداد تایم‌هایی که قیمت در این سطح بوده
                count = ((data['Low'] <= price) & (data['High'] >= price)).sum()
                tpo_profile.append({
                    'price_level': price,
                    'tpo_count': count
                })
            
            # شناسایی نقاط تی‌پی‌او بالا (Point of Control)
            poc = max(tpo_profile, key=lambda x: x['tpo_count'])
            
            return {
                'poc': poc,
                'tpo_profile': tpo_profile[:10]  # 10 سطح برتر
            }
        except Exception as e:
            logger.error(f"Error in market profile analysis: {e}")
            return {}
    
    def fibonacci_analysis(self, data):
        """تحلیل سطوح فیبوناچی"""
        try:
            if data.empty or len(data) < 50:
                return {}
            
            # شناسایی نقاط بالا و پایین اخیر
            high_point = data['High'][-50:].max()
            low_point = data['Low'][-50:].min()
            
            # محاسبه سطوح فیبوناچی
            diff = high_point - low_point
            fib_levels = {
                '0%': low_point,
                '23.6%': low_point + 0.236 * diff,
                '38.2%': low_point + 0.382 * diff,
                '50%': low_point + 0.5 * diff,
                '61.8%': low_point + 0.618 * diff,
                '78.6%': low_point + 0.786 * diff,
                '100%': high_point
            }
            
            # محاسبه سطوح گسترش فیبوناچی
            ext_levels = {
                '127.2%': high_point + 0.272 * diff,
                '161.8%': high_point + 0.618 * diff,
                '261.8%': high_point + 1.618 * diff
            }
            
            return {
                'retracement': fib_levels,
                'extension': ext_levels,
                'high_point': high_point,
                'low_point': low_point
            }
        except Exception as e:
            logger.error(f"Error in Fibonacci analysis: {e}")
            return {}
    
    def harmonic_patterns_analysis(self, data):
        """تحلیل الگوهای هارمونیک"""
        try:
            if data.empty or len(data) < 100:
                return {}
            
            # شناسایی نقاط چرخش محلی
            highs = data['High'].rolling(5, center=True).max()
            lows = data['Low'].rolling(5, center=True).min()
            
            # شناسایی الگوهای هارمونیک
            patterns_found = []
            
            # بررسی الگوی گارتلی
            if len(highs) >= 4 and len(lows) >= 3:
                # شناسایی نقاط X, A, B, C, D
                # این یک پیاده‌سازی ساده است
                X = lows[-100:].idxmin()
                A = highs[X:].idxmax()
                B = lows[A:].idxmin()
                C = highs[B:].idxmax()
                D = lows[C:].idxmin()
                
                # محاسبه نسبت‌ها
                XA = data.loc[A, 'High'] - data.loc[X, 'Low']
                AB = data.loc[A, 'High'] - data.loc[B, 'Low']
                BC = data.loc[C, 'High'] - data.loc[B, 'Low']
                CD = data.loc[C, 'High'] - data.loc[D, 'Low']
                
                # بررسی نسبت‌های گارتلی
                if (abs(AB / XA - 0.618) < 0.1 and 
                    abs(BC / AB - 0.382) < 0.1 and 
                    abs(CD / BC - 1.27) < 0.2):
                    patterns_found.append({
                        'pattern': 'gartley',
                        'points': {'X': X, 'A': A, 'B': B, 'C': C, 'D': D},
                        'type': 'bullish' if data.loc[D, 'Low'] < data.loc[X, 'Low'] else 'bearish'
                    })
            
            return {
                'patterns_found': patterns_found,
                'pattern_count': len(patterns_found)
            }
        except Exception as e:
            logger.error(f"Error in harmonic patterns analysis: {e}")
            return {}
    
    def ichimoku_analysis(self, data):
        """تحلیل ابر ایچیموکو"""
        try:
            if data.empty or len(data) < 60:
                return {}
            
            # محاسبه اجزای ابر ایچیموکو
            high_9 = data['High'].rolling(window=9).max()
            low_9 = data['Low'].rolling(window=9).min()
            high_26 = data['High'].rolling(window=26).max()
            low_26 = data['Low'].rolling(window=26).min()
            high_52 = data['High'].rolling(window=52).max()
            low_52 = data['Low'].rolling(window=52).min()
            
            # Tenkan-sen (Conversion Line)
            tenkan_sen = (high_9 + low_9) / 2
            
            # Kijun-sen (Base Line)
            kijun_sen = (high_26 + low_26) / 2
            
            # Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            
            # Senkou Span B (Leading Span B)
            senkou_span_b = ((high_52 + low_52) / 2).shift(26)
            
            # Chikou Span (Lagging Span)
            chikou_span = data['Close'].shift(-26)
            
            # ابر ایچیموکو (Kumo)
            kumo_upper = senkou_span_a.combine(senkou_span_b, max)
            kumo_lower = senkou_span_a.combine(senkou_span_b, min)
            
            return {
                'tenkan_sen': tenkan_sen.iloc[-1],
                'kijun_sen': kijun_sen.iloc[-1],
                'senkou_span_a': senkou_span_a.iloc[-1],
                'senkou_span_b': senkou_span_b.iloc[-1],
                'chikou_span': chikou_span.iloc[-1],
                'kumo_upper': kumo_upper.iloc[-1],
                'kumo_lower': kumo_lower.iloc[-1],
                'price_above_kumo': data['Close'][-1] > kumo_upper.iloc[-1],
                'price_below_kumo': data['Close'][-1] < kumo_lower.iloc[-1]
            }
        except Exception as e:
            logger.error(f"Error in Ichimoku analysis: {e}")
            return {}
    
    def support_resistance_analysis(self, data):
        """تحلیل سطوح حمایت و مقاومت"""
        try:
            if data.empty:
                return {}
            
            # شناسایی سطوح حمایت و مقاومت با استفاده از نقاط چرخش
            highs = data['High'].rolling(5, center=True).max()
            lows = data['Low'].rolling(5, center=True).min()
            
            # شناسایی نقاط چرخش محلی
            pivot_highs = []
            pivot_lows = []
            
            for i in range(5, len(data) - 5):
                if highs[i] == data['High'][i] and highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    pivot_highs.append((data.index[i], data['High'][i]))
                
                if lows[i] == data['Low'][i] and lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    pivot_lows.append((data.index[i], data['Low'][i]))
            
            # گروه‌بندی سطوح مشابه
            resistance_levels = self.group_similar_levels([level[1] for level in pivot_highs])
            support_levels = self.group_similar_levels([level[1] for level in pivot_lows])
            
            return {
                'resistance': resistance_levels[:5],  # 5 سطح مقاومت برتر
                'support': support_levels[:5]  # 5 سطح حمایت برتر
            }
        except Exception as e:
            logger.error(f"Error in support resistance analysis: {e}")
            return {}
    
    def group_similar_levels(self, levels, threshold=0.02):
        """گروه‌بندی سطوح مشابه"""
        if not levels:
            return []
        
        levels.sort(reverse=True)
        grouped = []
        current_group = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_group[-1]) / current_group[-1] <= threshold:
                current_group.append(level)
            else:
                grouped.append(sum(current_group) / len(current_group))
                current_group = [level]
        
        if current_group:
            grouped.append(sum(current_group) / len(current_group))
        
        return grouped
    
    def trend_lines_analysis(self, data):
        """تحلیل خطوط روند"""
        try:
            if data.empty or len(data) < 20:
                return {}
            
            # شناسایی خطوط روند صعودی
            uptrend_lines = []
            for i in range(10, len(data) - 10):
                # پیدا کردن دو کف پایین‌تر
                if (data['Low'][i] < data['Low'][i-10:i].min() and 
                    data['Low'][i] < data['Low'][i+1:i+11].min()):
                    
                    # پیدا کردن کف قبلی
                    for j in range(i-10, 0, -1):
                        if (data['Low'][j] < data['Low'][j-10:j].min() and 
                            data['Low'][j] < data['Low'][j+1:j+11].min()):
                            
                            # محاسبه خط روند
                            slope = (data['Low'][i] - data['Low'][j]) / (i - j)
                            intercept = data['Low'][j] - slope * j
                            
                            uptrend_lines.append({
                                'start': (j, data['Low'][j]),
                                'end': (i, data['Low'][i]),
                                'slope': slope,
                                'intercept': intercept
                            })
                            break
            
            # شناسایی خطوط روند نزولی
            downtrend_lines = []
            for i in range(10, len(data) - 10):
                # پیدا کردن دو سقف بالاتر
                if (data['High'][i] > data['High'][i-10:i].max() and 
                    data['High'][i] > data['High'][i+1:i+11].max()):
                    
                    # پیدا کردن سقف قبلی
                    for j in range(i-10, 0, -1):
                        if (data['High'][j] > data['High'][j-10:j].max() and 
                            data['High'][j] > data['High'][j+1:j+11].max()):
                            
                            # محاسبه خط روند
                            slope = (data['High'][i] - data['High'][j]) / (i - j)
                            intercept = data['High'][j] - slope * j
                            
                            downtrend_lines.append({
                                'start': (j, data['High'][j]),
                                'end': (i, data['High'][i]),
                                'slope': slope,
                                'intercept': intercept
                            })
                            break
            
            return {
                'uptrend_lines': uptrend_lines[-3:],  # 3 خط روند صعودی اخیر
                'downtrend_lines': downtrend_lines[-3:],  # 3 خط روند نزولی اخیر
                'current_trend': 'uptrend' if uptrend_lines else 'downtrend' if downtrend_lines else 'sideways'
            }
        except Exception as e:
            logger.error(f"Error in trend lines analysis: {e}")
            return {}
    
    def order_flow_analysis(self, data):
        """تحلیل جریان سفارش"""
        try:
            if data.empty:
                return {}
            
            # این یک پیاده‌سازی ساده از تحلیل جریان سفارش است
            # در واقعیت، این تحلیل نیاز به داده‌های لول 2 دارد
            
            # محاسبه حجم خرید و فروش
            buy_volume = data[data['Close'] > data['Open']]['Volume'].sum()
            sell_volume = data[data['Close'] < data['Open']]['Volume'].sum()
            
            # محاسبه نسبت خرید به فروش
            buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else float('inf')
            
            # شناسایی نواحی جذاب برای سفارش‌ها
            order_blocks = []
            
            for i in range(10, len(data) - 10):
                # شناسایی Order Block صعودی
                if (data['Close'][i] > data['Open'][i] and 
                    data['Volume'][i] > data['Volume'][i-5:i+5].mean() * 1.5 and
                    data['Close'][i+1] < data['Close'][i]):
                    
                    order_blocks.append({
                        'type': 'bullish_ob',
                        'price': data['Close'][i],
                        'volume': data['Volume'][i],
                        'time': data.index[i]
                    })
                
                # شناسایی Order Block نزولی
                elif (data['Close'][i] < data['Open'][i] and 
                      data['Volume'][i] > data['Volume'][i-5:i+5].mean() * 1.5 and
                      data['Close'][i+1] > data['Close'][i]):
                    
                    order_blocks.append({
                        'type': 'bearish_ob',
                        'price': data['Close'][i],
                        'volume': data['Volume'][i],
                        'time': data.index[i]
                    })
            
            return {
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'buy_sell_ratio': buy_sell_ratio,
                'order_blocks': order_blocks[-5:]  # 5 Order Block اخیر
            }
        except Exception as e:
            logger.error(f"Error in order flow analysis: {e}")
            return {}
    
    def vwap_analysis(self, data):
        """تحلیل میانگین وزنی حجمی (VWAP)"""
        try:
            if data.empty:
                return {}
            
            # محاسبه VWAP
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
            
            # محاسبه انحراف معیار VWAP
            vwap_std = typical_price.rolling(window=20).std()
            
            # محاسبه باندهای VWAP
            vwap_upper = vwap + vwap_std
            vwap_lower = vwap - vwap_std
            
            return {
                'vwap': vwap.iloc[-1],
                'vwap_upper': vwap_upper.iloc[-1],
                'vwap_lower': vwap_lower.iloc[-1],
                'price_above_vwap': data['Close'][-1] > vwap.iloc[-1],
                'price_below_vwap': data['Close'][-1] < vwap.iloc[-1]
            }
        except Exception as e:
            logger.error(f"Error in VWAP analysis: {e}")
            return {}
    
    def pivot_points_analysis(self, data):
        """تحلیل نقاط پیوت"""
        try:
            if data.empty:
                return {}
            
            # محاسبه نقاط پیوت کلاسیک
            high = data['High'][-1]
            low = data['Low'][-1]
            close = data['Close'][-1]
            
            pivot = (high + low + close) / 3
            
            # محاسبه سطوح مقاومت
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            # محاسبه سطوح حمایت
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                'pivot': pivot,
                'resistance': {'r1': r1, 'r2': r2, 'r3': r3},
                'support': {'s1': s1, 's2': s2, 's3': s3}
            }
        except Exception as e:
            logger.error(f"Error in pivot points analysis: {e}")
            return {}
    
    def advanced_candlestick_patterns(self, data):
        """تحلیل الگوهای شمعی پیشرفته"""
        try:
            if data.empty or len(data) < 5:
                return {}
            
            patterns_found = []
            
            # بررسی الگوهای شمعی پیشرفته
            # این یک پیاده‌سازی ساده است
            
            # الگوی سه سرباز سفید
            if (len(data) >= 3 and
                data['Close'][-1] > data['Open'][-1] and
                data['Close'][-2] > data['Open'][-2] and
                data['Close'][-3] > data['Open'][-3] and
                data['Close'][-1] > data['Close'][-2] > data['Close'][-3]):
                patterns_found.append('three_white_soldiers')
            
            # الگوی سه کلاغ سیاه
            if (len(data) >= 3 and
                data['Close'][-1] < data['Open'][-1] and
                data['Close'][-2] < data['Open'][-2] and
                data['Close'][-3] < data['Open'][-3] and
                data['Close'][-1] < data['Close'][-2] < data['Close'][-3]):
                patterns_found.append('three_black_crows')
            
            # الگوی ستاره صبحگاهی
            if (len(data) >= 3 and
                data['Close'][-3] < data['Open'][-3] and  # شمع اول نزولی
                abs(data['Close'][-2] - data['Open'][-2]) < abs(data['Close'][-3] - data['Open'][-3]) * 0.3 and  # دوجی
                data['Close'][-1] > data['Open'][-1] and  # شمع سوم صعودی
                data['Close'][-1] > (data['Open'][-3] + data['Close'][-3]) / 2):  # بسته شدن بالاتر از نیمه شمع اول
                patterns_found.append('morning_star')
            
            # الگوی ستاره عصرگاهی
            if (len(data) >= 3 and
                data['Close'][-3] > data['Open'][-3] and  # شمع اول صعودی
                abs(data['Close'][-2] - data['Open'][-2]) < abs(data['Close'][-3] - data['Open'][-3]) * 0.3 and  # دوجی
                data['Close'][-1] < data['Open'][-1] and  # شمع سوم نزولی
                data['Close'][-1] < (data['Open'][-3] + data['Close'][-3]) / 2):  # بسته شدن پایین‌تر از نیمه شمع اول
                patterns_found.append('evening_star')
            
            return {
                'patterns_found': patterns_found,
                'pattern_count': len(patterns_found),
                'descriptions': [self.advanced_candlesticks.get(p, '') for p in patterns_found]
            }
        except Exception as e:
            logger.error(f"Error in advanced candlestick patterns analysis: {e}")
            return {}
    
    def advanced_elliott_wave(self, data):
        """تحلیل امواج الیوت پیشرفته"""
        try:
            if data.empty:
                return {'current_pattern': 'unknown'}
            
            # شناسایی قله‌ها و دره‌ها
            peaks, _ = find_peaks(data['Close'], distance=5)
            troughs, _ = find_peaks(-data['Close'], distance=5)
            
            # تحلیل الگوی موجی
            if len(peaks) >= 3 and len(troughs) >= 2:
                # بررسی الگوی 5 موجی
                if (data['Close'][peaks[0]] < data['Close'][peaks[1]] > data['Close'][peaks[2]] and
                    data['Close'][troughs[0]] < data['Close'][troughs[1]]):
                    return {
                        'current_pattern': 'impulse_wave',
                        'wave_count': 3,
                        'confidence': 0.75
                    }
            
            # بررسی الگوی 3 موجی اصلاحی
            if len(peaks) >= 2 and len(troughs) >= 2:
                if (data['Close'][peaks[0]] > data['Close'][peaks[1]] and
                    data['Close'][troughs[0]] > data['Close'][troughs[1]]):
                    return {
                        'current_pattern': 'corrective_wave',
                        'wave_count': 2,
                        'confidence': 0.65
                    }
            
            return {
                'current_pattern': 'unknown',
                'wave_count': 0,
                'confidence': 0.5
            }
        except Exception as e:
            logger.error(f"Error in Elliott wave analysis: {e}")
            return {'current_pattern': 'unknown'}
    
    def market_structure_analysis(self, data):
        """تحلیل ساختار بازار"""
        try:
            if data.empty:
                return {}
            
            # شناسایی نقاط چرخش
            highs = data['High'].rolling(5, center=True).max()
            lows = data['Low'].rolling(5, center=True).min()
            
            # شناسایی Order Blocks
            order_blocks = []
            for i in range(10, len(data)-10):
                if data['Close'][i] > highs[i-5:i].max() and data['Volume'][i] > data['Volume'][i-5:i].mean():
                    order_blocks.append({
                        'type': 'bullish_ob',
                        'price': data['Close'][i],
                        'time': data.index[i]
                    })
                elif data['Close'][i] < lows[i-5:i].min() and data['Volume'][i] > data['Volume'][i-5:i].mean():
                    order_blocks.append({
                        'type': 'bearish_ob',
                        'price': data['Close'][i],
                        'time': data.index[i]
                    })
            
            # تحلیل روند فعلی
            short_trend = 'uptrend' if data['Close'][-20:].mean() > data['Close'][-40:-20].mean() else 'downtrend'
            
            return {
                'market_trend': short_trend,
                'order_blocks': order_blocks[-5:],  # 5 Order Block برتر
                'liquidity_zones': self.identify_liquidity_zones(data)
            }
        except Exception as e:
            logger.error(f"Error in market structure analysis: {e}")
            return {}
    
    def identify_liquidity_zones(self, data):
        """شناسایی نواحی نقدینگی"""
        try:
            if data.empty:
                return []
            
            # شناسایی نواحی نقدینگی بالای فروش
            sell_liquidity = []
            for i in range(10, len(data)-10):
                if (data['High'][i] > data['High'][i-5:i+5].max() and 
                    data['Volume'][i] > data['Volume'][i-5:i+5].mean() * 1.5):
                    sell_liquidity.append({
                        'price': data['High'][i],
                        'volume': data['Volume'][i],
                        'type': 'sell_liquidity'
                    })
            
            # شناسایی نواحی نقدینگی بالای خرید
            buy_liquidity = []
            for i in range(10, len(data)-10):
                if (data['Low'][i] < data['Low'][i-5:i+5].min() and 
                    data['Volume'][i] > data['Volume'][i-5:i+5].mean() * 1.5):
                    buy_liquidity.append({
                        'price': data['Low'][i],
                        'volume': data['Volume'][i],
                        'type': 'buy_liquidity'
                    })
            
            return sell_liquidity[-3:] + buy_liquidity[-3:]  # 3 ناحیه از هر نوع
        except Exception as e:
            logger.error(f"Error in identifying liquidity zones: {e}")
            return []
    
    def advanced_supply_demand(self, symbol):
        """تحلیل نواحی عرضه و تقاضا"""
        try:
            # دریافت داده‌های اخیر
            data = self.get_historical_data(symbol, period='3mo')
            
            if data.empty:
                return {'imbalance': 0}
            
            # شناسایی نواحی عرضه و تقاضا
            supply_zones = []
            demand_zones = []
            
            # محاسبه حجم معاملات در نواحی مختلف
            for i in range(10, len(data)-10):
                if data['Volume'][i] > data['Volume'][i-10:i+10].mean() * 1.5:
                    if data['Close'][i] > data['Close'][i-5:i].mean():
                        supply_zones.append((data.index[i], data['Close'][i]))
                    else:
                        demand_zones.append((data.index[i], data['Close'][i]))
            
            # محاسبه عدم تعادل عرضه و تقاضا
            supply_strength = sum([zone[1] for zone in supply_zones]) if supply_zones else 0
            demand_strength = sum([zone[1] for zone in demand_zones]) if demand_zones else 0
            
            imbalance = (demand_strength - supply_strength) / (supply_strength + demand_strength) if (supply_strength + demand_strength) > 0 else 0
            
            return {
                'imbalance': imbalance,
                'supply_zones': supply_zones[:3],  # 3 ناحیه عرضه برتر
                'demand_zones': demand_zones[:3],  # 3 ناحیه تقاضا برتر
                'signal': 'BUY' if imbalance > 0.2 else 'SELL' if imbalance < -0.2 else 'NEUTRAL'
            }
        except Exception as e:
            logger.error(f"Error in supply demand analysis: {e}")
            return {'imbalance': 0}
    
    def analyze_multi_timeframe(self, symbol):
        """تحلیل چند زمانی"""
        try:
            timeframes = ['1h', '4h', '1d', '1w']
            analysis = {}
            
            for tf in timeframes:
                try:
                    # دریافت داده‌های تاریخی برای این تایم‌فریم
                    data = self.get_historical_data(symbol, period='3mo')
                    
                    if not data.empty:
                        # تحلیل تکنیکال ساده برای این تایم‌فریم
                        rsi = talib.RSI(data['Close'], timeperiod=14)[-1] if LIBRARIES['talib'] else 50
                        sma20 = data['Close'].rolling(window=20).mean().iloc[-1]
                        sma50 = data['Close'].rolling(window=50).mean().iloc[-1]
                        
                        # تعیین روند
                        trend = 'uptrend' if data['Close'][-1] > sma20 > sma50 else 'downtrend'
                        
                        analysis[tf] = {
                            'rsi': rsi,
                            'trend': trend,
                            'price': data['Close'][-1],
                            'signal': 'BUY' if rsi < 30 and trend == 'uptrend' else 'SELL' if rsi > 70 and trend == 'downtrend' else 'HOLD'
                        }
                except Exception as e:
                    logger.error(f"Error analyzing timeframe {tf}: {e}")
                    analysis[tf] = {}
            
            return analysis
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            return {}
    
    def analyze_trading_session(self, symbol):
        """تحلیل جلسه معاملاتی"""
        try:
            # دریافت داده‌های اخیر
            data = self.get_historical_data(symbol, period='1w')
            
            if data.empty:
                return {}
            
            # تعریف جلسات معاملاتی
            sessions = {
                'asian': {'start': 22, 'end': 6},  # 22:00 UTC تا 06:00 UTC
                'london': {'start': 7, 'end': 16},  # 07:00 UTC تا 16:00 UTC
                'new_york': {'start': 12, 'end': 21}  # 12:00 UTC تا 21:00 UTC
            }
            
            session_analysis = {}
            
            for session_name, session_time in sessions.items():
                # فیلتر داده‌های این جلسه
                session_data = data.between_time(f"{session_time['start']}:00", f"{session_time['end']}:00")
                
                if not session_data.empty:
                    session_analysis[session_name] = {
                        'high': session_data['High'].max(),
                        'low': session_data['Low'].min(),
                        'volume': session_data['Volume'].sum(),
                        'range': session_data['High'].max() - session_data['Low'].min(),
                        'volatility': session_data['Close'].std()
                    }
            
            return session_analysis
        except Exception as e:
            logger.error(f"Error in trading session analysis: {e}")
            return {}
    
    def analyze_decision_zones(self, data):
        """تحلیل نواحی تصمیم‌گیری"""
        try:
            if data.empty:
                return {}
            
            # شناسایی نواحی تصمیم‌گیری بر اساس حجم و نوسان
            decision_zones = []
            
            for i in range(20, len(data)-20):
                # محاسبه نوسان و حجم
                volatility = data['Close'][i-20:i+20].std()
                volume = data['Volume'][i-20:i+20].mean()
                
                # شناسایی نواحی با نوسان و حجم بالا
                if (volatility > data['Close'].std() * 1.5 and 
                    volume > data['Volume'].mean() * 1.5):
                    
                    decision_zones.append({
                        'price': data['Close'][i],
                        'time': data.index[i],
                        'volatility': volatility,
                        'volume': volume,
                        'type': 'high_volatility_zone'
                    })
            
            return {
                'decision_zones': decision_zones[-5:],  # 5 ناحیه تصمیم‌گیری اخیر
                'zone_count': len(decision_zones)
            }
        except Exception as e:
            logger.error(f"Error in decision zones analysis: {e}")
            return {}
    
    def analyze_risk_management(self, data, market_data):
        """تحلیل مدیریت ریسک"""
        try:
            if data.empty or not market_data:
                return {}
            
            # محاسبه ATR (Average True Range)
            atr = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)[-1] if LIBRARIES['talib'] else data['Close'].std()
            
            # محاسبه اندازه پوزیشن پیشنهادی
            account_size = 10000  # فرض حساب 10000 دلاری
            risk_percent = 0.02  # ریسک 2%
            
            stop_loss_distance = atr * 2  # فاصله حد ضرر
            risk_amount = account_size * risk_percent
            
            position_size = risk_amount / stop_loss_distance if stop_loss_distance > 0 else 0
            
            # محاسبه سطوح حد ضرر و سود
            current_price = market_data.get('price', 0)
            stop_loss = current_price - stop_loss_distance
            take_profit = current_price + (stop_loss_distance * 2)  # نسبت ریسک به سود 1:2
            
            return {
                'atr': atr,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': 2.0,
                'risk_amount': risk_amount,
                'account_risk_percent': risk_percent * 100
            }
        except Exception as e:
            logger.error(f"Error in risk management analysis: {e}")
            return {}
    
    def perform_ai_analysis(self, historical_data, market_data, sentiment, economic_sentiment):
        """تحلیل هوش مصنوعی"""
        try:
            if historical_data.empty:
                return {}
            
            # آماده‌سازی داده‌ها برای مدل‌های یادگیری ماشین
            features = self.prepare_features(historical_data)
            
            if features.empty:
                return {}
            
            # پیش‌بینی با مدل‌های مختلف
            predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    if model is not None:
                        if model_name in ['lstm', 'gru']:
                            # برای مدل‌های عمیق
                            pred = self.predict_with_deep_model(model, features)
                        else:
                            # برای مدل‌های کلاسیک
                            X = features.iloc[:-1]
                            y = historical_data['Close'].iloc[1:]
                            
                            if len(X) > 10:  # حداقل داده برای آموزش
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                model.fit(X_train, y_train)
                                pred = model.predict(X_test.iloc[-1:].values.reshape(1, -1))[0]
                            else:
                                pred = historical_data['Close'][-1]
                        
                        predictions[model_name] = pred
                except Exception as e:
                    logger.error(f"Error with {model_name} model: {e}")
                    predictions[model_name] = historical_data['Close'][-1]
            
            # محاسبه میانگین پیش‌بینی‌ها
            avg_prediction = np.mean(list(predictions.values())) if predictions else historical_data['Close'][-1]
            
            # ترکیب با تحلیل احساسات
            sentiment_weight = 0.2
            economic_weight = 0.1
            
            sentiment_adjustment = sentiment.get('average_sentiment', 0) * sentiment_weight
            economic_adjustment = economic_sentiment.get('average_sentiment', 0) * economic_weight
            
            final_prediction = avg_prediction * (1 + sentiment_adjustment + economic_adjustment)
            
            return {
                'predictions': predictions,
                'average_prediction': avg_prediction,
                'final_prediction': final_prediction,
                'sentiment_adjustment': sentiment_adjustment,
                'economic_adjustment': economic_adjustment
            }
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return {}
    
    def prepare_features(self, data):
        """آماده‌سازی ویژگی‌ها برای مدل‌های یادگیری ماشین"""
        try:
            if data.empty:
                return pd.DataFrame()
            
            features = pd.DataFrame(index=data.index)
            
            # ویژگی‌های قیمت
            features['price'] = data['Close']
            features['price_change'] = data['Close'].pct_change()
            features['high_low_ratio'] = data['High'] / data['Low']
            
            # ویژگی‌های حجم
            features['volume'] = data['Volume']
            features['volume_change'] = data['Volume'].pct_change()
            features['volume_sma'] = data['Volume'].rolling(window=20).mean()
            
            # ویژگی‌های تکنیکال
            if LIBRARIES['talib']:
                features['rsi'] = talib.RSI(data['Close'], timeperiod=14)
                features['macd'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)[0]
                features['atr'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
            else:
                features['rsi'] = 50  # مقدار پیش‌فرض
                features['macd'] = 0
                features['atr'] = data['Close'].std()
            
            # ویژگی‌های زمانی
            features['day_of_week'] = data.index.dayofweek
            features['hour'] = data.index.hour
            
            # حذف ردیف‌های با مقادیر NaN
            features = features.dropna()
            
            return features
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def predict_with_deep_model(self, model, features):
        """پیش‌بینی با مدل‌های عمیق"""
        try:
            if model is None or features.empty:
                return 0
            
            # آماده‌سازی داده‌ها برای مدل‌های عمیق
            # این یک پیاده‌سازی ساده است
            sequence_length = 60
            if len(features) < sequence_length:
                return features['price'].iloc[-1]
            
            # ایجاد دنباله‌ها
            sequences = []
            for i in range(sequence_length, len(features)):
                sequences.append(features.iloc[i-sequence_length:i].values)
            
            if not sequences:
                return features['price'].iloc[-1]
            
            # پیش‌بینی
            X = np.array(sequences)
            predictions = model.predict(X)
            
            return predictions[-1][0]
        except Exception as e:
            logger.error(f"Error predicting with deep model: {e}")
            return 0
    
    def calculate_signal_score(self, analysis):
        """محاسبه امتیاز سیگنال نهایی"""
        try:
            score = 0.5  # امتیاز پایه
            
            # تحلیل تکنیکال
            technical = analysis.get('technical', {})
            classical = technical.get('classical', {})
            
            # RSI
            rsi = classical.get('rsi', {}).get('14', 50)
            if rsi < 30:
                score += 0.1  # سیگنال خرید
            elif rsi > 70:
                score -= 0.1  # سیگنال فروش
            
            # MACD
            macd = classical.get('macd', {})
            if macd.get('macd', 0) > macd.get('signal', 0):
                score += 0.1  # سیگنال خرید
            else:
                score -= 0.1  # سیگنال فروش
            
            # تحلیل احساسات
            sentiment = analysis.get('sentiment', {})
            avg_sentiment = sentiment.get('average_sentiment', 0)
            score += avg_sentiment * 0.2  # وزن احساسات
            
            # تحلیل اقتصادی
            economic_sentiment = analysis.get('economic_sentiment', {})
            avg_economic = economic_sentiment.get('average_sentiment', 0)
            score += avg_economic * 0.1  # وزن اخبار اقتصادی
            
            # تحلیل‌های پیشرفته
            advanced = analysis.get('advanced_analysis', {})
            
            # تحلیل ویچاف
            wyckoff = advanced.get('wyckoff', {})
            if wyckoff.get('accumulation_phase', False):
                score += 0.15  # سیگنال خرید قوی
            elif wyckoff.get('distribution_phase', False):
                score -= 0.15  # سیگنال فروش قوی
            
            # تحلیل ساختار بازار
            market_structure = advanced.get('market_structure', {})
            if market_structure.get('market_trend') == 'uptrend':
                score += 0.1
            elif market_structure.get('market_trend') == 'downtrend':
                score -= 0.1
            
            # تحلیل جریان سفارش
            order_flow = advanced.get('order_flow', {})
            buy_sell_ratio = order_flow.get('buy_sell_ratio', 1)
            if buy_sell_ratio > 1.5:
                score += 0.1
            elif buy_sell_ratio < 0.5:
                score -= 0.1
            
            # تحلیل هوش مصنوعی
            ai_analysis = analysis.get('ai_analysis', {})
            final_prediction = ai_analysis.get('final_prediction', 0)
            current_price = analysis.get('market_data', {}).get('price', 1)
            
            if final_prediction > current_price * 1.05:  # پیش‌بینی رشد بیش از 5%
                score += 0.15
            elif final_prediction < current_price * 0.95:  # پیش‌بینی کاهش بیش از 5%
                score -= 0.15
            
            # نرمال‌سازی امتیاز بین 0 و 1
            score = max(0, min(1, score))
            
            return score
        except Exception as e:
            logger.error(f"Error calculating signal score: {e}")
            return 0.5

async def main():
    """نقطه ورودی اصلی برنامه"""
    logger.info("Starting Advanced Trading Bot...")
    
    try:
        # ایجاد نمونه از ربات
        bot = AdvancedTradingBot()
        
        # دریافت توکن تلگرام
        token = os.getenv("TELEGRAM_TOKEN")
        if not token:
            logger.error("TELEGRAM_TOKEN not set in environment variables")
            return
        
        # ایجاد درخواست با پروکسی اگر تنظیم شده باشد
        if PROXY_SETTINGS:
            request = HTTPXRequest(
                proxy=PROXY_SETTINGS['proxy']['url'],
                connect_timeout=30.0,
                read_timeout=30.0,
                pool_timeout=30.0,
            )
        else:
            request = HTTPXRequest()
        
        # ایجاد اپلیکیشن تلگرام
        application = Application.builder().token(token).request(request).build()
        
        # تنظیم هندلرها
        from telegram_handlers import setup_handlers
        setup_handlers(application, bot)
        
        # اجرای ربات
        logger.info("Bot started successfully")
        await application.run_polling()
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")