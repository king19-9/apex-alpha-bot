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

# مدیریت خطای pandas_ta به دلیل عدم سازگاری با numpy جدید
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except (ImportError, ValueError) as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"pandas_ta not available due to compatibility issues: {e}")
    PANDAS_TA_AVAILABLE = False
    ta = None

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
    'pandas_ta': PANDAS_TA_AVAILABLE,
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
        if lib == 'pandas_ta':
            # pandas_ta را قبلاً مدیریت کردیم
            if PANDAS_TA_AVAILABLE:
                LIBRARIES[lib] = True
        else:
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
                self.is_postgres = True
                logger.info("PostgreSQL connection established")
            else:
                self.conn = sqlite3.connect('trading_bot.db', check_same_thread=False)
                self.is_postgres = False
                logger.warning("Using SQLite as fallback")
            
            self.create_tables()
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            # فallback به SQLite
            self.conn = sqlite3.connect('trading_bot.db', check_same_thread=False)
            self.is_postgres = False
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
        }
    
    def create_tables(self):
        """ایجاد جداول پایگاه داده"""
        cursor = self.conn.cursor()
        
        # تعیین نوع کلید اصلی بر اساس نوع پایگاه داده
        if self.is_postgres:
            id_type = "SERIAL PRIMARY KEY"
            timestamp_type = "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        else:
            id_type = "INTEGER PRIMARY KEY AUTOINCREMENT"
            timestamp_type = "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        
        # جدول کاربران
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            language TEXT DEFAULT 'fa',
            preferences TEXT,
            created_at {timestamp_type}
        )
        ''')
        
        # جدول تحلیل‌ها
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS analyses (
            id {id_type},
            user_id INTEGER,
            symbol TEXT,
            analysis_type TEXT,
            result TEXT,
            timestamp {timestamp_type},
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # جدول سیگنال‌ها
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS signals (
            id {id_type},
            user_id INTEGER,
            symbol TEXT,
            signal_type TEXT,
            signal_value TEXT,
            confidence REAL,
            timestamp {timestamp_type},
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # جدول واچ‌لیست
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS watchlist (
            id {id_type},
            user_id INTEGER,
            symbol TEXT,
            added_at {timestamp_type},
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # جدول عملکرد
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS performance (
            id {id_type},
            user_id INTEGER,
            symbol TEXT,
            strategy TEXT,
            entry_price REAL,
            exit_price REAL,
            profit_loss REAL,
            duration INTEGER,
            timestamp {timestamp_type},
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # جدول داده‌های بازار
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS market_data (
            id {id_type},
            symbol TEXT,
            source TEXT,
            price REAL,
            volume_24h REAL,
            market_cap REAL,
            price_change_24h REAL,
            timestamp {timestamp_type}
        )
        ''')
        
        # جدول اخبار
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS news (
            id {id_type},
            title TEXT,
            content TEXT,
            source TEXT,
            url TEXT,
            published_at TIMESTAMP,
            sentiment_score REAL,
            symbols TEXT,
            timestamp {timestamp_type}
        )
        ''')
        
        # جدول تحلیل‌های هوش مصنوعی
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS ai_analysis (
            id {id_type},
            symbol TEXT,
            analysis_type TEXT,
            result TEXT,
            confidence REAL,
            timestamp {timestamp_type}
        )
        ''')
        
        # جدول داده‌های اقتصادی
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS economic_data (
            id {id_type},
            event_type TEXT,
            event_name TEXT,
            event_date TIMESTAMP,
            actual_value REAL,
            forecast_value REAL,
            previous_value REAL,
            impact TEXT,
            timestamp {timestamp_type}
        )
        ''')
        
        # جدول داده‌های جلسه معاملاتی
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS trading_sessions (
            id {id_type},
            symbol TEXT,
            session_type TEXT,
            session_start TIMESTAMP,
            session_end TIMESTAMP,
            high_price REAL,
            low_price REAL,
            volume REAL,
            timestamp {timestamp_type}
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
    
    def get_historical_data(self, symbol, period='1y', interval='1d'):
        """دریافت داده‌های تاریخی"""
        try:
            # تلاش برای دریافت داده از Yahoo Finance
            data = yf.download(f'{symbol}-USD', period=period, interval=interval)
            if data.empty:
                # اگر داده‌ای دریافت نشد، داده‌های ساختگی برگردان
                return self.generate_dummy_data(symbol)
            return data
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return self.generate_dummy_data(symbol)
    
    def generate_dummy_data(self, symbol):
        """تولید داده‌های ساختگی برای تست"""
        logger.info(f"Generating dummy data for {symbol}")
        
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
        
        # ایجاد داده‌های ساختگی
        dates = pd.date_range(start='2023-01-01', end='2023-12-31')
        prices = []
        
        current_price = base_price
        for _ in range(len(dates)):
            change = np.random.uniform(-0.03, 0.03)
            current_price *= (1 + change)
            prices.append(current_price)
        
        # ایجاد DataFrame
        data = pd.DataFrame({
            'Open': prices,
            'High': [p * np.random.uniform(1.0, 1.03) for p in prices],
            'Low': [p * np.random.uniform(0.97, 1.0) for p in prices],
            'Close': prices,
            'Volume': [np.random.randint(1000000, 5000000) for _ in prices]
        }, index=dates)
        
        return data
    
    def advanced_technical_analysis(self, data):
        """تحلیل تکنیکال پیشرفته"""
        if data.empty:
            return {}
        
        try:
            # محاسبه شاخص‌های تکنیکال
            close = data['Close']
            high = data['High']
            low = data['Low']
            
            # RSI
            rsi = talib.RSI(close, timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            
            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close)
            
            # Williams %R
            williams_r = talib.WILLR(high, low, close)
            
            # Commodity Channel Index
            cci = talib.CCI(high, low, close)
            
            # Average Directional Index
            adx = talib.ADX(high, low, close)
            
            # تحلیل روند
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            
            trend_direction = 'صعودی' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'نزولی'
            
            return {
                'classical': {
                    'rsi': {'14': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50},
                    'macd': {
                        'macd': macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0,
                        'signal': macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else 0,
                        'histogram': macd_hist.iloc[-1] if not pd.isna(macd_hist.iloc[-1]) else 0
                    },
                    'bollinger': {
                        'upper': bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else 0,
                        'middle': bb_middle.iloc[-1] if not pd.isna(bb_middle.iloc[-1]) else 0,
                        'lower': bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else 0
                    },
                    'stochastic': {
                        'slowk': slowk.iloc[-1] if not pd.isna(slowk.iloc[-1]) else 50,
                        'slowd': slowd.iloc[-1] if not pd.isna(slowd.iloc[-1]) else 50
                    },
                    'williams_r': williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else -50,
                    'cci': cci.iloc[-1] if not pd.isna(cci.iloc[-1]) else 0,
                    'adx': adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25,
                    'trend': {
                        'direction': trend_direction,
                        'sma_20': sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else 0,
                        'sma_50': sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else 0
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {}
    
    def ichimoku_analysis(self, data):
        """تحلیل ابر ایچیموکو"""
        if data.empty:
            return {}
        
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # محاسبه خطوط ایچیموکو
            tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
            kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
            chikou_span = close.shift(-26)
            
            # مقادیر فعلی
            current_tenkan = tenkan_sen.iloc[-1] if not pd.isna(tenkan_sen.iloc[-1]) else 0
            current_kijun = kijun_sen.iloc[-1] if not pd.isna(kijun_sen.iloc[-1]) else 0
            current_senkou_a = senkou_span_a.iloc[-1] if not pd.isna(senkou_span_a.iloc[-1]) else 0
            current_senkou_b = senkou_span_b.iloc[-1] if not pd.isna(senkou_span_b.iloc[-1]) else 0
            current_chikou = chikou_span.iloc[-1] if not pd.isna(chikou_span.iloc[-1]) else 0
            current_close = close.iloc[-1]
            
            # بررسی موقعیت قیمت نسبت به ابر
            price_above_kumo = current_close > max(current_senkou_a, current_senkou_b)
            
            return {
                'tenkan_sen': current_tenkan,
                'kijun_sen': current_kijun,
                'senkou_span_a': current_senkou_a,
                'senkou_span_b': current_senkou_b,
                'chikou_span': current_chikou,
                'price_above_kumo': price_above_kumo
            }
        except Exception as e:
            logger.error(f"Error in Ichimoku analysis: {e}")
            return {}
    
    def wyckoff_analysis(self, data):
        """تحلیل ویچاف"""
        if data.empty:
            return {}
        
        try:
            close = data['Close']
            volume = data['Volume']
            
            # تحلیل ساده ویچاف
            recent_close = close.iloc[-1]
            avg_volume = volume.rolling(window=20).mean().iloc[-1]
            current_volume = volume.iloc[-1]
            
            # شناسایی فازها
            accumulation_phase = current_volume > avg_volume * 1.2 and recent_close > close.iloc[-5]
            distribution_phase = current_volume > avg_volume * 1.2 and recent_close < close.iloc[-5]
            
            return {
                'phase': 'انباشت' if accumulation_phase else 'توزیع' if distribution_phase else 'خنثی',
                'accumulation_phase': accumulation_phase,
                'distribution_phase': distribution_phase,
                'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1
            }
        except Exception as e:
            logger.error(f"Error in Wyckoff analysis: {e}")
            return {}
    
    def volume_profile_analysis(self, data):
        """تحلیل پروفایل حجمی"""
        if data.empty:
            return {}
        
        try:
            close = data['Close']
            volume = data['Volume']
            
            # محاسبه پروفایل حجمی ساده
            price_levels = np.linspace(close.min(), close.max(), 20)
            volume_profile = {}
            
            for i in range(len(price_levels) - 1):
                lower = price_levels[i]
                upper = price_levels[i + 1]
                mask = (close >= lower) & (close < upper)
                total_volume = volume[mask].sum()
                volume_profile[float(f"{(lower + upper) / 2:.2f}")] = total_volume
            
            # پیدا کردن POC (Point of Control)
            poc_price = max(volume_profile, key=volume_profile.get) if volume_profile else close.iloc[-1]
            
            # محاسبه محدوده ارزش (Value Area)
            total_volume = sum(volume_profile.values())
            if total_volume > 0:
                cumulative_volume = 0
                value_area_low = None
                value_area_high = None
                
                for price in sorted(volume_profile.keys()):
                    cumulative_volume += volume_profile[price]
                    if cumulative_volume >= total_volume * 0.3 and value_area_low is None:
                        value_area_low = price
                    if cumulative_volume >= total_volume * 0.7:
                        value_area_high = price
                        break
            else:
                value_area_low = close.min()
                value_area_high = close.max()
            
            return {
                'poc': {'price_level': poc_price},
                'value_area_low': value_area_low,
                'value_area_high': value_area_high,
                'volume_profile': volume_profile
            }
        except Exception as e:
            logger.error(f"Error in volume profile analysis: {e}")
            return {}
    
    def fibonacci_analysis(self, data):
        """تحلیل فیبوناچی"""
        if data.empty:
            return {}
        
        try:
            high = data['High'].max()
            low = data['Low'].min()
            
            # محاسبه سطوح فیبوناچی
            diff = high - low
            levels = {
                '0%': low,
                '23.6%': low + diff * 0.236,
                '38.2%': low + diff * 0.382,
                '50%': low + diff * 0.5,
                '61.8%': low + diff * 0.618,
                '78.6%': low + diff * 0.786,
                '100%': high
            }
            
            current_price = data['Close'].iloc[-1]
            
            # بررسی موقعیت فعلی قیمت
            nearest_level = None
            min_distance = float('inf')
            
            for level, price in levels.items():
                distance = abs(current_price - price)
                if distance < min_distance:
                    min_distance = distance
                    nearest_level = level
            
            return {
                'levels': levels,
                'current_price': current_price,
                'nearest_level': nearest_level,
                'high': high,
                'low': low
            }
        except Exception as e:
            logger.error(f"Error in Fibonacci analysis: {e}")
            return {}
    
    def harmonic_patterns_analysis(self, data):
        """تحلیل الگوهای هارمونیک"""
        if data.empty:
            return {}
        
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # پیدا کردن نقاط چرخش (پیک و دره)
            peaks, _ = find_peaks(high.values, distance=5)
            troughs, _ = find_peaks(-low.values, distance=5)
            
            # ترکیب نقاط
            pivot_points = sorted(list(peaks) + list(troughs))
            
            patterns_found = []
            
            # بررسی الگوهای هارمونیک
            for pattern_name, ratios in self.harmonic_patterns.items():
                if len(pivot_points) >= 4:
                    # محاسبه نسبت‌ها
                    XA = high.iloc[pivot_points[0]] - low.iloc[pivot_points[1]]
                    AB = high.iloc[pivot_points[1]] - low.iloc[pivot_points[2]]
                    BC = high.iloc[pivot_points[2]] - low.iloc[pivot_points[3]]
                    
                    # بررسی نسبت‌ها
                    ab_ratio = AB / XA if XA != 0 else 0
                    bc_ratio = BC / AB if AB != 0 else 0
                    
                    # بررسی تطابق با الگو
                    if (abs(ab_ratio - ratios['AB']) < 0.1 and 
                        abs(bc_ratio - ratios['BC']) < 0.1):
                        patterns_found.append({
                            'pattern': pattern_name,
                            'type': 'صعودی' if XA > 0 else 'نزولی',
                            'confidence': 1 - (abs(ab_ratio - ratios['AB']) + abs(bc_ratio - ratios['BC'])) / 2
                        })
            
            return {
                'pattern_count': len(patterns_found),
                'patterns_found': patterns_found
            }
        except Exception as e:
            logger.error(f"Error in harmonic patterns analysis: {e}")
            return {}
    
    def support_resistance_analysis(self, data):
        """تحلیل حمایت و مقاومت"""
        if data.empty:
            return {}
        
        try:
            high = data['High']
            low = data['Low']
            
            # پیدا کردن سطوح حمایت و مقاومت
            peaks, _ = find_peaks(high.values, distance=5)
            troughs, _ = find_peaks(-low.values, distance=5)
            
            # محاسبه سطوح
            resistance_levels = [high.iloc[i] for i in peaks]
            support_levels = [low.iloc[i] for i in troughs]
            
            # فیلتر کردن سطوح نزدیک به هم
            resistance_levels = self.filter_levels(resistance_levels)
            support_levels = self.filter_levels(support_levels)
            
            current_price = data['Close'].iloc[-1]
            
            return {
                'resistance': resistance_levels,
                'support': support_levels,
                'current_price': current_price,
                'nearest_resistance': min([r for r in resistance_levels if r > current_price], default=None),
                'nearest_support': max([s for s in support_levels if s < current_price], default=None)
            }
        except Exception as e:
            logger.error(f"Error in support resistance analysis: {e}")
            return {}
    
    def filter_levels(self, levels, threshold=0.02):
        """فیلتر کردن سطوح نزدیک به هم"""
        if not levels:
            return []
        
        filtered = []
        levels_sorted = sorted(levels)
        
        for level in levels_sorted:
            if not filtered or abs(level - filtered[-1]) / filtered[-1] > threshold:
                filtered.append(level)
        
        return filtered
    
    def trend_lines_analysis(self, data):
        """تحلیل خطوط روند"""
        if data.empty:
            return {}
        
        try:
            close = data['Close']
            
            # محاسبه شیب خط روند
            x = np.arange(len(close))
            y = close.values
            
            # رگرسیون خطی برای پیدا کردن شیب
            slope, intercept = np.polyfit(x, y, 1)
            
            # تعیین نوع روند
            trend_type = 'صعودی' if slope > 0 else 'نزولی'
            
            # محاسبه قدرت روند
            trend_strength = abs(slope) / close.mean() * 100 if close.mean() > 0 else 0
            
            return {
                'trend_type': trend_type,
                'slope': slope,
                'intercept': intercept,
                'strength': trend_strength,
                'r_squared': np.corrcoef(x, y)[0, 1]**2
            }
        except Exception as e:
            logger.error(f"Error in trend lines analysis: {e}")
            return {}
    
    def order_flow_analysis(self, data):
        """تحلیل جریان سفارش"""
        if data.empty:
            return {}
        
        try:
            close = data['Close']
            volume = data['Volume']
            
            # محاسبه حجم خرید و فروش (تخمینی)
            price_change = close.diff()
            buy_volume = volume[price_change > 0].sum()
            sell_volume = volume[price_change < 0].sum()
            
            # محاسبه نسبت خرید به فروش
            buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else float('inf')
            
            return {
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'buy_sell_ratio': buy_sell_ratio,
                'net_volume': buy_volume - sell_volume,
                'volume_pressure': 'خرید' if buy_volume > sell_volume else 'فروش'
            }
        except Exception as e:
            logger.error(f"Error in order flow analysis: {e}")
            return {}
    
    def vwap_analysis(self, data):
        """تحلیل VWAP"""
        if data.empty:
            return {}
        
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            volume = data['Volume']
            
            # محاسبه VWAP
            typical_price = (high + low + close) / 3
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            
            current_vwap = vwap.iloc[-1] if not pd.isna(vwap.iloc[-1]) else close.iloc[-1]
            current_price = close.iloc[-1]
            
            return {
                'vwap': current_vwap,
                'current_price': current_price,
                'price_above_vwap': current_price > current_vwap,
                'deviation': (current_price - current_vwap) / current_vwap * 100 if current_vwap > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error in VWAP analysis: {e}")
            return {}
    
    def pivot_points_analysis(self, data):
        """تحلیل پیوت پوینت‌ها"""
        if data.empty:
            return {}
        
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # محاسبه پیوت پوینت استاندارد
            pivot = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
            
            # محاسبه سطوح حمایت و مقاومت
            r1 = 2 * pivot - low.iloc[-1]
            s1 = 2 * pivot - high.iloc[-1]
            r2 = pivot + (high.iloc[-1] - low.iloc[-1])
            s2 = pivot - (high.iloc[-1] - low.iloc[-1])
            r3 = high.iloc[-1] + 2 * (pivot - low.iloc[-1])
            s3 = low.iloc[-1] - 2 * (high.iloc[-1] - pivot)
            
            current_price = close.iloc[-1]
            
            return {
                'pivot': pivot,
                'resistance': {'r1': r1, 'r2': r2, 'r3': r3},
                'support': {'s1': s1, 's2': s2, 's3': s3},
                'current_price': current_price,
                'position': 'بالا' if current_price > pivot else 'پایین'
            }
        except Exception as e:
            logger.error(f"Error in pivot points analysis: {e}")
            return {}
    
    def advanced_candlestick_patterns(self, data):
        """تحلیل الگوهای شمعی پیشرفته"""
        if data.empty:
            return {}
        
        try:
            open_price = data['Open'].values
            high = data['High'].values
            low = data['Low'].values
            close = data['Close'].values
            
            patterns_found = []
            
            # بررسی الگوهای شمعی پیشرفته
            for pattern_name, description in self.advanced_candlesticks.items():
                pattern_func = getattr(talib, f'CDL{pattern_name.upper()}', None)
                if pattern_func:
                    pattern_result = pattern_func(open_price, high, low, close)
                    if pattern_result[-1] != 0:
                        patterns_found.append({
                            'pattern': pattern_name,
                            'description': description,
                            'strength': abs(pattern_result[-1]) / 100
                        })
            
            return {
                'patterns_count': len(patterns_found),
                'patterns_found': patterns_found
            }
        except Exception as e:
            logger.error(f"Error in advanced candlestick patterns: {e}")
            return {}
    
    def advanced_elliott_wave(self, data):
        """تحلیل امواج الیوت پیشرفته"""
        if data.empty:
            return {}
        
        try:
            close = data['Close']
            
            # تحلیل ساده امواج الیوت
            # پیدا کردن نقاط چرخش
            peaks, _ = find_peaks(close.values, distance=5)
            troughs, _ = find_peaks(-close.values, distance=5)
            
            # ترکیب نقاط
            pivot_points = sorted(list(peaks) + list(troughs))
            
            # شناسایی موج فعلی
            if len(pivot_points) >= 3:
                # محاسبه جهت موج
                wave_direction = 'صعودی' if close.iloc[pivot_points[-1]] > close.iloc[pivot_points[-2]] else 'نزولی'
                
                # تخمین شماره موج
                wave_count = len([p for p in pivot_points[-5:] if p in peaks]) if wave_direction == 'صعودی' else len([p for p in pivot_points[-5:] if p in troughs])
                
                return {
                    'current_pattern': f'موج {wave_count}',
                    'wave_direction': wave_direction,
                    'pivot_points': pivot_points[-5:],
                    'confidence': min(wave_count / 5, 1.0)
                }
            else:
                return {
                    'current_pattern': 'نامشخص',
                    'wave_direction': 'خنثی',
                    'pivot_points': [],
                    'confidence': 0
                }
        except Exception as e:
            logger.error(f"Error in advanced Elliott wave: {e}")
            return {}
    
    def market_structure_analysis(self, data):
        """تحلیل ساختار بازار"""
        if data.empty:
            return {}
        
        try:
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            
            # پیدا کردن نقاط چرخش
            peaks, _ = find_peaks(high.values, distance=5)
            troughs, _ = find_peaks(-low.values, distance=5)
            
            # شناسایی ساختار بازار
            market_trend = 'صعودی' if close.iloc[-1] > close.iloc[-20] else 'نزولی'
            
            # پیدا کردن Order Block‌ها
            order_blocks = []
            
            for i in range(1, len(close)):
                # بررسی برای Order Block صعودی
                if (close.iloc[i] > close.iloc[i-1] and 
                    close.iloc[i-1] < close.iloc[i-2] and 
                    volume.iloc[i] > volume.iloc[i-1]):
                    order_blocks.append({
                        'type': 'bullish',
                        'price': close.iloc[i-1],
                        'index': i-1
                    })
                
                # بررسی برای Order Block نزولی
                if (close.iloc[i] < close.iloc[i-1] and 
                    close.iloc[i-1] > close.iloc[i-2] and 
                    volume.iloc[i] > volume.iloc[i-1]):
                    order_blocks.append({
                        'type': 'bearish',
                        'price': close.iloc[i-1],
                        'index': i-1
                    })
            
            return {
                'market_trend': market_trend,
                'order_blocks': order_blocks[-5:],  # 5 Order Block آخر
                'swing_highs': [high.iloc[i] for i in peaks[-3:]],
                'swing_lows': [low.iloc[i] for i in troughs[-3:]]
            }
        except Exception as e:
            logger.error(f"Error in market structure analysis: {e}")
            return {}
    
    def analyze_multi_timeframe(self, symbol):
        """تحلیل چند زمانی"""
        try:
            # دریافت داده‌ها در تایم‌فریم‌های مختلف
            timeframes = {
                '1h': '60m',
                '4h': '4h',
                '1d': '1d'
            }
            
            multi_tf_analysis = {}
            
            for tf_name, tf_value in timeframes.items():
                try:
                    tf_data = self.get_historical_data(symbol, period='60d', interval=tf_value)
                    if not tf_data.empty:
                        # تحلیل ساده روند
                        close = tf_data['Close']
                        sma_20 = talib.SMA(close, timeperiod=20)
                        sma_50 = talib.SMA(close, timeperiod=50)
                        
                        trend = 'صعودی' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'نزولی'
                        
                        multi_tf_analysis[tf_name] = {
                            'trend': trend,
                            'price': close.iloc[-1],
                            'sma_20': sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else 0,
                            'sma_50': sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else 0
                        }
                except Exception as e:
                    logger.error(f"Error in {tf_name} timeframe analysis: {e}")
            
            return multi_tf_analysis
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            return {}
    
    def analyze_trading_session(self, symbol):
        """تحلیل جلسه معاملاتی"""
        try:
            # دریافت داده‌های اخیر
            data = self.get_historical_data(symbol, period='5d')
            if data.empty:
                return {}
            
            # تقسیم داده‌ها بر اساس جلسات معاملاتی
            data.index = pd.to_datetime(data.index)
            data['hour'] = data.index.hour
            
            # تعریف جلسات معاملاتی
            sessions = {
                'Asia': (0, 8),
                'Europe': (8, 16),
                'America': (16, 24)
            }
            
            session_analysis = {}
            
            for session_name, (start_hour, end_hour) in sessions.items():
                session_data = data[(data['hour'] >= start_hour) & (data['hour'] < end_hour)]
                
                if not session_data.empty:
                    session_analysis[session_name] = {
                        'high': session_data['High'].max(),
                        'low': session_data['Low'].min(),
                        'volume': session_data['Volume'].sum(),
                        'volatility': (session_data['High'].max() - session_data['Low'].min()) / session_data['Close'].mean() * 100
                    }
            
            return session_analysis
        except Exception as e:
            logger.error(f"Error in trading session analysis: {e}")
            return {}
    
    def analyze_decision_zones(self, data):
        """تحلیل نواحی تصمیم‌گیری"""
        if data.empty:
            return {}
        
        try:
            close = data['Close']
            volume = data['Volume']
            
            # محاسبه میانگین متحرک حجم
            volume_ma = volume.rolling(window=20).mean()
            
            # پیدا کردن نواحی با حجم بالا
            high_volume_zones = data[volume > volume_ma * 1.5]
            
            # گروه‌بندی نواحی نزدیک به هم
            decision_zones = []
            
            if not high_volume_zones.empty:
                # مرتب‌سازی بر اساس قیمت
                sorted_zones = high_volume_zones.sort_values('Close')
                
                current_zone = {
                    'low': sorted_zones.iloc[0]['Close'],
                    'high': sorted_zones.iloc[0]['Close'],
                    'volume': sorted_zones.iloc[0]['Volume']
                }
                
                for _, row in sorted_zones.iloc[1:].iterrows():
                    if row['Close'] - current_zone['high'] < current_zone['high'] * 0.02:
                        current_zone['high'] = row['Close']
                        current_zone['volume'] += row['Volume']
                    else:
                        decision_zones.append(current_zone)
                        current_zone = {
                            'low': row['Close'],
                            'high': row['Close'],
                            'volume': row['Volume']
                        }
                
                decision_zones.append(current_zone)
            
            current_price = close.iloc[-1]
            
            # پیدا کردن نزدیک‌ترین ناحیه تصمیم‌گیری
            nearest_zone = None
            min_distance = float('inf')
            
            for zone in decision_zones:
                if zone['low'] <= current_price <= zone['high']:
                    nearest_zone = zone
                    break
                else:
                    distance = min(abs(current_price - zone['low']), abs(current_price - zone['high']))
                    if distance < min_distance:
                        min_distance = distance
                        nearest_zone = zone
            
            return {
                'decision_zones': decision_zones,
                'nearest_zone': nearest_zone,
                'current_price': current_price,
                'in_decision_zone': nearest_zone and nearest_zone['low'] <= current_price <= nearest_zone['high']
            }
        except Exception as e:
            logger.error(f"Error in decision zones analysis: {e}")
            return {}
    
    def analyze_risk_management(self, data, market_data):
        """تحلیل مدیریت ریسک"""
        if data.empty or not market_data:
            return {}
        
        try:
            close = data['Close']
            current_price = market_data.get('price', close.iloc[-1])
            
            # محاسبه ATR (Average True Range)
            high = data['High']
            low = data['Low']
            atr = talib.ATR(high, low, close, timeperiod=14)
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
            
            # محاسبه نوسانات
            returns = close.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # نوسانات سالانه
            
            # محاسبه سطوح حد ضرر و سود
            stop_loss = current_price - current_atr * 2
            take_profit = current_price + current_atr * 3
            
            # محاسبه نسبت ریسک به پاداش
            risk_reward_ratio = (take_profit - current_price) / (current_price - stop_loss) if stop_loss < current_price else 0
            
            return {
                'atr': current_atr,
                'volatility': volatility,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': risk_reward_ratio,
                'position_size': 0.02 / (current_atr / current_price) if current_atr > 0 and current_price > 0 else 0  # 2% ریسک
            }
        except Exception as e:
            logger.error(f"Error in risk management analysis: {e}")
            return {}
    
    def perform_ai_analysis(self, historical_data, market_data, sentiment, economic_sentiment):
        """تحلیل هوش مصنوعی"""
        try:
            if historical_data.empty:
                return {}
            
            close = historical_data['Close']
            
            # آماده‌سازی داده‌ها برای مدل‌ها
            X = []
            y = []
            
            # ایجاد ویژگی‌ها
            for i in range(20, len(close)):
                features = [
                    close.iloc[i-1],
                    close.iloc[i-5],
                    close.iloc[i-10],
                    close.iloc[i-20],
                    (close.iloc[i-1] - close.iloc[i-20]) / close.iloc[i-20]  # تغییر 20 روزه
                ]
                X.append(features)
                y.append(close.iloc[i])
            
            if len(X) < 10:
                return {}
            
            X = np.array(X)
            y = np.array(y)
            
            # تقسیم داده‌ها
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            predictions = {}
            
            # اجرای مدل‌های مختلف
            for model_name, model in self.models.items():
                try:
                    if model is not None:
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test[-1:].reshape(1, -1))[0]
                        predictions[model_name] = pred
                except Exception as e:
                    logger.error(f"Error in {model_name} prediction: {e}")
            
            # محاسبه میانگین پیش‌بینی‌ها
            if predictions:
                final_prediction = np.mean(list(predictions.values()))
                confidence = 1 - np.std(list(predictions.values())) / final_prediction if final_prediction > 0 else 0
            else:
                final_prediction = close.iloc[-1]
                confidence = 0
            
            return {
                'predictions': predictions,
                'final_prediction': final_prediction,
                'confidence': min(confidence, 1.0),
                'current_price': close.iloc[-1],
                'prediction_change': (final_prediction - close.iloc[-1]) / close.iloc[-1] * 100
            }
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return {}
    
    def advanced_supply_demand(self, symbol):
        """تحلیل پیشرفته عرضه و تقاضا"""
        try:
            # دریافت داده‌های اخیر
            data = self.get_historical_data(symbol, period='30d')
            if data.empty:
                return {}
            
            close = data['Close']
            volume = data['Volume']
            
            # محاسبه شاخص‌های عرضه و تقاضا
            # 1. حجم در نقاط چرخش
            peaks, _ = find_peaks(close.values, distance=5)
            troughs, _ = find_peaks(-close.values, distance=5)
            
            supply_volume = volume.iloc[peaks].mean() if len(peaks) > 0 else 0
            demand_volume = volume.iloc[troughs].mean() if len(troughs) > 0 else 0
            
            # 2. تحلیل فشار خرید/فروش
            price_change = close.diff()
            buy_pressure = volume[price_change > 0].sum()
            sell_pressure = volume[price_change < 0].sum()
            
            # 3. تحلیل تعادل عرضه و تقاضا
            imbalance = (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure) if (buy_pressure + sell_pressure) > 0 else 0
            
            return {
                'supply_volume': supply_volume,
                'demand_volume': demand_volume,
                'buy_pressure': buy_pressure,
                'sell_pressure': sell_pressure,
                'imbalance': imbalance,
                'market_condition': 'تقاضا' if imbalance > 0.1 else 'عرضه' if imbalance < -0.1 else 'تعادل'
            }
        except Exception as e:
            logger.error(f"Error in advanced supply demand analysis: {e}")
            return {}
    
    def calculate_signal_score(self, analysis):
        """محاسبه امتیاز سیگنال نهایی"""
        try:
            score = 0.5  # امتیاز پایه
            
            # تحلیل تکنیکال
            technical = analysis.get('technical', {}).get('classical', {})
            if 'rsi' in technical:
                rsi = technical['rsi'].get('14', 50)
                if rsi < 30:
                    score += 0.15  # سیگنال خرید
                elif rsi > 70:
                    score -= 0.15  # سیگنال فروش
            
            if 'macd' in technical:
                macd = technical['macd']
                if macd.get('macd', 0) > macd.get('signal', 0):
                    score += 0.1  # سیگنال خرید
                else:
                    score -= 0.1  # سیگنال فروش
            
            # تحلیل احساسات
            sentiment = analysis.get('sentiment', {})
            avg_sentiment = sentiment.get('average_sentiment', 0)
            score += avg_sentiment * 0.2
            
            # تحلیل اقتصادی
            economic_sentiment = analysis.get('economic_sentiment', {})
            avg_economic_sentiment = economic_sentiment.get('average_sentiment', 0)
            score += avg_economic_sentiment * 0.1
            
            # تحلیل هوش مصنوعی
            ai_analysis = analysis.get('ai_analysis', {})
            if 'prediction_change' in ai_analysis:
                prediction_change = ai_analysis['prediction_change']
                score += prediction_change / 100 * 0.3
            
            # تحلیل ساختار بازار
            market_structure = analysis.get('market_structure', {})
            if market_structure.get('market_trend') == 'صعودی':
                score += 0.1
            elif market_structure.get('market_trend') == 'نزولی':
                score -= 0.1
            
            # تحلیل چند زمانی
            multi_timeframe = analysis.get('multi_timeframe', {})
            if multi_timeframe:
                bullish_count = sum(1 for tf in multi_timeframe.values() if tf.get('trend') == 'صعودی')
                bearish_count = sum(1 for tf in multi_timeframe.values() if tf.get('trend') == 'نزولی')
                
                if bullish_count > bearish_count:
                    score += 0.1
                elif bearish_count > bullish_count:
                    score -= 0.1
            
            # تحلیل عرضه و تقاضا
            supply_demand = analysis.get('supply_demand', {})
            imbalance = supply_demand.get('imbalance', 0)
            score += imbalance * 0.15
            
            # محدود کردن امتیاز بین 0 و 1
            return max(0, min(1, score))
        except Exception as e:
            logger.error(f"Error calculating signal score: {e}")
            return 0.5

# اجرای برنامه
if __name__ == "__main__":
    import os
    from telegram.ext import Application
    
    # ایجاد نمونه از ربات
    bot = AdvancedTradingBot()
    
    # ساخت اپلیکیشن تلگرام
    application = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()
    
    # تنظیم هندلرها
    from telegram_handlers import setup_handlers
    setup_handlers(application, bot)
    
    # اجرای ربات
    application.run_polling()