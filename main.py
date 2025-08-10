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
    'psycopg2': False
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
            'coinmarketcap': os.getenv('COINMARKETCAP_API_KEY')
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
        
        logger.info("AdvancedTradingBot initialized successfully")
    
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
            'bybit': ccxt.bybit(PROXY_SETTINGS)
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
            'rally', 'surge', 'boom', 'breakthrough', 'upgrade', 'adoption', 'partnership'
        ]
        
        self.negative_keywords = [
            'نزول', 'کاهش', 'افت', 'ضرر', 'پایین', 'فروش', 'bearish', 'decrease', 'drop', 
            'loss', 'low', 'sell', 'negative', 'pessimistic', 'bear', 'crash', 'dump', 
            'decline', 'fall', 'slump', 'recession', 'risk', 'warning', 'fraud', 'hack'
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
            'descending_triangle': ['مثلث نزولی', 'ادامه روند نزولی']
        }
        
        # استراتژی‌های معاملاتی
        self.trading_strategies = {
            'scalping': {
                'description': 'اسکالپینگ - معاملات کوتاه مدت',
                'timeframe': '1m-15m',
                'risk_level': 'بالا',
                'profit_target': '0.5-1%'
            },
            'day_trading': {
                'description': 'معامله روزانه',
                'timeframe': '15m-4h',
                'risk_level': 'متوسط',
                'profit_target': '1-3%'
            },
            'swing_trading': {
                'description': 'سوینگ تریدینگ',
                'timeframe': '4h-1d',
                'risk_level': 'متوسط',
                'profit_target': '3-10%'
            },
            'position_trading': {
                'description': 'معامله پوزیشنی',
                'timeframe': '1d-1w',
                'risk_level': 'پایین',
                'profit_target': '10%+'
            }
        }
    
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
        
        # جدول داده‌های اقتصادی (NFP, CPI, FOMC)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS economic_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            event_name TEXT,
            event_date TIMESTAMP,
            impact_level TEXT,
            actual_value REAL,
            forecast_value REAL,
            previous_value REAL,
            currency_affected TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # جدول اردربلاک‌ها
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS order_blocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timeframe TEXT,
            block_type TEXT,  -- 'buy' or 'sell'
            price REAL,
            volume REAL,
            timestamp TIMESTAMP,
            is_valid BOOLEAN,
            confidence REAL
        )
        ''')
        
        # جدول نواحی عرضه و تقاضا
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS supply_demand_zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timeframe TEXT,
            zone_type TEXT,  -- 'supply' or 'demand'
            price_high REAL,
            price_low REAL,
            timestamp TIMESTAMP,
            strength REAL
        )
        ''')
        
        # جدول سشن‌های معاملاتی
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trading_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT,  -- 'Asian', 'European', 'American'
            start_time TIME,
            end_time TIME,
            timezone TEXT,
            volatility_level REAL
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
        
        # تلاش برای دریافت داده از DEX Screener
        try:
            data['dexscreener'] = await self.fetch_dexscreener_data(symbol)
        except Exception as e:
            logger.error(f"Error fetching from DEX Screener: {e}")
            data['dexscreener'] = {}
        
        # تلاش برای دریافت داده از TradingView (وب اسکرپینگ)
        try:
            data['tradingview'] = await self.fetch_tradingview_data(symbol)
        except Exception as e:
            logger.error(f"Error fetching from TradingView: {e}")
            data['tradingview'] = {}
        
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
    
    async def fetch_dexscreener_data(self, symbol):
        """دریافت داده‌ها از DEX Screener"""
        async with aiohttp.ClientSession() as session:
            url = f"https://api.dexscreener.com/latest/dex/search?q={symbol}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data['pairs']:
                        return data['pairs'][0]
                return {}
    
    async def fetch_tradingview_data(self, symbol):
        """دریافت داده‌ها از TradingView با وب اسکرپینگ"""
        try:
            url = f"https://www.tradingview.com/symbols/{symbol}/"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = self.session.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # استخراج داده‌ها با استفاده از الگوهای مشخص
                script_tags = soup.find_all('script')
                for script in script_tags:
                    if 'window.initData' in str(script):
                        # استخراج داده‌های JSON از اسکریپت
                        data_str = str(script)
                        start = data_str.find('{')
                        end = data_str.rfind('}') + 1
                        if start != -1 and end != -1:
                            json_data = json.loads(data_str[start:end])
                            return json_data
            return {}
        except Exception as e:
            logger.error(f"Error scraping TradingView: {e}")
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
        return f"{symbol}/USDT"    async def fetch_news_from_multiple_sources(self, symbol=None):
        """دریافت اخبار از چندین منبع"""
        news = []
        
        # دریافت اخبار از CryptoPanic
        try:
            news.extend(await self.fetch_cryptopanic_news(symbol))
        except Exception as e:
            logger.error(f"Error fetching from CryptoPanic: {e}")
        
        # دریافت اخبار از NewsAPI
        try:
            news.extend(await self.fetch_newsapi_news(symbol))
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
        
        # دریافت اخبار از CoinGecko
        try:
            news.extend(await self.fetch_coingecko_news(symbol))
        except Exception as e:
            logger.error(f"Error fetching from CoinGecko: {e}")
        
        # دریافت اخبار اقتصادی (NFP, CPI, FOMC)
        try:
            news.extend(await self.fetch_economic_news())
        except Exception as e:
            logger.error(f"Error fetching economic news: {e}")
        
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
    
    async def fetch_newsapi_news(self, symbol=None):
        """دریافت اخبار از NewsAPI"""
        if not self.api_keys['news']:
            return []
        
        async with aiohttp.ClientSession() as session:
            url = "https://newsapi.org/v2/everything"
            params = {
                'apiKey': self.api_keys['news'],
                'q': 'cryptocurrency OR bitcoin OR ethereum' if not symbol else symbol,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 10
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        {
                            'title': item['title'],
                            'content': item['description'],
                            'source': item['source']['name'],
                            'url': item['url'],
                            'published_at': datetime.strptime(item['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
                            'symbols': [symbol] if symbol else []
                        }
                        for item in data['articles']
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
        """دریافت اخبار اقتصادی (NFP, CPI, FOMC)"""
        # در یک پیاده‌سازی واقعی، این باید از APIهای اقتصادی استفاده کند
        # اینجا یک پیاده‌سازی ساده شده ارائه می‌شود
        
        economic_events = [
            {
                'title': 'FOMC Meeting Minutes',
                'content': 'Federal Open Market Committee meeting minutes released',
                'source': 'Federal Reserve',
                'url': 'https://www.federalreserve.gov',
                'published_at': datetime.now(),
                'symbols': ['USD', 'BTC', 'ETH'],
                'event_type': 'FOMC',
                'impact': 'high'
            },
            {
                'title': 'Non-Farm Payrolls Report',
                'content': 'Latest non-farm payrolls data shows unexpected results',
                'source': 'Bureau of Labor Statistics',
                'url': 'https://www.bls.gov',
                'published_at': datetime.now(),
                'symbols': ['USD', 'BTC', 'ETH'],
                'event_type': 'NFP',
                'impact': 'high'
            },
            {
                'title': 'CPI Inflation Data',
                'content': 'Consumer Price Index inflation data released',
                'source': 'Bureau of Labor Statistics',
                'url': 'https://www.bls.gov',
                'published_at': datetime.now(),
                'symbols': ['USD', 'BTC', 'ETH'],
                'event_type': 'CPI',
                'impact': 'high'
            }
        ]
        
        return economic_events
    
    async def advanced_sentiment_analysis(self, news_items):
        """تحلیل احساسات پیشرفته با هوش مصنوعی داخلی"""
        if not news_items:
            return {
                'average_sentiment': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'topics': [],
                'trends': [],
                'economic_impact': []
            }
        
        sentiments = []
        topics = []
        all_text = ""
        economic_impacts = []
        
        for news in news_items:
            text = f"{news['title']} {news['content']}"
            all_text += text + " "
            
            # تحلیل احساسات پیشرفته
            sentiment_score = self.analyze_text_sentiment(text)
            sentiments.append(sentiment_score)
            
            # استخراج موضوعات
            news_topics = self.extract_topics(text)
            topics.extend(news_topics)
            
            # تحلیل تاثیر اقتصادی
            if 'event_type' in news:
                impact = self.analyze_economic_impact(news)
                economic_impacts.append(impact)
        
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
                'trends': trends,
                'economic_impact': economic_impacts
            }
        return {'average_sentiment': 0, 'positive_count': 0, 'negative_count': 0, 'neutral_count': 0, 'topics': [], 'trends': [], 'economic_impact': []}
    
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
    
    def analyze_economic_impact(self, news_item):
        """تحلیل تاثیر رویدادهای اقتصادی"""
        event_type = news_item.get('event_type', '')
        impact = news_item.get('impact', 'medium')
        
        # تعیین تاثیر بر بازار ارزهای دیجیتال
        if event_type == 'FOMC':
            return {
                'event': 'FOMC',
                'impact': impact,
                'effect': 'High volatility expected, especially for USD pairs',
                'affected_pairs': ['BTC/USD', 'ETH/USD', 'XRP/USD']
            }
        elif event_type == 'NFP':
            return {
                'event': 'NFP',
                'impact': impact,
                'effect': 'Strong impact on USD and correlated assets',
                'affected_pairs': ['BTC/USD', 'ETH/USD', 'SOL/USD']
            }
        elif event_type == 'CPI':
            return {
                'event': 'CPI',
                'impact': impact,
                'effect': 'Inflation data affects all risk assets',
                'affected_pairs': ['BTC/USD', 'ETH/USD', 'ADA/USD']
            }
        
        return {
            'event': event_type,
            'impact': impact,
            'effect': 'Unknown economic event',
            'affected_pairs': []
        }
    
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
                if 'usd' in data and data['usd']:
                    prices.append(data['usd'])
                if 'usd_market_cap' in data and data['usd_market_cap']:
                    market_caps.append(data['usd_market_cap'])
                if 'usd_24h_vol' in data and data['usd_24h_vol']:
                    volumes.append(data['usd_24h_vol'])
                if 'usd_24h_change' in data and data['usd_24h_change']:
                    changes.append(data['usd_24h_change'])
        
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
            
            # تحلیل احساسات اخبار
            try:
                sentiment = await self.advanced_sentiment_analysis(news)
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {e}")
                sentiment = {'average_sentiment': 0, 'topics': [], 'economic_impact': []}
            
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
            
            # تحلیل اردربلاک‌ها
            try:
                order_blocks = self.analyze_order_blocks(historical_data)
            except Exception as e:
                logger.error(f"Error in order block analysis: {e}")
                order_blocks = []
            
            # تحلیل نواحی تصمیم‌گیری
            try:
                decision_zones = self.analyze_decision_zones(historical_data)
            except Exception as e:
                logger.error(f"Error in decision zone analysis: {e}")
                decision_zones = []
            
            # تحلیل سشن‌های معاملاتی
            try:
                session_analysis = self.analyze_trading_sessions(historical_data)
            except Exception as e:
                logger.error(f"Error in session analysis: {e}")
                session_analysis = {}
            
            # تحلیل هوش مصنوعی
            try:
                ai_analysis = self.perform_ai_analysis(historical_data, market_data, sentiment)
            except Exception as e:
                logger.error(f"Error in AI analysis: {e}")
                ai_analysis = {}
            
            # ترکیب همه تحلیل‌ها
            combined_analysis = {
                'symbol': symbol,
                'market_data': market_data,
                'sentiment': sentiment,
                'technical': technical_analysis,
                'elliott': elliott_analysis,
                'supply_demand': supply_demand,
                'order_blocks': order_blocks,
                'decision_zones': decision_zones,
                'session_analysis': session_analysis,
                'ai_analysis': ai_analysis,
                'news_count': len(news),
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
                'patterns': {},
                'multi_timeframe': {}
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
            
            # Bollinger Bands
            if LIBRARIES['talib']:
                upper, middle, lower = talib.BBANDS(data['Close'], timeperiod=20)
                result['classical']['bollinger'] = {
                    'upper': upper[-1],
                    'middle': middle[-1],
                    'lower': lower[-1]
                }
            
            # تحلیل روند
            if len(data) >= 50:
                # محاسبه میانگین متحرک‌ها
                sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
                sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
                
                # تعیین روند
                if sma_20 > sma_50:
                    trend_direction = 'صعودی'
                elif sma_20 < sma_50:
                    trend_direction = 'نزولی'
                else:
                    trend_direction = 'خنثی'
                
                result['classical']['trend'] = {
                    'direction': trend_direction,
                    'sma_20': sma_20,
                    'sma_50': sma_50
                }
            
            # شناسایی الگوهای شمعی
            if LIBRARIES['talib']:
                # الگوهای شمعی کلیدی
                patterns = {
                    'CDLDOJI': talib.CDLDOJI(data['Open'], data['High'], data['Low'], data['Close'])[-1],
                    'CDLENGULFING': talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])[-1],
                    'CDLHAMMER': talib.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])[-1],
                    'CDLSHOOTINGSTAR': talib.CDLSHOOTINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])[-1]
                }
                
                # فیلتر الگوهای معتبر
                valid_patterns = {pattern: value for pattern, value in patterns.items() if value != 0}
                
                if valid_patterns:
                    result['patterns'] = valid_patterns
            
            # تحلیل چند تایم فریم
            result['multi_timeframe'] = self.analyze_multiple_timeframes(data)
            
            return result
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {}
    
    def analyze_multiple_timeframes(self, data):
        """تحلیل چند تایم فریم"""
        try:
            multi_timeframe = {}
            
            # تحلیل تایم فریم‌های مختلف
            timeframes = {
                '1h': self.resample_data(data, '1H'),
                '4h': self.resample_data(data, '4H'),
                '1d': self.resample_data(data, '1D'),
                '1w': self.resample_data(data, '1W')
            }
            
            for tf, tf_data in timeframes.items():
                if not tf_data.empty:
                    # محاسبه RSI برای هر تایم فریم
                    delta = tf_data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    # محاسبه روند
                    sma_20 = tf_data['Close'].rolling(window=20).mean().iloc[-1]
                    sma_50 = tf_data['Close'].rolling(window=50).mean().iloc[-1] if len(tf_data) >= 50 else tf_data['Close'].iloc[-1]
                    
                    trend = 'صعودی' if sma_20 > sma_50 else 'نزولی' if sma_20 < sma_50 else 'خنثی'
                    
                    multi_timeframe[tf] = {
                        'rsi': rsi.iloc[-1],
                        'trend': trend,
                        'price': tf_data['Close'].iloc[-1],
                        'volume': tf_data['Volume'].iloc[-1]
                    }
            
            return multi_timeframe
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            return {}
    
    def resample_data(self, data, timeframe):
        """تغییر تایم فریم داده‌ها"""
        try:
            if timeframe == '1H':
                return data.resample('1H').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            elif timeframe == '4H':
                return data.resample('4H').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            elif timeframe == '1D':
                return data.resample('1D').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            elif timeframe == '1W':
                return data.resample('1W').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            return data
        except Exception as e:
            logger.error(f"Error resampling data to {timeframe}: {e}")
            return pd.DataFrame()
    
    def advanced_elliott_wave(self, data):
        """تحلیل امواج الیوت"""
        if data.empty:
            return {'current_pattern': 'unknown'}
        
        try:
            # در یک پیاده‌سازی واقعی، این باید از الگوریتم‌های پیچیده‌تری استفاده کند
            # اینجا یک پیاده‌سازی ساده شده ارائه می‌شود
            
            # پیدا کردن قله‌ها و دره‌ها
            peaks, _ = find_peaks(data['Close'], distance=20)
            troughs, _ = find_peaks(-data['Close'], distance=20)
            
            # ترکیب و مرتب‌سازی قله‌ها و دره‌ها
            extrema = []
            for peak in peaks:
                extrema.append((peak, data['Close'].iloc[peak], 'peak'))
            for trough in troughs:
                extrema.append((trough, data['Close'].iloc[trough], 'trough'))
            
            extrema.sort(key=lambda x: x[0])
            
            # تحلیل امواج
            if len(extrema) >= 5:
                # الگوی 5 موجی صعودی
                if (extrema[0][2] == 'trough' and 
                    extrema[1][2] == 'peak' and 
                    extrema[2][2] == 'trough' and 
                    extrema[3][2] == 'peak' and 
                    extrema[4][2] == 'trough'):
                    
                    # بررسی قوانین الیوت
                    if (extrema[1][1] > extrema[3][1] and 
                        extrema[2][1] > extrema[0][1] and 
                        extrema[4][1] > extrema[2][1]):
                        return {'current_pattern': 'impulse_up', 'confidence': 0.7}
                
                # الگوی 3 موجی اصلاحی
                elif (extrema[0][2] == 'peak' and 
                      extrema[1][2] == 'trough' and 
                      extrema[2][2] == 'peak'):
                    
                    if (extrema[1][1] < extrema[0][1] and 
                        extrema[2][1] < extrema[0][1]):
                        return {'current_pattern': 'corrective_down', 'confidence': 0.6}
            
            return {'current_pattern': 'unknown', 'confidence': 0.3}
        except Exception as e:
            logger.error(f"Error in Elliott wave analysis: {e}")
            return {'current_pattern': 'unknown', 'error': str(e)}
    
    def advanced_supply_demand(self, symbol):
        """تحلیل عرضه و تقاضا"""
        try:
            # در یک پیاده‌سازی واقعی، این باید از داده‌های زنجیره بلوکی و سفارشات استفاده کند
            # اینجا یک پیاده‌سازی ساده شده ارائه می‌شود
            
            # دریافت داده‌های بازار
            market_data = asyncio.run(self.get_market_data(symbol))
            
            # محاسبه نسبت حجم به ارزش بازار
            volume_to_cap_ratio = market_data.get('volume_24h', 0) / market_data.get('market_cap', 1)
            
            # تحلیل عرضه و تقاضا
            if volume_to_cap_ratio > 0.1:
                imbalance = 'تقاضای بالا'
                score = 0.8
            elif volume_to_cap_ratio > 0.05:
                imbalance = 'تعادل'
                score = 0.5
            else:
                imbalance = 'عرضه بالا'
                score = 0.2
            
            return {
                'imbalance': imbalance,
                'score': score,
                'volume_to_cap_ratio': volume_to_cap_ratio
            }
        except Exception as e:
            logger.error(f"Error in supply demand analysis: {e}")
            return {'imbalance': 'unknown', 'score': 0.5, 'error': str(e)}
    
    def analyze_order_blocks(self, data):
        """تحلیل اردربلاک‌ها"""
        try:
            order_blocks = []
            
            if len(data) < 20:
                return order_blocks
            
            # شناسایی اردربلاک‌های خرید (Buy Side Liquidity)
            # اردربلاک خرید معمولاً پس از یک حرکت نزولی قوی شکل می‌گیرد
            for i in range(2, len(data)-2):
                # بررسی الگوی اردربلاک خرید
                if (data['Close'].iloc[i-2] > data['Close'].iloc[i-1] and 
                    data['Close'].iloc[i-1] < data['Close'].iloc[i] and
                    data['Volume'].iloc[i] > data['Volume'].iloc[i-1] * 1.5):
                    
                    order_blocks.append({
                        'type': 'buy',
                        'price': data['Close'].iloc[i],
                        'volume': data['Volume'].iloc[i],
                        'timestamp': data.index[i],
                        'strength': self.calculate_order_block_strength(data, i, 'buy')
                    })
            
            # شناسایی اردربلاک‌های فروش (Sell Side Liquidity)
            # اردربلاک فروش معمولاً پس از یک حرکت صعودی قوی شکل می‌گیرد
            for i in range(2, len(data)-2):
                # بررسی الگوی اردربلاک فروش
                if (data['Close'].iloc[i-2] < data['Close'].iloc[i-1] and 
                    data['Close'].iloc[i-1] > data['Close'].iloc[i] and
                    data['Volume'].iloc[i] > data['Volume'].iloc[i-1] * 1.5):
                    
                    order_blocks.append({
                        'type': 'sell',
                        'price': data['Close'].iloc[i],
                        'volume': data['Volume'].iloc[i],
                        'timestamp': data.index[i],
                        'strength': self.calculate_order_block_strength(data, i, 'sell')
                    })
            
            return order_blocks[-5:] if order_blocks else []  # برگرداندن 5 اردربلاک آخر
        except Exception as e:
            logger.error(f"Error in order block analysis: {e}")
            return []
    
    def calculate_order_block_strength(self, data, index, block_type):
        """محاسبه قدرت اردربلاک"""
        try:
            # محاسبه قدرت بر اساس حجم و تغییر قیمت
            volume_strength = data['Volume'].iloc[index] / data['Volume'].rolling(window=20).mean().iloc[index]
            
            if block_type == 'buy':
                price_change = (data['Close'].iloc[index+1] - data['Close'].iloc[index]) / data['Close'].iloc[index]
            else:
                price_change = (data['Close'].iloc[index] - data['Close'].iloc[index+1]) / data['Close'].iloc[index+1]
            
            # ترکیب حجم و تغییر قیمت برای محاسبه قدرت
            strength = min(1.0, volume_strength * (1 + abs(price_change)))
            
            return strength
        except Exception as e:
            logger.error(f"Error calculating order block strength: {e}")
            return 0.5
    
    def analyze_decision_zones(self, data):
        """تحلیل نواحی تصمیم‌گیری"""
        try:
            decision_zones = []
            
            if len(data) < 50:
                return decision_zones
            
            # شناسایی نواحی حمایت و مقاومت
            # استفاده از روش کف و سقف محلی
            for i in range(20, len(data)-20):
                window = data.iloc[i-20:i+20]
                
                # شناسایی ناحیه حمایت
                if data['Low'].iloc[i] == window['Low'].min():
                    decision_zones.append({
                        'type': 'support',
                        'price': data['Low'].iloc[i],
                        'strength': self.calculate_zone_strength(data, i, 'support'),
                        'timestamp': data.index[i]
                    })
                
                # شناسایی ناحیه مقاومت
                if data['High'].iloc[i] == window['High'].max():
                    decision_zones.append({
                        'type': 'resistance',
                        'price': data['High'].iloc[i],
                        'strength': self.calculate_zone_strength(data, i, 'resistance'),
                        'timestamp': data.index[i]
                    })
            
            return decision_zones[-5:] if decision_zones else []  # برگرداندن 5 ناحیه آخر
        except Exception as e:
            logger.error(f"Error in decision zone analysis: {e}")
            return []
    
    def calculate_zone_strength(self, data, index, zone_type):
        """محاسبه قدرت ناحیه تصمیم‌گیری"""
        try:
            # محاسبه تعداد دفعات تست ناحیه
            test_count = 0
            successful_tests = 0
            
            if zone_type == 'support':
                # برای ناحیه حمایت، بررسی واکنش قیمت به ناحیه
                for i in range(max(0, index-10), min(len(data), index+10)):
                    if data['Low'].iloc[i] <= data['Low'].iloc[index] * 1.01:
                        test_count += 1
                        if data['Close'].iloc[i] > data['Close'].iloc[index]:
                            successful_tests += 1
            else:
                # برای ناحیه مقاومت، بررسی واکنش قیمت به ناحیه
                for i in range(max(0, index-10), min(len(data), index+10)):
                    if data['High'].iloc[i] >= data['High'].iloc[index] * 0.99:
                        test_count += 1
                        if data['Close'].iloc[i] < data['Close'].iloc[index]:
                            successful_tests += 1
            
            # محاسبه قدرت بر اساس موفقیت تست‌ها
            strength = successful_tests / test_count if test_count > 0 else 0.5
            
            return strength
        except Exception as e:
            logger.error(f"Error calculating zone strength: {e}")
            return 0.5
    
    def analyze_trading_sessions(self, data):
        """تحلیل سشن‌های معاملاتی"""
        try:
            session_analysis = {
                'asian_session': {
                    'start': '22:00',
                    'end': '08:00',
                    'volatility': 0.0,
                    'volume_ratio': 0.0
                },
                'european_session': {
                    'start': '08:00',
                    'end': '16:00',
                    'volatility': 0.0,
                    'volume_ratio': 0.0
                },
                'american_session': {
                    'start': '13:00',
                    'end': '22:00',
                    'volatility': 0.0,
                    'volume_ratio': 0.0
                }
            }
            
            # تبدیل زمان به UTC
            data_utc = data.copy()
            data_utc.index = data_utc.index.tz_localize(None).tz_convert('UTC')
            
            # محاسبه نوسان و حجم برای هر سشن
            for session_name, session_info in session_analysis.items():
                session_data = data_utc.between_time(session_info['start'], session_info['end'])
                
                if not session_data.empty:
                    # محاسبه نوسان
                    returns = session_data['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
                    
                    # محاسبه نسبت حجم
                    total_volume = data['Volume'].sum()
                    session_volume = session_data['Volume'].sum()
                    volume_ratio = session_volume / total_volume if total_volume > 0 else 0
                    
                    session_analysis[session_name]['volatility'] = volatility
                    session_analysis[session_name]['volume_ratio'] = volume_ratio
            
            return session_analysis
        except Exception as e:
            logger.error(f"Error in session analysis: {e}")
            return {}
    
    def perform_ai_analysis(self, historical_data, market_data, sentiment):
        """انجام تحلیل هوش مصنوعی پیشرفته"""
        ai_results = {}
        
        # پیش‌بینی قیمت با مدل‌های مختلف
        predictions = self.predict_price(historical_data)
        ai_results['predictions'] = predictions
        
        # تحلیل ریسک
        risk_analysis = self.analyze_risk(historical_data, market_data)
        ai_results['risk_analysis'] = risk_analysis
        
        # تحلیل فرصت‌ها
        opportunities = self.analyze_opportunities(historical_data, market_data, sentiment)
        ai_results['opportunities'] = opportunities
        
        # تحلیل زمان‌بندی بازار
        timing_analysis = self.analyze_market_timing(historical_data)
        ai_results['timing_analysis'] = timing_analysis
        
        # تحلیل رفتار قیمت
        behavior_analysis = self.analyze_price_behavior(historical_data)
        ai_results['behavior_analysis'] = behavior_analysis
        
        return ai_results
    
    def predict_price(self, historical_data):
        """پیش‌بینی قیمت با مدل‌های مختلف"""
        if len(historical_data) < 30:
            return {'error': 'Not enough data for prediction'}
        
        try:
            # آماده‌سازی داده‌ها
            data = historical_data.copy()
            data['Target'] = data['Close'].shift(-1)
            data = data.dropna()
            
            if len(data) < 20:
                return {'error': 'Not enough data after preparation'}
            
            # ویژگی‌ها
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            X = data[features]
            y = data['Target']
            
            # تقسیم داده‌ها
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            predictions = {}
            
            # پیش‌بینی با مدل‌های مختلف
            for model_name, model in self.models.items():
                if model_name in ['prophet', 'lstm', 'gru']:
                    continue  # این مدل‌ها نیاز به پردازش خاص دارند
                
                try:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test[-1:])  # پیش‌بینی برای آخرین داده
                    predictions[model_name] = float(pred[0])
                except Exception as e:
                    logger.error(f"Error in {model_name} prediction: {e}")
            
            # محاسبه میانگین پیش‌بینی‌ها
            if predictions:
                avg_prediction = np.mean(list(predictions.values()))
                predictions['average'] = avg_prediction
                predictions['confidence'] = self.calculate_prediction_confidence(predictions)
            
            return predictions
        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            return {'error': str(e)}
    
    def calculate_prediction_confidence(self, predictions):
        """محاسبه اطمینان پیش‌بینی"""
        if len(predictions) < 2:
            return 0.5
        
        values = list(predictions.values())
        std_dev = np.std(values)
        mean_val = np.mean(values)
        
        # اطمینان بر اساس انحراف معیار
        if std_dev / mean_val < 0.05:  # انحراف معیار کم
            return 0.9
        elif std_dev / mean_val < 0.1:  # انحراف معیار متوسط
            return 0.7
        else:  # انحراف معیار زیاد
            return 0.5
    
    def analyze_risk(self, historical_data, market_data):
        """تحلیل ریسک"""
        try:
            returns = historical_data['Close'].pct_change().dropna()
            
            risk_metrics = {
                'volatility': returns.std() * np.sqrt(252),  # نوسان سالانه
                'max_drawdown': self.calculate_max_drawdown(historical_data['Close']),
                'var_95': returns.quantile(0.05),  # Value at Risk 95%
                'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0,
                'beta': self.calculate_beta(historical_data['Close']),
                'liquidity_risk': self.analyze_liquidity_risk(market_data)
            }
            
            # محاسبه امتیاز ریسک کلی
            risk_score = self.calculate_risk_score(risk_metrics)
            risk_metrics['risk_score'] = risk_score
            
            return risk_metrics
        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            return {'error': str(e)}
    
    def calculate_max_drawdown(self, prices):
        """محاسبه حداکثر افت"""
        cumulative = (1 + prices.pct_change()).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
    
    def calculate_beta(self, prices):
        """محاسه بتا نسبت به بازار"""
        try:
            # دریافت داده‌های بازار (BTC به عنوان شاخص بازار)
            market_data = yf.download('BTC-USD', period='1y', interval='1d')
            if market_data.empty:
                return 1.0
            
            # محاسبه بازده‌ها
            asset_returns = prices.pct_change().dropna()
            market_returns = market_data['Close'].pct_change().dropna()
            
            # هم‌ترازسازی داده‌ها
            min_length = min(len(asset_returns), len(market_returns))
            asset_returns = asset_returns[-min_length:]
            market_returns = market_returns[-min_length:]
            
            # محاسبه کوواریانس و واریانس
            covariance = np.cov(asset_returns, market_returns)[0, 1]
            variance = np.var(market_returns)
            
            return covariance / variance if variance != 0 else 1.0
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 1.0
    
    def analyze_liquidity_risk(self, market_data):
        """تحلیل ریسک نقدینگی"""
        try:
            volume = market_data.get('volume_24h', 0)
            price = market_data.get('price', 1)
            
            if volume == 0 or price == 0:
                return 1.0  # ریسک بالا
            
            # محاسبه نسبت حجم به ارزش بازار
            market_cap = market_data.get('market_cap', 1)
            volume_to_cap_ratio = volume / market_cap
            
            # امتیاز ریسک نقدینگی (هرچه کمتر بهتر)
            if volume_to_cap_ratio > 0.1:  # نقدینگی بالا
                return 0.2
            elif volume_to_cap_ratio > 0.05:  # نقدینگی متوسط
                return 0.5
            else:  # نقدینگی پایین
                return 0.8
        except Exception as e:
            logger.error(f"Error in liquidity risk analysis: {e}")
            return 0.5
    
    def calculate_risk_score(self, risk_metrics):
        """محاسبه امتیاز ریسک کلی"""
        try:
            # وزن‌ها برای هر معیار
            weights = {
                'volatility': 0.2,
                'max_drawdown': 0.25,
                'var_95': 0.15,
                'sharpe_ratio': 0.2,
                'beta': 0.1,
                'liquidity_risk': 0.1
            }
            
            # نرمال‌سازی معیارها
            normalized = {}
            
            # نوسان (هر چه کمتر بهتر)
            normalized['volatility'] = min(risk_metrics['volatility'] / 2.0, 1.0)
            
            # حداکثر افت (هر چه کمتر بهتر)
            normalized['max_drawdown'] = min(abs(risk_metrics['max_drawdown']) / 0.5, 1.0)
            
            # VaR (هر چه کمتر بهتر)
            normalized['var_95'] = min(abs(risk_metrics['var_95']) / 0.1, 1.0)
            
            # نسبت شارپ (هر چه بیشتر بهتر)
            normalized['sharpe_ratio'] = max(0, 1.0 - min(risk_metrics['sharpe_ratio'] / 2.0, 1.0))
            
            # بتا (هر چه نزدیک‌تر به 1 بهتر)
            normalized['beta'] = abs(risk_metrics['beta'] - 1.0)
            
            # ریسک نقدینگی (هر چه کمتر بهتر)
            normalized['liquidity_risk'] = risk_metrics['liquidity_risk']
            
            # محاسبه امتیاز نهایی
            risk_score = sum(normalized[metric] * weight for metric, weight in weights.items())
            
            return min(1.0, max(0.0, risk_score))
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5  # مقدار پیش‌فرض در صورت خطا
    
    def analyze_opportunities(self, historical_data, market_data, sentiment):
        """تحلیل فرصت‌ها"""
        try:
            opportunities = {}
            
            # تحلیل فرصت‌های کوتاه مدت
            short_term_opportunities = self.analyze_short_term_opportunities(historical_data)
            opportunities['short_term'] = short_term_opportunities
            
            # تحلیل فرصت‌های بلند مدت
            long_term_opportunities = self.analyze_long_term_opportunities(historical_data, market_data)
            opportunities['long_term'] = long_term_opportunities
            
            # تحلیل فرصت‌های مبتنی بر اخبار
            news_opportunities = self.analyze_news_opportunities(sentiment)
            opportunities['news_based'] = news_opportunities
            
            # تحلیل فرصت‌های مبتنی بر اردربلاک‌ها
            order_block_opportunities = self.analyze_order_block_opportunities(historical_data)
            opportunities['order_block'] = order_block_opportunities
            
            # تحلیل فرصت‌های مبتنی بر نواحی عرضه و تقاضا
            supply_demand_opportunities = self.analyze_supply_demand_opportunities(historical_data)
            opportunities['supply_demand'] = supply_demand_opportunities
            
            # محاسبه امتیاز کلی فرصت‌ها
            opportunity_score = self.calculate_opportunity_score(opportunities)
            opportunities['overall_score'] = opportunity_score
            
            return opportunities
        except Exception as e:
            logger.error(f"Error in opportunity analysis: {e}")
            return {'error': str(e)}
    
    def