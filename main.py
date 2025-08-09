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
        return f"{symbol}/USDT"
    
    async def fetch_news_from_multiple_sources(self, symbol=None):
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
            
            # تحلیل احساسات اخبار
            try:
                sentiment = await self.advanced_sentiment_analysis(news)
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {e}")
                sentiment = {'average_sentiment': 0, 'topics': []}
            
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
            logger.error(f"Error getting historical data: {e}")
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
        dates = pd.date_range(end=datetime.now(), periods=365)
        prices = [base_price * (1 + np.random.normal(0, 0.02)) for _ in range(365)]
        
        return pd.DataFrame({
            'Open': prices,
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.99 for p in prices],
            'Close': prices,
            'Volume': [1000000 * np.random.uniform(0.8, 1.2) for _ in range(365)]
        }, index=dates)
    
    def advanced_technical_analysis(self, data):
        """تحلیل تکنیکال پیشرفته با تمام ابزارها"""
        try:
            analysis = {}
            
            # 1. تحلیل کلاسیک
            analysis['classical'] = self.classical_analysis(data)
            
            # 2. تحلیل پرایس اکشن
            analysis['price_action'] = self.price_action_analysis(data)
            
            # 3. تحلیل چند تایم‌فریم
            analysis['multi_timeframe'] = self.multi_timeframe_analysis(data)
            
            # 4. تحلیل حجم و نقدینگی
            analysis['volume'] = self.volume_analysis(data)
            
            # 5. تحلیل واگرایی
            analysis['divergence'] = self.divergence_analysis(data)
            
            # 6. تحلیل فیبوناچی
            analysis['fibonacci'] = self.fibonacci_analysis(data)
            
            # 7. تحلیل امواج
            analysis['waves'] = self.wave_analysis(data)
            
            # 8. تحلیل مارکت پروفایل
            analysis['market_profile'] = self.market_profile_analysis(data)
            
            # 9. تحلیل مومنتوم پیشرفته
            analysis['advanced_momentum'] = self.advanced_momentum_analysis(data)
            
            # 10. تحلیل اینترمارکت
            analysis['intermarket'] = self.intermarket_analysis(data)
            
            # 11. سیستم هشدار هوشمند
            analysis['alerts'] = self.intelligent_alert_system(data)
            
            # 12. یادگیری ماشین برای تحلیل تکنیکال
            analysis['ml_analysis'] = self.ml_technical_analysis(data)
            
            return analysis
        except Exception as e:
            logger.error(f"Error in advanced technical analysis: {e}")
            return {}
    
    def classical_analysis(self, data):
        """تحلیل تکنیکال کلاسیک"""
        try:
            classical = {}
            
            # شاخص‌های اصلی
            if LIBRARIES['pandas_ta']:
                df = data.copy()
                
                # RSI در چند دوره
                classical['rsi'] = {
                    '14': pandas_ta.rsi(df['Close'], length=14).iloc[-1],
                    '21': pandas_ta.rsi(df['Close'], length=21).iloc[-1],
                    '50': pandas_ta.rsi(df['Close'], length=50).iloc[-1]
                }
                
                # MACD
                macd = pandas_ta.macd(df['Close'])
                classical['macd'] = {
                    'value': macd['MACD_12_26_9'].iloc[-1],
                    'signal': macd['MACDs_12_26_9'].iloc[-1],
                    'histogram': macd['MACDh_12_26_9'].iloc[-1]
                }
                
                # بولینگر بندز
                bb = pandas_ta.bbands(df['Close'], length=20, std=2)
                classical['bollinger'] = {
                    'upper': bb['BBU_20_2.0'].iloc[-1],
                    'middle': bb['BBM_20_2.0'].iloc[-1],
                    'lower': bb['BBL_20_2.0'].iloc[-1],
                    'position': self.get_bollinger_position(df['Close'].iloc[-1], bb),
                    'width': (bb['BBU_20_2.0'].iloc[-1] - bb['BBL_20_2.0'].iloc[-1]) / bb['BBM_20_2.0'].iloc[-1]
                }
                
                # استوکاستیک
                stoch = pandas_ta.stoch(df['High'], df['Low'], df['Close'])
                classical['stochastic'] = {
                    'k': stoch['STOCHk_14_3_3'].iloc[-1],
                    'd': stoch['STOCHd_14_3_3'].iloc[-1]
                }
            
            # میانگین متحرک‌ها
            classical['moving_averages'] = {
                'sma_20': data['Close'].rolling(20).mean().iloc[-1],
                'sma_50': data['Close'].rolling(50).mean().iloc[-1],
                'sma_200': data['Close'].rolling(200).mean().iloc[-1],
                'ema_12': data['Close'].ewm(span=12).mean().iloc[-1],
                'ema_26': data['Close'].ewm(span=26).mean().iloc[-1],
                'ema_50': data['Close'].ewm(span=50).mean().iloc[-1]
            }
            
            # تحلیل روند
            classical['trend'] = self.analyze_trend(data)
            
            # سطوح کلیدی
            classical['key_levels'] = self.find_key_levels(data)
            
            return classical
        except Exception as e:
            logger.error(f"Error in classical analysis: {e}")
            return {}
    
    def get_bollinger_position(self, price, bb_data):
        """تعیین موقعیت قیمت نسبت به بولینگر"""
        upper = bb_data['BBU_20_2.0'].iloc[-1]
        middle = bb_data['BBM_20_2.0'].iloc[-1]
        lower = bb_data['BBL_20_2.0'].iloc[-1]
        
        if price > upper:
            return "above"
        elif price < lower:
            return "below"
        else:
            return "inside"
    
    def analyze_trend(self, data):
        """تحلیل روند قیمت"""
        try:
            # محاسبه شیب خط روند با رگرسیون خطی
            x = np.arange(len(data))
            y = data['Close'].values
            
            slope, intercept = np.polyfit(x, y, 1)
            
            # تعیین روند
            if slope > 0.1:
                trend = 'صعودی'
            elif slope < -0.1:
                trend = 'نزولی'
            else:
                trend = 'خنثی'
            
            return {
                'direction': trend,
                'slope': slope,
                'strength': abs(slope)
            }
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return {'direction': 'unknown', 'slope': 0, 'strength': 0}
    
    def find_key_levels(self, data):
        """یافتن سطوح کلیدی حمایت و مقاومت"""
        try:
            # یافتن قله‌ها و دره‌ها
            highs = data['High'].rolling(window=20, center=True).max()
            lows = data['Low'].rolling(window=20, center=True).min()
            
            # یافتن نقاط محوری
            pivot_highs = data['High'][(data['High'] == highs) & (highs > highs.shift(1)) & (highs > highs.shift(-1))]
            pivot_lows = data['Low'][(data['Low'] == lows) & (lows < lows.shift(1)) & (lows < lows.shift(-1))]
            
            return {
                'resistance': pivot_highs.tail(5).tolist(),
                'support': pivot_lows.tail(5).tolist()
            }
        except Exception as e:
            logger.error(f"Error finding key levels: {e}")
            return {'resistance': [], 'support': []}
    
    def price_action_analysis(self, data):
        """تحلیل پرایس اکشن به سبک ال بروکس"""
        try:
            price_action = {}
            
            # الگوهای کندلی
            price_action['candle_patterns'] = self.identify_candlestick_patterns(data)
            
            # ساختار بازار
            price_action['market_structure'] = self.analyze_market_structure(data)
            
            # شکست ساختار
            price_action['break_of_structure'] = self.detect_break_of_structure(data)
            
            # نواحی عرضه و تقاضا
            price_action['supply_demand'] = self.find_supply_demand_zones(data)
            
            # الگوهای قیمت
            price_action['price_patterns'] = self.identify_price_patterns(data)
            
            # تحلیل نوسانات
            price_action['volatility'] = self.analyze_volatility(data)
            
            return price_action
        except Exception as e:
            logger.error(f"Error in price action analysis: {e}")
            return {}
    
    def identify_candlestick_patterns(self, data):
        """شناسایی الگوهای کندلی"""
        try:
            patterns = {}
            last_candle = data.iloc[-1]
            prev_candle = data.iloc[-2] if len(data) > 1 else None
            
            # پین بار
            if prev_candle is not None:
                body_size = abs(last_candle['Close'] - last_candle['Open'])
                total_size = last_candle['High'] - last_candle['Low']
                
                if body_size < total_size * 0.3:  # بدنه کوچک
                    if last_candle['Close'] > last_candle['Open']:  # صعودی
                        if last_candle['Low'] < min(prev_candle['Open'], prev_candle['Close']):
                            patterns['pin_bar'] = 'bullish'
                    else:  # نزولی
                        if last_candle['High'] > max(prev_candle['Open'], prev_candle['Close']):
                            patterns['pin_bar'] = 'bearish'
            
            # الگوی پوشاننده
            if prev_candle is not None:
                if (last_candle['Open'] < prev_candle['Close'] and 
                    last_candle['Close'] > prev_candle['Open'] and
                    abs(last_candle['Close'] - last_candle['Open']) > abs(prev_candle['Close'] - prev_candle['Open'])):
                    patterns['engulfing'] = 'bullish'
                elif (last_candle['Open'] > prev_candle['Close'] and 
                      last_candle['Close'] < prev_candle['Open'] and
                      abs(last_candle['Close'] - last_candle['Open']) > abs(prev_candle['Close'] - prev_candle['Open'])):
                    patterns['engulfing'] = 'bearish'
            
            # ستاره صبحگاهی/شبگاهی
            if len(data) >= 3:
                third_candle = data.iloc[-3]
                if (third_candle['Close'] < third_candle['Open'] and  # شمع نزولی
                    prev_candle['Open'] < prev_candle['Close'] and   # دوجی یا شمع کوچک
                    abs(prev_candle['Close'] - prev_candle['Open']) < (third_candle['High'] - third_candle['Low']) * 0.3 and
                    last_candle['Close'] > last_candle['Open'] and    # شمع صعودی
                    last_candle['Close'] > third_candle['Open']):
                    patterns['morning_star'] = True
                
                elif (third_candle['Close'] > third_candle['Open'] and  # شمع صعودی
                      prev_candle['Open'] > prev_candle['Close'] and   # دوجی یا شمع کوچک
                      abs(prev_candle['Close'] - prev_candle['Open']) < (third_candle['High'] - third_candle['Low']) * 0.3 and
                      last_candle['Close'] < last_candle['Open'] and    # شمع نزولی
                      last_candle['Close'] < third_candle['Open']):
                    patterns['evening_star'] = True
            
            return patterns
        except Exception as e:
            logger.error(f"Error identifying candlestick patterns: {e}")
            return {}
    
    def analyze_market_structure(self, data):
        """تحلیل ساختار بازار (HH, HL, LH, LL)"""
        try:
            structure = {
                'higher_highs': [],
                'higher_lows': [],
                'lower_highs': [],
                'lower_lows': []
            }
            
            # یافتن قله‌ها و دره‌ها
            highs = data['High'][(data['High'] > data['High'].shift(1)) & (data['High'] > data['High'].shift(-1))]
            lows = data['Low'][(data['Low'] < data['Low'].shift(1)) & (data['Low'] < data['Low'].shift(-1))]
            
            # تحلیل ساختار
            for i in range(1, len(highs)):
                if highs.iloc[i] > highs.iloc[i-1]:
                    structure['higher_highs'].append(highs.index[i])
                else:
                    structure['lower_highs'].append(highs.index[i])
            
            for i in range(1, len(lows)):
                if lows.iloc[i] > lows.iloc[i-1]:
                    structure['higher_lows'].append(lows.index[i])
                else:
                    structure['lower_lows'].append(lows.index[i])
            
            # تعیین روند فعلی
            if len(structure['higher_highs']) > len(structure['lower_highs']):
                structure['trend'] = 'uptrend'
            elif len(structure['lower_highs']) > len(structure['higher_highs']):
                structure['trend'] = 'downtrend'
            else:
                structure['trend'] = 'ranging'
            
            return structure
        except Exception as e:
            logger.error(f"Error analyzing market structure: {e}")
            return {}
    
    def detect_break_of_structure(self, data):
        """تشخیص شکست ساختار"""
        try:
            bos = {
                'breaks': [],
                'confirmed': [],
                'failed': []
            }
            
            # تحلیل شکست‌های اخیر
            structure = self.analyze_market_structure(data)
            
            # بررسی شکست قله‌ها و دره‌ها
            for i in range(1, len(data)):
                # شکست مقاومت
                if data['High'].iloc[i] > data['High'].iloc[i-1] and data['Close'].iloc[i] > data['High'].iloc[i-1]:
                    bos['breaks'].append({
                        'type': 'resistance',
                        'price': data['High'].iloc[i-1],
                        'time': data.index[i],
                        'confirmed': data['Close'].iloc[i+1] > data['High'].iloc[i-1] if i+1 < len(data) else False
                    })
                
                # شکست حمایت
                if data['Low'].iloc[i] < data['Low'].iloc[i-1] and data['Close'].iloc[i] < data['Low'].iloc[i-1]:
                    bos['breaks'].append({
                        'type': 'support',
                        'price': data['Low'].iloc[i-1],
                        'time': data.index[i],
                        'confirmed': data['Close'].iloc[i+1] < data['Low'].iloc[i-1] if i+1 < len(data) else False
                    })
            
            return bos
        except Exception as e:
            logger.error(f"Error detecting break of structure: {e}")
            return {}
    
    def find_supply_demand_zones(self, data):
        """یافتن نواحی عرضه و تقاضا"""
        try:
            zones = {
                'supply': [],
                'demand': []
            }
            
            # یافتن نواحی تقاضا (حمایت)
            for i in range(2, len(data)):
                # ناحیه تقاضا: دره با حجم بالا
                if (data['Low'].iloc[i] < data['Low'].iloc[i-1] and 
                    data['Low'].iloc[i] < data['Low'].iloc[i-2] and
                    data['Volume'].iloc[i] > data['Volume'].mean() * 1.5):
                    
                    zones['demand'].append({
                        'price': data['Low'].iloc[i],
                        'strength': data['Volume'].iloc[i] / data['Volume'].mean(),
                        'time': data.index[i]
                    })
            
            # یافتن نواحی عرضه (مقاومت)
            for i in range(2, len(data)):
                # ناحیه عرضه: قله با حجم بالا
                if (data['High'].iloc[i] > data['High'].iloc[i-1] and 
                    data['High'].iloc[i] > data['High'].iloc[i-2] and
                    data['Volume'].iloc[i] > data['Volume'].mean() * 1.5):
                    
                    zones['supply'].append({
                        'price': data['High'].iloc[i],
                        'strength': data['Volume'].iloc[i] / data['Volume'].mean(),
                        'time': data.index[i]
                    })
            
            # حذف نواحی نزدیک به هم
            zones['supply'] = self.merge_nearby_zones(zones['supply'])
            zones['demand'] = self.merge_nearby_zones(zones['demand'])
            
            return zones
        except Exception as e:
            logger.error(f"Error finding supply/demand zones: {e}")
            return {}
    
    def merge_nearby_zones(self, zones, threshold=0.02):
        """ادغام نواحی نزدیک به هم"""
        if not zones:
            return []
        
        merged = [zones[0]]
        for zone in zones[1:]:
            last = merged[-1]
            if abs(zone['price'] - last['price']) / last['price'] < threshold:
                # ادغام دو ناحیه
                last['price'] = (last['price'] + zone['price']) / 2
                last['strength'] = max(last['strength'], zone['strength'])
            else:
                merged.append(zone)
        
        return merged
    
    def identify_price_patterns(self, data):
        """شناسایی الگوهای قیمت"""
        try:
            patterns = {}
            
            # الگوی سر و شانه
            patterns['head_and_shoulders'] = self.detect_head_and_shoulders(data)
            
            # الگوی دو قله/دو کف
            patterns['double_top_bottom'] = self.detect_double_top_bottom(data)
            
            # الگوی مثلث
            patterns['triangle'] = self.detect_triangle_pattern(data)
            
            # الگوی پرچم
            patterns['flag'] = self.detect_flag_pattern(data)
            
            return patterns
        except Exception as e:
            logger.error(f"Error identifying price patterns: {e}")
            return {}
    
    def analyze_volatility(self, data):
        """تحلیل نوسانات"""
        try:
            volatility = {}
            
            # ATR (Average True Range)
            if LIBRARIES['pandas_ta']:
                atr = pandas_ta.atr(data['High'], data['Low'], data['Close'])
                volatility['atr'] = atr.iloc[-1]
                volatility['atr_percent'] = (atr.iloc[-1] / data['Close'].iloc[-1]) * 100
            
            # نوسان استاندارد
            returns = data['Close'].pct_change().dropna()
            volatility['std_dev'] = returns.std() * np.sqrt(252)  # سالانه
            
            # باندهای بولینگر
            bb_width = (data['High'].rolling(20).max() - data['Low'].rolling(20).min()) / data['Close'].rolling(20).mean()
            volatility['bb_width'] = bb_width.iloc[-1]
            
            # شاخص نوسان چایکین
            if LIBRARIES['pandas_ta']:
                volatility['chaikin'] = pandas_ta.chop(data['High'], data['Low'], data['Close']).iloc[-1]
            
            return volatility
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return {}
    
    def detect_head_and_shoulders(self, data):
        """تشخیص الگوی سر و شانه"""
        try:
            # یافتن قله‌ها
            peaks, _ = find_peaks(data['High'], distance=5)
            
            if len(peaks) >= 3:
                # بررسی شرایط سر و شانه
                p1 = data['High'].iloc[peaks[-3]]
                p2 = data['High'].iloc[peaks[-2]]
                p3 = data['High'].iloc[peaks[-1]]
                
                if p2 > p1 and p2 > p3 and abs(p1 - p3) / p2 < 0.1:
                    return {
                        'type': 'head_and_shoulders',
                        'left_shoulder': peaks[-3],
                        'head': peaks[-2],
                        'right_shoulder': peaks[-1],
                        'neckline': (data['Low'].iloc[peaks[-3]] + data['Low'].iloc[peaks[-1]]) / 2
                    }
            
            return None
        except Exception as e:
            logger.error(f"Error detecting head and shoulders: {e}")
            return None
    
    def detect_double_top_bottom(self, data):
        """تشخیص الگوی دو قله/دو کف"""
        try:
            # دو قله
            peaks, _ = find_peaks(data['High'], distance=10)
            if len(peaks) >= 2:
                p1 = data['High'].iloc[peaks[-2]]
                p2 = data['High'].iloc[peaks[-1]]
                
                if abs(p1 - p2) / p1 < 0.05:  # تفاوت کمتر از 5%
                    return {
                        'type': 'double_top',
                        'first_peak': peaks[-2],
                        'second_peak': peaks[-1],
                        'neckline': min(data['Low'].iloc[peaks[-2]:peaks[-1]])
                    }
            
            # دو کف
            troughs, _ = find_peaks(-data['Low'], distance=10)
            if len(troughs) >= 2:
                t1 = data['Low'].iloc[troughs[-2]]
                t2 = data['Low'].iloc[troughs[-1]]
                
                if abs(t1 - t2) / t1 < 0.05:  # تفاوت کمتر از 5%
                    return {
                        'type': 'double_bottom',
                        'first_trough': troughs[-2],
                        'second_trough': troughs[-1],
                        'neckline': max(data['High'].iloc[troughs[-2]:troughs[-1]])
                    }
            
            return None
        except Exception as e:
            logger.error(f"Error detecting double top/bottom: {e}")
            return None
    
    def detect_triangle_pattern(self, data):
        """تشخیص الگوی مثلث"""
        try:
            # این یک پیاده‌سازی ساده است
            # در عمل نیاز به تشخیص خطوط روند است
            
            # یافتن قله‌ها و دره‌ها
            peaks, _ = find_peaks(data['High'], distance=5)
            troughs, _ = find_peaks(-data['Low'], distance=5)
            
            if len(peaks) >= 2 and len(troughs) >= 2:
                # بررسی همگرایی قله‌ها و دره‌ها
                peak_slope = (data['High'].iloc[peaks[-1]] - data['High'].iloc[peaks[-2]]) / (peaks[-1] - peaks[-2])
                trough_slope = (data['Low'].iloc[troughs[-1]] - data['Low'].iloc[troughs[-2]]) / (troughs[-1] - troughs[-2])
                
                if abs(peak_slope) < 0.01 and abs(trough_slope) < 0.01:
                    return {
                        'type': 'symmetrical_triangle',
                        'upper_trend': (peaks[-2], data['High'].iloc[peaks[-2]]),
                        'lower_trend': (troughs[-2], data['Low'].iloc[troughs[-2]])
                    }
            
            return None
        except Exception as e:
            logger.error(f"Error detecting triangle pattern: {e}")
            return None
    
    def detect_flag_pattern(self, data):
        """تشخیص الگوی پرچم"""
        try:
            # این یک پیاده‌سازی ساده است
            # در عمل نیاز به تشخیص حرکت سریع و سپس تثبیت است
            
            # بررسی حرکت سریع اخیر
            recent_change = (data['Close'].iloc[-1] - data['Close'].iloc[-10]) / data['Close'].iloc[-10]
            
            if abs(recent_change) > 0.05:  # حرکت بیش از 5%
                # بررسی تثبیت
                recent_volatility = data['Close'].iloc[-10:].std()
                overall_volatility = data['Close'].std()
                
                if recent_volatility < overall_volatility * 0.5:
                    return {
                        'type': 'flag',
                        'direction': 'bullish' if recent_change > 0 else 'bearish',
                        'pole_height': abs(recent_change),
                        'flag_length': 10
                    }
            
            return None
        except Exception as e:
            logger.error(f"Error detecting flag pattern: {e}")
            return None
    
    def multi_timeframe_analysis(self, data):
        """تحلیل چند تایم‌فریم"""
        try:
            mtf = {}
            timeframes = ['1mo', '1wk', '1d', '4h', '1h', '15m']
            
            for tf in timeframes:
                try:
                    # دریافت داده برای تایم‌فریم مورد نظر
                    tf_data = self.get_historical_data(data.name.split('-')[0], period='1y', interval=tf)
                    if not tf_data.empty:
                        mtf[tf] = {
                            'trend': self.analyze_trend(tf_data),
                            'rsi': pandas_ta.rsi(tf_data['Close']).iloc[-1] if LIBRARIES['pandas_ta'] else None,
                            'key_level': self.find_key_levels(tf_data)[-1] if len(self.find_key_levels(tf_data)) > 0 else None,
                            'structure': self.analyze_market_structure(tf_data)
                        }
                except Exception as e:
                    logger.warning(f"Error in {tf} timeframe: {e}")
                    mtf[tf] = {'error': str(e)}
            
            return mtf
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            return {}
    
    def volume_analysis(self, data):
        """تحلیل حجم و نقدینگی"""
        try:
            volume = {}
            
            # حجم‌های کلیدی
            volume['profile'] = {
                'avg_volume': data['Volume'].mean(),
                'current_volume': data['Volume'].iloc[-1],
                'volume_ratio': data['Volume'].iloc[-1] / data['Volume'].mean(),
                'high_volume_days': len(data[data['Volume'] > data['Volume'].quantile(0.9)]),
                'low_volume_days': len(data[data['Volume'] < data['Volume'].quantile(0.1)])
            }
            
            # تحلیل حجم-قیمت
            volume['price_volume'] = {
                'up_volume': data[data['Close'] > data['Open']]['Volume'].sum(),
                'down_volume': data[data['Close'] < data['Open']]['Volume'].sum(),
                'volume_balance': (data[data['Close'] > data['Open']]['Volume'].sum() - 
                              data[data['Close'] < data['Open']]['Volume'].sum()) / data['Volume'].sum()
            }
            
            # الگوهای حجمی
            volume['patterns'] = self.identify_volume_patterns(data)
            
            return volume
        except Exception as e:
            logger.error(f"Error in volume analysis: {e}")
            return {}
    
    def identify_volume_patterns(self, data):
        """شناسایی الگوهای حجمی"""
        try:
            patterns = {}
            
            # افزایش حجم در روند
            volume_trend = data['Volume'].rolling(5).mean()
            price_trend = data['Close'].rolling(5).mean()
            
            if volume_trend.iloc[-1] > volume_trend.iloc[-5] and price_trend.iloc[-1] > price_trend.iloc[-5]:
                patterns['volume_trend'] = 'increasing_uptrend'
            elif volume_trend.iloc[-1] > volume_trend.iloc[-5] and price_trend.iloc[-1] < price_trend.iloc[-5]:
                patterns['volume_trend'] = 'increasing_downtrend'
            
            # حجم غیرعادی
            avg_volume = data['Volume'].mean()
            current_volume = data['Volume'].iloc[-1]
            
            if current_volume > avg_volume * 2:
                patterns['unusual_volume'] = 'high'
            elif current_volume < avg_volume * 0.5:
                patterns['unusual_volume'] = 'low'
            
            return patterns
        except Exception as e:
            logger.error(f"Error identifying volume patterns: {e}")
            return {}
    
    def divergence_analysis(self, data):
        """تحلیل واگرایی"""
        try:
            divergence = {}
            
            if LIBRARIES['pandas_ta']:
                # واگرایی RSI
                rsi = pandas_ta.rsi(data['Close'])
                divergence['rsi'] = self.detect_divergence(data['Close'], rsi)
                
                # واگرایی MACD
                macd = pandas_ta.macd(data['Close'])['MACD_12_26_9']
                divergence['macd'] = self.detect_divergence(data['Close'], macd)
                
                # واگرایی استوکاستیک
                stoch = pandas_ta.stoch(data['High'], data['Low'], data['Close'])['STOCHk_14_3_3']
                divergence['stochastic'] = self.detect_divergence(data['Close'], stoch)
            
            return divergence
        except Exception as e:
            logger.error(f"Error in divergence analysis: {e}")
            return {}
    
    def detect_divergence(self, price, indicator):
        """تشخیص واگرایی بین قیمت و اندیکاتور"""
        try:
            divergence = {
                'bullish': False,
                'bearish': False,
                'strength': 0
            }
            
            # یافتن قله‌ها و دره‌ها در قیمت و اندیکاتور
            price_peaks, _ = find_peaks(price, distance=5)
            indicator_peaks, _ = find_peaks(indicator, distance=5)
            
            price_troughs, _ = find_peaks(-price, distance=5)
            indicator_troughs, _ = find_peaks(-indicator, distance=5)
            
            # بررسی واگرایی صعودی
            if len(price_troughs) >= 2 and len(indicator_troughs) >= 2:
                if (price[price_troughs[-1]] < price[price_troughs[-2]] and 
                    indicator[indicator_troughs[-1]] > indicator[indicator_troughs[-2]]):
                    divergence['bullish'] = True
                    divergence['strength'] = abs(indicator[indicator_troughs[-1]] - indicator[indicator_troughs[-2]])
            
            # بررسی واگرایی نزولی
            if len(price_peaks) >= 2 and len(indicator_peaks) >= 2:
                if (price[price_peaks[-1]] > price[price_peaks[-2]] and 
                    indicator[indicator_peaks[-1]] < indicator[indicator_peaks[-2]]):
                    divergence['bearish'] = True
                    divergence['strength'] = abs(indicator[indicator_peaks[-1]] - indicator[indicator_peaks[-2]])
            
            return divergence
        except Exception as e:
            logger.error(f"Error detecting divergence: {e}")
            return {}
    
    def fibonacci_analysis(self, data):
        """تحلیل فیبوناچی"""
        try:
            fib = {}
            
            # یافتن سطوح کلیدی برای فیبوناچی
            high = data['High'].max()
            low = data['Low'].min()
            
            # محاسبه سطوح فیبوناچی
            levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            fib['retracement'] = {
                'high': high,
                'low': low,
                'levels': {f"{int(level*100)}%": high - (high - low) * level for level in levels}
            }
            
            # فیبوناچی اکستنشن
            fib['extension'] = {
                'levels': {f"{int(level*100)}%": high + (high - low) * level for level in [1.272, 1.618, 2.0]}
            }
            
            # مناطق فیبوناچی
            fib['zones'] = self.find_fibonacci_zones(data)
            
            return fib
        except Exception as e:
            logger.error(f"Error in Fibonacci analysis: {e}")
            return {}
    
    def find_fibonacci_zones(self, data):
        """یافتن مناطق فیبوناچی"""
        try:
            zones = []
            
            # یافتن حرکات قیمت قابل توجه
            swings = self.find_price_swings(data)
            
            for swing in swings:
                high = swing['high']
                low = swing['low']
                
                # محاسبه سطوح فیبوناچی
                levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                fib_levels = [high - (high - low) * level for level in levels]
                
                zones.append({
                    'start': swing['start_time'],
                    'end': swing['end_time'],
                    'high': high,
                    'low': low,
                    'levels': fib_levels
                })
            
            return zones
        except Exception as e:
            logger.error(f"Error finding Fibonacci zones: {e}")
            return []
    
    def find_price_swings(self, data):
        """یافتن نوسانات قیمت"""
        try:
            swings = []
            
            # یافتن قله‌ها و دره‌ها
            peaks, _ = find_peaks(data['High'], distance=5)
            troughs, _ = find_peaks(-data['Low'], distance=5)
            
            # ترکیب نوسانات
            points = []
            for p in peaks:
                points.append((data.index[p], data['High'].iloc[p], 'high'))
            for t in troughs:
                points.append((data.index[t], data['Low'].iloc[t], 'low'))
            
            points.sort(key=lambda x: x[0])
            
            # تشکیل نوسانات
            for i in range(1, len(points)-1):
                if points[i][2] == 'high' and points[i-1][2] == 'low' and points[i+1][2] == 'low':
                    swings.append({
                        'start_time': points[i-1][0],
                        'end_time': points[i+1][0],
                        'high': points[i][1],
                        'low': min(points[i-1][1], points[i+1][1])
                    })
            
            return swings
        except Exception as e:
            logger.error(f"Error finding price swings: {e}")
            return []
    
    def wave_analysis(self, data):
        """تحلیل امواج (الیوت و ولف)"""
        try:
            waves = {}
            
            # امواج الیوت
            waves['elliott'] = self.advanced_elliott_wave(data)
            
            # امواج ولف
            waves['wolfe'] = self.detect_wolfe_waves(data)
            
            # امواج هارمونیک
            waves['harmonic'] = self.detect_harmonic_patterns(data)
            
            return waves
        except Exception as e:
            logger.error(f"Error in wave analysis: {e}")
            return {}
    
    def advanced_elliott_wave(self, data):
        """تحلیل امواج الیوت پیشرفته"""
        try:
            # این یک تحلیل ساده از امواج الیوت است
            # برای تحلیل پیشرفته نیاز به کتابخانه‌های تخصصی داره
            
            # یافتن قله‌ها و دره‌ها
            peaks, _ = find_peaks(data['High'], distance=5)
            troughs, _ = find_peaks(-data['Low'], distance=5)
            
            # ترکیب نقاط
            points = []
            for p in peaks:
                points.append((data.index[p], data['High'].iloc[p], 'high'))
            for t in troughs:
                points.append((data.index[t], data['Low'].iloc[t], 'low'))
            
            # مرتب‌سازی بر اساس تاریخ
            points.sort(key=lambda x: x[0])
            
            # تحلیل ساده الگو
            if len(points) >= 5:
                # بررسی الگوی 5 موجی
                pattern = 'impulse' if points[0][2] == 'low' else 'corrective'
            else:
                pattern = 'incomplete'
            
            return {
                'current_pattern': pattern,
                'points': points[-10:],  # 10 نقطه آخر
                'confidence': min(len(points) / 10, 1.0)  # اطمینان بر اساس تعداد نقاط
            }
        except Exception as e:
            logger.error(f"Error in Elliott wave analysis: {e}")
            return {'current_pattern': 'unknown', 'points': [], 'confidence': 0}
    
    def detect_wolfe_waves(self, data):
        """تشخیص امواج ولف"""
        try:
            waves = []
            
            # این یک پیاده‌سازی ساده است
            # برای پیاده‌سازی کامل نیاز به الگوریتم‌های پیچیده‌تر است
            
            # یافتن 5 نقطه برای تشکیل موج
            points = self.find_swing_points(data)
            
            if len(points) >= 5:
                # بررسی شرایط موج ولف
                if (points[0][1] > points[1][1] and 
                    points[1][1] < points[2][1] and 
                    points[2][1] > points[3][1] and 
                    points[3][1] < points[4][1] and
                    points[4][1] < points[1][1]):
                    
                    waves.append({
                        'type': 'bullish',
                        'points': points[:5],
                        'target': points[1][1] + (points[2][1] - points[1][1])
                    })
            
            return waves
        except Exception as e:
            logger.error(f"Error detecting Wolfe waves: {e}")
            return []
    
    def detect_harmonic_patterns(self, data):
        """تشخیص الگوهای هارمونیک"""
        try:
            patterns = []
            
            # الگوی گارتلی
            gartley = self.detect_gartley_pattern(data)
            if gartley:
                patterns.append({'type': 'gartley', 'points': gartley})
            
            # الگوی پروانه
            butterfly = self.detect_butterfly_pattern(data)
            if butterfly:
                patterns.append({'type': 'butterfly', 'points': butterfly})
            
            # الگوی خفاش
            bat = self.detect_bat_pattern(data)
            if bat:
                patterns.append({'type': 'bat', 'points': bat})
            
            return patterns
        except Exception as e:
            logger.error(f"Error detecting harmonic patterns: {e}")
            return []
    
    def detect_gartley_pattern(self, data):
        """تشخیص الگوی گارتلی"""
        # پیاده‌سازی ساده شده
        # در عمل نیاز به محاسبات دقیق‌تر است
        points = self.find_swing_points(data)
        
        if len(points) >= 5:
            # بررسی نسبت‌های فیبوناچی
            # XA, AB, BC, CD
            # اینجا باید نسبت‌ها را بررسی کنیم
            
            # برای سادگی، فقط یک مثال می‌زنم
            return points[:5]
        
        return None
    
    def detect_butterfly_pattern(self, data):
        """تشخیص الگوی پروانه"""
        # مشابه الگوی گارتلی با نسبت‌های متفاوت
        points = self.find_swing_points(data)
        
        if len(points) >= 5:
            return points[:5]
        
        return None
    
    def detect_bat_pattern(self, data):
        """تشخیص الگوی خفاش"""
        # مشابه الگوی گارتلی با نسبت‌های متفاوت
        points = self.find_swing_points(data)
        
        if len(points) >= 5:
            return points[:5]
        
        return None
    
    def find_swing_points(self, data):
        """یافتن نقاط چرخش قیمت"""
        try:
            points = []
            
            # یافتن قله‌ها و دره‌ها
            peaks, _ = find_peaks(data['High'], distance=5)
            troughs, _ = find_peaks(-data['Low'], distance=5)
            
            for p in peaks:
                points.append((data.index[p], data['High'].iloc[p], 'high'))
            
            for t in troughs:
                points.append((data.index[t], data['Low'].iloc[t], 'low'))
            
            # مرتب‌سازی بر اساس زمان
            points.sort(key=lambda x: x[0])
            
            return points
        except Exception as e:
            logger.error(f"Error finding swing points: {e}")
            return []
    
    def market_profile_analysis(self, data):
        """تحلیل مارکت پروفایل (TPO, Value Area, POC)"""
        try:
            profile = {}
            
            # تقسیم قیمت به رنج‌ها
            price_range = data['High'].max() - data['Low'].min()
            num_bins = 20
            bin_size = price_range / num_bins
            
            # محاسبه TPO (Time Price Opportunity)
            tpo_bins = {}
            for i in range(num_bins):
                lower = data['Low'].min() + i * bin_size
                upper = lower + bin_size
                tpo_bins[f"{lower:.2f}-{upper:.2f}"] = 0
            
            # شمارش تایم‌ها در هر رنج قیمتی
            for _, row in data.iterrows():
                price = row['Close']
                for bin_range in tpo_bins:
                    lower, upper = map(float, bin_range.split('-'))
                    if lower <= price <= upper:
                        tpo_bins[bin_range] += 1
                        break
            
            profile['tpo'] = tpo_bins
            
            # محاسبه Value Area (70% از TPO)
            total_tpo = sum(tpo_bins.values())
            value_area_tpo = total_tpo * 0.7
            
            sorted_bins = sorted(tpo_bins.items(), key=lambda x: x[1], reverse=True)
            cumulative_tpo = 0
            value_area_bins = []
            
            for bin_range, count in sorted_bins:
                if cumulative_tpo < value_area_tpo:
                    value_area_bins.append(bin_range)
                    cumulative_tpo += count
            
            profile['value_area'] = {
                'bins': value_area_bins,
                'high': max(map(float, [b.split('-')[1] for b in value_area_bins])),
                'low': min(map(float, [b.split('-')[0] for b in value_area_bins]))
            }
            
            # محاسبه Point of Control (POC)
            if tpo_bins:
                poc_bin = max(tpo_bins.items(), key=lambda x: x[1])[0]
                profile['poc'] = float(poc_bin.split('-')[0])
            else:
                profile['poc'] = 0.0
            
            return profile
        except Exception as e:
            logger.error(f"Error in market profile analysis: {e}")
            return {}
    
    def advanced_momentum_analysis(self, data):
        """تحلیل مومنتوم پیشرفته"""
        try:
            momentum = {}
            
            if LIBRARIES['pandas_ta']:
                # شاخص‌های مومنتوم
                momentum['rsi'] = pandas_ta.rsi(data['Close']).iloc[-1]
                momentum['stoch'] = pandas_ta.stoch(data['High'], data['Low'], data['Close'])['STOCHk_14_3_3'].iloc[-1]
                momentum['cci'] = pandas_ta.cci(data['High'], data['Low'], data['Close']).iloc[-1]
                momentum['mfi'] = pandas_ta.mfi(data['High'], data['Low'], data['Close'], data['Volume']).iloc[-1]
                
                # تحلیل مومنتوم چند لایه
                momentum['multi_layer'] = self.multi_layer_momentum_analysis(data)
                
                # واگرایی‌های مومنتوم
                momentum['divergences'] = self.momentum_divergence_analysis(data)
            
            return momentum
        except Exception as e:
            logger.error(f"Error in advanced momentum analysis: {e}")
            return {}
    
    def multi_layer_momentum_analysis(self, data):
        """تحلیل مومنتوم چند لایه"""
        try:
            layers = {}
            
            # لایه‌های زمانی مختلف
            timeframes = [5, 10, 20, 50]
            
            for tf in timeframes:
                if LIBRARIES['pandas_ta']:
                    layers[f'rsi_{tf}'] = pandas_ta.rsi(data['Close'], length=tf).iloc[-1]
                    layers[f'stoch_{tf}'] = pandas_ta.stoch(data['High'], data['Low'], data['Close'], length=tf)['STOCHk_14_3_3'].iloc[-1]
            
            # تحلیل همگرایی/واگرایی بین لایه‌ها
            layers['convergence'] = self.analyze_momentum_convergence(layers)
            
            return layers
        except Exception as e:
            logger.error(f"Error in multi-layer momentum analysis: {e}")
            return {}
    
    def analyze_momentum_convergence(self, layers):
        """تحلیل همگرایی مومنتوم"""
        try:
            # استخراج مقادیر RSI از لایه‌های مختلف
            rsi_values = [v for k, v in layers.items() if 'rsi_' in k]
            
            if len(rsi_values) < 2:
                return {'status': 'insufficient_data'}
            
            # محاسبه انحراف معیار
            std_dev = np.std(rsi_values)
            mean_val = np.mean(rsi_values)
            
            # تعیین وضعیت همگرایی
            if std_dev / mean_val < 0.1:  # همگرایی قوی
                return {'status': 'strong_convergence', 'direction': 'bullish' if mean_val > 50 else 'bearish'}
            elif std_dev / mean_val < 0.2:  # همگرایی متوسط
                return {'status': 'moderate_convergence', 'direction': 'bullish' if mean_val > 50 else 'bearish'}
            else:  # واگرایی
                return {'status': 'divergence', 'strength': std_dev / mean_val}
        except Exception as e:
            logger.error(f"Error analyzing momentum convergence: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def momentum_divergence_analysis(self, data):
        """تحلیل واگرایی مومنتوم"""
        try:
            divergences = {}
            
            if LIBRARIES['pandas_ta']:
                # واگرایی RSI
                rsi = pandas_ta.rsi(data['Close'])
                divergences['rsi'] = self.detect_divergence(data['Close'], rsi)
                
                # واگرایی CCI
                cci = pandas_ta.cci(data['High'], data['Low'], data['Close'])
                divergences['cci'] = self.detect_divergence(data['Close'], cci)
                
                # واگرایی MFI
                mfi = pandas_ta.mfi(data['High'], data['Low'], data['Close'], data['Volume'])
                divergences['mfi'] = self.detect_divergence(data['Close'], mfi)
            
            return divergences
        except Exception as e:
            logger.error(f"Error in momentum divergence analysis: {e}")
            return {}
    
    def intermarket_analysis(self, data):
        """تحلیل اینترمارکت"""
        try:
            intermarket = {}
            
            # همبستگی با شاخص‌های اصلی
            intermarket['correlations'] = self.calculate_market_correlations(data)
            
            # تحلیل جریان نقدینگی بین بازارها
            intermarket['liquidity_flows'] = self.analyze_liquidity_flows()
            
            # تحلیل چرخه‌های بازار
            intermarket['market_cycles'] = self.analyze_market_cycles(data)
            
            return intermarket
        except Exception as e:
            logger.error(f"Error in intermarket analysis: {e}")
            return {}
    
    def calculate_market_correlations(self, data):
        """محاسبه همبستگی با شاخص‌های اصلی"""
        try:
            correlations = {}
            
            # دریافت داده‌های شاخص‌های اصلی
            indices = {
                'BTC': 'Bitcoin',
                'SPY': 'S&P 500',
                'QQQ': 'NASDAQ',
                'GLD': 'Gold',
                'USO': 'Oil'
            }
            
            for symbol, name in indices.items():
                try:
                    # دریافت داده‌های شاخص
                    index_data = yf.download(f'{symbol}-USD', period='1y', interval='1d')
                    
                    if not index_data.empty:
                        # محاسبه بازده‌ها
                        asset_returns = data['Close'].pct_change().dropna()
                        index_returns = index_data['Close'].pct_change().dropna()
                        
                        # هم‌ترازسازی داده‌ها
                        min_length = min(len(asset_returns), len(index_returns))
                        asset_returns = asset_returns[-min_length:]
                        index_returns = index_returns[-min_length:]
                        
                        # محاسبه همبستگی
                        correlation, p_value = pearsonr(asset_returns, index_returns)
                        
                        correlations[name] = {
                            'correlation': correlation,
                            'p_value': p_value,
                            'strength': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.3 else 'weak'
                        }
                except Exception as e:
                    logger.warning(f"Error calculating correlation with {name}: {e}")
                    correlations[name] = {'error': str(e)}
            
            return correlations
        except Exception as e:
            logger.error(f"Error calculating market correlations: {e}")
            return {}
    
    def analyze_liquidity_flows(self):
        """تحلیل جریان نقدینگی بین بازارها"""
        try:
            flows = {}
            
            # تحلیل جریان نقدینگی بین ارزهای دیجیتال
            flows['crypto_flows'] = self.analyze_crypto_flows()
            
            # تحلیل جریان نقدینگی بین بازارهای سنتی
            flows['traditional_flows'] = self.analyze_traditional_flows()
            
            return flows
        except Exception as e:
            logger.error(f"Error analyzing liquidity flows: {e}")
            return {}
    
    def analyze_crypto_flows(self):
        """تحلیل جریان نقدینگی در بازار ارزهای دیجیتال"""
        try:
            flows = {}
            
            # دریافت داده‌های حجم معاملات صرافی‌های اصلی
            for exchange_name, exchange in self.exchanges.items():
                try:
                    # دریافت حجم 24 ساعته
                    markets = exchange.load_markets()
                    total_volume = 0
                    
                    # محاسبه حجم کل
                    for symbol in markets:
                        if symbol.endswith('/USDT'):
                            try:
                                ticker = exchange.fetch_ticker(symbol)
                                total_volume += ticker['quoteVolume']
                            except:
                                pass
                    
                    flows[exchange_name] = {
                        'total_volume': total_volume,
                        'change_24h': 0  # در عمل باید تغییرات محاسبه شود
                    }
                except Exception as e:
                    logger.warning(f"Error analyzing {exchange_name} flows: {e}")
                    flows[exchange_name] = {'error': str(e)}
            
            return flows
        except Exception as e:
            logger.error(f"Error analyzing crypto flows: {e}")
            return {}
    
    def analyze_traditional_flows(self):
        """تحلیل جریان نقدینگی در بازارهای سنتی"""
        try:
            flows = {}
            
            # دریافت داده‌های شاخص‌های اصلی
            indices = {
                'SPY': 'S&P 500',
                'QQQ': 'NASDAQ',
                'GLD': 'Gold',
                'USO': 'Oil'
            }
            
            for symbol, name in indices.items():
                try:
                    # دریافت داده‌های شاخص
                    data = yf.download(f'{symbol}-USD', period='1mo', interval='1d')
                    
                    if not data.empty:
                        # محاسبه حجم معاملات
                        volume = data['Volume'].iloc[-1]
                        volume_change = (data['Volume'].iloc[-1] - data['Volume'].iloc[-5]) / data['Volume'].iloc[-5] if len(data) > 5 else 0
                        
                        flows[name] = {
                            'volume': volume,
                            'volume_change': volume_change,
                            'trend': 'increasing' if volume_change > 0 else 'decreasing'
                        }
                except Exception as e:
                    logger.warning(f"Error analyzing {name} flows: {e}")
                    flows[name] = {'error': str(e)}
            
            return flows
        except Exception as e:
            logger.error(f"Error analyzing traditional flows: {e}")
            return {}
    
    def analyze_market_cycles(self, data):
        """تحلیل چرخه‌های بازار"""
        try:
            cycles = {}
            
            # تحلیل چرخه‌های کوتاه مدت
            cycles['short_term'] = self.detect_short_term_cycles(data)
            
            # تحلیل چرخه‌های بلند مدت
            cycles['long_term'] = self.detect_long_term_cycles(data)
            
            # پیش‌بینی چرخه بعدی
            cycles['next_cycle'] = self.predict_next_cycle(data)
            
            return cycles
        except Exception as e:
            logger.error(f"Error analyzing market cycles: {e}")
            return {}
    
    def detect_short_term_cycles(self, data):
        """تشخیص چرخه‌های کوتاه مدت"""
        try:
            # استفاده از تحلیل فوریه برای تشخیص چرخه‌ها
            close_prices = data['Close'].values
            
            # اعمال تبدیل فوریه
            fft = np.fft.fft(close_prices)
            freq = np.fft.fftfreq(len(close_prices))
            
            # یافتن فرکانس‌های اصلی
            power = np.abs(fft) ** 2
            dominant_freq_idx = np.argsort(power)[-5:]  # 5 فرکانس برتر
            
            cycles = []
            for idx in dominant_freq_idx:
                if freq[idx] > 0:  # نادیده گرفتن فرکانس صفر
                    period = 1 / freq[idx]
                    if period < len(data) / 2:  # چرخه‌های معتبر
                        cycles.append({
                            'period': period,
                            'power': power[idx],
                            'strength': power[idx] / np.sum(power)
                        })
            
            return sorted(cycles, key=lambda x: x['strength'], reverse=True)
        except Exception as e:
            logger.error(f"Error detecting short-term cycles: {e}")
            return []
    
    def detect_long_term_cycles(self, data):
        """تشخیص چرخه‌های بلند مدت"""
        try:
            # تحلیل روند بلند مدت
            long_term_trend = self.analyze_trend(data)
            
            # تحلیل چرخه‌های تاریخی
            historical_cycles = self.analyze_historical_cycles(data)
            
            return {
                'trend': long_term_trend,
                'historical_patterns': historical_cycles
            }
        except Exception as e:
            logger.error(f"Error detecting long-term cycles: {e}")
            return {}
    
    def analyze_historical_cycles(self, data):
        """تحلیل چرخه‌های تاریخی"""
        try:
            # در یک پیاده‌سازی واقعی، این بخش باید داده‌های تاریخی بیشتری را تحلیل کند
            # اینجا یک تحلیل ساده ارائه می‌شود
            
            # شناسایی نقاط عطف تاریخی
            turning_points = self.identify_turning_points(data)
            
            # تحلیل فواصل بین نقاط عطف
            intervals = []
            for i in range(1, len(turning_points)):
                interval = (turning_points[i]['time'] - turning_points[i-1]['time']).days
                intervals.append(interval)
            
            if intervals:
                avg_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                return {
                    'avg_interval_days': avg_interval,
                    'std_interval_days': std_interval,
                    'turning_points': turning_points[-5:]  # 5 نقطه عطف آخر
                }
            else:
                return {'message': 'No clear turning points detected'}
        except Exception as e:
            logger.error(f"Error analyzing historical cycles: {e}")
            return {}
    
    def identify_turning_points(self, data):
        """شناسایی نقاط عطف تاریخی"""
        try:
            turning_points = []
            
            # یافتن قله‌ها و دره‌های اصلی
            peaks, _ = find_peaks(data['High'], distance=30, prominence=data['High'].std() * 2)
            troughs, _ = find_peaks(-data['Low'], distance=30, prominence=data['Low'].std() * 2)
            
            # افزودن قله‌ها
            for peak in peaks:
                turning_points.append({
                    'time': data.index[peak],
                    'price': data['High'].iloc[peak],
                    'type': 'peak'
                })
            
            # افزودن دره‌ها
            for trough in troughs:
                turning_points.append({
                    'time': data.index[trough],
                    'price': data['Low'].iloc[trough],
                    'type': 'trough'
                })
            
            # مرتب‌سازی بر اساس زمان
            turning_points.sort(key=lambda x: x['time'])
            
            return turning_points
        except Exception as e:
            logger.error(f"Error identifying turning points: {e}")
            return []
    
    def predict_next_cycle(self, data):
        """پیش‌بینی چرخه بعدی"""
        try:
            # تحلیل چرخه‌های قبلی
            cycles = self.detect_short_term_cycles(data)
            
            if not cycles:
                return {'message': 'Insufficient data for cycle prediction'}
            
            # محاسبه میانگین چرخه‌های اصلی
            dominant_cycles = [c for c in cycles if c['strength'] > 0.1]
            
            if dominant_cycles:
                avg_period = np.mean([c['period'] for c in dominant_cycles])
                
                # پیش‌بینی زمان چرخه بعدی
                last_date = data.index[-1]
                next_cycle_date = last_date + pd.Timedelta(days=int(avg_period))
                
                return {
                    'predicted_date': next_cycle_date,
                    'confidence': np.mean([c['strength'] for c in dominant_cycles]),
                    'based_on_cycles': len(dominant_cycles)
                }
            else:
                return {'message': 'No dominant cycles detected'}
        except Exception as e:
            logger.error(f"Error predicting next cycle: {e}")
            return {}
    
    def intelligent_alert_system(self, data):
        """سیستم هشدار هوشمند"""
        try:
            alerts = []
            
            # هشدارهای تکنیکال
            alerts.extend(self.generate_technical_alerts(data))
            
            # هشدارهای حجمی
            alerts.extend(self.generate_volume_alerts(data))
            
            # هشدارهای الگویی
            alerts.extend(self.generate_pattern_alerts(data))
            
            # هشدارهای مومنتوم
            alerts.extend(self.generate_momentum_alerts(data))
            
            # هشدارهای چند تایم‌فریم
            alerts.extend(self.generate_multitimeframe_alerts(data))
            
            return alerts
        except Exception as e:
            logger.error(f"Error in intelligent alert system: {e}")
            return []
    
    def generate_technical_alerts(self, data):
        """تولید هشدارهای تکنیکال"""
        try:
            alerts = []
            
            # هشدار RSI
            if LIBRARIES['pandas_ta']:
                rsi = pandas_ta.rsi(data['Close']).iloc[-1]
                if rsi > 70:
                    alerts.append({
                        'type': 'RSI',
                        'message': 'RSI in overbought zone',
                        'severity': 'medium',
                        'value': rsi
                    })
                elif rsi < 30:
                    alerts.append({
                        'type': 'RSI',
                        'message': 'RSI in oversold zone',
                        'severity': 'medium',
                        'value': rsi
                    })
            
            # هشدار بولینگر بندز
            if LIBRARIES['pandas_ta']:
                bb = pandas_ta.bbands(data['Close'], length=20, std=2)
                current_price = data['Close'].iloc[-1]
                upper_band = bb['BBU_20_2.0'].iloc[-1]
                lower_band = bb['BBL_20_2.0'].iloc[-1]
                
                if current_price > upper_band:
                    alerts.append({
                        'type': 'Bollinger',
                        'message': 'Price above upper Bollinger Band',
                        'severity': 'medium',
                        'value': current_price
                    })
                elif current_price < lower_band:
                    alerts.append({
                        'type': 'Bollinger',
                        'message': 'Price below lower Bollinger Band',
                        'severity': 'medium',
                        'value': current_price
                    })
            
            # هشدار شکست مقاومت/حمایت
            key_levels = self.find_key_levels(data)
            current_price = data['Close'].iloc[-1]
            
            # بررسی شکست مقاومت
            for resistance in key_levels.get('resistance', []):
                if current_price > resistance:
                    alerts.append({
                        'type': 'Breakout',
                        'message': f'Price broke resistance at {resistance}',
                        'severity': 'high',
                        'value': resistance
                    })
            
            # بررسی شکست حمایت
            for support in key_levels.get('support', []):
                if current_price < support:
                    alerts.append({
                        'type': 'Breakdown',
                        'message': f'Price broke support at {support}',
                        'severity': 'high',
                        'value': support
                    })
            
            return alerts
        except Exception as e:
            logger.error(f"Error generating technical alerts: {e}")
            return []
    
    def generate_volume_alerts(self, data):
        """تولید هشدارهای حجمی"""
        try:
            alerts = []
            
            # حجم غیرعادی
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].mean()
            
            if current_volume > avg_volume * 2:
                alerts.append({
                    'type': 'Volume',
                    'message': 'Unusually high volume detected',
                    'severity': 'high',
                    'value': current_volume / avg_volume
                })
            elif current_volume < avg_volume * 0.5:
                alerts.append({
                    'type': 'Volume',
                    'message': 'Unusually low volume detected',
                    'severity': 'medium',
                    'value': current_volume / avg_volume
                })
            
            # واگرایی حجم-قیمت
            price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
            volume_change = (data['Volume'].iloc[-1] - data['Volume'].iloc[-5]) / data['Volume'].iloc[-5]
            
            # قیمت افزایش یافته اما حجم کاهش یافته
            if price_change > 0.05 and volume_change < -0.2:
                alerts.append({
                    'type': 'Divergence',
                    'message': 'Price increasing but volume decreasing',
                    'severity': 'medium',
                    'value': abs(volume_change)
                })
            
            # قیمت کاهش یافته اما حجم کاهش یافته
            elif price_change < -0.05 and volume_change < -0.2:
                alerts.append({
                    'type': 'Divergence',
                    'message': 'Price decreasing but volume decreasing',
                    'severity': 'medium',
                    'value': abs(volume_change)
                })
            
            return alerts
        except Exception as e:
            logger.error(f"Error generating volume alerts: {e}")
            return []
    
    def generate_pattern_alerts(self, data):
        """تولید هشدارهای الگویی"""
        try:
            alerts = []
            
            # الگوهای کندلی
            patterns = self.identify_candlestick_patterns(data)
            
            if 'pin_bar' in patterns:
                alerts.append({
                    'type': 'Pattern',
                    'message': f'{patterns["pin_bar"].capitalize()} Pin Bar detected',
                    'severity': 'medium',
                    'value': patterns["pin_bar"]
                })
            
            if 'engulfing' in patterns:
                alerts.append({
                    'type': 'Pattern',
                    'message': f'{patterns["engulfing"].capitalize()} Engulfing pattern detected',
                    'severity': 'high',
                    'value': patterns["engulfing"]
                })
            
            if 'morning_star' in patterns:
                alerts.append({
                    'type': 'Pattern',
                    'message': 'Morning Star pattern detected',
                    'severity': 'high',
                    'value': 'bullish'
                })
            
            if 'evening_star' in patterns:
                alerts.append({
                    'type': 'Pattern',
                    'message': 'Evening Star pattern detected',
                    'severity': 'high',
                    'value': 'bearish'
                })
            
            # الگوهای قیمتی
            price_patterns = self.identify_price_patterns(data)
            
            if price_patterns.get('head_and_shoulders'):
                alerts.append({
                    'type': 'Pattern',
                    'message': 'Head and Shoulders pattern detected',
                    'severity': 'high',
                    'value': 'bearish'
                })
            
            if price_patterns.get('double_top_bottom'):
                pattern = price_patterns['double_top_bottom']
                alerts.append({
                    'type': 'Pattern',
                    'message': f'{pattern["type"].replace("_", " ").title()} pattern detected',
                    'severity': 'high',
                    'value': 'bearish' if pattern['type'] == 'double_top' else 'bullish'
                })
            
            return alerts
        except Exception as e:
            logger.error(f"Error generating pattern alerts: {e}")
            return []
    
    def generate_momentum_alerts(self, data):
        """تولید هشدارهای مومنتوم"""
        try:
            alerts = []
            
            if not LIBRARIES['pandas_ta']:
                return alerts
            
            # واگرایی RSI
            rsi = pandas_ta.rsi(data['Close'])
            rsi_divergence = self.detect_divergence(data['Close'], rsi)
            
            if rsi_divergence['bullish']:
                alerts.append({
                    'type': 'Divergence',
                    'message': 'Bullish RSI divergence detected',
                    'severity': 'high',
                    'value': rsi_divergence['strength']
                })
            
            if rsi_divergence['bearish']:
                alerts.append({
                    'type': 'Divergence',
                    'message': 'Bearish RSI divergence detected',
                    'severity': 'high',
                    'value': rsi_divergence['strength']
                })
            
            # واگرایی MACD
            macd = pandas_ta.macd(data['Close'])['MACD_12_26_9']
            macd_divergence = self.detect_divergence(data['Close'], macd)
            
            if macd_divergence['bullish']:
                alerts.append({
                    'type': 'Divergence',
                    'message': 'Bullish MACD divergence detected',
                    'severity': 'high',
                    'value': macd_divergence['strength']
                })
            
            if macd_divergence['bearish']:
                alerts.append({
                    'type': 'Divergence',
                    'message': 'Bearish MACD divergence detected',
                    'severity': 'high',
                    'value': macd_divergence['strength']
                })
            
            return alerts
        except Exception as e:
            logger.error(f"Error generating momentum alerts: {e}")
            return []
    
    def generate_multitimeframe_alerts(self, data):
        """تولید هشدارهای چند تایم‌فریم"""
        try:
            alerts = []
            
            # تحلیل چند تایم‌فریم
            mtf = self.multi_timeframe_analysis(data)
            
            # بررسی همگرایی سیگنال‌ها در تایم‌فریم‌های مختلف
            bullish_signals = 0
            bearish_signals = 0
            
            for tf, analysis in mtf.items():
                if isinstance(analysis, dict) and 'trend' in analysis:
                    if analysis['trend']['direction'] == 'صعودی':
                        bullish_signals += 1
                    elif analysis['trend']['direction'] == 'نزولی':
                        bearish_signals += 1
            
            # اگر اکثر تایم‌فریم‌ها یک جهت را نشان می‌دهند
            if bullish_signals > len(mtf) * 0.7:
                alerts.append({
                    'type': 'Multi-timeframe',
                    'message': 'Bullish convergence across multiple timeframes',
                    'severity': 'high',
                    'value': bullish_signals / len(mtf)
                })
            elif bearish_signals > len(mtf) * 0.7:
                alerts.append({
                    'type': 'Multi-timeframe',
                    'message': 'Bearish convergence across multiple timeframes',
                    'severity': 'high',
                    'value': bearish_signals / len(mtf)
                })
            
            return alerts
        except Exception as e:
            logger.error(f"Error generating multi-timeframe alerts: {e}")
            return []
    
    def ml_technical_analysis(self, data):
        """تحلیل تکنیکال با یادگیری ماشین"""
        try:
            ml_analysis = {}
            
            # طبقه‌بندی الگوها با یادگیری ماشین
            ml_analysis['pattern_classification'] = self.classify_patterns_ml(data)
            
            # پیش‌بینی روند با یادگیری ماشین
            ml_analysis['trend_prediction'] = self.predict_trend_ml(data)
            
            # تحلیل نوسانات با یادگیری ماشین
            ml_analysis['volatility_forecast'] = self.forecast_volatility_ml(data)
            
            return ml_analysis
        except Exception as e:
            logger.error(f"Error in ML technical analysis: {e}")
            return {}
    
    def classify_patterns_ml(self, data):
        """طبقه‌بندی الگوها با یادگیری ماشین"""
        try:
            # استخراج ویژگی‌ها از داده‌ها
            features = self.extract_pattern_features(data)
            
            if len(features) < 10:
                return {'message': 'Insufficient data for pattern classification'}
            
            # تبدیل به DataFrame
            df = pd.DataFrame(features)
            
            # جدا کردن ویژگی‌ها و برچسب‌ها
            X = df.drop('pattern_type', axis=1)
            y = df['pattern_type']
            
            # تقسیم داده‌ها
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # آموزش مدل
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # پیش‌بینی برای آخرین داده
            last_features = X.iloc[-1:].values.reshape(1, -1)
            prediction = model.predict(last_features)[0]
            confidence = model.predict_proba(last_features)[0].max()
            
            return {
                'predicted_pattern': prediction,
                'confidence': confidence,
                'model_accuracy': model.score(X_test, y_test)
            }
        except Exception as e:
            logger.error(f"Error in pattern classification with ML: {e}")
            return {'error': str(e)}
    
    def extract_pattern_features(self, data):
        """استخراج ویژگی‌ها برای طبقه‌بندی الگوها"""
        try:
            features = []
            
            # حرکت روی داده‌ها برای استخراج الگوها
            window_size = 10
            for i in range(window_size, len(data)):
                window_data = data.iloc[i-window_size:i]
                
                # محاسبه ویژگی‌ها
                feature_set = {
                    'open_change': (window_data['Open'].iloc[-1] - window_data['Open'].iloc[0]) / window_data['Open'].iloc[0],
                    'high_change': (window_data['High'].iloc[-1] - window_data['High'].iloc[0]) / window_data['High'].iloc[0],
                    'low_change': (window_data['Low'].iloc[-1] - window_data['Low'].iloc[0]) / window_data['Low'].iloc[0],
                    'close_change': (window_data['Close'].iloc[-1] - window_data['Close'].iloc[0]) / window_data['Close'].iloc[0],
                    'volume_change': (window_data['Volume'].iloc[-1] - window_data['Volume'].iloc[0]) / window_data['Volume'].iloc[0],
                    'volatility': window_data['Close'].std() / window_data['Close'].mean(),
                    'body_to_range': abs(window_data['Close'].iloc[-1] - window_data['Open'].iloc[-1]) / (window_data['High'].iloc[-1] - window_data['Low'].iloc[-1]),
                    'upper_shadow': (window_data['High'].iloc[-1] - max(window_data['Open'].iloc[-1], window_data['Close'].iloc[-1])) / (window_data['High'].iloc[-1] - window_data['Low'].iloc[-1]),
                    'lower_shadow': (min(window_data['Open'].iloc[-1], window_data['Close'].iloc[-1]) - window_data['Low'].iloc[-1]) / (window_data['High'].iloc[-1] - window_data['Low'].iloc[-1]),
                    'pattern_type': self.identify_pattern_type(window_data)
                }
                
                features.append(feature_set)
            
            return features
        except Exception as e:
            logger.error(f"Error extracting pattern features: {e}")
            return []
    
    def identify_pattern_type(self, window_data):
        """شناسایی نوع الگو برای داده‌های پنجره"""
        try:
            # در یک پیاده‌سازی واقعی، این باید از الگوریتم‌های پیچیده‌تری استفاده کند
            # اینجا یک شناسایی ساده ارائه می‌شود
            
            first_close = window_data['Close'].iloc[0]
            last_close = window_data['Close'].iloc[-1]
            change = (last_close - first_close) / first_close
            
            if change > 0.05:
                return 'bullish'
            elif change < -0.05:
                return 'bearish'
            else:
                return 'neutral'
        except Exception as e:
            logger.error(f"Error identifying pattern type: {e}")
            return 'unknown'
    
    def predict_trend_ml(self, data):
        """پیش‌بینی روند با یادگیری ماشین"""
        try:
            # آماده‌سازی داده‌ها
            features = self.extract_trend_features(data)
            
            if len(features) < 20:
                return {'message': 'Insufficient data for trend prediction'}
            
            # تبدیل به DataFrame
            df = pd.DataFrame(features)
            
            # جدا کردن ویژگی‌ها و برچسب‌ها
            X = df.drop('trend_direction', axis=1)
            y = df['trend_direction']
            
            # تقسیم داده‌ها
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # آموزش مدل
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # پیش‌بینی برای آخرین داده
            last_features = X.iloc[-1:].values.reshape(1, -1)
            prediction = model.predict(last_features)[0]
            confidence = model.predict_proba(last_features)[0].max()
            
            return {
                'predicted_trend': prediction,
                'confidence': confidence,
                'model_accuracy': model.score(X_test, y_test)
            }
        except Exception as e:
            logger.error(f"Error in trend prediction with ML: {e}")
            return {'error': str(e)}
    
    def extract_trend_features(self, data):
        """استخراج ویژگی‌ها برای پیش‌بینی روند"""
        try:
            features = []
            
            # حرکت روی داده‌ها برای استخراج ویژگی‌ها
            window_size = 20
            for i in range(window_size, len(data)):
                window_data = data.iloc[i-window_size:i]
                
                # محاسبه ویژگی‌ها
                feature_set = {
                    'sma_5': window_data['Close'].rolling(5).mean().iloc[-1],
                    'sma_10': window_data['Close'].rolling(10).mean().iloc[-1],
                    'sma_20': window_data['Close'].rolling(20).mean().iloc[-1],
                    'ema_5': window_data['Close'].ewm(span=5).mean().iloc[-1],
                    'ema_10': window_data['Close'].ewm(span=10).mean().iloc[-1],
                    'ema_20': window_data['Close'].ewm(span=20).mean().iloc[-1],
                    'momentum': (window_data['Close'].iloc[-1] - window_data['Close'].iloc[-5]) / window_data['Close'].iloc[-5],
                    'volatility': window_data['Close'].std() / window_data['Close'].mean(),
                    'volume_sma': window_data['Volume'].rolling(10).mean().iloc[-1],
                    'price_change': (window_data['Close'].iloc[-1] - window_data['Close'].iloc[0]) / window_data['Close'].iloc[0],
                    'trend_direction': self.determine_trend_direction(window_data)
                }
                
                features.append(feature_set)
            
            return features
        except Exception as e:
            logger.error(f"Error extracting trend features: {e}")
            return []
    
    def determine_trend_direction(self, window_data):
        """تعیین جهت روند برای داده‌های پنجره"""
        try:
            # محاسبه شیب خط روند با رگرسیون خطی
            x = np.arange(len(window_data))
            y = window_data['Close'].values
            
            slope, _ = np.polyfit(x, y, 1)
            
            # تعیین روند
            if slope > 0.01:
                return 'up'
            elif slope < -0.01:
                return 'down'
            else:
                return 'sideways'
        except Exception as e:
            logger.error(f"Error determining trend direction: {e}")
            return 'unknown'
    
    def forecast_volatility_ml(self, data):
        """پیش‌بینی نوسانات با یادگیری ماشین"""
        try:
            # آماده‌سازی داده‌ها
            features = self.extract_volatility_features(data)
            
            if len(features) < 20:
                return {'message': 'Insufficient data for volatility forecasting'}
            
            # تبدیل به DataFrame
            df = pd.DataFrame(features)
            
            # جدا کردن ویژگی‌ها و برچسب‌ها
            X = df.drop('future_volatility', axis=1)
            y = df['future_volatility']
            
            # تقسیم داده‌ها
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # آموزش مدل
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # پیش‌بینی برای آخرین داده
            last_features = X.iloc[-1:].values.reshape(1, -1)
            prediction = model.predict(last_features)[0]
            
            # محاسبه خطا
            mse = mean_squared_error(y_test, model.predict(X_test))
            
            return {
                'predicted_volatility': prediction,
                'model_mse': mse,
                'model_r2': model.score(X_test, y_test)
            }
        except Exception as e:
            logger.error(f"Error in volatility forecasting with ML: {e}")
            return {'error': str(e)}
    
    def extract_volatility_features(self, data):
        """استخراج ویژگی‌ها برای پیش‌بینی نوسانات"""
        try:
            features = []
            
            # حرکت روی داده‌ها برای استخراج ویژگی‌ها
            window_size = 20
            for i in range(window_size, len(data) - 5):  # 5 روز آینده را پیش‌بینی می‌کنیم
                window_data = data.iloc[i-window_size:i]
                future_data = data.iloc[i:i+5]
                
                # محاسبه ویژگی‌ها
                feature_set = {
                    'current_volatility': window_data['Close'].std() / window_data['Close'].mean(),
                    'volatility_sma': window_data['Close'].rolling(10).std().mean(),
                    'volume_volatility': window_data['Volume'].std() / window_data['Volume'].mean(),
                    'price_range': (window_data['High'].max() - window_data['Low'].min()) / window_data['Close'].mean(),
                    'body_avg': abs(window_data['Close'] - window_data['Open']).mean() / (window_data['High'] - window_data['Low']).mean(),
                    'high_low_ratio': window_data['High'].mean() / window_data['Low'].mean(),
                    'close_position': (window_data['Close'] - window_data['Low'].mean()) / (window_data['High'].mean() - window_data['Low'].mean()),
                    'volume_change': (window_data['Volume'].iloc[-1] - window_data['Volume'].iloc[0]) / window_data['Volume'].iloc[0],
                    'price_acceleration': self.calculate_acceleration(window_data['Close']),
                    'future_volatility': future_data['Close'].std() / future_data['Close'].mean()
                }
                
                features.append(feature_set)
            
            return features
        except Exception as e:
            logger.error(f"Error extracting volatility features: {e}")
            return []
    
    def calculate_acceleration(self, prices):
        """محاسبه شتاب قیمت"""
        try:
            # محاسبه تغییرات قیمت
            changes = prices.pct_change().dropna()
            
            # محاسبه تغییرات تغییرات (شتاب)
            acceleration = changes.diff().dropna()
            
            return acceleration.mean()
        except Exception as e:
            logger.error(f"Error calculating acceleration: {e}")
            return 0
    
    def advanced_supply_demand(self, symbol):
        """تحلیل پیشرفته عرضه و تقاضا"""
        try:
            supply_demand = {}
            
            # تحلیل سفارشات صرافی
            supply_demand['order_book'] = self.analyze_order_book(symbol)
            
            # تحلیل جریان نقدینگی
            supply_demand['liquidity'] = self.analyze_liquidity_flows_symbol(symbol)
            
            # تحلیل توکنومیکس
            supply_demand['tokenomics'] = self.analyze_tokenomics(symbol)
            
            # تحلیل نهنگ‌ها
            supply_demand['whales'] = self.analyze_whale_activity(symbol)
            
            return supply_demand
        except Exception as e:
            logger.error(f"Error in advanced supply demand analysis: {e}")
            return {}
    
    def analyze_order_book(self, symbol):
        """تحلیل دفتر سفارشات"""
        try:
            order_book = {}
            
            # تلاش برای دریافت دفتر سفارشات از صرافی‌ها
            for exchange_name, exchange in self.exchanges.items():
                try:
                    # تبدیل نماد به فرمت مناسب برای صرافی
                    exchange_symbol = self.convert_symbol_for_exchange(symbol, exchange_name)
                    
                    # دریافت دفتر سفارشات
                    ob = exchange.fetch_order_book(exchange_symbol, limit=20)
                    
                    # محاسبه شاخص‌های دفتر سفارشات
                    bids_total = sum([bid[1] for bid in ob['bids']])
                    asks_total = sum([ask[1] for ask in ob['asks']])
                    
                    # محاسبه عدم تعادل
                    imbalance = (bids_total - asks_total) / (bids_total + asks_total)
                    
                    # محاسبه فشار خرید/فروش
                    buy_pressure = sum([bid[0] * bid[1] for bid in ob['bids'][:10]])
                    sell_pressure = sum([ask[0] * ask[1] for ask in ob['asks'][:10]])
                    
                    order_book[exchange_name] = {
                        'bids_total': bids_total,
                        'asks_total': asks_total,
                        'imbalance': imbalance,
                        'buy_pressure': buy_pressure,
                        'sell_pressure': sell_pressure,
                        'spread': (ob['asks'][0][0] - ob['bids'][0][0]) / ob['bids'][0][0] if ob['asks'] and ob['bids'] else 0
                    }
                except Exception as e:
                    logger.warning(f"Error analyzing order book for {exchange_name}: {e}")
                    order_book[exchange_name] = {'error': str(e)}
            
            return order_book
        except Exception as e:
            logger.error(f"Error in order book analysis: {e}")
            return {}
    
    def analyze_liquidity_flows_symbol(self, symbol):
        """تحلیل جریان نقدینگی برای یک نماد خاص"""
        try:
            liquidity = {}
            
            # دریافت داده‌های تاریخی
            historical_data = self.get_historical_data(symbol)
            
            if historical_data.empty:
                return {'error': 'No historical data available'}
            
            # تحلیل حجم‌ها
            liquidity['volume_analysis'] = {
                'avg_volume': historical_data['Volume'].mean(),
                'current_volume': historical_data['Volume'].iloc[-1],
                'volume_trend': (historical_data['Volume'].iloc[-1] - historical_data['Volume'].iloc[-5]) / historical_data['Volume'].iloc[-5] if len(historical_data) > 5 else 0,
                'volume_spike': historical_data['Volume'].iloc[-1] / historical_data['Volume'].mean() - 1
            }
            
            # تحلیل نقدینگی در صرافی‌ها
            exchange_liquidity = {}
            for exchange_name, exchange in self.exchanges.items():
                try:
                    # تبدیل نماد به فرمت مناسب برای صرافی
                    exchange_symbol = self.convert_symbol_for_exchange(symbol, exchange_name)
                    
                    # دریافت تیکر
                    ticker = exchange.fetch_ticker(exchange_symbol)
                    
                    # محاسبه شاخص‌های نقدینگی
                    exchange_liquidity[exchange_name] = {
                        'volume': ticker['quoteVolume'],
                        'spread': (ticker['ask'] - ticker['bid']) / ticker['bid'] if ticker['ask'] and ticker['bid'] else 0,
                        'liquidity_score': min(ticker['quoteVolume'] / 1000000, 1.0)  # نمره نقدینگی تا 1
                    }
                except Exception as e:
                    logger.warning(f"Error analyzing liquidity for {exchange_name}: {e}")
                    exchange_liquidity[exchange_name] = {'error': str(e)}
            
            liquidity['exchange_liquidity'] = exchange_liquidity
            
            return liquidity
        except Exception as e:
            logger.error(f"Error in liquidity flow analysis: {e}")
            return {}
    
    def analyze_tokenomics(self, symbol):
        """تحلیل توکنومیکس"""
        try:
            tokenomics = {}
            
            # دریافت داده‌های توکنومیکس از CoinGecko
            if self.api_keys['coingecko']:
                try:
                    async with aiohttp.ClientSession() as session:
                        url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}"
                        params = {
                            'localization': 'false',
                            'tickers': 'false',
                            'market_data': 'true',
                            'community_data': 'false',
                            'developer_data': 'true',
                            'sparkline': 'false'
                        }
                        
                        if self.api_keys['coingecko']:
                            params['x_cg_demo_api_key'] = self.api_keys['coingecko']
                        
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                # استخراج داده‌های توکنومیکس
                                tokenomics['market_data'] = {
                                    'circulating_supply': data.get('market_data', {}).get('circulating_supply', 0),
                                    'total_supply': data.get('market_data', {}).get('total_supply', 0),
                                    'max_supply': data.get('market_data', {}).get('max_supply', 0),
                                    'fully_diluted_valuation': data.get('market_data', {}).get('fully_diluted_valuation', {}).get('usd', 0)
                                }
                                
                                # محاسبه شاخص‌های توکنومیکس
                                circulating_supply = tokenomics['market_data']['circulating_supply']
                                total_supply = tokenomics['market_data']['total_supply']
                                
                                if total_supply > 0:
                                    tokenomics['supply_metrics'] = {
                                        'circulating_ratio': circulating_supply / total_supply,
                                        'inflation_rate': self.calculate_inflation_rate(data),
                                        'supply_concentration': self.analyze_supply_concentration(data)
                                    }
                except Exception as e:
                    logger.warning(f"Error fetching tokenomics from CoinGecko: {e}")
            
            return tokenomics
        except Exception as e:
            logger.error(f"Error in tokenomics analysis: {e}")
            return {}
    
    def calculate_inflation_rate(self, coin_data):
        """محاسبه نرخ تورم"""
        try:
            # در یک پیاده‌سازی واقعی، این باید از داده‌های تاریخی عرضه استفاده کند
            # اینجا یک تخمین ساده ارائه می‌شود
            
            # دریافت داده‌های تاریخی عرضه
            # این بخش نیاز به API دیگری دارد
            
            # برای سادگی، یک مقدار پیش‌فرض برمی‌گردانیم
            return 0.02  # 2% نرخ تورم سالانه
        except Exception as e:
            logger.error(f"Error calculating inflation rate: {e}")
            return 0
    
    def analyze_supply_concentration(self, coin_data):
        """تحلیل تمرکز عرضه"""
        try:
            # در یک پیاده‌سازی واقعی، این باید از داده‌های توزیع آدرس‌ها استفاده کند
            # اینجا یک تحلیل ساده ارائه می‌شود
            
            # دریافت داده‌های توزیع آدرس‌ها
            # این بخش نیاز به API دیگری دارد
            
            # برای سادگی، یک مقدار پیش‌فرض برمی‌گردانیم
            return {
                'gini_coefficient': 0.6,  # ضریب جینی (0 تا 1، هر چه بیشتر، تمرکز بیشتر)
                'top_10_percent': 0.8,  # درصد عرضه در دسترس 10% برتر
                'top_1_percent': 0.4    # درصد عرضه در دسترس 1% برتر
            }
        except Exception as e:
            logger.error(f"Error analyzing supply concentration: {e}")
            return {}
    
    def analyze_whale_activity(self, symbol):
        """تحلیل فعالیت نهنگ‌ها"""
        try:
            whale_activity = {}
            
            # در یک پیاده‌سازی واقعی، این باید از داده‌های بلاکچین استفاده کند
            # اینجا یک تحلیل ساده ارائه می‌شود
            
            # دریافت داده‌های تراکنش‌های بزرگ
            # این بخش نیاز به API دیگری دارد
            
            # برای سادگی، داده‌های ساختگی تولید می‌کنیم
            whale_activity['large_transactions'] = {
                'count_24h': np.random.randint(5, 20),
                'volume_24h': np.random.uniform(1000000, 10000000),
                'avg_size': np.random.uniform(10000, 100000),
                'trend': 'increasing' if np.random.random() > 0.5 else 'decreasing'
            }
            
            whale_activity['wallet_movements'] = {
                'exchange_inflows': np.random.uniform(100000, 1000000),
                'exchange_outflows': np.random.uniform(100000, 1000000),
                'net_flow': np.random.uniform(-500000, 500000),
                'whale_balance': np.random.uniform(10000000, 100000000)
            }
            
            return whale_activity
        except Exception as e:
            logger.error(f"Error in whale activity analysis: {e}")
            return {}
    
    def analyze_opportunities(self, historical_data, market_data, sentiment):
        """تحلیل فرصت‌ها"""
        try:
            opportunities = {}
            
            # فرصت‌های معاملاتی
            opportunities['trading'] = self.analyze_trading_opportunities(historical_data, market_data)
            
            # فرصت‌های سرمایه‌گذاری
            opportunities['investment'] = self.analyze_investment_opportunities(historical_data, market_data, sentiment)
            
            # فرصت‌های آربیتراژ
            opportunities['arbitrage'] = self.analyze_arbitrage_opportunities(market_data)
            
            return opportunities
        except Exception as e:
            logger.error(f"Error in opportunity analysis: {e}")
            return {}
    
    def analyze_trading_opportunities(self, historical_data, market_data):
        """تحلیل فرصت‌های معاملاتی"""
        try:
            trading_opps = {}
            
            # تحلیل شکاف‌ها
            trading_opps['gaps'] = self.analyze_gaps(historical_data)
            
            # تحلیل محدوده‌های معاملاتی
            trading_opps['ranges'] = self.analyze_trading_ranges(historical_data)
            
            # تحلیل شکست‌ها
            trading_opps['breakouts'] = self.analyze_breakouts(historical_data)
            
            # تحلیل بازگشت‌ها
            trading_opps['reversals'] = self.analyze_reversals(historical_data)
            
            return trading_opps
        except Exception as e:
            logger.error(f"Error in trading opportunity analysis: {e}")
            return {}
    
    def analyze_gaps(self, data):
        """تحلیل شکاف‌های قیمتی"""
        try:
            gaps = []
            
            # یافتن شکاف‌ها
            for i in range(1, len(data)):
                prev_high = data['High'].iloc[i-1]
                prev_low = data['Low'].iloc[i-1]
                curr_open = data['Open'].iloc[i]
                
                # شکاف صعودی
                if curr_open > prev_high:
                    gap_size = (curr_open - prev_high) / prev_high
                    gaps.append({
                        'type': 'up',
                        'size': gap_size,
                        'date': data.index[i],
                        'filled': data['Low'].iloc[i] <= prev_high
                    })
                
                # شکاف نزولی
                elif curr_open < prev_low:
                    gap_size = (prev_low - curr_open) / prev_low
                    gaps.append({
                        'type': 'down',
                        'size': gap_size,
                        'date': data.index[i],
                        'filled': data['High'].iloc[i] >= prev_low
                    })
            
            # تحلیل آماری شکاف‌ها
            if gaps:
                gap_stats = {
                    'total_gaps': len(gaps),
                    'up_gaps': len([g for g in gaps if g['type'] == 'up']),
                    'down_gaps': len([g for g in gaps if g['type'] == 'down']),
                    'filled_gaps': len([g for g in gaps if g['filled']]),
                    'avg_gap_size': np.mean([g['size'] for g in gaps]),
                    'recent_gaps': gaps[-5:]  # 5 شکاف آخر
                }
            else:
                gap_stats = {'message': 'No gaps detected'}
            
            return gap_stats
        except Exception as e:
            logger.error(f"Error in gap analysis: {e}")
            return {}
    
    def analyze_trading_ranges(self, data):
        """تحلیل محدوده‌های معاملاتی"""
        try:
            ranges = []
            
            # یافتن محدوده‌های معاملاتی
            in_range = False
            range_start = None
            range_high = None
            range_low = None
            
            for i in range(1, len(data)):
                current_high = data['High'].iloc[i]
                current_low = data['Low'].iloc[i]
                
                if not in_range:
                    # شروع یک محدوده جدید
                    if abs(data['Close'].iloc[i] - data['Close'].iloc[i-1]) / data['Close'].iloc[i-1] < 0.01:  # تغییر قیمت کم
                        in_range = True
                        range_start = data.index[i]
                        range_high = current_high
                        range_low = current_low
                else:
                    # به‌روزرسانی محدوده
                    range_high = max(range_high, current_high)
                    range_low = min(range_low, current_low)
                    
                    # پایان محدوده
                    if (current_high > range_high * 1.05) or (current_low < range_low * 0.95):  # شکست 5%
                        ranges.append({
                            'start': range_start,
                            'end': data.index[i-1],
                            'high': range_high,
                            'low': range_low,
                            'width': (range_high - range_low) / range_low,
                            'duration': (data.index[i-1] - range_start).days
                        })
                        in_range = False
            
            # اگر در محدوده هستیم، آن را اضافه کن
            if in_range:
                ranges.append({
                    'start': range_start,
                    'end': data.index[-1],
                    'high': range_high,
                    'low': range_low,
                    'width': (range_high - range_low) / range_low,
                    'duration': (data.index[-1] - range_start).days
                })
            
            # تحلیل آماری محدوده‌ها
            if ranges:
                range_stats = {
                    'total_ranges': len(ranges),
                    'avg_duration': np.mean([r['duration'] for r in ranges]),
                    'avg_width': np.mean([r['width'] for r in ranges]),
                    'current_range': ranges[-1] if ranges else None
                }
            else:
                range_stats = {'message': 'No trading ranges detected'}
            
            return range_stats
        except Exception as e:
            logger.error(f"Error in trading range analysis: {e}")
            return {}
    
    def analyze_breakouts(self, data):
        """تحلیل شکست‌ها"""
        try:
            breakouts = []
            
            # یافتن سطوح مقاومت
            resistance_levels = self.find_key_levels(data).get('resistance', [])
            
            # بررسی شکست مقاومت
            for level in resistance_levels:
                for i in range(1, len(data)):
                    if data['High'].iloc[i] > level and data['Close'].iloc[i-1] <= level:
                        # شکست مقاومت
                        breakout_strength = (data['High'].iloc[i] - level) / level
                        breakout_volume = data['Volume'].iloc[i] / data['Volume'].mean()
                        
                        breakouts.append({
                            'type': 'resistance',
                            'level': level,
                            'date': data.index[i],
                            'strength': breakout_strength,
                            'volume_ratio': breakout_volume,
                            'confirmed': data['Close'].iloc[i+1] > level if i+1 < len(data) else False
                        })
                        break
            
            # یافتن سطوح حمایت
            support_levels = self.find_key_levels(data).get('support', [])
            
            # بررسی شکست حمایت
            for level in support_levels:
                for i in range(1, len(data)):
                    if data['Low'].iloc[i] < level and data['Close'].iloc[i-1] >= level:
                        # شکست حمایت
                        breakout_strength = (level - data['Low'].iloc[i]) / level
                        breakout_volume = data['Volume'].iloc[i] / data['Volume'].mean()
                        
                        breakouts.append({
                            'type': 'support',
                            'level': level,
                            'date': data.index[i],
                            'strength': breakout_strength,
                            'volume_ratio': breakout_volume,
                            'confirmed': data['Close'].iloc[i+1] < level if i+1 < len(data) else False
                        })
                        break
            
            # تحلیل آماری شکست‌ها
            if breakouts:
                breakout_stats = {
                    'total_breakouts': len(breakouts),
                    'resistance_breakouts': len([b for b in breakouts if b['type'] == 'resistance']),
                    'support_breakouts': len([b for b in breakouts if b['type'] == 'support']),
                    'confirmed_breakouts': len([b for b in breakouts if b['confirmed']]),
                    'avg_strength': np.mean([b['strength'] for b in breakouts]),
                    'recent_breakouts': breakouts[-5:]  # 5 شکست آخر
                }
            else:
                breakout_stats = {'message': 'No breakouts detected'}
            
            return breakout_stats
        except Exception as e:
            logger.error(f"Error in breakout analysis: {e}")
            return {}
    
    def analyze_reversals(self, data):
        """تحلیل بازگشت‌ها"""
        try:
            reversals = []
            
            # یافتن نقاط بازگشتی بالقوه
            for i in range(2, len(data)-2):
                # بازگشت صعودی
                if (data['Low'].iloc[i] < data['Low'].iloc[i-1] and 
                    data['Low'].iloc[i] < data['Low'].iloc[i-2] and
                    data['Low'].iloc[i] < data['Low'].iloc[i+1] and
                    data['Low'].iloc[i] < data['Low'].iloc[i+2]):
                    
                    # محاسبه قدرت بازگشت
                    reversal_strength = (data['High'].iloc[i+2] - data['Low'].iloc[i]) / data['Low'].iloc[i]
                    reversal_volume = data['Volume'].iloc[i] / data['Volume'].mean()
                    
                    reversals.append({
                        'type': 'bullish',
                        'price': data['Low'].iloc[i],
                        'date': data.index[i],
                        'strength': reversal_strength,
                        'volume_ratio': reversal_volume
                    })
                
                # بازگشت نزولی
                elif (data['High'].iloc[i] > data['High'].iloc[i-1] and 
                      data['High'].iloc[i] > data['High'].iloc[i-2] and
                      data['High'].iloc[i] > data['High'].iloc[i+1] and
                      data['High'].iloc[i] > data['High'].iloc[i+2]):
                    
                    # محاسبه قدرت بازگشت
                    reversal_strength = (data['High'].iloc[i] - data['Low'].iloc[i+2]) / data['High'].iloc[i]
                    reversal_volume = data['Volume'].iloc[i] / data['Volume'].mean()
                    
                    reversals.append({
                        'type': 'bearish',
                        'price': data['High'].iloc[i],
                        'date': data.index[i],
                        'strength': reversal_strength,
                        'volume_ratio': reversal_volume
                    })
            
            # تحلیل آماری بازگشت‌ها
            if reversals:
                reversal_stats = {
                    'total_reversals': len(reversals),
                    'bullish_reversals': len([r for r in reversals if r['type'] == 'bullish']),
                    'bearish_reversals': len([r for r in reversals if r['type'] == 'bearish']),
                    'avg_strength': np.mean([r['strength'] for r in reversals]),
                    'recent_reversals': reversals[-5:]  # 5 بازگشت آخر
                }
            else:
                reversal_stats = {'message': 'No reversals detected'}
            
            return reversal_stats
        except Exception as e:
            logger.error(f"Error in reversal analysis: {e}")
            return {}
    
    def analyze_investment_opportunities(self, historical_data, market_data, sentiment):
        """تحلیل فرصت‌های سرمایه‌گذاری"""
        try:
            investment_opps = {}
            
            # تحلیل ارزش ذاتی
            investment_opps['intrinsic_value'] = self.analyze_intrinsic_value(historical_data, market_data)
            
            # تحلیل رشد بلندمدت
            investment_opps['long_term_growth'] = self.analyze_long_term_growth(historical_data)
            
            # تحلیل ریسک-پاداش
            investment_opps['risk_reward'] = self.analyze_risk_reward(historical_data, market_data)
            
            # تحلیل زمان‌بندی سرمایه‌گذاری
            investment_opps['timing'] = self.analyze_investment_timing(historical_data, sentiment)
            
            return investment_opps
        except Exception as e:
            logger.error(f"Error in investment opportunity analysis: {e}")
            return {}
    
    def analyze_intrinsic_value(self, historical_data, market_data):
        """تحلیل ارزش ذاتی"""
        try:
            intrinsic_value = {}
            
            # مدل‌های ارزشگذاری
            current_price = market_data.get('price', 1)
            
            # مدل تنزیل جریان‌های نقدی (DCF)
            dcf_value = self.calculate_dcf_value(historical_data)
            intrinsic_value['dcf'] = {
                'value': dcf_value,
                'premium_discount': (dcf_value - current_price) / current_price,
                'signal': 'undervalued' if dcf_value > current_price * 1.1 else 'overvalued' if dcf_value < current_price * 0.9 else 'fair'
            }
            
            # مدل نسبت‌های مالی
            pe_ratio = self.calculate_pe_ratio(historical_data, market_data)
            pb_ratio = self.calculate_pb_ratio(historical_data, market_data)
            ps_ratio = self.calculate_ps_ratio(historical_data, market_data)
            
            intrinsic_value['ratios'] = {
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'ps_ratio': ps_ratio,
                'avg_pe': self.industry_average_pe_ratio(),
                'avg_pb': self.industry_average_pb_ratio(),
                'avg_ps': self.industry_average_ps_ratio()
            }
            
            # تحلیل ترکیبی
            intrinsic_value['combined'] = {
                'value': (dcf_value + self.value_from_ratios(pe_ratio, pb_ratio, ps_ratio)) / 2,
                'confidence': 0.7  # در یک پیاده‌سازی واقعی، این باید محاسبه شود
            }
            
            return intrinsic_value
        except Exception as e:
            logger.error(f"Error in intrinsic value analysis: {e}")
            return {}
    
    def calculate_dcf_value(self, data):
        """محاسبه ارزش با مدل DCF"""
        try:
            # در یک پیاده‌سازی واقعی، این باید از داده‌های مالی استفاده کند
            # اینجا یک تخمین ساده ارائه می‌شود
            
            # دریافت داده‌های رشد
            growth_rates = data['Close'].pct_change().dropna()
            avg_growth = growth_rates.mean()
            
            # نرخ تنزیل
            discount_rate = 0.1  # 10%
            
            # پیش‌بینی جریان‌های نقدی آینده
            current_value = data['Close'].iloc[-1]
            future_values = [current_value * (1 + avg_growth) ** i for i in range(1, 6)]
            
            # تنزیل جریان‌های نقدی
            discounted_values = [fv / (1 + discount_rate) ** i for i, fv in enumerate(future_values, 1)]
            
            # محاسبه ارزش فعلی
            dcf_value = sum(discounted_values)
            
            return dcf_value
        except Exception as e:
            logger.error(f"Error calculating DCF value: {e}")
            return data['Close'].iloc[-1]
    
    def calculate_pe_ratio(self, data, market_data):
        """محاسبه نسبت P/E"""
        try:
            # در یک پیاده‌سازی واقعی، این باید از داده‌های مالی استفاده کند
            # اینجا یک تخمین ساده ارائه می‌شود
            
            current_price = market_data.get('price', 1)
            
            # تخمین سود هر سهم
            earnings_estimate = current_price * 0.05  # فرض سود 5%
            
            if earnings_estimate > 0:
                return current_price / earnings_estimate
            else:
                return 0
        except Exception as e:
            logger.error(f"Error calculating P/E ratio: {e}")
            return 0
    
    def calculate_pb_ratio(self, data, market_data):
        """محاسبه نسبت P/B"""
        try:
            # در یک پیاده‌سازی واقعی، این باید از داده‌های مالی استفاده کند
            # اینجا یک تخمین ساده ارائه می‌شود
            
            current_price = market_data.get('price', 1)
            
            # تخمین ارزش دفتری هر سهم
            book_value_estimate = current_price * 0.8  # فرض ارزش دفتری 80% قیمت
            
            if book_value_estimate > 0:
                return current_price / book_value_estimate
            else:
                return 0
        except Exception as e:
            logger.error(f"Error calculating P/B ratio: {e}")
            return 0
    
    def calculate_ps_ratio(self, data, market_data):
        """محاسبه نسبت P/S"""
        try:
            # در یک پیاده‌سازی واقعی، این باید از داده‌های مالی استفاده کند
            # اینجا یک تخمین ساده ارائه می‌شود
            
            current_price = market_data.get('price', 1)
            
            # تخمین فروش هر سهم
            sales_estimate = current_price * 0.5  # فرض فروش 50% قیمت
            
            if sales_estimate > 0:
                return current_price / sales_estimate
            else:
                return 0
        except Exception as e:
            logger.error(f"Error calculating P/S ratio: {e}")
            return 0
    
    def industry_average_pe_ratio(self):
        """میانگین نسبت P/E صنعت"""
        # در یک پیاده‌سازی واقعی، این باید از داده‌های صنعت استفاده کند
        return 25  # مقدار پیش‌فرض
    
    def industry_average_pb_ratio(self):
        """میانگین نسبت P/B صنعت"""
        # در یک پیاده‌سازی واقعی، این باید از داده‌های صنعت استفاده کند
        return 3  # مقدار پیش‌فرض
    
    def industry_average_ps_ratio(self):
        """میانگین نسبت P/S صنعت"""
        # در یک پیاده‌سازی واقعی، این باید از داده‌های صنعت استفاده کند
        return 5  # مقدار پیش‌فرض
    
    def value_from_ratios(self, pe_ratio, pb_ratio, ps_ratio):
        """محاسبه ارزش از نسبت‌ها"""
        try:
            # وزن‌ها برای هر نسبت
            weights = {
                'pe': 0.4,
                'pb': 0.3,
                'ps': 0.3
            }
            
            # مقادیر مرجع
            industry_pe = self.industry_average_pe_ratio()
            industry_pb = self.industry_average_pb_ratio()
            industry_ps = self.industry_average_ps_ratio()
            
            # محاسبه ارزش نسبی
            pe_value = industry_pe / pe_ratio if pe_ratio > 0 else 1
            pb_value = industry_pb / pb_ratio if pb_ratio > 0 else 1
            ps_value = industry_ps / ps_ratio if ps_ratio > 0 else 1
            
            # محاسبه ارزش ترکیبی
            combined_value = (pe_value * weights['pe'] + pb_value * weights['pb'] + ps_value * weights['ps'])
            
            return combined_value
        except Exception as e:
            logger.error(f"Error calculating value from ratios: {e}")
            return 1
    
    def analyze_long_term_growth(self, data):
        """تحلیل رشد بلندمدت"""
        try:
            growth_analysis = {}
            
            # محاسبه نرخ رشد تاریخی
            returns = data['Close'].pct_change().dropna()
            
            # رشد سالانه
            annual_returns = (1 + returns).resample('Y').prod() - 1
            growth_analysis['annual_growth'] = {
                'mean': annual_returns.mean(),
                'median': annual_returns.median(),
                'std': annual_returns.std(),
                'max': annual_returns.max(),
                'min': annual_returns.min()
            }
            
            # رشد مرکب سالانه (CAGR)
            years = len(data) / 365  # فرض داده‌های روزانه
            cagr = (data['Close'].iloc[-1] / data['Close'].iloc[0]) ** (1 / years) - 1
            growth_analysis['cagr'] = cagr
            
            # تحلیل روندهای رشد
            growth_analysis['growth_trends'] = self.analyze_growth_trends(data)
            
            # پیش‌بینی رشد آینده
            growth_analysis['future_growth'] = self.predict_future_growth(data)
            
            return growth_analysis
        except Exception as e:
            logger.error(f"Error in long-term growth analysis: {e}")
            return {}
    
    def analyze_growth_trends(self, data):
        """تحلیل روندهای رشد"""
        try:
            trends = {}
            
            # تحلیل رشد در بازه‌های زمانی مختلف
            periods = {
                '1m': 30,
                '3m': 90,
                '6m': 180,
                '1y': 365,
                '2y': 730
            }
            
            for period_name, days in periods.items():
                if len(data) > days:
                    period_data = data.iloc[-days:]
                    period_return = (period_data['Close'].iloc[-1] / period_data['Close'].iloc[0]) - 1
                    
                    # محاسبه نرخ رشد سالانه
                    years = days / 365
                    annualized_return = (1 + period_return) ** (1 / years) - 1
                    
                    trends[period_name] = {
                        'total_return': period_return,
                        'annualized_return': annualized_return,
                        'volatility': period_data['Close'].pct_change().std() * np.sqrt(252)
                    }
            
            return trends
        except Exception as e:
            logger.error(f"Error analyzing growth trends: {e}")
            return {}
    
    def predict_future_growth(self, data):
        """پیش‌بینی رشد آینده"""
        try:
            # در یک پیاده‌سازی واقعی، این باید از مدل‌های پیشرفته‌تر استفاده کند
            # اینجا یک تخمین ساده ارائه می‌شود
            
            # محاسبه نرخ رشد تاریخی
            returns = data['Close'].pct_change().dropna()
            avg_return = returns.mean()
            std_return = returns.std()
            
            # پیش‌بینی رشد برای 1 سال آینده
            future_return_1y = avg_return
            future_return_1y_upper = avg_return + std_return
            future_return_1y_lower = avg_return - std_return
            
            # پیش‌بینی رشد برای 3 سال آینده
            future_return_3y = avg_return * 3
            future_return_3y_upper = (avg_return + std_return) * 3
            future_return_3y_lower = (avg_return - std_return) * 3
            
            return {
                '1_year': {
                    'expected': future_return_1y,
                    'optimistic': future_return_1y_upper,
                    'pessimistic': future_return_1y_lower
                },
                '3_years': {
                    'expected': future_return_3y,
                    'optimistic': future_return_3y_upper,
                    'pessimistic': future_return_3y_lower
                }
            }
        except Exception as e:
            logger.error(f"Error predicting future growth: {e}")
            return {}
    
    def analyze_risk_reward(self, data, market_data):
        """تحلیل ریسک-پاداش"""
        try:
            risk_reward = {}
            
            # محاسبه نوسان
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # نوسان سالانه
            
            # محاسبه بازده مورد انتظار
            expected_return = returns.mean() * 252  # بازده سالانه
            
            # محاسبه نسبت شارپ
            sharpe_ratio = expected_return / volatility if volatility > 0 else 0
            
            # محاسبه حداکثر افت
            max_drawdown = self.calculate_max_drawdown(data['Close'])
            
            # محاسبه نسبت کالمار
            calmar_ratio = expected_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # تحلیل سناریو
            scenarios = self.analyze_scenarios(data)
            
            risk_reward = {
                'volatility': volatility,
                'expected_return': expected_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'scenarios': scenarios
            }
            
            return risk_reward
        except Exception as e:
            logger.error(f"Error in risk-reward analysis: {e}")
            return {}
    
    def analyze_scenarios(self, data):
        """تحلیل سناریوها"""
        try:
            scenarios = {}
            
            # محاسبه بازده‌ها
            returns = data['Close'].pct_change().dropna()
            
            # سناریوی بهینه
            best_return = returns.quantile(0.95) * 252  # بهترین 5% بازده‌ها
            
            # سناریوی بدبینانه
            worst_return = returns.quantile(0.05) * 252  # بدترین 5% بازده‌ها
            
            # سناریوی محتمل
            likely_return = returns.median() * 252  # میانه بازده‌ها
            
            # محاسبه ارزش در معرض ریسک (VaR)
            var_95 = returns.quantile(0.05)  # Value at Risk 95%
            
            # محاسبه شرطی ارزش در معرض ریسک (CVaR)
            cvar_95 = returns[returns <= var_95].mean()  # Conditional Value at Risk 95%
            
            scenarios = {
                'optimistic': {
                    'return': best_return,
                    'probability': 0.05
                },
                'pessimistic': {
                    'return': worst_return,
                    'probability': 0.05
                },
                'likely': {
                    'return': likely_return,
                    'probability': 0.5
                },
                'var_95': var_95,
                'cvar_95': cvar_95
            }
            
            return scenarios
        except Exception as e:
            logger.error(f"Error in scenario analysis: {e}")
            return {}
    
    def analyze_investment_timing(self, data, sentiment):
        """تحلیل زمان‌بندی سرمایه‌گذاری"""
        try:
            timing = {}
            
            # تحلیل تکنیکال زمان‌بندی
            timing['technical'] = self.analyze_technical_timing(data)
            
            # تحلیل احساسات بازار
            timing['sentiment'] = self.analyze_sentiment_timing(sentiment)
            
            # تحلیل چرخه‌های بازار
            timing['cycles'] = self.analyze_market_cycles_timing(data)
            
            # تحلیل ترکیبی
            timing['combined'] = self.combine_timing_signals(timing)
            
            return timing
        except Exception as e:
            logger.error(f"Error in investment timing analysis: {e}")
            return {}
    
    def analyze_technical_timing(self, data):
        """تحلیل زمان‌بندی تکنیکال"""
        try:
            timing_signals = {}
            
            # شاخص‌های زمان‌بندی
            if LIBRARIES['pandas_ta']:
                # RSI
                rsi = pandas_ta.rsi(data['Close']).iloc[-1]
                timing_signals['rsi'] = 'buy' if rsi < 30 else 'sell' if rsi > 70 else 'neutral'
                
                # MACD
                macd = pandas_ta.macd(data['Close'])
                macd_value = macd['MACD_12_26_9'].iloc[-1]
                macd_signal = macd['MACDs_12_26_9'].iloc[-1]
                timing_signals['macd'] = 'buy' if macd_value > macd_signal else 'sell' if macd_value < macd_signal else 'neutral'
                
                # بولینگر بندز
                bb = pandas_ta.bbands(data['Close'], length=20, std=2)
                current_price = data['Close'].iloc[-1]
                lower_band = bb['BBL_20_2.0'].iloc[-1]
                upper_band = bb['BBU_20_2.0'].iloc[-1]
                
                if current_price < lower_band:
                    timing_signals['bollinger'] = 'buy'
                elif current_price > upper_band:
                    timing_signals['bollinger'] = 'sell'
                else:
                    timing_signals['bollinger'] = 'neutral'
            
            # تحلیل روند
            trend = self.analyze_trend(data)
            timing_signals['trend'] = 'buy' if trend['direction'] == 'صعودی' else 'sell' if trend['direction'] == 'نزولی' else 'neutral'
            
            # تحلیل حجم
            volume_trend = data['Volume'].rolling(10).mean().iloc[-1] / data['Volume'].rolling(50).mean().iloc[-1] - 1
            timing_signals['volume'] = 'buy' if volume_trend > 0.1 else 'sell' if volume_trend < -0.1 else 'neutral'
            
            return timing_signals
        except Exception as e:
            logger.error(f"Error in technical timing analysis: {e}")
            return {}
    
    def analyze_sentiment_timing(self, sentiment):
        """تحلیل زمان‌بندی بر اساس احساسات"""
        try:
            sentiment_signals = {}
            
            # تحلیل احساسات کلی
            avg_sentiment = sentiment.get('average_sentiment', 0)
            
            if avg_sentiment > 0.5:
                sentiment_signals['overall'] = 'buy'
            elif avg_sentiment < -0.5:
                sentiment_signals['overall'] = 'sell'
            else:
                sentiment_signals['overall'] = 'neutral'
            
            # تحلیل شاخص ترس و طمع
            # در یک پیاده‌سازی واقعی، این باید از داده‌های خارجی استفاده کند
            fear_greed_index = np.random.uniform(0, 100)  # مقدار تصادفی برای مثال
            
            if fear_greed_index < 25:
                sentiment_signals['fear_greed'] = 'buy'  # ترس شدید - فرصت خرید
            elif fear_greed_index > 75:
                sentiment_signals['fear_greed'] = 'sell'  # طمع شدید - هشدار فروش
            else:
                sentiment_signals['fear_greed'] = 'neutral'
            
            return sentiment_signals
        except Exception as e:
            logger.error(f"Error in sentiment timing analysis: {e}")
            return {}
    
    def analyze_market_cycles_timing(self, data):
        """تحلیل زمان‌بندی بر اساس چرخه‌های بازار"""
        try:
            cycle_signals = {}
            
            # تحلیل چرخه‌های کوتاه مدت
            short_term_cycles = self.detect_short_term_cycles(data)
            
            if short_term_cycles:
                # چرخه فعلی
                current_cycle = short_term_cycles[0]
                cycle_position = (len(data) % current_cycle['period']) / current_cycle['period']
                
                # تعیین فاز چرخه
                if cycle_position < 0.25:
                    phase = 'accumulation'  # تجمیع
                elif cycle_position < 0.5:
                    phase = 'markup'  # افزایش قیمت
                elif cycle_position < 0.75:
                    phase = 'distribution'  # توزیع
                else:
                    phase = 'markdown'  # کاهش قیمت
                
                cycle_signals['short_term'] = {
                    'current_cycle': {
                        'period': current_cycle['period'],
                        'strength': current_cycle['strength'],
                        'position': cycle_position,
                        'phase': phase
                    },
                    'signal': 'buy' if phase in ['accumulation', 'markup'] else 'sell' if phase in ['distribution', 'markdown'] else 'neutral'
                }
            else:
                cycle_signals['short_term'] = {'message': 'No clear cycles detected'}
            
            # تحلیل چرخه‌های بلند مدت
            long_term_trend = self.analyze_trend(data)
            
            if long_term_trend['direction'] == 'صعودی':
                cycle_signals['long_term'] = 'buy'
            elif long_term_trend['direction'] == 'نزولی':
                cycle_signals['long_term'] = 'sell'
            else:
                cycle_signals['long_term'] = 'neutral'
            
            return cycle_signals
        except Exception as e:
            logger.error(f"Error in market cycles timing analysis: {e}")
            return {}
    
    def combine_timing_signals(self, timing):
        """ترکیب سیگنال‌های زمان‌بندی"""
        try:
            # وزن‌ها برای هر نوع سیگنال
            weights = {
                'technical': 0.5,
                'sentiment': 0.3,
                'cycles': 0.2
            }
            
            # تبدیل سیگنال‌ها به عدد
            signal_values = {
                'buy': 1,
                'neutral': 0,
                'sell': -1
            }
            
            # محاسبه امتیاز هر نوع سیگنال
            technical_score = np.mean([signal_values.get(timing['technical'].get(indicator, 'neutral'), 0) 
                                     for indicator in timing['technical']]) if 'technical' in timing else 0
            
            sentiment_score = np.mean([signal_values.get(timing['sentiment'].get(indicator, 'neutral'), 0) 
                                     for indicator in timing['sentiment']]) if 'sentiment' in timing else 0
            
            cycles_score = np.mean([signal_values.get(timing['cycles'].get(indicator, 'neutral'), 0) 
                                   for indicator in timing['cycles']]) if 'cycles' in timing else 0
            
            # محاسبه امتیاز نهایی
            combined_score = (technical_score * weights['technical'] + 
                             sentiment_score * weights['sentiment'] + 
                             cycles_score * weights['cycles'])
            
            # تعیین سیگنال نهایی
            if combined_score > 0.3:
                final_signal = 'buy'
            elif combined_score < -0.3:
                final_signal = 'sell'
            else:
                final_signal = 'neutral'
            
            return {
                'signal': final_signal,
                'score': combined_score,
                'confidence': abs(combined_score)
            }
        except Exception as e:
            logger.error(f"Error combining timing signals: {e}")
            return {'signal': 'neutral', 'score': 0, 'confidence': 0}
    
    def analyze_arbitrage_opportunities(self, market_data):
        """تحلیل فرصت‌های آربیتراژ"""
        try:
            arbitrage = {}
            
            # آربیتراژ بین صرافی‌ها
            arbitrage['exchange'] = self.analyze_exchange_arbitrage(market_data)
            
            # آربیتراژ سه‌وجهی
            arbitrage['triangular'] = self.analyze_triangular_arbitrage()
            
            # آربیتراژ زمانی
            arbitrage['temporal'] = self.analyze_temporal_arbitrage()
            
            return arbitrage
        except Exception as e:
            logger.error(f"Error in arbitrage opportunity analysis: {e}")
            return {}
    
    def analyze_exchange_arbitrage(self, market_data):
        """تحلیل آربیتراژ بین صرافی‌ها"""
        try:
            opportunities = []
            
            # دریافت داده‌های صرافی‌ها
            exchange_data = market_data.get('exchanges', {})
            
            # اگر داده‌های صرافی موجود است
            if exchange_data:
                # استخراج قیمت‌ها
                prices = {}
                for exchange, data in exchange_data.items():
                    if isinstance(data, dict) and 'price' in data and data['price'] > 0:
                        prices[exchange] = data['price']
                
                # یافتن فرصت‌های آربیتراژ
                if len(prices) >= 2:
                    # محاسبه کمترین و بیشترین قیمت
                    min_price = min(prices.values())
                    max_price = max(prices.values())
                    
                    # محاسبه تفاوت قیمت
                    price_diff = max_price - min_price
                    price_diff_percent = (price_diff / min_price) * 100
                    
                    # اگر تفاوت قیمت قابل توجه باشد
                    if price_diff_percent > 0.5:  # تفاوت بیش از 0.5%
                        # یافتن صرافی‌ها با کمترین و بیشترین قیمت
                        min_exchange = min(prices, key=prices.get)
                        max_exchange = max(prices, key=prices.get)
                        
                        opportunities.append({
                            'type': 'exchange',
                            'buy_exchange': min_exchange,
                            'sell_exchange': max_exchange,
                            'buy_price': prices[min_exchange],
                            'sell_price': prices[max_exchange],
                            'price_diff': price_diff,
                            'price_diff_percent': price_diff_percent,
                            'potential_profit': price_diff_percent - 0.2  # کسر کارمزد
                        })
            
            return opportunities
        except Exception as e:
            logger.error(f"Error in exchange arbitrage analysis: {e}")
            return []
    
    def analyze_triangular_arbitrage(self):
        """تحلیل آربیتراژ سه‌وجهی"""
        try:
            opportunities = []
            
            # در یک پیاده‌سازی واقعی، این باید از داده‌های زنده صرافی‌ها استفاده کند
            # اینجا یک تحلیل مفهومی ارائه می‌شود
            
            # جفت‌ارزهای اصلی برای آربیتراژ سه‌وجهی
            triangles = [
                ('BTC/USDT', 'ETH/USDT', 'ETH/BTC'),
                ('BTC/USDT', 'BNB/USDT', 'BNB/BTC'),
                ('ETH/USDT', 'BNB/USDT', 'BNB/ETH')
            ]
            
            for triangle in triangles:
                # دریافت داده‌های تیکر
                tickers = {}
                for pair in triangle:
                    for exchange_name, exchange in self.exchanges.items():
                        try:
                            ticker = exchange.fetch_ticker(pair)
                            tickers[pair] = {
                                'exchange': exchange_name,
                                'price': ticker['last']
                            }
                            break
                        except:
                            pass
                
                # اگر داده‌های همه جفت‌ارزها موجود است
                if len(tickers) == 3:
                    # محاسبه قیمت محاسبه شده
                    # برای مثلث A/B, B/C, A/C: (A/B) * (B/C) باید برابر با A/C باشد
                    calculated_price = tickers[triangle[0]]['price'] / tickers[triangle[2]]['price']
                    actual_price = tickers[triangle[1]]['price']
                    
                    # محاسبه تفاوت قیمت
                    price_diff = abs(calculated_price - actual_price)
                    price_diff_percent = (price_diff / actual_price) * 100
                    
                    # اگر تفاوت قیمت قابل توجه باشد
                    if price_diff_percent > 0.5:  # تفاوت بیش از 0.5%
                        opportunities.append({
                            'type': 'triangular',
                            'pairs': triangle,
                            'calculated_price': calculated_price,
                            'actual_price': actual_price,
                            'price_diff_percent': price_diff_percent,
                            'potential_profit': price_diff_percent - 0.3  # کسر کارمزد
                        })
            
            return opportunities
        except Exception as e:
            logger.error(f"Error in triangular arbitrage analysis: {e}")
            return []
    
    def analyze_temporal_arbitrage(self):
        """تحلیل آربیتراژ زمانی"""
        try:
            opportunities = []
            
            # در یک پیاده‌سازی واقعی، این باید از داده‌های تاریخی و پیش‌بینی‌ها استفاده کند
            # اینجا یک تحلیل مفهومی ارائه می‌شود
            
            # برای هر صرافی، بررسی تفاوت قیمت بین لحظه و آینده
            for exchange_name, exchange in self.exchanges.items():
                try:
                    # دریافت داده‌های تاریخی کوتاه مدت
                    # در عمل باید از داده‌های واقعی استفاده شود
                    
                    # برای مثال، فرض می‌کنیم قیمت آینده را داریم
                    current_price = 100  # مقدار فرضی
                    future_price = 101   # مقدار فرضی
                    
                    # محاسبه تفاوت قیمت
                    price_diff = future_price - current_price
                    price_diff_percent = (price_diff / current_price) * 100
                    
                    # اگر تفاوت قیمت قابل توجه باشد
                    if abs(price_diff_percent) > 0.5:  # تفاوت بیش از 0.5%
                        opportunities.append({
                            'type': 'temporal',
                            'exchange': exchange_name,
                            'current_price': current_price,
                            'future_price': future_price,
                            'price_diff_percent': price_diff_percent,
                            'direction': 'buy' if price_diff > 0 else 'sell',
                            'potential_profit': abs(price_diff_percent) - 0.1  # کسر کارمزد
                        })
                except Exception as e:
                    logger.warning(f"Error analyzing temporal arbitrage for {exchange_name}: {e}")
            
            return opportunities
        except Exception as e:
            logger.error(f"Error in temporal arbitrage analysis: {e}")
            return []
    
    def analyze_market_timing(self, data):
        """تحلیل زمان‌بندی بازار"""
        try:
            timing = {}
            
            # تحلیل زمان‌بندی تکنیکال
            timing['technical'] = self.analyze_technical_timing(data)
            
            # تحلیل زمان‌بندی چرخه‌ای
            timing['cyclical'] = self.analyze_cyclical_timing(data)
            
            # تحلیل زمان‌بندی فصلی
            timing['seasonal'] = self.analyze_seasonal_timing(data)
            
            # تحلیل زمان‌بندی مبتنی بر رویداد
            timing['event'] = self.analyze_event_timing(data)
            
            return timing
        except Exception as e:
            logger.error(f"Error in market timing analysis: {e}")
            return {}
    
    def analyze_cyclical_timing(self, data):
        """تحلیل زمان‌بندی چرخه‌ای"""
        try:
            cyclical_timing = {}
            
            # تحلیل چرخه‌های بازار
            cycles = self.detect_short_term_cycles(data)
            
            if cycles:
                # چرخه فعلی
                current_cycle = cycles[0]
                cycle_position = (len(data) % current_cycle['period']) / current_cycle['period']
                
                # تعیین فاز چرخه
                if cycle_position < 0.25:
                    phase = 'accumulation'  # تجمیع
                elif cycle_position < 0.5:
                    phase = 'markup'  # افزایش قیمت
                elif cycle_position < 0.75:
                    phase = 'distribution'  # توزیع
                else:
                    phase = 'markdown'  # کاهش قیمت
                
                cyclical_timing = {
                    'current_cycle': {
                        'period': current_cycle['period'],
                        'strength': current_cycle['strength'],
                        'position': cycle_position,
                        'phase': phase
                    },
                    'signal': 'buy' if phase in ['accumulation', 'markup'] else 'sell' if phase in ['distribution', 'markdown'] else 'neutral'
                }
            else:
                cyclical_timing = {'message': 'No clear cycles detected'}
            
            return cyclical_timing
        except Exception as e:
            logger.error(f"Error in cyclical timing analysis: {e}")
            return {}
    
    def analyze_seasonal_timing(self, data):
        """تحلیل زمان‌بندی فصلی"""
        try:
            seasonal_timing = {}
            
            # استخراج ماه از داده‌ها
            data_with_month = data.copy()
            data_with_month['month'] = data_with_month.index.month
            
            # محاسبه میانگین بازده ماهانه
            monthly_returns = data_with_month.groupby('month')['Close'].apply(lambda x: x.pct_change().mean())
            
            # ماه فعلی
            current_month = data.index[-1].month
            
            # بازده ماه فعلی
            current_month_return = monthly_returns.get(current_month, 0)
            
            # تعیین سیگنال بر اساس بازده ماهانه
            if current_month_return > 0.02:  # بازده مثبت قوی
                signal = 'buy'
            elif current_month_return < -0.02:  # بازده منفی قوی
                signal = 'sell'
            else:
                signal = 'neutral'
            
            seasonal_timing = {
                'monthly_returns': monthly_returns.to_dict(),
                'current_month': current_month,
                'current_month_return': current_month_return,
                'signal': signal
            }
            
            return seasonal_timing
        except Exception as e:
            logger.error(f"Error in seasonal timing analysis: {e}")
            return {}
    
    def analyze_event_timing(self, data):
        """تحلیل زمان‌بندی مبتنی بر رویداد"""
        try:
            event_timing = {}
            
            # در یک پیاده‌سازی واقعی، این باید از تقویم رویدادها استفاده کند
            # اینجا یک تحلیل مفهومی ارائه می‌شود
            
            # رویدادهای آینده
            future_events = [
                {'date': datetime.now() + timedelta(days=30), 'type': 'halving', 'impact': 'high'},
                {'date': datetime.now() + timedelta(days=60), 'type': 'listing', 'impact': 'medium'},
                {'date': datetime.now() + timedelta(days=90), 'type': 'upgrade', 'impact': 'medium'}
            ]
            
            # تحلیل تأثیر رویدادها
            event_impacts = []
            for event in future_events:
                # در یک پیاده‌سازی واقعی، این باید بر اساس داده‌های تاریخی محاسبه شود
                # اینجا یک تخمین ساده ارائه می‌شود
                
                if event['type'] == 'halving':
                    expected_impact = 0.2  # افزایش 20% قیمت
                elif event['type'] == 'listing':
                    expected_impact = 0.1  # افزایش 10% قیمت
                elif event['type'] == 'upgrade':
                    expected_impact = 0.05  # افزایش 5% قیمت
                else:
                    expected_impact = 0
                
                event_impacts.append({
                    'date': event['date'],
                    'type': event['type'],
                    'impact': event['impact'],
                    'expected_impact': expected_impact,
                    'days_until': (event['date'] - datetime.now()).days
                })
            
            event_timing = {
                'future_events': event_impacts,
                'overall_signal': 'buy' if any(e['expected_impact'] > 0.1 for e in event_impacts) else 'neutral'
            }
            
            return event_timing
        except Exception as e:
            logger.error(f"Error in event timing analysis: {e}")
            return {}
    
    def analyze_price_behavior(self, data):
        """تحلیل رفتار قیمت"""
        try:
            behavior = {}
            
            # تحلیل الگوهای رفتاری
            behavior['patterns'] = self.analyze_behavioral_patterns(data)
            
            # تحلیل روانشناسی بازار
            behavior['psychology'] = self.analyze_market_psychology(data)
            
            # تحلیل رفتار نهنگ‌ها
            behavior['whale_behavior'] = self.analyze_whale_behavior(data)
            
            return behavior
        except Exception as e:
            logger.error(f"Error in price behavior analysis: {e}")
            return {}
    
    def analyze_behavioral_patterns(self, data):
        """تحلیل الگوهای رفتاری"""
        try:
            patterns = {}
            
            # تحلیل الگوی FOMO (ترس از دست دادن)
            patterns['fomo'] = self.detect_fomo_pattern(data)
            
            # تحلیل الگوی پانیک (هراس)
            patterns['panic'] = self.detect_panic_pattern(data)
            
            # تحلیل الگوی FUD (ترس، عدم قطعیت و تردید)
            patterns['fud'] = self.detect_fud_pattern(data)
            
            # تحلیل الگوی HODL (نگه داشتن)
            patterns['hodl'] = self.detect_hodl_pattern(data)
            
            return patterns
        except Exception as e:
            logger.error(f"Error in behavioral pattern analysis: {e}")
            return {}
    
    def detect_fomo_pattern(self, data):
        """تشخیص الگوی FOMO"""
        try:
            # شاخص‌های FOMO
            returns = data['Close'].pct_change().dropna()
            
            # افزایش سریع قیمت
            rapid_increase = returns.rolling(5).mean().iloc[-1] > 0.05  # افزایش متوسط 5% در 5 روز
            
            # افزایش حجم
            volume_increase = data['Volume'].rolling(5).mean().iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1] > 1.5  # افزایش 50% حجم
            
            # RSI بالا
            rsi_high = False
            if LIBRARIES['pandas_ta']:
                rsi = pandas_ta.rsi(data['Close']).iloc[-1]
                rsi_high = rsi > 70
            
            # ترکیب شاخص‌ها
            fomo_score = (1 if rapid_increase else 0) + (1 if volume_increase else 0) + (1 if rsi_high else 0)
            
            return {
                'detected': fomo_score >= 2,
                'score': fomo_score,
                'rapid_increase': rapid_increase,
                'volume_increase': volume_increase,
                'rsi_high': rsi_high
            }
        except Exception as e:
            logger.error(f"Error detecting FOMO pattern: {e}")
            return {'detected': False, 'error': str(e)}
    
    def detect_panic_pattern(self, data):
        """تشخیص الگوی پانیک"""
        try:
            # شاخص‌های پانیک
            returns = data['Close'].pct_change().dropna()
            
            # کاهش سریع قیمت
            rapid_decrease = returns.rolling(5).mean().iloc[-1] < -0.05  # کاهش متوسط 5% در 5 روز
            
            # افزایش حجم
            volume_increase = data['Volume'].rolling(5).mean().iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1] > 1.5  # افزایش 50% حجم
            
            # RSI پایین
            rsi_low = False
            if LIBRARIES['pandas_ta']:
                rsi = pandas_ta.rsi(data['Close']).iloc[-1]
                rsi_low = rsi < 30
            
            # ترکیب شاخص‌ها
            panic_score = (1 if rapid_decrease else 0) + (1 if volume_increase else 0) + (1 if rsi_low else 0)
            
            return {
                'detected': panic_score >= 2,
                'score': panic_score,
                'rapid_decrease': rapid_decrease,
                'volume_increase': volume_increase,
                'rsi_low': rsi_low
            }
        except Exception as e:
            logger.error(f"Error detecting panic pattern: {e}")
            return {'detected': False, 'error': str(e)}
    
    def detect_fud_pattern(self, data):
        """تشخیص الگوی FUD"""
        try:
            # شاخص‌های FUD
            returns = data['Close'].pct_change().dropna()
            
            # کاهش تدریجی قیمت
            gradual_decrease = returns.rolling(10).mean().iloc[-1] < -0.02  # کاهش متوسط 2% در 10 روز
            
            # کاهش حجم
            volume_decrease = data['Volume'].rolling(10).mean().iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1] < 0.8  # کاهش 20% حجم
            
            # نوسانات پایین
            low_volatility = returns.rolling(10).std().iloc[-1] < 0.02  # نوسان کمتر از 2%
            
            # ترکیب شاخص‌ها
            fud_score = (1 if gradual_decrease else 0) + (1 if volume_decrease else 0) + (1 if low_volatility else 0)
            
            return {
                'detected': fud_score >= 2,
                'score': fud_score,
                'gradual_decrease': gradual_decrease,
                'volume_decrease': volume_decrease,
                'low_volatility': low_volatility
            }
        except Exception as e:
            logger.error(f"Error detecting FUD pattern: {e}")
            return {'detected': False, 'error': str(e)}
    
    def detect_hodl_pattern(self, data):
        """تشخیص الگوی HODL"""
        try:
            # شاخص‌های HODL
            returns = data['Close'].pct_change().dropna()
            
            # ثبات قیمت
            price_stability = returns.rolling(20).std().iloc[-1] < 0.01  # نوسان کمتر از 1%
            
            # حجم پایین
            low_volume = data['Volume'].rolling(20).mean().iloc[-1] / data['Volume'].rolling(50).mean().iloc[-1] < 0.8  # کاهش 20% حجم
            
            # عدم وجود روند قوی
            no_strong_trend = abs(self.analyze_trend(data)['slope']) < 0.01
            
            # ترکیب شاخص‌ها
            hodl_score = (1 if price_stability else 0) + (1 if low_volume else 0) + (1 if no_strong_trend else 0)
            
            return {
                'detected': hodl_score >= 2,
                'score': hodl_score,
                'price_stability': price_stability,
                'low_volume': low_volume,
                'no_strong_trend': no_strong_trend
            }
        except Exception as e:
            logger.error(f"Error detecting HODL pattern: {e}")
            return {'detected': False, 'error': str(e)}
    
    def analyze_market_psychology(self, data):
        """تحلیل روانشناسی بازار"""
        try:
            psychology = {}
            
            # تحلیل شاخص ترس و طمع
            psychology['fear_greed'] = self.analyze_fear_greed_index(data)
            
            # تحلیل شاخص آگاهی بازار
            psychology['market_awareness'] = self.analyze_market_awareness(data)
            
            # تحلیل شاخص هیجان بازار
            psychology['market_sentiment'] = self.analyze_market_sentiment(data)
            
            return psychology
        except Exception as e:
            logger.error(f"Error in market psychology analysis: {e}")
            return {}
    
    def analyze_fear_greed_index(self, data):
        """تحلیل شاخص ترس و طمع"""
        try:
            # در یک پیاده‌سازی واقعی، این باید از داده‌های خارجی استفاده کند
            # اینجا یک تخمین ساده ارائه می‌شود
            
            # شاخص‌های جزئی
            returns = data['Close'].pct_change().dropna()
            
            # نوسانات (25% شاخص)
            volatility = returns.std() * np.sqrt(252)
            volatility_score = max(0, min(100, 50 - volatility * 2500))  # نوسانات کمتر = طمع بیشتر
            
            # مومنتوم قیمت (25% شاخص)
            momentum = returns.rolling(14).mean().iloc[-1] * 252
            momentum_score = max(0, min(100, 50 + momentum * 1000))  # مومنتوم مثبت = طمع بیشتر
            
            # حجم (25% شاخص)
            volume_ratio = data['Volume'].rolling(30).mean().iloc[-1] / data['Volume'].rolling(90).mean().iloc[-1]
            volume_score = max(0, min(100, 50 + (volume_ratio - 1) * 100))  # حجم بیشتر = طمع بیشتر
            
            # قدرت خریداران در مقابل فروشندگان (25% شاخص)
            # در یک پیاده‌سازی واقعی، این باید از داده‌های سفارشات استفاده کند
            buyer_seller_score = np.random.randint(0, 100)  # مقدار تصادفی برای مثال
            
            # محاسبه شاخص نهایی
            fear_greed_index = (
                volatility_score * 0.25 +
                momentum_score * 0.25 +
                volume_score * 0.25 +
                buyer_seller_score * 0.25
            )
            
            # تعیین وضعیت
            if fear_greed_index > 75:
                status = 'extreme greed'
            elif fear_greed_index > 55:
                status = 'greed'
            elif fear_greed_index > 45:
                status = 'neutral'
            elif fear_greed_index > 25:
                status = 'fear'
            else:
                status = 'extreme fear'
            
            return {
                'index': fear_greed_index,
                'status': status,
                'components': {
                    'volatility': volatility_score,
                    'momentum': momentum_score,
                    'volume': volume_score,
                    'buyer_seller': buyer_seller_score
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing fear and greed index: {e}")
            return {}
    
    def analyze_market_awareness(self, data):
        """تحلیل شاخص آگاهی بازار"""
        try:
            # در یک پیاده‌سازی واقعی، این باید از داده‌های خارجی استفاده کند
            # اینجا یک تخمین ساده ارائه می‌شود
            
            # شاخص‌های جزئی
            returns = data['Close'].pct_change().dropna()
            
            # توجه رسانه‌ها (40% شاخص)
            # در یک پیاده‌سازی واقعی، این باید از تحلیل اخبار استفاده کند
            media_attention = np.random.uniform(0, 100)  # مقدار تصادفی برای مثال
            
            # جستجوهای گوگل (30% شاخص)
            # در یک پیاده‌سازی واقعی، این باید از داده‌های گوگل ترندز استفاده کند
            google_searches = np.random.uniform(0, 100)  # مقدار تصادفی برای مثال
            
            # فعالیت شبکه‌های اجتماعی (30% شاخص)
            # در یک پیاده‌سازی واقعی، این باید از داده‌های شبکه‌های اجتماعی استفاده کند
            social_media = np.random.uniform(0, 100)  # مقدار تصادفی برای مثال
            
            # محاسبه شاخص نهایی
            market_awareness = (
                media_attention * 0.4 +
                google_searches * 0.3 +
                social_media * 0.3
            )
            
            # تعیین وضعیت
            if market_awareness > 80:
                status = 'very high'
            elif market_awareness > 60:
                status = 'high'
            elif market_awareness > 40:
                status = 'moderate'
            elif market_awareness > 20:
                status = 'low'
            else:
                status = 'very low'
            
            return {
                'index': market_awareness,
                'status': status,
                'components': {
                    'media_attention': media_attention,
                    'google_searches': google_searches,
                    'social_media': social_media
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing market awareness: {e}")
            return {}
    
    def analyze_market_sentiment(self, data):
        """تحلیل شاخص هیجان بازار"""
        try:
            # در یک پیاده‌سازی واقعی، این باید از داده‌های خارجی استفاده کند
            # اینجا یک تخمین ساده ارائه می‌شود
            
            # شاخص‌های جزئی
            returns = data['Close'].pct_change().dropna()
            
            # احساسات سرمایه‌گذاران (50% شاخص)
            # در یک پیاده‌سازی واقعی، این باید از نظرسنجی‌ها استفاده کند
            investor_sentiment = np.random.uniform(0, 100)  # مقدار تصادفی برای مثال
            
            # تحلیل اخبار (30% شاخص)
            # در یک پیاده‌سازی واقعی، این باید از تحلیل اخبار استفاده کند
            news_sentiment = np.random.uniform(0, 100)  # مقدار تصادفی برای مثال
            
            # تحلیل شبکه‌های اجتماعی (20% شاخص)
            # در یک پیاده‌سازی واقعی، این باید از تحلیل شبکه‌های اجتماعی استفاده کند
            social_sentiment = np.random.uniform(0, 100)  # مقدار تصادفی برای مثال
            
            # محاسبه شاخص نهایی
            market_sentiment = (
                investor_sentiment * 0.5 +
                news_sentiment * 0.3 +
                social_sentiment * 0.2
            )
            
            # تعیین وضعیت
            if market_sentiment > 80:
                status = 'very bullish'
            elif market_sentiment > 60:
                status = 'bullish'
            elif market_sentiment > 40:
                status = 'neutral'
            elif market_sentiment > 20:
                status = 'bearish'
            else:
                status = 'very bearish'
            
            return {
                'index': market_sentiment,
                'status': status,
                'components': {
                    'investor_sentiment': investor_sentiment,
                    'news_sentiment': news_sentiment,
                    'social_sentiment': social_sentiment
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
            return {}
    
    def analyze_whale_behavior(self, data):
        """تحلیل رفتار نهنگ‌ها"""
        try:
            whale_behavior = {}
            
            # تحلیل تراکنش‌های بزرگ
            whale_behavior['large_transactions'] = self.analyze_large_transactions(data)
            
            # تحلیل جریان نقدینگی نهنگ‌ها
            whale_behavior['whale_flows'] = self.analyze_whale_flows(data)
            
            # تحلیل توزیع دارایی
            whale_behavior['wealth_distribution'] = self.analyze_wealth_distribution(data)
            
            return whale_behavior
        except Exception as e:
            logger.error(f"Error in whale behavior analysis: {e}")
            return {}
    
    def analyze_large_transactions(self, data):
        """تحلیل تراکنش‌های بزرگ"""
        try:
            # در یک پیاده‌سازی واقعی، این باید از داده‌های بلاکچین استفاده کند
            # اینجا یک تحلیل مفهویی ارائه می‌شود
            
            # برای مثال، فرض می‌کنیم داده‌های تراکنش‌های بزرگ را داریم
            large_txs = [
                {'date': data.index[-5], 'amount': 1000000, 'type': 'inflow'},
                {'date': data.index[-4], 'amount': 500000, 'type': 'outflow'},
                {'date': data.index[-3], 'amount': 2000000, 'type': 'inflow'},
                {'date': data.index[-2], 'amount': 750000, 'type': 'outflow'},
                {'date': data.index[-1], 'amount': 1500000, 'type': 'inflow'}
            ]
            
            # محاسبه خالص جریان
            net_flow = sum(tx['amount'] if tx['type'] == 'inflow' else -tx['amount'] for tx in large_txs)
            
            # تحلیل الگو
            if net_flow > 0:
                pattern = 'accumulation'  # تجمیع
            elif net_flow < 0:
                pattern = 'distribution'  # توزیع
            else:
                pattern = 'neutral'  # خنثی
            
            return {
                'large_transactions': large_txs,
                'net_flow': net_flow,
                'pattern': pattern,
                'signal': 'buy' if pattern == 'accumulation' else 'sell' if pattern == 'distribution' else 'neutral'
            }
        except Exception as e:
            logger.error(f"Error analyzing large transactions: {e}")
            return {}
    
    def analyze_whale_flows(self, data):
        """تحلیل جریان نقدینگی نهنگ‌ها"""
        try:
            # در یک پیاده‌سازی واقعی، این باید از داده‌های بلاکچین استفاده کند
            # اینجا یک تحلیل مفهویی ارائه می‌شود
            
            # جریان‌های نقدینگی نهنگ‌ها
            whale_flows = {
                'exchange_inflows': np.random.uniform(100000, 1000000),  # ورودی به صرافی‌ها
                'exchange_outflows': np.random.uniform(100000, 1000000),  # خروجی از صرافی‌ها
                'wallet_transfers': np.random.uniform(50000, 500000),  # انتقال بین کیف‌پول‌ها
                'miner_movements': np.random.uniform(10000, 100000)  # حرکت ماینرها
            }
            
            # محاسبه خالص جریان
            net_flow = whale_flows['exchange_outflows'] - whale_flows['exchange_inflows']
            
            # تحلیل الگو
            if net_flow > 0:
                pattern = 'accumulation'  # تجمیع
            elif net_flow < 0:
                pattern = 'distribution'  # توزیع
            else:
                pattern = 'neutral'  # خنثی
            
            return {
                'flows': whale_flows,
                'net_flow': net_flow,
                'pattern': pattern,
                'signal': 'buy' if pattern == 'accumulation' else 'sell' if pattern == 'distribution' else 'neutral'
            }
        except Exception as e:
            logger.error(f"Error analyzing whale flows: {e}")
            return {}
    
    def analyze_wealth_distribution(self, data):
        """تحلیل توزیع ثروت"""
        try:
            # در یک پیاده‌سازی واقعی، این باید از داده‌های بلاکچین استفاده کند
            # اینجا یک تحلیل مفهویی ارائه می‌شود
            
            # توزیع ثروت
            wealth_distribution = {
                'top_1_percent': np.random.uniform(20, 50),  # درصد دارایی در دسترس 1% برتر
                'top_10_percent': np.random.uniform(50, 80),  # درصد دارایی در دسترس 10% برتر
                'top_100_wallets': np.random.uniform(10, 30),  # درصد دارایی در دسترس 100 کیف‌پول برتر
                'gini_coefficient': np.random.uniform(0.5, 0.9)  # ضریب جینی (0 تا 1، هر چه بیشتر، نابرابری بیشتر)
            }
            
            # تحلیل تمرکز
            if wealth_distribution['gini_coefficient'] > 0.7:
                concentration = 'very high'
            elif wealth_distribution['gini_coefficient'] > 0.5:
                concentration = 'high'
            elif wealth_distribution['gini_coefficient'] > 0.3:
                concentration = 'moderate'
            else:
                concentration = 'low'
            
            return {
                'distribution': wealth_distribution,
                'concentration': concentration,
                'risk_level': 'high' if concentration in ['very high', 'high'] else 'medium' if concentration == 'moderate' else 'low'
            }
        except Exception as e:
            logger.error(f"Error analyzing wealth distribution: {e}")
            return {}
    
    def calculate_signal_score(self, analysis):
        """محاسبه امتیاز سیگنال نهایی"""
        try:
            # وزن‌ها برای هر نوع تحلیل
            weights = {
                'market_data': 0.15,
                'sentiment': 0.15,
                'technical': 0.3,
                'elliott': 0.1,
                'supply_demand': 0.1,
                'ai_analysis': 0.2
            }
            
            # محاسبه امتیاز هر بخش
            scores = {}
            
            # امتیاز داده‌های بازار
            market_data = analysis.get('market_data', {})
            price_change = market_data.get('price_change_24h', 0)
            scores['market_data'] = max(0, min(1, 0.5 + price_change / 10))  # نرمال‌سازی بین 0 و 1
            
            # امتیاز احساسات
            sentiment = analysis.get('sentiment', {})
            avg_sentiment = sentiment.get('average_sentiment', 0)
            scores['sentiment'] = max(0, min(1, (avg_sentiment + 1) / 2))  # تبدیل از [-1,1] به [0,1]
            
            # امتیاز تحلیل تکنیکال
            technical = analysis.get('technical', {})
            classical = technical.get('classical', {})
            rsi = classical.get('rsi', {}).get('14', 50)
            macd = classical.get('macd', {})
            macd_value = macd.get('value', 0)
            macd_signal = macd.get('signal', 0)
            
            # محاسبه امتیاز تکنیکال
            rsi_score = 1 - abs(rsi - 50) / 50  # RSI نزدیک به 50 امتیاز بالاتر دارد
            macd_score = 0.5 if macd_value > macd_signal else 0.5  # MACD بالاتر از سیگنال امتیاز بالاتر دارد
            
            scores['technical'] = (rsi_score + macd_score) / 2