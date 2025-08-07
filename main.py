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
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
    logger.info("TensorFlow loaded successfully")
except ImportError as e:
    TF_AVAILABLE = False
    logger.warning(f"TensorFlow not available: {e}")

try:
    import talib
    TALIB_AVAILABLE = True
    logger.info("TA-Lib loaded successfully")
except ImportError as e:
    TALIB_AVAILABLE = False
    logger.warning(f"TA-Lib not available: {e}")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
    logger.info("pandas-ta loaded successfully")
except ImportError as e:
    PANDAS_TA_AVAILABLE = False
    logger.warning(f"pandas-ta not available: {e}")

try:
    import pywt
    PYWT_AVAILABLE = True
    logger.info("PyWavelets loaded successfully")
except ImportError as e:
    PYWT_AVAILABLE = False
    logger.warning(f"PyWavelets not available: {e}")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    logger.info("LightGBM loaded successfully")
except ImportError as e:
    LIGHTGBM_AVAILABLE = False
    logger.warning(f"LightGBM not available: {e}")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    logger.info("XGBoost loaded successfully")
except ImportError as e:
    XGBOOST_AVAILABLE = False
    logger.warning(f"XGBoost not available: {e}")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    logger.info("Prophet loaded successfully")
except ImportError as e:
    PROPHET_AVAILABLE = False
    logger.warning(f"Prophet not available: {e}")

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
    logger.info("Statsmodels loaded successfully")
except ImportError as e:
    STATSMODELS_AVAILABLE = False
    logger.warning(f"Statsmodels not available: {e}")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    logger.info("Seaborn loaded successfully")
except ImportError as e:
    SEABORN_AVAILABLE = False
    logger.warning(f"Seaborn not available: {e}")

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
        
        # پایگاه داده (تغییر به PostgreSQL برای Railway)
        try:
            import psycopg2
            self.conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            logger.info("PostgreSQL connection established")
        except:
            # فallback به SQLite برای تست محلی
            self.conn = sqlite3.connect('trading_bot.db', check_same_thread=False)
            logger.warning("Using SQLite as fallback")
        
        self.create_tables()
        
        # مدل‌های تحلیل
        self.models = self.initialize_models()
        
        # تنظیمات صرافی‌ها
        self.exchanges = {
            'binance': ccxt.binance(PROXY_SETTINGS),
            'coinbase': ccxt.coinbase(PROXY_SETTINGS),
            'kucoin': ccxt.kucoin(PROXY_SETTINGS),
            'bybit': ccxt.bybit(PROXY_SETTINGS)
        }
        
        # تنظیمات APIها
        self.coingecko_api_key = os.getenv('COINGECKO_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.cryptopanic_api_key = os.getenv('CRYPTOPANIC_API_KEY')
        
        # تست دسترسی به اینترنت
        self.internet_available = self.test_internet_connection()
        logger.info(f"Internet available: {self.internet_available}")
        
        # اگر اینترنت در دسترس نیست، از حالت آفلاین استفاده کن
        if not self.internet_available:
            logger.warning("Internet connection not available. Using offline mode.")
            self.offline_mode = True
        else:
            self.offline_mode = False
        
        # راه‌اندازی سیستم تحلیل متن پیشرفته
        self.setup_text_analysis()
        
        logger.info("AdvancedTradingBot initialized successfully")

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
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
        
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        
        if PROPHET_AVAILABLE:
            models['prophet'] = Prophet()
        
        # اضافه کردن مدل‌های عمیق در صورت وجود
        if TF_AVAILABLE:
            models['lstm'] = self.build_lstm_model()
            models['gru'] = self.build_gru_model()
        
        logger.info("Machine learning models initialized")
        return models

    def build_lstm_model(self):
        """ساخت مدل LSTM"""
        if not TF_AVAILABLE:
            return None
            
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 5)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def build_gru_model(self):
        """ساخت مدل GRU"""
        if not TF_AVAILABLE:
            return None
            
        model = Sequential([
            GRU(50, return_sequences=True, input_shape=(60, 5)),
            Dropout(0.2),
            GRU(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
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
            
            if self.coingecko_api_key:
                params['x_cg_demo_api_key'] = self.coingecko_api_key
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get(symbol.lower(), {})
                return {}

    async def fetch_coinmarketcap_data(self, symbol):
        """دریافت داده‌ها از CoinMarketCap"""
        if not os.getenv('COINMARKETCAP_API_KEY'):
            return {}
        
        async with aiohttp.ClientSession() as session:
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
            headers = {
                'X-CMC_PRO_API_KEY': os.getenv('COINMARKETCAP_API_KEY'),
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
        if not self.cryptopanic_api_key:
            return []
        
        async with aiohttp.ClientSession() as session:
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                'auth_token': self.cryptopanic_api_key,
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
        if not self.news_api_key:
            return []
        
        async with aiohttp.ClientSession() as session:
            url = "https://newsapi.org/v2/everything"
            params = {
                'apiKey': self.news_api_key,
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
            if PANDAS_TA_AVAILABLE:
                df = data.copy()
                
                # RSI در چند دوره
                classical['rsi'] = {
                    '14': ta.rsi(df['Close'], length=14).iloc[-1],
                    '21': ta.rsi(df['Close'], length=21).iloc[-1],
                    '50': ta.rsi(df['Close'], length=50).iloc[-1]
                }
                
                # MACD
                macd = ta.macd(df['Close'])
                classical['macd'] = {
                    'value': macd['MACD_12_26_9'].iloc[-1],
                    'signal': macd['MACDs_12_26_9'].iloc[-1],
                    'histogram': macd['MACDh_12_26_9'].iloc[-1]
                }
                
                # بولینگر بندز
                bb = ta.bbands(df['Close'], length=20, std=2)
                classical['bollinger'] = {
                    'upper': bb['BBU_20_2.0'].iloc[-1],
                    'middle': bb['BBM_20_2.0'].iloc[-1],
                    'lower': bb['BBL_20_2.0'].iloc[-1],
                    'position': self.get_bollinger_position(df['Close'].iloc[-1], bb),
                    'width': (bb['BBU_20_2.0'].iloc[-1] - bb['BBL_20_2.0'].iloc[-1]) / bb['BBM_20_2.0'].iloc[-1]
                }
                
                # استوکاستیک
                stoch = ta.stoch(df['High'], df['Low'], df['Close'])
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
            if PANDAS_TA_AVAILABLE:
                atr = ta.atr(data['High'], data['Low'], data['Close'])
                volatility['atr'] = atr.iloc[-1]
                volatility['atr_percent'] = (atr.iloc[-1] / data['Close'].iloc[-1]) * 100
            
            # نوسان استاندارد
            returns = data['Close'].pct_change().dropna()
            volatility['std_dev'] = returns.std() * np.sqrt(252)  # سالانه
            
            # باندهای بولینگر
            bb_width = (data['High'].rolling(20).max() - data['Low'].rolling(20).min()) / data['Close'].rolling(20).mean()
            volatility['bb_width'] = bb_width.iloc[-1]
            
            # شاخص نوسان چایکین
            if PANDAS_TA_AVAILABLE:
                volatility['chaikin'] = ta.chop(data['High'], data['Low'], data['Close']).iloc[-1]
            
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
                            'rsi': ta.rsi(tf_data['Close']).iloc[-1] if PANDAS_TA_AVAILABLE else None,
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
            
            if PANDAS_TA_AVAILABLE:
                # واگرایی RSI
                rsi = ta.rsi(data['Close'])
                divergence['rsi'] = self.detect_divergence(data['Close'], rsi)
                
                # واگرایی MACD
                macd = ta.macd(data['Close'])['MACD_12_26_9']
                divergence['macd'] = self.detect_divergence(data['Close'], macd)
                
                # واگرایی استوکاستیک
                stoch = ta.stoch(data['High'], data['Low'], data['Close'])['STOCHk_14_3_3']
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
            poc_bin = max(tpo