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
import schedule
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
    level=logging.DEBUG,
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
        self.conn = sqlite3.connect('trading_bot.db', check_same_thread=False)
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
        
        # شروع وظایف زمان‌بندی شده
        self.start_scheduled_tasks()
        
        # راه‌اندازی سیستم تحلیل متن پیشرفته
        self.setup_text_analysis()
    
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
            
            # تحلیل تکنیکال
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
            return 0.5
    
    def analyze_opportunities(self, historical_data, market_data, sentiment):
        """تحلیل فرصت‌های معاملاتی"""
        opportunities = []
        
        try:
            current_price = market_data.get('price', 0)
            if current_price == 0:
                return opportunities
            
            # تحلیل فرصت‌های بر اساس تحلیل تکنیکال
            tech_opportunities = self.analyze_technical_opportunities(historical_data)
            opportunities.extend(tech_opportunities)
            
            # تحلیل فرصت‌های بر اساس احساسات
            sentiment_opportunities = self.analyze_sentiment_opportunities(sentiment)
            opportunities.extend(sentiment_opportunities)
            
            # تحلیل فرصت‌های بر اساس رفتار قیمت
            price_opportunities = self.analyze_price_opportunities(historical_data, current_price)
            opportunities.extend(price_opportunities)
            
            # مرتب‌سازی بر اساس امتیاز
            opportunities.sort(key=lambda x: x['score'], reverse=True)
            
            return opportunities[:5]  # برگرداندن 5 فرصت برتر
        except Exception as e:
            logger.error(f"Error in opportunity analysis: {e}")
            return opportunities
    
    def analyze_technical_opportunities(self, historical_data):
        """تحلیل فرصت‌های تکنیکال"""
        opportunities = []
        
        try:
            # محاسبه اندیکاتورها
            rsi = self.calculate_rsi(historical_data)
            macd = self.calculate_macd(historical_data)
            bb = self.calculate_bollinger_bands(historical_data)
            
            current_price = historical_data['Close'].iloc[-1]
            
            # فرصت‌های RSI
            if rsi < 30:  # اشباع فروش
                opportunities.append({
                    'type': 'RSI Oversold',
                    'description': 'RSI در منطقه اشباع فروش - فرصت خرید',
                    'action': 'BUY',
                    'confidence': 0.8,
                    'score': 0.8
                })
            elif rsi > 70:  # اشباع خرید
                opportunities.append({
                    'type': 'RSI Overbought',
                    'description': 'RSI در منطقه اشباع خرید - فرصت فروش',
                    'action': 'SELL',
                    'confidence': 0.8,
                    'score': 0.8
                })
            
            # فرصت‌های MACD
            if macd['histogram'] > 0 and macd['macd'] > macd['signal']:
                opportunities.append({
                    'type': 'MACD Bullish',
                    'description': 'MACD سیگنال صعودی داده است',
                    'action': 'BUY',
                    'confidence': 0.7,
                    'score': 0.7
                })
            elif macd['histogram'] < 0 and macd['macd'] < macd['signal']:
                opportunities.append({
                    'type': 'MACD Bearish',
                    'description': 'MACD سیگنال نزولی داده است',
                    'action': 'SELL',
                    'confidence': 0.7,
                    'score': 0.7
                })
            
            # فرصت‌های بولینگر باند
            if current_price < bb['lower']:
                opportunities.append({
                    'type': 'Bollinger Lower',
                    'description': 'قیمت به کران پایینی بولینگر برخورد کرده',
                    'action': 'BUY',
                    'confidence': 0.75,
                    'score': 0.75
                })
            elif current_price > bb['upper']:
                opportunities.append({
                    'type': 'Bollinger Upper',
                    'description': 'قیمت به کران بالایی بولینگر برخورد کرده',
                    'action': 'SELL',
                    'confidence': 0.75,
                    'score': 0.75
                })
            
            return opportunities
        except Exception as e:
            logger.error(f"Error in technical opportunity analysis: {e}")
            return opportunities
    
    def analyze_sentiment_opportunities(self, sentiment):
        """تحلیل فرصت‌های بر اساس احساسات"""
        opportunities = []
        
        try:
            avg_sentiment = sentiment.get('average_sentiment', 0)
            
            if avg_sentiment > 0.3:  # احساسات بسیار مثبت
                opportunities.append({
                    'type': 'Positive Sentiment',
                    'description': 'احساسات بازار بسیار مثبت است',
                    'action': 'BUY',
                    'confidence': 0.6,
                    'score': 0.6
                })
            elif avg_sentiment < -0.3:  # احساسات بسیار منفی
                opportunities.append({
                    'type': 'Negative Sentiment',
                    'description': 'احساسات بازار بسیار منفی است',
                    'action': 'SELL',
                    'confidence': 0.6,
                    'score': 0.6
                })
            
            # تحلیل بر اساس موضوعات
            topics = sentiment.get('topics', [])
            if 'پذیرش' in topics:
                opportunities.append({
                    'type': 'Adoption Trend',
                    'description': 'روند پذیرش در حال افزایش است',
                    'action': 'BUY',
                    'confidence': 0.7,
                    'score': 0.7
                })
            elif 'تنظیم' in topics:
                opportunities.append({
                    'type': 'Regulation Risk',
                    'description': 'ریسک مقرراتی افزایش یافته',
                    'action': 'SELL',
                    'confidence': 0.65,
                    'score': 0.65
                })
            
            return opportunities
        except Exception as e:
            logger.error(f"Error in sentiment opportunity analysis: {e}")
            return opportunities
    
    def analyze_price_opportunities(self, historical_data, current_price):
        """تحلیل فرصت‌های بر اساس رفتار قیمت"""
        opportunities = []
        
        try:
            # محاسبه سطوح حمایت و مقاومت
            support, resistance = self.calculate_support_resistance(historical_data)
            
            # فرصت‌های نزدیک به حمایت
            if current_price <= support * 1.02:  # نزدیک به حمایت
                opportunities.append({
                    'type': 'Near Support',
                    'description': f'قیمت نزدیک به سطح حمایت {support:.2f} است',
                    'action': 'BUY',
                    'confidence': 0.75,
                    'score': 0.75
                })
            
            # فرصت‌های نزدیک به مقاومت
            if current_price >= resistance * 0.98:  # نزدیک به مقاومت
                opportunities.append({
                    'type': 'Near Resistance',
                    'description': f'قیمت نزدیک به سطح مقاومت {resistance:.2f} است',
                    'action': 'SELL',
                    'confidence': 0.75,
                    'score': 0.75
                })
            
            # تحلیل شکست مقاومت
            if current_price > resistance * 1.02:  # شکست مقاومت
                opportunities.append({
                    'type': 'Resistance Break',
                    'description': f'قیمت مقاومت {resistance:.2f} را شکسته',
                    'action': 'BUY',
                    'confidence': 0.8,
                    'score': 0.8
                })
            
            # تحلیل شکست حمایت
            if current_price < support * 0.98:  # شکست حمایت
                opportunities.append({
                    'type': 'Support Break',
                    'description': f'قیمت حمایت {support:.2f} را شکسته',
                    'action': 'SELL',
                    'confidence': 0.8,
                    'score': 0.8
                })
            
            return opportunities
        except Exception as e:
            logger.error(f"Error in price opportunity analysis: {e}")
            return opportunities
    
    def calculate_support_resistance(self, historical_data):
        """محاسبه سطوح حمایت و مقاومت"""
        try:
            prices = historical_data['Close']
            current_price = prices.iloc[-1]
            
            # محاسبه نقاط چرخش (Swing Points)
            swing_highs = []
            swing_lows = []
            
            for i in range(2, len(prices) - 2):
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    swing_highs.append(prices[i])
                elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                    swing_lows.append(prices[i])
            
            # محاسبه حمایت و مقاومت
            if swing_lows:
                support = np.mean(swing_lows[-3:])  # میانگین 3 حمایت اخیر
            else:
                support = prices.min()
            
            if swing_highs:
                resistance = np.mean(swing_highs[-3:])  # میانگین 3 مقاومت اخیر
            else:
                resistance = prices.max()
            
            return support, resistance
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return current_price * 0.95, current_price * 1.05
    
    def analyze_market_timing(self, historical_data):
        """تحلیل زمان‌بندی بازار"""
        try:
            # تحلیل چرخه‌های زمانی
            timing_analysis = {
                'daily_cycle': self.analyze_daily_cycle(historical_data),
                'weekly_cycle': self.analyze_weekly_cycle(historical_data),
                'monthly_cycle': self.analyze_monthly_cycle(historical_data),
                'seasonal_patterns': self.analyze_seasonal_patterns(historical_data)
            }
            
            # محاسبه امتیاز زمان‌بندی
            timing_score = self.calculate_timing_score(timing_analysis)
            timing_analysis['timing_score'] = timing_score
            
            return timing_analysis
        except Exception as e:
            logger.error(f"Error in market timing analysis: {e}")
            return {'error': str(e)}
    
    def analyze_daily_cycle(self, historical_data):
        """تحلیل چرخه روزانه"""
        try:
            # گروه‌بندی داده‌ها بر اساس ساعت
            hourly_data = historical_data.groupby(historical_data.index.hour)['Close'].mean()
            
            # پیدا کردن بهترین و بدترین ساعت‌ها
            best_hour = hourly_data.idxmax()
            worst_hour = hourly_data.idxmin()
            
            return {
                'best_hour': best_hour,
                'worst_hour': worst_hour,
                'hourly_performance': hourly_data.to_dict()
            }
        except Exception as e:
            logger.error(f"Error in daily cycle analysis: {e}")
            return {}
    
    def analyze_weekly_cycle(self, historical_data):
        """تحلیل چرخه هفتگی"""
        try:
            # گروه‌بندی داده‌ها بر اساس روز هفته
            weekly_data = historical_data.groupby(historical_data.index.dayofweek)['Close'].mean()
            
            # پیدا کردن بهترین و بدترین روزها
            best_day = weekly_data.idxmax()
            worst_day = weekly_data.idxmin()
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            return {
                'best_day': day_names[best_day],
                'worst_day': day_names[worst_day],
                'daily_performance': {day_names[i]: val for i, val in weekly_data.items()}
            }
        except Exception as e:
            logger.error(f"Error in weekly cycle analysis: {e}")
            return {}
    
    def analyze_monthly_cycle(self, historical_data):
        """تحلیل چرخه ماهانه"""
        try:
            # گروه‌بندی داده‌ها بر اساس روز ماه
            monthly_data = historical_data.groupby(historical_data.index.day)['Close'].mean()
            
            # پیدا کردن بهترین و بدترین روزهای ماه
            best_day = monthly_data.idxmax()
            worst_day = monthly_data.idxmin()
            
            return {
                'best_day_of_month': best_day,
                'worst_day_of_month': worst_day,
                'monthly_performance': monthly_data.to_dict()
            }
        except Exception as e:
            logger.error(f"Error in monthly cycle analysis: {e}")
            return {}
    
    def analyze_seasonal_patterns(self, historical_data):
        """تحلیل الگوهای فصلی"""
        try:
            # گروه‌بندی داده‌ها بر اساس ماه
            seasonal_data = historical_data.groupby(historical_data.index.month)['Close'].mean()
            
            # پیدا کردن بهترین و بدترین ماه‌ها
            best_month = seasonal_data.idxmax()
            worst_month = seasonal_data.idxmin()
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            return {
                'best_month': month_names[best_month - 1],
                'worst_month': month_names[worst_month - 1],
                'seasonal_performance': {month_names[i - 1]: val for i, val in seasonal_data.items()}
            }
        except Exception as e:
            logger.error(f"Error in seasonal patterns analysis: {e}")
            return {}
    
    def calculate_timing_score(self, timing_analysis):
        """محاسبه امتیاز زمان‌بندی"""
        try:
            # دریافت زمان فعلی
            now = datetime.now()
            current_hour = now.hour
            current_day = now.weekday()
            current_day_of_month = now.day
            current_month = now.month
            
            score = 0.5  # امتیاز پایه
            
            # امتیاز بر اساس چرخه روزانه
            daily_cycle = timing_analysis.get('daily_cycle', {})
            if current_hour in daily_cycle.get('hourly_performance', {}):
                hourly_perf = daily_cycle['hourly_performance'][current_hour]
                if hourly_perf > np.mean(list(daily_cycle.get('hourly_performance', {}).values())):
                    score += 0.1
            
            # امتیاز بر اساس چرخه هفتگی
            weekly_cycle = timing_analysis.get('weekly_cycle', {})
            if current_day in weekly_cycle.get('daily_performance', {}):
                daily_perf = weekly_cycle['daily_performance'][current_day]
                if daily_perf > np.mean(list(weekly_cycle.get('daily_performance', {}).values())):
                    score += 0.1
            
            # امتیاز بر اساس چرخه ماهانه
            monthly_cycle = timing_analysis.get('monthly_cycle', {})
            if current_day_of_month in monthly_cycle.get('monthly_performance', {}):
                monthly_perf = monthly_cycle['monthly_performance'][current_day_of_month]
                if monthly_perf > np.mean(list(monthly_cycle.get('monthly_performance', {}).values())):
                    score += 0.1
            
            # امتیاز بر اساس الگوهای فصلی
            seasonal_patterns = timing_analysis.get('seasonal_patterns', {})
            if current_month in seasonal_patterns.get('seasonal_performance', {}):
                seasonal_perf = seasonal_patterns['seasonal_performance'][current_month]
                if seasonal_perf > np.mean(list(seasonal_patterns.get('seasonal_performance', {}).values())):
                    score += 0.1
            
            return min(1.0, max(0.0, score))
        except Exception as e:
            logger.error(f"Error calculating timing score: {e}")
            return 0.5
    
    def analyze_price_behavior(self, historical_data):
        """تحلیل رفتار قیمت"""
        try:
            behavior_analysis = {
                'volatility_regime': self.analyze_volatility_regime(historical_data),
                'trend_strength': self.analyze_trend_strength(historical_data),
                'momentum': self.analyze_momentum(historical_data),
                'mean_reversion': self.analyze_mean_reversion(historical_data),
                'price_efficiency': self.analyze_price_efficiency(historical_data)
            }
            
            return behavior_analysis
        except Exception as e:
            logger.error(f"Error in price behavior analysis: {e}")
            return {'error': str(e)}
    
    def analyze_volatility_regime(self, historical_data):
        """تحلیل رژیم نوسانی"""
        try:
            returns = historical_data['Close'].pct_change().dropna()
            
            # محاسبه نوسان‌های متحرک
            volatility_20 = returns.rolling(window=20).std() * np.sqrt(252)
            volatility_50 = returns.rolling(window=50).std() * np.sqrt(252)
            
            current_volatility = volatility_20.iloc[-1]
            avg_volatility = volatility_50.mean()
            
            # تعیین رژیم نوسانی
            if current_volatility > avg_volatility * 1.5:
                regime = 'High Volatility'
            elif current_volatility < avg_volatility * 0.5:
                regime = 'Low Volatility'
            else:
                regime = 'Normal Volatility'
            
            return {
                'regime': regime,
                'current_volatility': current_volatility,
                'average_volatility': avg_volatility,
                'volatility_ratio': current_volatility / avg_volatility
            }
        except Exception as e:
            logger.error(f"Error in volatility regime analysis: {e}")
            return {}
    
    def analyze_trend_strength(self, historical_data):
        """تحلیل قدرت روند"""
        try:
            prices = historical_data['Close']
            
            # محاسبه شیب روند با رگرسیون خطی
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            
            # محاسبه R-squared
            y_pred = slope * x + np.interp(0, x, prices)
            ss_res = np.sum((prices - y_pred) ** 2)
            ss_tot = np.sum((prices - np.mean(prices)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # تعیین قدرت روند
            if abs(slope) > 0.1 and r_squared > 0.7:
                strength = 'Strong'
            elif abs(slope) > 0.05 and r_squared > 0.5:
                strength = 'Moderate'
            else:
                strength = 'Weak'
            
            direction = 'Upward' if slope > 0 else 'Downward'
            
            return {
                'direction': direction,
                'strength': strength,
                'slope': slope,
                'r_squared': r_squared
            }
        except Exception as e:
            logger.error(f"Error in trend strength analysis: {e}")
            return {}
    
    def analyze_momentum(self, historical_data):
        """تحلیل مومنتوم"""
        try:
            prices = historical_data['Close']
            
            # محاسبه شاخص مومنتوم
            momentum_5 = prices.iloc[-1] / prices.iloc[-6] - 1 if len(prices) >= 6 else 0
            momentum_10 = prices.iloc[-1] / prices.iloc[-11] - 1 if len(prices) >= 11 else 0
            momentum_20 = prices.iloc[-1] / prices.iloc[-21] - 1 if len(prices) >= 21 else 0
            
            # محاسبه شاخص قدرت نسبی (RSI)
            rsi = self.calculate_rsi(historical_data)
            
            # تعیین وضعیت مومنتوم
            if momentum_5 > 0.02 and momentum_10 > 0.02 and rsi > 60:
                momentum_status = 'Strong Bullish'
            elif momentum_5 < -0.02 and momentum_10 < -0.02 and rsi < 40:
                momentum_status = 'Strong Bearish'
            elif momentum_5 > 0 and momentum_10 < 0:
                momentum_status = 'Weakening'
            elif momentum_5 < 0 and momentum_10 > 0:
                momentum_status = 'Strengthening'
            else:
                momentum_status = 'Neutral'
            
            return {
                'status': momentum_status,
                'momentum_5': momentum_5,
                'momentum_10': momentum_10,
                'momentum_20': momentum_20,
                'rsi': rsi
            }
        except Exception as e:
            logger.error(f"Error in momentum analysis: {e}")
            return {}
    
    def analyze_mean_reversion(self, historical_data):
        """تحلیل بازگشت به میانگین"""
        try:
            prices = historical_data['Close']
            
            # محاسبه میانگین متحرک
            ma_20 = prices.rolling(window=20).mean()
            ma_50 = prices.rolling(window=50).mean()
            
            current_price = prices.iloc[-1]
            current_ma_20 = ma_20.iloc[-1]
            current_ma_50 = ma_50.iloc[-1]
            
            # محاسبه فاصله از میانگین
            deviation_20 = (current_price - current_ma_20) / current_ma_20
            deviation_50 = (current_price - current_ma_50) / current_ma_50
            
            # تعیین وضعیت بازگشت به میانگین
            if abs(deviation_20) > 0.05:  # انحراف بیش از 5%
                if deviation_20 > 0:
                    reversion_status = 'Overbought - Likely to revert down'
                else:
                    reversion_status = 'Oversold - Likely to revert up'
            else:
                reversion_status = 'Normal - No strong reversion signal'
            
            return {
                'status': reversion_status,
                'deviation_from_ma20': deviation_20,
                'deviation_from_ma50': deviation_50,
                'current_price': current_price,
                'ma_20': current_ma_20,
                'ma_50': current_ma_50
            }
        except Exception as e:
            logger.error(f"Error in mean reversion analysis: {e}")
            return {}
    
    def analyze_price_efficiency(self, historical_data):
        """تحلیل کارایی قیمت"""
        try:
            prices = historical_data['Close']
            
            # محاسبه کارایی بازار با استفاده از آزمون Runs
            returns = prices.pct_change().dropna()
            positive_returns = returns > 0
            runs = 0
            prev_sign = None
            
            for sign in positive_returns:
                if sign != prev_sign:
                    runs += 1
                    prev_sign = sign
            
            # محاسبه آماره Runs
            n = len(returns)
            expected_runs = (2 * n * positive_returns.sum() * (1 - positive_returns.mean())) + 1
            std_runs = np.sqrt(2 * n * positive_returns.sum() * (1 - positive_returns.mean()) * 
                             (2 * n * positive_returns.sum() * (1 - positive_returns.mean()) - n))
            
            z_score = (runs - expected_runs) / std_runs if std_runs > 0 else 0
            
            # تعیین کارایی بازار
            if abs(z_score) < 1.96:
                efficiency = 'Efficient Market'
            elif z_score > 1.96:
                efficiency = 'Momentum Market'
            else:
                efficiency = 'Mean Reverting Market'
            
            return {
                'efficiency_type': efficiency,
                'runs': runs,
                'expected_runs': expected_runs,
                'z_score': z_score
            }
        except Exception as e:
            logger.error(f"Error in price efficiency analysis: {e}")
            return {}
    
    def get_historical_data(self, symbol, period="1y", interval="1d"):
        """دریافت داده‌های تاریخی با مدیریت خطا"""
        try:
            # تلاش برای دریافت داده از yfinance
            data = yf.download(symbol, period=period, interval=interval)
            if not data.empty:
                return data
        except Exception as e:
            logger.error(f"Error fetching historical data from yfinance: {e}")
        
        # اگر yfinance کار نکرد، از صرافی‌ها استفاده کن
        try:
            exchange_symbol = self.convert_symbol_for_exchange(symbol, 'binance')
            exchange = ccxt.binance()
            
            # دریافت داده‌های تاریخی
            ohlcv = exchange.fetch_ohlcv(exchange_symbol, timeframe=interval, limit=500)
            
            # تبدیل به DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data from exchange: {e}")
        
        # اگر هیچ‌کدام کار نکرد، داده‌های ساختگی برگردان
        logger.warning(f"Using dummy data for {symbol}")
        return self.generate_dummy_data(symbol)
    
    def generate_dummy_data(self, symbol):
        """تولید داده‌های ساختگی برای تست"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = np.random.normal(loc=100, scale=10, size=100).cumsum()
        
        return pd.DataFrame({
            'Open': prices + np.random.normal(0, 1, 100),
            'High': prices + np.random.normal(1, 1, 100),
            'Low': prices