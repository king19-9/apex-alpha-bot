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

# بارگذاری متغیرهای محیطی
load_dotenv()

# تنظیمات لاگینگ
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
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
        
        # شروع وظایف زمان‌بندی شده
        self.start_scheduled_tasks()
        
        # راه‌اندازی سیستم تحلیل متن پیشرفته
        self.setup_text_analysis()
    
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
        """دریافت داده‌ها از چندین منبع با مکانیزم بازگشتی"""
        data = {}
        
        # تلاش برای دریافت داده از CoinGecko
        try:
            data['coingecko'] = await self.fetch_coingecko_data(symbol)
        except Exception as e:
            logger.error(f"Error fetching from CoinGecko: {e}")
        
        # تلاش برای دریافت داده از CoinMarketCap
        try:
            data['coinmarketcap'] = await self.fetch_coinmarketcap_data(symbol)
        except Exception as e:
            logger.error(f"Error fetching from CoinMarketCap: {e}")
        
        # تلاش برای دریافت داده از DEX Screener
        try:
            data['dexscreener'] = await self.fetch_dexscreener_data(symbol)
        except Exception as e:
            logger.error(f"Error fetching from DEX Screener: {e}")
        
        # تلاش برای دریافت داده از TradingView (وب اسکرپینگ)
        try:
            data['tradingview'] = await self.fetch_tradingview_data(symbol)
        except Exception as e:
            logger.error(f"Error fetching from TradingView: {e}")
        
        # تلاش برای دریافت داده از صرافی‌ها
        try:
            data['exchanges'] = await self.fetch_exchange_data(symbol)
        except Exception as e:
            logger.error(f"Error fetching from exchanges: {e}")
        
        return data
    
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
        """دریافت داده‌ها از صرافی‌های مختلف"""
        exchange_data = {}
        
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
        
        return exchange_data
    
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
                            'published_at': datetime.strptime(item['published_at'], '%Y-%m-%dT%H:%M:%S%z'),
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
    
    async def perform_advanced_analysis(self, symbol):
        """انجام تحلیل پیشرفته با استفاده از همه منابع داده"""
        # دریافت داده‌های بازار
        market_data = await self.get_market_data(symbol)
        
        # دریافت اخبار مرتبط
        news = await self.fetch_news_from_multiple_sources(symbol)
        
        # تحلیل احساسات اخبار
        sentiment = await self.advanced_sentiment_analysis(news)
        
        # دریافت داده‌های تاریخی برای تحلیل تکنیکال
        historical_data = self.get_historical_data(symbol)
        
        # تحلیل تکنیکال
        technical_analysis = self.advanced_technical_analysis(historical_data)
        
        # تحلیل امواج الیوت
        elliott_analysis = self.advanced_elliott_wave(historical_data)
        
        # تحلیل عرضه و تقاضا
        supply_demand = self.advanced_supply_demand(symbol)
        
        # تحلیل هوش مصنوعی پیشرفته
        ai_analysis = self.perform_ai_analysis(historical_data, market_data, sentiment)
        
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
        signal_score = self.calculate_signal_score(combined_analysis)
        signal = 'BUY' if signal_score > 0.7 else 'SELL' if signal_score < 0.3 else 'HOLD'
        
        combined_analysis['signal'] = signal
        combined_analysis['confidence'] = signal_score
        
        return combined_analysis
    
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
        """پیش‌بینی قیمت با مدل‌های یادگیری ماشین"""
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
            
            # تحلیل فرصت‌های بر اساس احساسات بازار
            sentiment_opportunities = self.analyze_sentiment_opportunities(sentiment)
            opportunities.extend(sentiment_opportunities)
            
            # تحلیل فرصت‌های بر اساس رفتار قیمت
            price_opportunities = self.analyze_price_opportunities(historical_data, current_price)
            opportunities.extend(price_opportportunities)
            
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
        """دریافت داده‌های تاریخی"""
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
        return self.generate_dummy_data(symbol)
    
    def generate_dummy_data(self, symbol):
        """تولید داده‌های ساختگی برای تست"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = np.random.normal(loc=100, scale=10, size=100).cumsum()
        
        return pd.DataFrame({
            'Open': prices + np.random.normal(0, 1, 100),
            'High': prices + np.random.normal(1, 1, 100),
            'Low': prices + np.random.normal(-1, 1, 100),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    def calculate_signal_score(self, analysis):
        """محاسبه امتیاز سیگنال بر اساس تحلیل‌های مختلف"""
        score = 0
        
        # امتیاز از تحلیل تکنیکال
        if analysis.get('technical', {}).get('rsi', 50) < 30:
            score += 0.2  # اشباع فروش
        elif analysis.get('technical', {}).get('rsi', 50) > 70:
            score -= 0.2  # اشباع خرید
        
        if analysis.get('technical', {}).get('macd', {}).get('histogram', 0) > 0:
            score += 0.1
        else:
            score -= 0.1
        
        # امتیاز از تحلیل احساسات
        sentiment = analysis.get('sentiment', {})
        if sentiment.get('average_sentiment', 0) > 0.2:
            score += 0.2
        elif sentiment.get('average_sentiment', 0) < -0.2:
            score -= 0.2
        
        # امتیاز از تحلیل امواج الیوت
        if analysis.get('elliott', {}).get('current_pattern') == 'bullish':
            score += 0.3
        elif analysis.get('elliott', {}).get('current_pattern') == 'bearish':
            score -= 0.3
        
        # امتیاز از تحلیل عرضه و تقاضا
        if analysis.get('supply_demand', {}).get('imbalance', 0) > 0.2:
            score += 0.2
        elif analysis.get('supply_demand', {}).get('imbalance', 0) < -0.2:
            score -= 0.2
        
        # امتیاز از تحلیل هوش مصنوعی
        ai_analysis = analysis.get('ai_analysis', {})
        if ai_analysis.get('risk_analysis', {}).get('risk_score', 0.5) < 0.3:
            score += 0.1  # ریسک پایین
        
        if ai_analysis.get('timing_analysis', {}).get('timing_score', 0.5) > 0.7:
            score += 0.1  # زمان‌بندی خوب
        
        # محدود کردن امتیاز بین 0 و 1
        return max(0, min(1, score))
    
    def advanced_elliott_wave(self, data):
        """تحلیل امواج الیوت با الگوریتم‌های پیشرفته"""
        close_prices = data['Close'].values
        
        # استفاده از تبدیل ویولت در صورت وجود
        if PYWT_AVAILABLE:
            try:
                coeffs = pywt.wavedec(close_prices, 'db1', level=5)
            except Exception as e:
                logger.error(f"Error in wavelet transform: {e}")
                coeffs = None
        else:
            coeffs = None
        
        # شناسایی قله‌ها و دره‌ها
        peaks, _ = find_peaks(close_prices, distance=5)
        troughs, _ = find_peaks(-close_prices, distance=5)
        
        # تحلیل امواج
        waves = []
        for i in range(1, len(peaks)):
            if peaks[i] > peaks[i-1] and troughs[i] > troughs[i-1]:
                waves.append({
                    'type': 'impulse',
                    'start': troughs[i-1],
                    'end': peaks[i],
                    'strength': (close_prices[peaks[i]] - close_prices[troughs[i-1]]) / close_prices[troughs[i-1]]
                })
            elif peaks[i] < peaks[i-1] and troughs[i] < troughs[i-1]:
                waves.append({
                    'type': 'corrective',
                    'start': peaks[i-1],
                    'end': troughs[i],
                    'strength': (close_prices[peaks[i-1]] - close_prices[troughs[i]]) / close_prices[troughs[i]]
                })
        
        return {
            'waves': waves[-10:],  # 10 موج آخر
            'current_pattern': 'bullish' if len([w for w in waves if w['type'] == 'impulse']) > 5 else 'bearish',
            'wavelet_coeffs': coeffs,
            'dominant_cycle': self.detect_dominant_cycle(close_prices)
        }
    
    def detect_dominant_cycle(self, prices):
        """شناسایی چرخه غالب با تبدیل فوریه"""
        try:
            fft = np.fft.fft(prices)
            freqs = np.fft.fftfreq(len(prices))
            dominant_freq = freqs[np.argmax(np.abs(fft[1:])) + 1]
            return 1 / dominant_freq if dominant_freq != 0 else 0
        except Exception as e:
            logger.error(f"Error in dominant cycle detection: {e}")
            return 0
    
    def advanced_supply_demand(self, symbol):
        """تحلیل پیشرفته عرضه و تقاضا"""
        try:
            # استفاده از چندین صرافی برای تحلیل بهتر
            orderbooks = {}
            for exchange_name, exchange in self.exchanges.items():
                try:
                    exchange_symbol = self.convert_symbol_for_exchange(symbol, exchange_name)
                    orderbook = exchange.fetch_order_book(exchange_symbol, limit=50)
                    orderbooks[exchange_name] = orderbook
                except Exception as e:
                    logger.error(f"Error fetching orderbook from {exchange_name}: {e}")
            
            if not orderbooks:
                return {'error': 'No orderbook data available'}
            
            # ترکیب داده‌های همه صرافی‌ها
            all_bids = []
            all_asks = []
            
            for exchange_name, orderbook in orderbooks.items():
                all_bids.extend(orderbook['bids'])
                all_asks.extend(orderbook['asks'])
            
            # تحلیل عمق بازار ترکیبی
            bids = sorted(all_bids, key=lambda x: x[0], reverse=True)
            asks = sorted(all_asks, key=lambda x: x[0])
            
            # محاسبه ناحیه ارزش (Value Area)
            total_volume = sum([bid[1] for bid in bids] + [ask[1] for ask in asks])
            value_area_high = None
            value_area_low = None
            cumulative_volume = 0
            
            # شناسایی ناحیه ارزش (70% حجم)
            for level in sorted(bids + asks, key=lambda x: x[0]):
                cumulative_volume += level[1]
                if cumulative_volume >= total_volume * 0.7:
                    value_area_high = level[0]
                    break
            
            cumulative_volume = 0
            for level in sorted(bids + asks, key=lambda x: x[0], reverse=True):
                cumulative_volume += level[1]
                if cumulative_volume >= total_volume * 0.7:
                    value_area_low = level[0]
                    break
            
            # تحلیل نقاط کنترل (Point of Control)
            poc = max(bids + asks, key=lambda x: x[1])[0]
            
            # تحلیل عدم تعادل سفارشات
            bid_volume = sum([bid[1] for bid in bids])
            ask_volume = sum([ask[1] for ask in asks])
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
            
            # تحلیل لیکوییدیتی
            liquidity_score = self.calculate_liquidity_score(bids, asks)
            
            return {
                'value_area': {'high': value_area_high, 'low': value_area_low},
                'point_of_control': poc,
                'imbalance': imbalance,
                'liquidity_score': liquidity_score,
                'bid_levels': bids[:10],
                'ask_levels': asks[:10],
                'market_depth': self.calculate_market_depth(bids, asks),
                'exchange_count': len(orderbooks)
            }
        except Exception as e:
            logger.error(f"Error in supply demand analysis: {e}")
            return {'error': str(e)}
    
    def calculate_liquidity_score(self, bids, asks):
        """محاسبه امتیاز نقدینگی"""
        try:
            spread = asks[0][0] - bids[0][0]
            depth_1percent = sum([bid[1] for bid in bids if bid[0] >= bids[0][0] * 0.99]) + \
                           sum([ask[1] for ask in asks if ask[0] <= asks[0][0] * 1.01])
            
            spread_score = 1 / (1 + spread)
            depth_score = depth_1percent / 1000000
            
            return (spread_score + depth_score) / 2
        except Exception as e:
            logger.error(f"Error in liquidity score calculation: {e}")
            return 0.5
    
    def calculate_market_depth(self, bids, asks):
        """محاسبه عمق بازار"""
        try:
            depth = {}
            for percent in [0.1, 0.5, 1, 2, 5]:
                bid_depth = sum([bid[1] for bid in bids if bid[0] >= bids[0][0] * (1 - percent/100)])
                ask_depth = sum([ask[1] for ask in asks if ask[0] <= asks[0][0] * (1 + percent/100)])
                depth[f'{percent}%'] = {'bid': bid_depth, 'ask': ask_depth}
            return depth
        except Exception as e:
            logger.error(f"Error in market depth calculation: {e}")
            return {}
    
    def advanced_technical_analysis(self, data):
        """تحلیل تکنیکال پیشرفته با اندیکاتورهای متعدد"""
        try:
            # اندیکاتورهای اصلی
            ichimoku = self.calculate_ichimoku(data)
            macd = self.calculate_macd(data)
            rsi = self.calculate_rsi(data)
            stoch = self.calculate_stochastic(data)
            bb = self.calculate_bollinger_bands(data)
            atr = self.calculate_atr(data)
            vwap = self.calculate_vwap(data)
            
            # اندیکاتورهای تالاب با pandas-ta در صورت عدم وجود TA-Lib
            williams_r = None
            cci = None
            
            if TALIB_AVAILABLE:
                try:
                    williams_r = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)
                    cci = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
                except Exception as e:
                    logger.error(f"Error calculating TA-Lib indicators: {e}")
            elif PANDAS_TA_AVAILABLE:
                try:
                    # استفاده از pandas-ta به جای TA-Lib
                    df = data.copy()
                    df.ta.williams_r(length=14, append=True)
                    df.ta.cci(length=14, append=True)
                    williams_r = df['WILLR_14'].values
                    cci = df['CCI_14'].values
                except Exception as e:
                    logger.error(f"Error calculating pandas-ta indicators: {e}")
            
            return {
                'ichimoku': ichimoku,
                'macd': macd,
                'rsi': rsi,
                'stochastic': stoch,
                'bollinger': bb,
                'atr': atr,
                'vwap': vwap,
                'williams_r': williams_r[-1] if williams_r is not None else None,
                'cci': cci[-1] if cci is not None else None
            }
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {}
    
    def calculate_ichimoku(self, data):
        """محاسبه اندیکاتور ایچیموکو"""
        try:
            high_9 = data['High'].rolling(window=9).max()
            low_9 = data['Low'].rolling(window=9).min()
            high_26 = data['High'].rolling(window=26).max()
            low_26 = data['Low'].rolling(window=26).min()
            high_52 = data['High'].rolling(window=52).max()
            low_52 = data['Low'].rolling(window=52).min()
            
            tenkan_sen = (high_9 + low_9) / 2
            kijun_sen = (high_26 + low_26) / 2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            senkou_span_b = ((high_52 + low_52) / 2).shift(26)
            chikou_span = data['Close'].shift(-26)
            
            return {
                'tenkan_sen': tenkan_sen.iloc[-1],
                'kijun_sen': kijun_sen.iloc[-1],
                'senkou_span_a': senkou_span_a.iloc[-1],
                'senkou_span_b': senkou_span_b.iloc[-1],
                'chikou_span': chikou_span.iloc[-1],
                'cloud_bullish': senkou_span_a.iloc[-1] > senkou_span_b.iloc[-1]
            }
        except Exception as e:
            logger.error(f"Error calculating Ichimoku: {e}")
            return {}
    
    def calculate_macd(self, data):
        """محاسبه اندیکاتور MACD"""
        try:
            exp12 = data['Close'].ewm(span=12, adjust=False).mean()
            exp26 = data['Close'].ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            return {
                'macd': macd.iloc[-1],
                'signal': signal.iloc[-1],
                'histogram': histogram.iloc[-1]
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {}
    
    def calculate_rsi(self, data):
        """محاسبه اندیکاتور RSI"""
        try:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50
    
    def calculate_stochastic(self, data):
        """محاسبه اندیکاتور Stochastic"""
        try:
            low_14 = data['Low'].rolling(window=14).min()
            high_14 = data['High'].rolling(window=14).max()
            k_percent = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
            d_percent = k_percent.rolling(window=3).mean()
            
            return {
                'stoch_k': k_percent.iloc[-1],
                'stoch_d': d_percent.iloc[-1]
            }
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return {'stoch_k': 50, 'stoch_d': 50}
    
    def calculate_bollinger_bands(self, data):
        """محاسبه اندیکاتور بولینگر باند"""
        try:
            ma_20 = data['Close'].rolling(window=20).mean()
            std_20 = data['Close'].rolling(window=20).std()
            upper_band = ma_20 + (std_20 * 2)
            lower_band = ma_20 - (std_20 * 2)
            
            position = "above_upper" if data['Close'].iloc[-1] > upper_band.iloc[-1] else \
                      "below_lower" if data['Close'].iloc[-1] < lower_band.iloc[-1] else "inside"
            
            return {
                'upper': upper_band.iloc[-1],
                'middle': ma_20.iloc[-1],
                'lower': lower_band.iloc[-1],
                'position': position
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {}
    
    def calculate_atr(self, data):
        """محاسبه اندیکاتور ATR"""
        try:
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=14).mean()
            return atr.iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0
    
    def calculate_vwap(self, data):
        """محاسبه اندیکاتور VWAP"""
        try:
            q = data['Volume']
            p = data['Close']
            vwap = (p * q).cumsum() / q.cumsum()
            return vwap.iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return data['Close'].iloc[-1]
    
    def generate_persian_explanation(self, analysis_data):
        """تولید توضیحات فارسی با هوش مصنوعی داخلی"""
        
        # تعیین نوع سیگنال و توضیحات مربوطه
        signal_descriptions = {
            'BUY': {
                'strong': 'سیگنال قوی خرید - فرصت عالی برای ورود به معامله',
                'medium': 'سیگنال خرید - شرایط مساعد برای ورود',
                'weak': 'سیگنال خرید - با احتیاط ورود کنید'
            },
            'SELL': {
                'strong': 'سیگنال قوی فروش - خروج فوری توصیه می‌شود',
                'medium': 'سیگنال فروش - کاهش قیمت محتمل است',
                'weak': 'سیگنال فروش - با احتیاط عمل کنید'
            },
            'HOLD': {
                'strong': 'نگهداری - وضعیت بازار پایدار است',
                'medium': 'نگهداری - منتظر سیگنال بعدی باشید',
                'weak': 'نگهداری - با احتیاط نظارت کنید'
            }
        }
        
        # تعیین قدرت سیگنال
        confidence = analysis_data['confidence']
        if confidence > 0.8:
            strength = 'strong'
        elif confidence > 0.6:
            strength = 'medium'
        else:
            strength = 'weak'
        
        # تحلیل تکنیکال
        rsi = analysis_data['technical'].get('rsi', 50)
        rsi_status = "اشباع خرید" if rsi > 70 else "اشباع فروش" if rsi < 30 else "طبیعی"
        
        macd = analysis_data['technical'].get('macd', {}).get('histogram', 0)
        macd_status = "صعودی" if macd > 0 else "نزولی"
        
        # تحلیل احساسات
        sentiment = analysis_data['sentiment'].get('average_sentiment', 0)
        if sentiment > 0.3:
            sentiment_text = "احساسات بسیار مثبت"
        elif sentiment > 0.1:
            sentiment_text = "احساسات مثبت"
        elif sentiment > -0.1:
            sentiment_text = "احساسات خنثی"
        elif sentiment > -0.3:
            sentiment_text = "احساسات منفی"
        else:
            sentiment_text = "احساسات بسیار منفی"
        
        # تحلیل هوش مصنوعی
        ai_analysis = analysis_data.get('ai_analysis', {})
        risk_score = ai_analysis.get('risk_analysis', {}).get('risk_score', 0.5)
        timing_score = ai_analysis.get('timing_analysis', {}).get('timing_score', 0.5)
        
        # تعیین سطح ریسک
        if risk_score < 0.3:
            risk_level = "پایین"
        elif risk_score < 0.7:
            risk_level = "متوسط"
        else:
            risk_level = "بالا"
        
        # تعیین کیفیت زمان‌بندی
        if timing_score > 0.7:
            timing_quality = "عالی"
        elif timing_score > 0.5:
            timing_quality = "خوب"
        else:
            timing_quality = "متوسط"
        
        # تحلیل فرصت‌ها
        opportunities = ai_analysis.get('opportunities', [])
        opportunity_text = ""
        if opportunities:
            top_opportunity = opportunities[0]
            opportunity_text = f"🎯 *فرصت برتر:* {top_opportunity['description']}"
        
        # تحلیل رفتار قیمت
        behavior_analysis = ai_analysis.get('behavior_analysis', {})
        trend_strength = behavior_analysis.get('trend_strength', {})
        momentum_status = behavior_analysis.get('momentum', {}).get('status', 'ناشناخته')
        
        # تولید توضیحات نهایی
        explanation = f"""
📊 *تحلیل جامع {analysis_data['symbol']}*

💰 *قیمت فعلی:* {analysis_data['market_data']['price']:,.2f} USD
📈 *تغییر 24 ساعته:* {analysis_data['market_data']['price_change_24h']:+.2f}%
🔄 *حجم معاملات:* {analysis_data['market_data']['volume_24h']:,.0f} USD

🤖 *سیگنال:* {analysis_data['signal']}
📈 *اطمینان:* {analysis_data['confidence']*100:.1f}%
🎯 *توصیه:* {signal_descriptions[analysis_data['signal']][strength]}

📈 *تحلیل تکنیکال:*
• الگوی البروکس: {analysis_data['elliott']['current_pattern']}
• ایچیموکو: {'صعودی' if analysis_data['technical'].get('ichimoku', {}).get('cloud_bullish', False) else 'نزولی'}
• RSI: {rsi:.1f} ({rsi_status})
• MACD: {macd:.2f} ({macd_status})
• مومنتوم: {momentum_status}

⚖️ *عرضه و تقاضا:*
• ناحیه ارزش: {analysis_data['supply_demand'].get('value_area', {}).get('low', 0):.2f} - {analysis_data['supply_demand'].get('value_area', {}).get('high', 0):.2f}
• نقطه کنترل: {analysis_data['supply_demand'].get('point_of_control', 0):.2f}
• عدم تعادل: {analysis_data['supply_demand'].get('imbalance', 0)*100:.1f}%

📰 *تحلیل اخبار:*
• تعداد اخبار: {analysis_data['news_count']}
• {sentiment_text}
• موضوعات اصلی: {', '.join(analysis_data['sentiment'].get('topics', []))}

🧠 *تحلیل هوش مصنوعی:*
• سطح ریسک: {risk_level}
• کیفیت زمان‌بندی: {timing_quality}
• قدرت روند: {trend_strength.get('strength', 'ناشناخته')} {trend_strength.get('direction', '')}
{opportunity_text}

📊 *نقاط کلیدی:*
• مقاومت: {analysis_data['supply_demand'].get('value_area', {}).get('high', 0):.2f}
• حمایت: {analysis_data['supply_demand'].get('value_area', {}).get('low', 0):.2f}
• حد ضرر پیشنهادی: {analysis_data['market_data']['price'] * 0.95:.2f}
• هدف سود پیشنهادی: {analysis_data['market_data']['price'] * 1.1:.2f}

⚠️ *هشدار:* این تحلیل صرفاً جنبه آموزشی دارد. همیشه ریسک را مدیریت کنید.
        """
        
        return explanation
    
    def save_analysis(self, user_id, symbol, analysis_type, result):
        """ذخیره تحلیل در پایگاه داده"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            INSERT INTO analyses (user_id, symbol, analysis_type, result)
            VALUES (?, ?, ?, ?)
            ''', (user_id, symbol, analysis_type, json.dumps(result)))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            return None
    
    def save_signal(self, user_id, symbol, signal_type, signal_value, confidence):
        """ذخیره سیگنال در پایگاه داده"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            INSERT INTO signals (user_id, symbol, signal_type, signal_value, confidence)
            VALUES (?, ?, ?, ?, ?)
            ''', (user_id, symbol, signal_type, signal_value, confidence))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            return None
    
    def get_user_watchlist(self, user_id):
        """دریافت واچ‌لیست کاربر"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT symbol FROM watchlist WHERE user_id = ?', (user_id,))
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting watchlist: {e}")
            return []
    
    def add_to_watchlist(self, user_id, symbol):
        """افزودن نماد به واچ‌لیست"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('INSERT INTO watchlist (user_id, symbol) VALUES (?, ?)', (user_id, symbol))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error adding to watchlist: {e}")
    
    def remove_from_watchlist(self, user_id, symbol):
        """حذف نماد از واچ‌لیست"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM watchlist WHERE user_id = ? AND symbol = ?', (user_id, symbol))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error removing from watchlist: {e}")
    
    def generate_performance_report(self, user_id):
        """تولید گزارش عملکرد کاربر"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            SELECT symbol, strategy, profit_loss, timestamp
            FROM performance
            WHERE user_id = ?
            ORDER BY timestamp DESC
            ''', (user_id,))
            
            performance = []
            for row in cursor.fetchall():
                performance.append({
                    'symbol': row[0],
                    'strategy': row[1],
                    'profit_loss': row[2],
                    'timestamp': row[3]
                })
            
            if not performance:
                return "هیچ معامله‌ای ثبت نشده است."
            
            df = pd.DataFrame(performance)
            
            # محاسبات آماری
            total_trades = len(df)
            profitable_trades = len(df[df['profit_loss'] > 0])
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            total_profit = df['profit_loss'].sum()
            avg_profit = df['profit_loss'].mean()
            max_profit = df['profit_loss'].max()
            max_loss = df['profit_loss'].min()
            
            # محاسبه شارپ ratio
            risk_free_rate = 0.02
            excess_returns = df['profit_loss'] - risk_free_rate/252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if len(excess_returns) > 1 else 0
            
            # محاسبه حداکثر افت
            cumulative = (1 + df['profit_loss']).cumprod()
            peak = cumulative.expanding().max()
            drawdown = (cumulative - peak) / peak
            max_drawdown = drawdown.min()
            
            # ایجاد نمودار عملکرد
            plt.figure(figsize=(12, 6))
            if SEABORN_AVAILABLE:
                sns.set()  # تنظیم استایل seaborn در صورت وجود
            plt.plot(cumulative.index, cumulative.values, label='Cumulative Returns')
            plt.fill_between(cumulative.index, drawdown.values, 0, color='red', alpha=0.3, label='Drawdown')
            plt.title('Performance Chart')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            
            # تبدیل نمودار به base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            report = f"""
📊 *گزارش عملکرد شما*

• تعداد کل معاملات: {total_trades}
• معاملات سودده: {profitable_trades}
• نرخ برد: {win_rate:.1%}

• سود کل: {total_profit:.2f}%
• میانگین سود: {avg_profit:.2f}%
• بیشترین سود: {max_profit:.2f}%
• بیشترین ضرر: {max_loss:.2f}%

• نسبت شارپ: {sharpe_ratio:.2f}
• حداکثر افت: {max_drawdown:.1%}

• بهترین استراتژی: {df.groupby('strategy')['profit_loss'].mean().idxmax()}
            """
            
            return {
                'text': report,
                'chart': chart_base64
            }
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return "خطا در تولید گزارش عملکرد"
    
    def start_scheduled_tasks(self):
        """شروع وظایف زمان‌بندی شده"""
        # به‌روزرسانی داده‌های بازار هر 5 دقیقه
        schedule.every(5).minutes.do(self.update_market_data)
        
        # به‌روزرسانی اخبار هر 15 دقیقه
        schedule.every(15).minutes.do(self.update_news)
        
        # اجرای وظایف در یک thread جداگانه
        def run_schedule():
            while True:
                schedule.run_pending()
                time.sleep(1)
        
        import threading
        thread = threading.Thread(target=run_schedule)
        thread.daemon = True
        thread.start()
    
    def update_market_data(self):
        """به‌روزرسانی داده‌های بازار"""
        logger.info("Updating market data...")
        # دریافت لیست نمادهای محبوب
        popular_symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOT', 'DOGE', 'AVAX', 'MATIC']
        
        for symbol in popular_symbols:
            try:
                market_data = asyncio.run(self.get_market_data(symbol))
                
                # ذخیره در پایگاه داده
                cursor = self.conn.cursor()
                cursor.execute('''
                INSERT INTO market_data (symbol, source, price, volume_24h, market_cap, price_change_24h)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    'combined',
                    market_data['price'],
                    market_data['volume_24h'],
                    market_data['market_cap'],
                    market_data['price_change_24h']
                ))
                self.conn.commit()
                
                logger.info(f"Updated market data for {symbol}")
            except Exception as e:
                logger.error(f"Error updating market data for {symbol}: {e}")
    
    def update_news(self):
        """به‌روزرسانی اخبار"""
        logger.info("Updating news...")
        try:
            news = asyncio.run(self.fetch_news_from_multiple_sources())
            
            for news_item in news:
                # تحلیل احساسات خبر
                sentiment = asyncio.run(self.advanced_sentiment_analysis([news_item]))
                
                # ذخیره در پایگاه داده
                cursor = self.conn.cursor()
                cursor.execute('''
                INSERT INTO news (title, content, source, url, published_at, sentiment_score, symbols)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    news_item['title'],
                    news_item['content'],
                    news_item['source'],
                    news_item['url'],
                    news_item['published_at'],
                    sentiment.get('average_sentiment', 0),
                    json.dumps(news_item.get('symbols', []))
                ))
                self.conn.commit()
            
            logger.info(f"Updated {len(news)} news items")
        except Exception as e:
            logger.error(f"Error updating news: {e}")
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """دستور شروع ربات"""
        keyboard = [
            [InlineKeyboardButton("📊 تحلیل عمیق", callback_data='deep_analysis')],
            [InlineKeyboardButton("🔥 سیگنال‌های طلایی", callback_data='golden_signals')],
            [InlineKeyboardButton("📋 مدیریت واچ‌لیست", callback_data='watchlist')],
            [InlineKeyboardButton("📈 گزارش عملکرد", callback_data='performance_report')],
            [InlineKeyboardButton("⚙️ تنظیمات", callback_data='settings')],
            [InlineKeyboardButton("❓ راهنما", callback_data='help')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🤖 *به ربات تریدینگ هوشمند خوش آمدید!*\\n\\nلطفاً یک گزینه را انتخاب کنید:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def deep_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """تحلیل عمیق نماد"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        context.user_data['state'] = 'deep_analysis'
        
        await query.edit_message_text(
            "لطفاً نماد مورد نظر را وارد کنید (مثال: BTC, ETH, AAPL):"
        )
    
    async def handle_symbol(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پردازش نماد ورودی"""
        if context.user_data.get('state') == 'deep_analysis':
            symbol = update.message.text.upper()
            user_id = update.effective_user.id
            
            # ارسال پیام در حال پردازش
            processing_msg = await update.message.reply_text("⏳ در حال تحلیل داده‌ها...")
            
            try:
                # انجام تحلیل پیشرفته
                analysis = await self.perform_advanced_analysis(symbol)
                
                # تولید توضیحات فارسی
                explanation = self.generate_persian_explanation(analysis)
                
                # ذخیره تحلیل
                self.save_analysis(user_id, symbol, 'deep_analysis', analysis)
                
                # حذف پیام پردازش
                await processing_msg.delete()
                
                # ارسال تحلیل
                await update.message.reply_text(explanation, parse_mode='Markdown')
                
            except Exception as e:
                logger.error(f"Error in deep analysis: {e}")
                await processing_msg.edit_text("❌ خطا در تحلیل. لطفاً دوباره تلاش کنید.")
            
            context.user_data['state'] = None
    
    async def golden_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """سیگنال‌های طلایی"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        watchlist = self.get_user_watchlist(user_id)
        
        if not watchlist:
            await query.edit_message_text("❌ واچ‌لیست شما خالی است. لطفاً ابتدا نمادها را به واچ‌لیست اضافه کنید.")
            return
        
        # ارسال پیام در حال پردازش
        processing_msg = await query.edit_message_text("⏳ در حال بررسی سیگنال‌ها...")
        
        signals = []
        for symbol in watchlist:
            try:
                # تحلیل سریع
                analysis = await self.perform_advanced_analysis(symbol)
                
                if analysis['confidence'] > 0.8:  # سیگنال طلایی
                    signals.append({
                        'symbol': symbol,
                        'score': analysis['confidence'],
                        'price': analysis['market_data']['price'],
                        'signal': analysis['signal']
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        if signals:
            response = "🔥 *سیگنال‌های طلایی (اطمینان >80%):*\\n\\n"
            for sig in signals:
                response += f"• {sig['symbol']}: {sig['signal']}\\n"
                response += f"  قیمت: {sig['price']:,.2f}\\n"
                response += f"  اطمینان: {sig['score']*100:.1f}%\\n\\n"
            
            await processing_msg.edit_text(response, parse_mode='Markdown')
        else:
            await processing_msg.edit_text("در حال حاضر سیگنال طلایی وجود ندارد.")
    
    async def watchlist_management(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """مدیریت واچ‌لیست"""
        query = update.callback_query
        await query.answer()
        
        keyboard = [
            [InlineKeyboardButton("➕ افزودن نماد", callback_data='add_to_watchlist')],
            [InlineKeyboardButton("➖ حذف نماد", callback_data='remove_from_watchlist')],
            [InlineKeyboardButton("📋 نمایش واچ‌لیست", callback_data='show_watchlist')],
            [InlineKeyboardButton("🔙 بازگشت", callback_data='back_to_main')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "📋 *مدیریت واچ‌لیست:*",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def add_to_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """افزودن نماد به واچ‌لیست"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        context.user_data['state'] = 'add_to_watchlist'
        
        await query.edit_message_text("لطفاً نماد مورد نظر را وارد کنید:")
    
    async def remove_from_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """حذف نماد از واچ‌لیست"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        context.user_data['state'] = 'remove_from_watchlist'
        
        await query.edit_message_text("لطفاً نماد مورد نظر برای حذف را وارد کنید:")
    
    async def show_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """نمایش واچ‌لیست"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        watchlist = self.get_user_watchlist(user_id)
        
        if watchlist:
            response = "📋 *واچ‌لیست شما:*\\n\\n"
            for i, symbol in enumerate(watchlist, 1):
                response += f"{i}. {symbol}\\n"
        else:
            response = "واچ‌لیست شما خالی است."
        
        await query.edit_message_text(response, parse_mode='Markdown')
    
    async def handle_watchlist_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پردازش عملیات واچ‌لیست"""
        user_id = update.effective_user.id
        symbol = update.message.text.upper()
        state = context.user_data.get('state')
        
        if state == 'add_to_watchlist':
            self.add_to_watchlist(user_id, symbol)
            await update.message.reply_text(f"✅ نماد {symbol} به واچ‌لیست اضافه شد.")
        elif state == 'remove_from_watchlist':
            self.remove_from_watchlist(user_id, symbol)
            await update.message.reply_text(f"✅ نماد {symbol} از واچ‌لیست حذف شد.")
        
        context.user_data['state'] = None
    
    async def performance_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """گزارش عملکرد"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        report = self.generate_performance_report(user_id)
        
        if isinstance(report, str):
            await query.edit_message_text(report)
        else:
            # ارسال نمودار
            try:
                chart_bytes = base64.b64decode(report['chart'])
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=chart_bytes,
                    caption="📈 نمودار عملکرد شما"
                )
                
                # ارسال متن گزارش
                await query.edit_message_text(report['text'], parse_mode='Markdown')
            except Exception as e:
                logger.error(f"Error sending performance report: {e}")
                await query.edit_message_text("خطا در ارسال گزارش عملکرد")
    
    async def settings_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """منوی تنظیمات"""
        query = update.callback_query
        await query.answer()
        
        keyboard = [
            [InlineKeyboardButton("🌐 تغییر زبان", callback_data='change_language')],
            [InlineKeyboardButton("🔔 مدیریت اعلان‌ها", callback_data='manage_notifications')],
            [InlineKeyboardButton("🔄 تنظیمات پروکسی", callback_data='proxy_settings')],
            [InlineKeyboardButton("🔙 بازگشت", callback_data='back_to_main')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "⚙️ *تنظیمات:*",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """دستور راهنما"""
        help_text = """
🤖 *راهنمای ربات تریدینگ هوشمند*

• /start - شروع ربات و نمایش منوی اصلی
• /analyze [symbol] - تحلیل عمیق نماد
• /signals - نمایش سیگنال‌های طلایی
• /watchlist - مدیریت واچ‌لیست
• /performance - گزارش عملکرد
• /settings - تنظیمات
• /help - نمایش این راهنما

📚 *قابلیت‌های ربات:*
- تحلیل تکنیکال پیشرفته با 15+ اندیکاتور
- تحلیل امواج الیوت با الگوریتم‌های هوش مصنوعی
- تحلیل عرضه و تقاضا با دفترچه سفارشات
- ترکیب 10+ مدل یادگیری ماشین
- تحلیل احساسات بازار از چند منبع
- پیش‌بینی قیمت با دقت بالا
- مدیریت ریسک هوشمند
- گزارش عملکرد جامع
- پشتیبانی از چند زبان

🌐 *منابع داده:*
- CoinGecko
- CoinMarketCap
- DEX Screener
- TradingView (وب اسکرپینگ)
- چندین صرافی معتبر
- منابع خبری متعدد

⚠️ *هشدار:* این ربات صرفاً جنبه آموزشی دارد و مسئولیت تصمیم‌گیری با شماست.
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """مدیریت callbackها"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data == 'deep_analysis':
            await self.deep_analysis(update, context)
        elif data == 'golden_signals':
            await self.golden_signals(update, context)
        elif data == 'watchlist':
            await self.watchlist_management(update, context)
        elif data == 'add_to_watchlist':
            await self.add_to_watchlist(update, context)
        elif data == 'remove_from_watchlist':
            await self.remove_from_watchlist(update, context)
        elif data == 'show_watchlist':
            await self.show_watchlist(update, context)
        elif data == 'performance_report':
            await self.performance_report(update, context)
        elif data == 'settings':
            await self.settings_menu(update, context)
        elif data == 'help':
            await self.help_command(update, context)
        elif data == 'back_to_main':
            await self.start(update, context)

def main():
    """تابع اصلی اجرای برنامه"""
    # ایجاد نمونه از ربات
    bot = AdvancedTradingBot()
    
    # تنظیمات تلگرام
    application = Application.builder().token(os.getenv('TELEGRAM_TOKEN')).build()
    
    # افزودن هندلرها
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("analyze", bot.handle_symbol))
    application.add_handler(CommandHandler("signals", bot.golden_signals))
    application.add_handler(CommandHandler("watchlist", bot.watchlist_management))
    application.add_handler(CommandHandler("performance", bot.performance_report))
    application.add_handler(CommandHandler("settings", bot.settings_menu))
    application.add_handler(CallbackQueryHandler(bot.handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_symbol))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_watchlist_action))
    
    # اجرای ربات
    application.run_polling()

if __name__ == '__main__':
    main()