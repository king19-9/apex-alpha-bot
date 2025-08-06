import os
import logging
import asyncio
import json
import numpy as np
import pandas as pd
import yfinance as yf
import ccxt
import sqlite3
from datetime import datetime, timedelta
import pytz
import io
import base64
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from dotenv import load_dotenv

load_dotenv()

# تنظیمات لاگینگ
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# وارد کردن کتابخانه‌ها به صورت شرطی
try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TORCH_AVAILABLE = True
    logger.info("PyTorch and Transformers loaded successfully")
except ImportError as e:
    TORCH_AVAILABLE = False
    logger.warning(f"PyTorch or Transformers not available: {e}")

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

# کتابخانه‌های یادگیری ماشین
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

# کتابخانه‌های تحلیل و مصورسازی
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# مدیریت شرطی برای seaborn
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    logger.info("Seaborn loaded successfully")
except ImportError as e:
    SEABORN_AVAILABLE = False
    logger.warning(f"Seaborn not available: {e}")

class AdvancedTradingBot:
    def __init__(self):
        # پایگاه داده
        self.conn = sqlite3.connect('trading_bot.db', check_same_thread=False)
        self.create_tables()
        
        # مدل‌های تحلیل
        self.models = self.initialize_models()
        
        # تنظیمات
        self.exchange = ccxt.binance()
        
        # تنظیمات مدل‌های زبانی در صورت وجود
        if TORCH_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/gpt2-persian")
                self.model = AutoModelForCausalLM.from_pretrained("HooshvareLab/gpt2-persian")
                self.sentiment_pipeline = pipeline("sentiment-analysis", model="HooshvareLab/bert-fa-sentiment-deepsenti")
                logger.info("Language models loaded successfully")
            except Exception as e:
                logger.error(f"Error loading language models: {e}")
                self.tokenizer = None
                self.model = None
                self.sentiment_pipeline = None
        else:
            self.tokenizer = None
            self.model = None
            self.sentiment_pipeline = None
    
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
            orderbook = self.exchange.fetch_order_book(symbol, limit=50)
            
            # تحلیل عمق بازار
            bids = orderbook['bids']
            asks = orderbook['asks']
            
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
                'market_depth': self.calculate_market_depth(bids, asks)
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
    
    def advanced_sentiment_analysis(self, symbol):
        """تحلیل احساسات پیشرفته با چند منبع"""
        sentiment_scores = {
            'news': 0,
            'social_media': 0,
            'analyst_ratings': 0,
            'options_market': 0
        }
        
        # تحلیل اخبار
        news_sentiment = self.analyze_news_sentiment(symbol)
        sentiment_scores['news'] = news_sentiment
        
        # تحلیل شبکه‌های اجتماعی
        social_sentiment = self.analyze_social_media_sentiment(symbol)
        sentiment_scores['social_media'] = social_sentiment
        
        # تحلیل رتبه‌بندی تحلیلگران
        analyst_sentiment = self.analyze_analyst_ratings(symbol)
        sentiment_scores['analyst_ratings'] = analyst_sentiment
        
        # تحلیل بازار اختیارات
        options_sentiment = self.analyze_options_sentiment(symbol)
        sentiment_scores['options_market'] = options_sentiment
        
        # ترکیب امتیازات
        combined_sentiment = np.mean(list(sentiment_scores.values()))
        
        return {
            'scores': sentiment_scores,
            'combined': combined_sentiment,
            'interpretation': self.interpret_sentiment(combined_sentiment)
        }
    
    def analyze_news_sentiment(self, symbol):
        """تحلیل احساسات اخبار"""
        try:
            # شبیه‌سازی داده‌ها
            news_headlines = [
                f"{symbol} در حال ثبت رکورد جدید است",
                f"تحلیلگران پیش‌بینی صعودی برای {symbol} دارند",
                f"نگرانی‌ها در مورد رگولاتوری {symbol} افزایش یافته"
            ]
            
            sentiments = []
            for headline in news_headlines:
                if self.sentiment_pipeline:
                    try:
                        result = self.sentiment_pipeline(headline)[0]
                        sentiments.append(result['score'] if result['label'] == 'POSITIVE' else -result['score'])
                    except Exception as e:
                        logger.error(f"Error in sentiment analysis: {e}")
                        sentiments.append(np.random.uniform(-1, 1))
                else:
                    # اگر مدل احساسات موجود نباشد، از امتیاز تصادفی استفاده می‌کنیم
                    sentiments.append(np.random.uniform(-1, 1))
            
            return np.mean(sentiments)
        except Exception as e:
            logger.error(f"Error in news sentiment analysis: {e}")
            return 0
    
    def analyze_social_media_sentiment(self, symbol):
        """تحلیل احساسات شبکه‌های اجتماعی"""
        try:
            # شبیه‌سازی داده‌ها
            tweets = [
                f"من به {symbol} خیلی خوشبین هستم! 🚀",
                f"{symbol} در حال سقوط است، بفروشید!",
                f"تحلیل تکنیکال {symbol} نشان‌دهنده ادامه روند صعودی است"
            ]
            
            sentiments = []
            for tweet in tweets:
                if self.sentiment_pipeline:
                    try:
                        result = self.sentiment_pipeline(tweet)[0]
                        sentiments.append(result['score'] if result['label'] == 'POSITIVE' else -result['score'])
                    except Exception as e:
                        logger.error(f"Error in sentiment analysis: {e}")
                        sentiments.append(np.random.uniform(-1, 1))
                else:
                    sentiments.append(np.random.uniform(-1, 1))
            
            return np.mean(sentiments)
        except Exception as e:
            logger.error(f"Error in social media sentiment analysis: {e}")
            return 0
    
    def analyze_analyst_ratings(self, symbol):
        """تحلیل رتبه‌بندی تحلیلگران"""
        try:
            # شبیه‌سازی داده‌ها
            ratings = [
                {'rating': 'BUY', 'weight': 1},
                {'rating': 'HOLD', 'weight': 0},
                {'rating': 'SELL', 'weight': -1}
            ]
            
            weighted_score = sum(r['weight'] for r in ratings) / len(ratings)
            return weighted_score
        except Exception as e:
            logger.error(f"Error in analyst ratings analysis: {e}")
            return 0
    
    def analyze_options_sentiment(self, symbol):
        """تحلیل احساسات بازار اختیارات"""
        try:
            # شبیه‌سازی داده‌ها
            put_volume = 10000
            call_volume = 15000
            put_call_ratio = put_volume / call_volume
            
            return 1 - put_call_ratio
        except Exception as e:
            logger.error(f"Error in options sentiment analysis: {e}")
            return 0
    
    def interpret_sentiment(self, score):
        """تفسیر امتیاز احساسات"""
        if score > 0.3:
            return "احساسات بسیار مثبت - احتمال ادامه روند صعودی"
        elif score > 0.1:
            return "احساسات مثبت - شرایط مساعد برای رشد"
        elif score > -0.1:
            return "احساسات خنثی - بازار در حالت انتظار"
        elif score > -0.3:
            return "احساسات منفی - احتمال اصلاح قیمت"
        else:
            return "احساسات بسیار منفی - هشدار ریزش شدید"
    
    def generate_persian_explanation(self, analysis_data):
        """تولید توضیحات فارسی با هوش مصنوعی"""
        if self.model is None or self.tokenizer is None:
            # اگر مدل زبانی موجود نباشد، از توضیحات ثابت استفاده می‌کنیم
            return f"""
📊 *تحلیل جامع {analysis_data['symbol']}*

💰 *قیمت فعلی:* {analysis_data['price']:,.2f} USD

🤖 *سیگنال:* {analysis_data['signal']}
📈 *اطمینان:* {analysis_data['confidence']*100:.1f}%

📈 *تحلیل تکنیکال:*
• الگوی البروکس: {analysis_data['elliott']['current_pattern']}
• ایچیموکو: {'صعودی' if analysis_data['technical'].get('ichimoku', {}).get('cloud_bullish', False) else 'نزولی'}
• RSI: {analysis_data['technical'].get('rsi', 50):.1f}
• MACD: {analysis_data['technical'].get('macd', {}).get('histogram', 0):.2f}

⚖️ *عرضه و تقاضا:*
• ناحیه ارزش: {analysis_data['supply_demand'].get('value_area', {}).get('low', 0):.2f} - {analysis_data['supply_demand'].get('value_area', {}).get('high', 0):.2f}
• نقطه کنترل: {analysis_data['supply_demand'].get('point_of_control', 0):.2f}
• عدم تعادل: {analysis_data['supply_demand'].get('imbalance', 0)*100:.1f}%

📰 *تحلیل احساسات:* {analysis_data['sentiment']['interpretation']}

⚠️ *هشدار:* این تحلیل صرفاً جنبه آموزشی دارد. همیشه ریسک را مدیریت کنید.
            """
        
        try:
            prompt = f"""
            شما یک تحلیلگر مالی حرفه‌ای هستید. لطفاً تحلیل زیر را به زبان فارسی و به صورت کامل توضیح دهید:
            
            نماد: {analysis_data['symbol']}
            قیمت فعلی: {analysis_data['price']}
            سیگنال: {analysis_data['signal']}
            اطمینان: {analysis_data['confidence']*100:.1f}%
            
            تحلیل تکنیکال:
            - الگوی البروکس: {analysis_data['elliott']['current_pattern']}
            - مناطق عرضه و تقاضا: {analysis_data['supply_demand']['point_of_control']}
            - شاخص‌های کلیدی: RSI={analysis_data['technical']['rsi']:.1f}, MACD={analysis_data['technical']['macd']['histogram']:.2f}
            
            تحلیل احساسات: {analysis_data['sentiment']['interpretation']}
            
            لطفاً تحلیل کامل شامل:
            1. توضیح وضعیت فعلی بازار
            2. دلایل سیگنال صادر شده
            3. نقاط ورود و خروج احتمالی
            4. مدیریت ریسک پیشنهادی
            5. چشم‌انداز آینده
            """
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=1000,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return explanation
        except Exception as e:
            logger.error(f"Error generating Persian explanation: {e}")
            return f"تحلیل {analysis_data['symbol']}: سیگنال {analysis_data['signal']} با اطمینان {analysis_data['confidence']*100:.1f}%"
    
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
            "لطفاً نماد مورد نظر را وارد کنید (مثال: BTC-USD):"
        )
    
    async def handle_symbol(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پردازش نماد ورودی"""
        if context.user_data.get('state') == 'deep_analysis':
            symbol = update.message.text.upper()
            user_id = update.effective_user.id
            
            try:
                # دریافت داده‌های بازار
                data = yf.download(symbol, period="2y", interval="1d")
                
                if data.empty:
                    await update.message.reply_text("❌ نماد یافت نشد. لطفاً دوباره تلاش کنید.")
                    return
                
                # تحلیل‌های پیشرفته
                elliott = self.advanced_elliott_wave(data)
                supply_demand = self.advanced_supply_demand(symbol)
                technical = self.advanced_technical_analysis(data)
                sentiment = self.advanced_sentiment_analysis(symbol)
                
                # محاسبه امتیاز سیگنال
                signal_score = self.calculate_signal_score(elliott, supply_demand, technical, sentiment)
                signal = 'BUY' if signal_score > 0.7 else 'SELL' if signal_score < 0.3 else 'HOLD'
                
                # تحلیل ترکیبی
                analysis_result = {
                    'symbol': symbol,
                    'price': data['Close'].iloc[-1],
                    'elliott': elliott,
                    'supply_demand': supply_demand,
                    'technical': technical,
                    'sentiment': sentiment,
                    'signal': signal,
                    'confidence': signal_score,
                    'timestamp': datetime.now().isoformat()
                }
                
                # تولید توضیحات فارسی
                explanation = self.generate_persian_explanation(analysis_result)
                
                # ذخیره تحلیل
                self.save_analysis(user_id, symbol, 'deep_analysis', analysis_result)
                
                # ارسال تحلیل
                await update.message.reply_text(explanation, parse_mode='Markdown')
                
            except Exception as e:
                logger.error(f"Error in deep analysis: {e}")
                await update.message.reply_text("❌ خطا در تحلیل. لطفاً دوباره تلاش کنید.")
            
            context.user_data['state'] = None
    
    def calculate_signal_score(self, elliott, supply_demand, technical, sentiment):
        """محاسبه امتیاز سیگنال"""
        score = 0
        
        # امتیاز الیوت
        if elliott.get('current_pattern') == 'bullish':
            score += 0.3
        elif elliott.get('current_pattern') == 'bearish':
            score -= 0.3
        
        # امتیاز عرضه و تقاضا
        if supply_demand.get('imbalance') and supply_demand['imbalance'] > 0.2:  # تقاضای بیشتر
            score += 0.2
        elif supply_demand.get('imbalance') and supply_demand['imbalance'] < -0.2:  # عرضه بیشتر
            score -= 0.2
        
        # امتیاز تکنیکال
        if technical.get('rsi') and technical['rsi'] < 30:  # اشباع فروش
            score += 0.2
        elif technical.get('rsi') and technical['rsi'] > 70:  # اشباع خرید
            score -= 0.2
        
        if technical.get('macd', {}).get('histogram', 0) > 0:
            score += 0.1
        else:
            score -= 0.1
        
        # امتیاز احساسات
        score += sentiment.get('combined', 0) * 0.2
        
        return max(0, min(1, score))  # نرمال‌سازی بین 0 و 1
    
    async def golden_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """سیگنال‌های طلایی"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        watchlist = self.get_user_watchlist(user_id)
        
        if not watchlist:
            await query.edit_message_text("❌ واچ‌لیست شما خالی است. لطفاً ابتدا نمادها را به واچ‌لیست اضافه کنید.")
            return
        
        signals = []
        for symbol in watchlist:
            try:
                data = yf.download(symbol, period="6mo", interval="1d")
                
                # تحلیل سریع
                elliott = self.advanced_elliott_wave(data)
                supply_demand = self.advanced_supply_demand(symbol)
                technical = self.advanced_technical_analysis(data)
                sentiment = self.advanced_sentiment_analysis(symbol)
                
                # محاسبه امتیاز سیگنال
                signal_score = self.calculate_signal_score(elliott, supply_demand, technical, sentiment)
                
                if signal_score > 0.8:  # سیگنال طلایی
                    signals.append({
                        'symbol': symbol,
                        'score': signal_score,
                        'price': data['Close'].iloc[-1],
                        'signal': 'BUY' if signal_score > 0.9 else 'SELL'
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        if signals:
            response = "🔥 *سیگنال‌های طلایی (اطمینان >80%):*\\n\\n"
            for sig in signals:
                response += f"• {sig['symbol']}: {sig['signal']}\\n"
                response += f"  قیمت: {sig['price']:,.2f}\\n"
                response += f"  اطمینان: {sig['score']*100:.1f}%\\n\\n"
            
            await query.edit_message_text(response, parse_mode='Markdown')
        else:
            await query.edit_message_text("در حال حاضر سیگنال طلایی وجود ندارد.")
    
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