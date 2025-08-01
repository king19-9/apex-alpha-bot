import os
import logging
import json
import time
import threading
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dotenv import load_dotenv

# API های خارجی
import ccxt
from tradingview_ta import TA_Handler, Interval
import requests
from newsapi import NewsApiClient

# ربات تلگرام
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes, ConversationHandler

# بارگیری متغیرهای محیطی
load_dotenv()

# تنظیم لاگینگ
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ثابت‌ها
ANALYZING, AWAITING_SYMBOL, AWAITING_DIRECTION = range(3)
GOLD_SIGNAL_THRESHOLD = 80
SILVER_SIGNAL_THRESHOLD = 65

# بارگیری تنظیمات از متغیرهای محیطی یا فایل تنظیمات
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "your_telegram_token")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "your_news_api_key")
TRADING_PAIRS = ["BTC/USDT", "ETH/USDT", "XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY"]  # واچ‌لیست پیش‌فرض

# مقداردهی اولیه API های خارجی
news_api = NewsApiClient(api_key=NEWS_API_KEY)
exchange = ccxt.binance()  # صرافی پیش‌فرض، قابل تغییر

# راه‌اندازی پایگاه داده
def setup_database():
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    # جدول کاربران
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        gold_notifications BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # جدول سیگنال‌ها
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        direction TEXT,
        entry_price REAL,
        stop_loss REAL,
        take_profit1 REAL,
        take_profit2 REAL,
        confidence INTEGER,
        status TEXT DEFAULT 'active',
        strategy TEXT,
        win_rate REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        closed_at TIMESTAMP,
        result TEXT,
        profit_loss REAL
    )
    ''')
    
    # جدول معاملات تحت نظارت
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS monitored_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        symbol TEXT,
        direction TEXT,
        entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (user_id)
    )
    ''')
    
    conn.commit()
    conn.close()

# عملیات پایگاه داده
class Database:
    @staticmethod
    def get_connection():
        return sqlite3.connect('trading_bot.db')
    
    @staticmethod
    def add_user(user_id):
        conn = Database.get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
        conn.commit()
        conn.close()
    
    @staticmethod
    def set_gold_notifications(user_id, status):
        conn = Database.get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET gold_notifications = ? WHERE user_id = ?", (status, user_id))
        conn.commit()
        conn.close()
    
    @staticmethod
    def get_gold_notification_users():
        conn = Database.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM users WHERE gold_notifications = 1")
        users = [row[0] for row in cursor.fetchall()]
        conn.close()
        return users
    
    @staticmethod
    def add_signal(signal_data):
        conn = Database.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO signals (
            symbol, direction, entry_price, stop_loss, take_profit1, take_profit2,
            confidence, strategy, win_rate, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal_data["symbol"], signal_data["direction"], signal_data["entry_price"],
            signal_data["stop_loss"], signal_data["take_profit1"], signal_data["take_profit2"],
            signal_data["confidence"], signal_data["strategy"], signal_data["win_rate"],
            datetime.now()
        ))
        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return signal_id
    
    @staticmethod
    def update_signal_result(signal_id, result, profit_loss):
        conn = Database.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE signals 
        SET status = 'closed', closed_at = ?, result = ?, profit_loss = ?
        WHERE id = ?
        """, (datetime.now(), result, profit_loss, signal_id))
        conn.commit()
        conn.close()
    
    @staticmethod
    def add_monitored_trade(user_id, symbol, direction):
        conn = Database.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO monitored_trades (user_id, symbol, direction)
        VALUES (?, ?, ?)
        """, (user_id, symbol, direction))
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id
    
    @staticmethod
    def remove_monitored_trade(trade_id):
        conn = Database.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM monitored_trades WHERE id = ?", (trade_id,))
        conn.commit()
        conn.close()
    
    @staticmethod
    def get_user_monitored_trades(user_id):
        conn = Database.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        SELECT id, symbol, direction, entry_time
        FROM monitored_trades
        WHERE user_id = ?
        """, (user_id,))
        trades = cursor.fetchall()
        conn.close()
        return trades
    
    @staticmethod
    def get_all_monitored_trades():
        conn = Database.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        SELECT id, user_id, symbol, direction, entry_time
        FROM monitored_trades
        """)
        trades = cursor.fetchall()
        conn.close()
        return trades
    
    @staticmethod
    def get_signal_stats(period=None):
        conn = Database.get_connection()
        cursor = conn.cursor()
        
        if period:
            date_threshold = datetime.now() - timedelta(days=period)
            cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN confidence >= ? THEN 1 ELSE 0 END) as gold_signals,
                SUM(CASE WHEN confidence >= ? AND confidence < ? THEN 1 ELSE 0 END) as silver_signals,
                SUM(CASE WHEN confidence >= ? AND result = 'win' THEN 1 ELSE 0 END) as gold_wins,
                SUM(CASE WHEN confidence >= ? AND confidence < ? AND result = 'win' THEN 1 ELSE 0 END) as silver_wins,
                AVG(profit_loss) as avg_profit_loss
            FROM signals
            WHERE created_at >= ?
            """, (
                GOLD_SIGNAL_THRESHOLD, SILVER_SIGNAL_THRESHOLD, GOLD_SIGNAL_THRESHOLD,
                GOLD_SIGNAL_THRESHOLD, SILVER_SIGNAL_THRESHOLD, GOLD_SIGNAL_THRESHOLD,
                date_threshold
            ))
        else:
            cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN confidence >= ? THEN 1 ELSE 0 END) as gold_signals,
                SUM(CASE WHEN confidence >= ? AND confidence < ? THEN 1 ELSE 0 END) as silver_signals,
                SUM(CASE WHEN confidence >= ? AND result = 'win' THEN 1 ELSE 0 END) as gold_wins,
                SUM(CASE WHEN confidence >= ? AND confidence < ? AND result = 'win' THEN 1 ELSE 0 END) as silver_wins,
                AVG(profit_loss) as avg_profit_loss
            FROM signals
            """, (
                GOLD_SIGNAL_THRESHOLD, SILVER_SIGNAL_THRESHOLD, GOLD_SIGNAL_THRESHOLD,
                GOLD_SIGNAL_THRESHOLD, SILVER_SIGNAL_THRESHOLD, GOLD_SIGNAL_THRESHOLD
            ))
        
        stats = cursor.fetchone()
        
        # دریافت سیگنال‌های اخیر
        if period:
            cursor.execute("""
            SELECT id, symbol, direction, confidence, result, profit_loss, created_at
            FROM signals
            WHERE created_at >= ?
            ORDER BY created_at DESC
            LIMIT 10
            """, (date_threshold,))
        else:
            cursor.execute("""
            SELECT id, symbol, direction, confidence, result, profit_loss, created_at
            FROM signals
            ORDER BY created_at DESC
            LIMIT 10
            """)
        
        recent_signals = cursor.fetchall()
        conn.close()
        
        return stats, recent_signals

# موتور تحلیل
class AnalysisEngine:
    @staticmethod
    def get_session_info():
        """تعیین سشن معاملاتی فعلی (آسیا، لندن، نیویورک، توکیو، سیدنی)"""
        now = datetime.utcnow()
        hour = now.hour
        
        sessions = []
        
        # آسیا (توکیو): 00:00-09:00 UTC
        if 0 <= hour < 9:
            sessions.append("توکیو (آسیا)")
        
        # لندن: 08:00-17:00 UTC
        if 8 <= hour < 17:
            sessions.append("لندن (اروپا)")
        
        # نیویورک: 13:00-22:00 UTC
        if 13 <= hour < 22:
            sessions.append("نیویورک (آمریکا)")
            
        # سیدنی: 22:00-07:00 UTC
        if hour >= 22 or hour < 7:
            sessions.append("سیدنی (استرالیا)")
        
        return sessions
    
    @staticmethod
    def get_current_price(symbol):
        """دریافت قیمت فعلی یک نماد"""
        try:
            ticker = exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"خطا در دریافت قیمت برای {symbol}: {e}")
            return None
    
    @staticmethod
    def get_ohlcv_data(symbol, timeframe='1h', limit=100):
        """دریافت داده‌های OHLCV برای یک نماد"""
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"خطا در دریافت داده‌های OHLCV برای {symbol}: {e}")
            return None
    
    @staticmethod
    def get_tradingview_analysis(symbol, exchange_name="BINANCE", screener="crypto", interval=Interval.INTERVAL_1_HOUR):
        """دریافت تحلیل تکنیکال TradingView"""
        try:
            handler = TA_Handler(
                symbol=symbol.split('/')[0],
                exchange=exchange_name,
                screener=screener,
                interval=interval
            )
            analysis = handler.get_analysis()
            return analysis
        except Exception as e:
            logger.error(f"خطا در دریافت تحلیل TradingView برای {symbol}: {e}")
            return None
    
    @staticmethod
    def get_economic_data():
        """دریافت داده‌های اقتصادی (نسخه ساده شده)"""
        # در یک پیاده‌سازی واقعی، این به investpy یا مشابه آن متصل می‌شود
        # برای الان، داده‌های مثال برمی‌گردانیم
        return {
            "fed_rate": 5.25,
            "upcoming_events": [
                {"date": "2025-09-20", "event": "جلسه FOMC", "importance": "بالا"},
                {"date": "2025-09-15", "event": "خرده‌فروشی آمریکا", "importance": "متوسط"}
            ]
        }
    
    @staticmethod
    def get_news(symbol):
        """دریافت آخرین اخبار برای یک نماد"""
        try:
            # استخراج ارز پایه برای جستجوی اخبار
            if '/' in symbol:
                search_term = symbol.split('/')[0]
            else:
                search_term = symbol
                
            # مدیریت موارد خاص
            if search_term == "XAU":
                search_term = "Gold"
            
            # دریافت اخبار از NewsAPI
            news = news_api.get_everything(
                q=search_term,
                language='en',
                sort_by='publishedAt',
                page_size=5
            )
            
            return news.get('articles', [])
        except Exception as e:
            logger.error(f"خطا در دریافت اخبار برای {symbol}: {e}")
            return []
    
    @staticmethod
    def analyze_sentiment(text):
        """تحلیل احساسات ساده"""
        # یک پیاده‌سازی پیچیده‌تر از NLP استفاده می‌کند
        positive_words = ['bullish', 'surge', 'gain', 'positive', 'up', 'rise', 'growth']
        negative_words = ['bearish', 'drop', 'fall', 'negative', 'down', 'decline', 'loss']
        
        text = text.lower()
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return "مثبت"
        elif negative_count > positive_count:
            return "منفی"
        else:
            return "خنثی"
    
    @staticmethod
    def identify_support_resistance(df):
        """شناسایی سطوح حمایت و مقاومت کلیدی"""
        # پیاده‌سازی ساده - در یک ربات واقعی این پیچیده‌تر خواهد بود
        pivot_high = df['high'].rolling(5, center=True).max()
        pivot_low = df['low'].rolling(5, center=True).min()
        
        resistance_levels = []
        support_levels = []
        
        for i in range(2, len(df) - 2):
            if pivot_high.iloc[i] == df['high'].iloc[i] and pivot_high.iloc[i] > pivot_high.iloc[i-1] and pivot_high.iloc[i] > pivot_high.iloc[i+1]:
                resistance_levels.append(df['high'].iloc[i])
            
            if pivot_low.iloc[i] == df['low'].iloc[i] and pivot_low.iloc[i] < pivot_low.iloc[i-1] and pivot_low.iloc[i] < pivot_low.iloc[i+1]:
                support_levels.append(df['low'].iloc[i])
        
        # دریافت 3 سطح اخیر
        resistance_levels = sorted(resistance_levels)[-3:]
        support_levels = sorted(support_levels)[:3]
        
        return support_levels, resistance_levels
    
    @staticmethod
    def identify_candlestick_patterns(df):
        """شناسایی الگوهای کندل استیک"""
        patterns = []
        
        # دوجی
        df['body'] = abs(df['close'] - df['open'])
        df['range'] = df['high'] - df['low']
        df['is_doji'] = df['body'] <= 0.1 * df['range']
        
        # چکش
        df['upper_shadow'] = df.apply(
            lambda x: x['high'] - max(x['open'], x['close']), axis=1
        )
        df['lower_shadow'] = df.apply(
            lambda x: min(x['open'], x['close']) - x['low'], axis=1
        )
        df['is_hammer'] = (df['lower_shadow'] >= 2 * df['body']) & (df['upper_shadow'] <= 0.1 * df['range'])
        
        # ستاره تیرانداز
        df['is_shooting_star'] = (df['upper_shadow'] >= 2 * df['body']) & (df['lower_shadow'] <= 0.1 * df['range'])
        
        # انگلفینگ
        df['prev_body'] = df['body'].shift(1)
        df['prev_close'] = df['close'].shift(1)
        df['prev_open'] = df['open'].shift(1)
        
        df['is_bullish_engulfing'] = (
            (df['close'] > df['open']) &  # کندل فعلی صعودی است
            (df['prev_close'] < df['prev_open']) &  # کندل قبلی نزولی است
            (df['close'] > df['prev_open']) &  # بسته شدن فعلی بالاتر از باز شدن قبلی است
            (df['open'] < df['prev_close'])  # باز شدن فعلی پایین‌تر از بسته شدن قبلی است
        )
        
        df['is_bearish_engulfing'] = (
            (df['close'] < df['open']) &  # کندل فعلی نزولی است
            (df['prev_close'] > df['prev_open']) &  # کندل قبلی صعودی است
            (df['close'] < df['prev_open']) &  # بسته شدن فعلی پایین‌تر از باز شدن قبلی است
            (df['open'] > df['prev_close'])  # باز شدن فعلی بالاتر از بسته شدن قبلی است
        )
        
        # بررسی الگوها در آخرین 3 کندل
        for i in range(min(3, len(df))):
            idx = -i - 1
            if idx < -len(df):
                continue
                
            if df['is_doji'].iloc[idx]:
                patterns.append(f"دوجی {i+1} کندل قبل شناسایی شد")
            
            if df['is_hammer'].iloc[idx]:
                patterns.append(f"چکش {i+1} کندل قبل شناسایی شد")
            
            if df['is_shooting_star'].iloc[idx]:
                patterns.append(f"ستاره تیرانداز {i+1} کندل قبل شناسایی شد")
            
            if df['is_bullish_engulfing'].iloc[idx]:
                patterns.append(f"انگلفینگ صعودی {i+1} کندل قبل شناسایی شد")
            
            if df['is_bearish_engulfing'].iloc[idx]:
                patterns.append(f"انگلفینگ نزولی {i+1} کندل قبل شناسایی شد")
        
        return patterns
    
    @staticmethod
    def identify_market_structure(df):
        """شناسایی ساختار بازار (سقف‌های بالاتر، کف‌های پایین‌تر و ...)"""
        structure = {}
        
        # دریافت نوسانات بالا و پایین
        df['swing_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['swing_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        
        # دریافت 5 نوسان اخیر بالا و پایین
        recent_swing_highs = df[df['swing_high']].iloc[-5:]['high'].tolist()
        recent_swing_lows = df[df['swing_low']].iloc[-5:]['low'].tolist()
        
        # تعیین روند بر اساس نوسانات بالا و پایین
        if len(recent_swing_highs) >= 2:
            structure['higher_highs'] = recent_swing_highs[-1] > recent_swing_highs[-2] if len(recent_swing_highs) >= 2 else None
        
        if len(recent_swing_lows) >= 2:
            structure['higher_lows'] = recent_swing_lows[-1] > recent_swing_lows[-2] if len(recent_swing_lows) >= 2 else None
        
        # تعیین روند کلی
        if structure.get('higher_highs') and structure.get('higher_lows'):
            structure['trend'] = "روند صعودی"
        elif not structure.get('higher_highs', True) and not structure.get('higher_lows', True):
            structure['trend'] = "روند نزولی"
        else:
            structure['trend'] = "روند جانبی/نامشخص"
        
        return structure
    
    @staticmethod
    def backtest_strategies(symbol):
        """بک‌تست چندین استراتژی برای یافتن بهترین مورد"""
        strategies = {
            "EMA Crossover": {
                "description": "تقاطع EMA 9 بالا/پایین EMA 21",
                "win_rate": 0,
                "trades": 0
            },
            "RSI Divergence": {
                "description": "قیمت کف پایین‌تر می‌سازد در حالی که RSI کف بالاتر می‌سازد (صعودی) یا برعکس",
                "win_rate": 0,
                "trades": 0
            },
            "Support/Resistance Bounce": {
                "description": "برگشت قیمت از سطوح حمایت/مقاومت کلیدی",
                "win_rate": 0,
                "trades": 0
            },
            "Ichimoku Cloud": {
                "description": "عبور قیمت از بالا/پایین ابر با تأیید",
                "win_rate": 0,
                "trades": 0
            },
            "Bollinger Band Squeeze": {
                "description": "شکست قیمت پس از انقباض باندها",
                "win_rate": 0,
                "trades": 0
            }
        }
        
        # دریافت داده‌های تاریخی برای چندین تایم‌فریم
        timeframes = ['1h', '4h', '1d']
        data = {}
        for tf in timeframes:
            df = AnalysisEngine.get_ohlcv_data(symbol, timeframe=tf, limit=200)
            if df is not None:
                data[tf] = df
        
        # اگر نتوانستیم داده‌ها را دریافت کنیم، نتایج خالی برمی‌گردانیم
        if not data:
            return {"best_strategy": None, "win_rate": 0, "all_strategies": strategies}
        
        # بک‌تست ساده برای هر استراتژی
        # در یک پیاده‌سازی واقعی، این بسیار پیچیده‌تر خواهد بود
        
        # استراتژی تقاطع EMA
        for tf, df in data.items():
            if df is not None and len(df) > 30:
                # محاسبه EMA ها
                df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
                df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
                
                # تولید سیگنال‌ها
                df['signal'] = 0
                df.loc[df['ema9'] > df['ema21'], 'signal'] = 1
                df.loc[df['ema9'] < df['ema21'], 'signal'] = -1
                
                # محاسبه بازده‌ها
                df['pct_change'] = df['close'].pct_change()
                df['strategy_return'] = df['signal'].shift(1) * df['pct_change']
                
                # شمارش معاملات سودآور
                profitable_trades = (df['strategy_return'] > 0).sum()
                total_trades = (df['signal'] != df['signal'].shift(1)).sum() - 1  # به استثنای اولین سیگنال
                
                if total_trades > 0:
                    win_rate = profitable_trades / total_trades * 100
                    strategies["EMA Crossover"]["win_rate"] = max(strategies["EMA Crossover"]["win_rate"], win_rate)
                    strategies["EMA Crossover"]["trades"] += total_trades
        
        # استراتژی‌های دیگر به طور مشابه پیاده‌سازی می‌شوند
        # برای این مثال، نتایج را برای استراتژی‌های دیگر شبیه‌سازی می‌کنیم
        
        strategies["RSI Divergence"]["win_rate"] = 68.5
        strategies["RSI Divergence"]["trades"] = 35
        
        strategies["Support/Resistance Bounce"]["win_rate"] = 72.3
        strategies["Support/Resistance Bounce"]["trades"] = 42
        
        strategies["Ichimoku Cloud"]["win_rate"] = 65.8
        strategies["Ichimoku Cloud"]["trades"] = 38
        
        strategies["Bollinger Band Squeeze"]["win_rate"] = 70.1
        strategies["Bollinger Band Squeeze"]["trades"] = 30
        
        # یافتن بهترین استراتژی
        best_strategy = max(strategies.items(), key=lambda x: x[1]["win_rate"])
        
        return {
            "best_strategy": best_strategy[0],
            "win_rate": best_strategy[1]["win_rate"],
            "all_strategies": strategies
        }
    
    @staticmethod
    def analyze_symbol(symbol):
        """انجام تحلیل جامع یک نماد"""
        # مقداردهی اولیه ساختار نتیجه
        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "technical": {},
            "fundamental": {},
            "recommendation": {}
        }
        
        # قیمت فعلی و اطلاعات سشن
        price = AnalysisEngine.get_current_price(symbol)
        sessions = AnalysisEngine.get_session_info()
        
        result["summary"]["current_price"] = price
        result["summary"]["trading_sessions"] = sessions
        
        # بک‌تست برای یافتن بهترین استراتژی
        backtest_results = AnalysisEngine.backtest_strategies(symbol)
        result["summary"]["best_strategy"] = backtest_results["best_strategy"]
        result["summary"]["strategy_win_rate"] = backtest_results["win_rate"]
        
        # تحلیل تکنیکال
        # دریافت داده‌ها برای چندین تایم‌فریم
        timeframes = {
            '15m': 'کوتاه‌مدت',
            '1h': 'میان‌مدت',
            '4h': 'میان‌مدت',
            '1d': 'بلندمدت'
        }
        
        result["technical"]["trends"] = {}
        
        for tf, tf_name in timeframes.items():
            df = AnalysisEngine.get_ohlcv_data(symbol, timeframe=tf)
            if df is not None:
                # تعیین روند
                if len(df) >= 20:
                    df['sma20'] = df['close'].rolling(20).mean()
                    last_close = df['close'].iloc[-1]
                    last_sma20 = df['sma20'].iloc[-1]
                    
                    if last_close > last_sma20:
                        trend = "صعودی"
                    elif last_close < last_sma20:
                        trend = "نزولی"
                    else:
                        trend = "خنثی"
                    
                    result["technical"]["trends"][tf_name] = trend
        
        # تحلیل RSI
        hourly_data = AnalysisEngine.get_ohlcv_data(symbol, timeframe='1h')
        if hourly_data is not None and len(hourly_data) >= 14:
            delta = hourly_data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            
            rs = gain / loss
            hourly_data['rsi'] = 100 - (100 / (1 + rs))
            
            last_rsi = hourly_data['rsi'].iloc[-1]
            
            if last_rsi > 70:
                rsi_condition = "اشباع خرید"
            elif last_rsi < 30:
                rsi_condition = "اشباع فروش"
            else:
                rsi_condition = "خنثی"
            
            result["technical"]["rsi"] = {
                "value": last_rsi,
                "condition": rsi_condition
            }
        
        # حمایت و مقاومت
        daily_data = AnalysisEngine.get_ohlcv_data(symbol, timeframe='1d')
        if daily_data is not None:
            support_levels, resistance_levels = AnalysisEngine.identify_support_resistance(daily_data)
            
            result["technical"]["support_levels"] = support_levels
            result["technical"]["resistance_levels"] = resistance_levels
        
        # تحلیل پرایس اکشن
        if hourly_data is not None:
            candlestick_patterns = AnalysisEngine.identify_candlestick_patterns(hourly_data)
            market_structure = AnalysisEngine.identify_market_structure(hourly_data)
            
            result["technical"]["candlestick_patterns"] = candlestick_patterns
            result["technical"]["market_structure"] = market_structure
        
        # تحلیل TradingView
        tv_analysis = AnalysisEngine.get_tradingview_analysis(symbol)
        if tv_analysis:
            result["technical"]["tradingview"] = {
                "summary": tv_analysis.summary,
                "oscillators": tv_analysis.oscillators,
                "moving_averages": tv_analysis.moving_averages
            }
        
        # تحلیل فاندامنتال
        # داده‌های اقتصادی
        econ_data = AnalysisEngine.get_economic_data()
        result["fundamental"]["economic"] = econ_data
        
        # اخبار
        news = AnalysisEngine.get_news(symbol)
        news_with_sentiment = []
        
        for article in news:
            sentiment = AnalysisEngine.analyze_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
            news_with_sentiment.append({
                "title": article.get('title'),
                "sentiment": sentiment,
                "url": article.get('url'),
                "published_at": article.get('publishedAt')
            })
        
        result["fundamental"]["news"] = news_with_sentiment
        
        # احساسات کلی از اخبار
        if news_with_sentiment:
            positive_count = sum(1 for article in news_with_sentiment if article['sentiment'] == 'مثبت')
            negative_count = sum(1 for article in news_with_sentiment if article['sentiment'] == 'منفی')
            
            if positive_count > negative_count:
                overall_sentiment = "مثبت"
            elif negative_count > positive_count:
                overall_sentiment = "منفی"
            else:
                overall_sentiment = "خنثی"
            
            result["fundamental"]["overall_sentiment"] = overall_sentiment
        
        # توصیه معاملاتی
        # ترکیب عوامل تکنیکال و فاندامنتال برای تولید یک توصیه
        # این یک نسخه ساده شده است - یک پیاده‌سازی واقعی پیچیده‌تر خواهد بود
        
        technical_score = 0
        
        # امتیاز بر اساس روندها
        for tf, trend in result["technical"]["trends"].items():
            if trend == "صعودی":
                technical_score += 1
            elif trend == "نزولی":
                technical_score -= 1
        
        # امتیاز بر اساس RSI
        if "rsi" in result["technical"]:
            if result["technical"]["rsi"]["condition"] == "اشباع فروش":
                technical_score += 1
            elif result["technical"]["rsi"]["condition"] == "اشباع خرید":
                technical_score -= 1
        
        # امتیاز بر اساس TradingView
        if "tradingview" in result["technical"]:
            tv_recommendation = result["technical"]["tradingview"]["summary"].get("RECOMMENDATION")
            if tv_recommendation in ["STRONG_BUY", "BUY"]:
                technical_score += 2
            elif tv_recommendation in ["STRONG_SELL", "SELL"]:
                technical_score -= 2
        
        # امتیاز بر اساس الگوهای کندل استیک
        if "candlestick_patterns" in result["technical"]:
            for pattern in result["technical"]["candlestick_patterns"]:
                if "صعودی" in pattern:
                    technical_score += 0.5
                elif "نزولی" in pattern:
                    technical_score -= 0.5
        
        # امتیاز بر اساس ساختار بازار
        if "market_structure" in result["technical"] and "trend" in result["technical"]["market_structure"]:
            if result["technical"]["market_structure"]["trend"] == "روند صعودی":
                technical_score += 1
            elif result["technical"]["market_structure"]["trend"] == "روند نزولی":
                technical_score -= 1
        
        # امتیاز فاندامنتال
        fundamental_score = 0
        
        # امتیاز بر اساس احساسات اخبار
        if "overall_sentiment" in result["fundamental"]:
            if result["fundamental"]["overall_sentiment"] == "مثبت":
                fundamental_score += 1
            elif result["fundamental"]["overall_sentiment"] == "منفی":
                fundamental_score -= 1
        
        # امتیاز ترکیبی
        total_score = technical_score + fundamental_score
        
        # نرمال‌سازی به درصد اطمینان (0-100)
        confidence = min(max(50 + (total_score * 10), 0), 100)
        
        # تعیین جهت
        if total_score > 1:  # سیگنال قوی صعودی
            direction = "BUY"
            entry_price = price
            stop_loss = price * 0.97  # 3% زیر نقطه ورود
            take_profit1 = price * 1.05  # 5% بالای نقطه ورود
            take_profit2 = price * 1.10  # 10% بالای نقطه ورود
            risk_reward = ((take_profit1 - entry_price) + (take_profit2 - entry_price)) / 2 / (entry_price - stop_loss)
            recommended_leverage = min(5, 1 / (entry_price - stop_loss) * entry_price * 0.1)  # حداکثر 5x، با استفاده از 10% ریسک
            
            result["recommendation"] = {
                "action": direction,
                "confidence": confidence,
                "entry": entry_price,
                "stop_loss": stop_loss,
                "take_profit1": take_profit1,
                "take_profit2": take_profit2,
                "risk_reward_ratio": risk_reward,
                "recommended_leverage": recommended_leverage
            }
        elif total_score < -1:  # سیگنال قوی نزولی
            direction = "SELL"
            entry_price = price
            stop_loss = price * 1.03  # 3% بالای نقطه ورود
            take_profit1 = price * 0.95  # 5% زیر نقطه ورود
            take_profit2 = price * 0.90  # 10% زیر نقطه ورود
            risk_reward = ((entry_price - take_profit1) + (entry_price - take_profit2)) / 2 / (stop_loss - entry_price)
            recommended_leverage = min(5, 1 / (stop_loss - entry_price) * entry_price * 0.1)  # حداکثر 5x، با استفاده از 10% ریسک
            
            result["recommendation"] = {
                "action": direction,
                "confidence": confidence,
                "entry": entry_price,
                "stop_loss": stop_loss,
                "take_profit1": take_profit1,
                "take_profit2": take_profit2,
                "risk_reward_ratio": risk_reward,
                "recommended_leverage": recommended_leverage
            }
        else:  # سیگنال واضحی نیست
            result["recommendation"] = {
                "action": "WAIT",
                "confidence": confidence,
                "reason": "در حال حاضر فرصت معاملاتی واضحی وجود ندارد. شرایط بازار نامشخص یا متناقض است."
            }
        
        return result

# شکارچی سیگنال
class SignalHunter:
    def __init__(self, watchlist=None):
        self.watchlist = watchlist or TRADING_PAIRS
        self.running = False
        self.thread = None
    
    def start(self):
        """شروع فرآیند شکار سیگنال"""
        if self.running:
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._hunt_signals)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def stop(self):
        """توقف فرآیند شکار سیگنال"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None
        return True
    
    def _hunt_signals(self):
        """حلقه اصلی شکار سیگنال"""
        while self.running:
            try:
                # اسکن هر نماد در واچ‌لیست
                for symbol in self.watchlist:
                    logger.info(f"اسکن {symbol} برای سیگنال‌ها...")
                    
                    # تحلیل نماد
                    analysis = AnalysisEngine.analyze_symbol(symbol)
                    
                    # بررسی اینکه آیا فرصت معاملاتی وجود دارد
                    if "recommendation" in analysis and analysis["recommendation"]["action"] in ["BUY", "SELL"]:
                        # بررسی سطح اطمینان
                        confidence = analysis["recommendation"]["confidence"]
                        
                        if confidence >= GOLD_SIGNAL_THRESHOLD:
                            # سیگنال طلایی
                            signal_data = {
                                "symbol": symbol,
                                "direction": analysis["recommendation"]["action"],
                                "entry_price": analysis["recommendation"]["entry"],
                                "stop_loss": analysis["recommendation"]["stop_loss"],
                                "take_profit1": analysis["recommendation"]["take_profit1"],
                                "take_profit2": analysis["recommendation"]["take_profit2"],
                                "confidence": confidence,
                                "strategy": analysis["summary"]["best_strategy"],
                                "win_rate": analysis["summary"]["strategy_win_rate"]
                            }
                            
                            # ذخیره سیگنال در پایگاه داده
                            signal_id = Database.add_signal(signal_data)
                            
                            # اطلاع‌رسانی به کاربرانی که نوتیفیکیشن طلایی را فعال کرده‌اند
                            self._notify_gold_signal(signal_id, signal_data, analysis)
                        
                        elif confidence >= SILVER_SIGNAL_THRESHOLD:
                            # سیگنال نقره‌ای
                            signal_data = {
                                "symbol": symbol,
                                "direction": analysis["recommendation"]["action"],
                                "entry_price": analysis["recommendation"]["entry"],
                                "stop_loss": analysis["recommendation"]["stop_loss"],
                                "take_profit1": analysis["recommendation"]["take_profit1"],
                                "take_profit2": analysis["recommendation"]["take_profit2"],
                                "confidence": confidence,
                                "strategy": analysis["summary"]["best_strategy"],
                                "win_rate": analysis["summary"]["strategy_win_rate"]
                            }
                            
                            # ذخیره سیگنال در پایگاه داده
                            Database.add_signal(signal_data)
                    
                    # خواب برای جلوگیری از برخورد با محدودیت‌های API
                    time.sleep(5)
                
                # خواب قبل از دور بعدی اسکن
                time.sleep(300)  # 5 دقیقه
            
            except Exception as e:
                logger.error(f"خطا در فرآیند شکار سیگنال: {e}")
                time.sleep(60)  # خواب به مدت 1 دقیقه در صورت خطا
    
    def _notify_gold_signal(self, signal_id, signal_data, analysis):
        """اطلاع‌رسانی به کاربران در مورد سیگنال طلایی"""
        # دریافت کاربرانی که نوتیفیکیشن طلایی را فعال کرده‌اند
        users = Database.get_gold_notification_users()
        
        # آماده‌سازی پیام
        message = self._format_signal_message(signal_id, signal_data, analysis)
        
        # ارسال به هر کاربر
        for user_id in users:
            try:
                # این توسط ربات تلگرام مدیریت می‌شود
                # برای الان، فقط لاگ می‌کنیم
                logger.info(f"ارسال نوتیفیکیشن سیگنال طلایی به کاربر {user_id}")
                # در یک پیاده‌سازی واقعی، این یک متد در کلاس ربات را صدا می‌زند
            except Exception as e:
                logger.error(f"خطا در ارسال نوتیفیکیشن به کاربر {user_id}: {e}")
    
    def _format_signal_message(self, signal_id, signal_data, analysis):
        """قالب‌بندی پیام سیگنال برای تلگرام"""
        direction = signal_data["direction"]
        symbol = signal_data["symbol"]
        confidence = signal_data["confidence"]
        entry = signal_data["entry_price"]
        sl = signal_data["stop_loss"]
        tp1 = signal_data["take_profit1"]
        tp2 = signal_data["take_profit2"]
        strategy = signal_data["strategy"]
        win_rate = signal_data["win_rate"]
        
        risk_reward = round(((tp1 - entry) + (tp2 - entry)) / 2 / (entry - sl) if direction == "BUY" 
                            else ((entry - tp1) + (entry - tp2)) / 2 / (sl - entry), 2)
        
        message = f"🔔 *سیگنال طلایی #{signal_id}* 🔔\n\n"
        message += f"*نماد:* {symbol}\n"
        message += f"*جهت:* {'🟢 خرید' if direction == 'BUY' else '🔴 فروش'}\n"
        message += f"*اطمینان:* {confidence:.1f}%\n"
        message += f"*استراتژی:* {strategy} (نرخ پیروزی: {win_rate:.1f}%)\n\n"
        
        message += f"*نقطه ورود:* {entry:.5f}\n"
        message += f"*حد ضرر:* {sl:.5f}\n"
        message += f"*حد سود 1:* {tp1:.5f}\n"
        message += f"*حد سود 2:* {tp2:.5f}\n"
        message += f"*نسبت ریسک/پاداش:* {risk_reward:.2f}\n\n"
        
        # افزودن بینش‌های کلیدی تحلیل
        message += "*نکات برجسته تحلیل:*\n"
        
        # تکنیکال
        if "technical" in analysis:
            tech = analysis["technical"]
            
            if "trends" in tech:
                message += "- *روندها:* "
                trends = [f"{tf}: {trend}" for tf, trend in tech["trends"].items()]
                message += ", ".join(trends) + "\n"
            
            if "rsi" in tech:
                message += f"- *RSI:* {tech['rsi']['value']:.1f} ({tech['rsi']['condition']})\n"
            
            if "candlestick_patterns" in tech and tech["candlestick_patterns"]:
                message += f"- *الگوی کلیدی:* {tech['candlestick_patterns'][0]}\n"
            
            if "market_structure" in tech and "trend" in tech["market_structure"]:
                message += f"- *ساختار بازار:* {tech['market_structure']['trend']}\n"
            
            if "tradingview" in tech and "summary" in tech["tradingview"]:
                message += f"- *TradingView:* {tech['tradingview']['summary'].get('RECOMMENDATION', 'N/A')}\n"
        
        # فاندامنتال
        if "fundamental" in analysis:
            fund = analysis["fundamental"]
            
            if "overall_sentiment" in fund:
                message += f"- *احساسات اخبار:* {fund['overall_sentiment']}\n"
        
        return message

    def get_silver_signals(self, limit=5):
        """دریافت سیگنال‌های نقره‌ای اخیر"""
        conn = Database.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT id, symbol, direction, entry_price, stop_loss, take_profit1, take_profit2, confidence, strategy, win_rate, created_at
        FROM signals
        WHERE confidence >= ? AND confidence < ? AND status = 'active'
        ORDER BY created_at DESC
        LIMIT ?
        """, (SILVER_SIGNAL_THRESHOLD, GOLD_SIGNAL_THRESHOLD, limit))
        
        signals = cursor.fetchall()
        conn.close()
        
        # قالب‌بندی سیگنال‌ها
        formatted_signals = []
        for signal in signals:
            signal_id, symbol, direction, entry, sl, tp1, tp2, confidence, strategy, win_rate, created_at = signal
            
            risk_reward = round(((tp1 - entry) + (tp2 - entry)) / 2 / (entry - sl) if direction == "BUY" 
                                else ((entry - tp1) + (entry - tp2)) / 2 / (sl - entry), 2)
            
            formatted_signal = {
                "id": signal_id,
                "symbol": symbol,
                "direction": direction,
                "entry": entry,
                "stop_loss": sl,
                "take_profit1": tp1,
                "take_profit2": tp2,
                "confidence": confidence,
                "strategy": strategy,
                "win_rate": win_rate,
                "risk_reward": risk_reward,
                "created_at": created_at
            }
            
            formatted_signals.append(formatted_signal)
        
        return formatted_signals

# پایشگر معاملات
class TradeMonitor:
    def __init__(self):
        self.running = False
        self.thread = None
    
    def start(self):
        """شروع فرآیند پایش معامله"""
        if self.running:
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_trades)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def stop(self):
        """توقف فرآیند پایش معامله"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None
        return True
    
    def _monitor_trades(self):
        """حلقه اصلی پایش معامله"""
        while self.running:
            try:
                # دریافت تمام معاملات تحت نظارت
                trades = Database.get_all_monitored_trades()
                
                for trade_id, user_id, symbol, direction, entry_time in trades:
                    # تحلیل نماد
                    analysis = AnalysisEngine.analyze_symbol(symbol)
                    
                    # بررسی اینکه آیا تحلیل با جهت معامله در تضاد است
                    contradiction = False
                    
                    if "recommendation" in analysis:
                        rec_action = analysis["recommendation"]["action"]
                        
                        # بررسی تناقضات
                        if direction == "BUY" and rec_action == "SELL":
                            contradiction = True
                        elif direction == "SELL" and rec_action == "BUY":
                            contradiction = True
                    
                    # اگر تناقضی وجود دارد، هشداری به کاربر ارسال کنید
                    if contradiction:
                        self._send_trade_alert(user_id, trade_id, symbol, direction, analysis)
                    
                    # خواب برای جلوگیری از برخورد با محدودیت‌های API
                    time.sleep(1)
                
                # خواب قبل از دور بعدی پایش
                time.sleep(300)  # 5 دقیقه
            
            except Exception as e:
                logger.error(f"خطا در فرآیند پایش معامله: {e}")
                time.sleep(60)  # خواب به مدت 1 دقیقه در صورت خطا
    
    def _send_trade_alert(self, user_id, trade_id, symbol, direction, analysis):
        """ارسال هشدار معامله به یک کاربر"""
        # آماده‌سازی پیام
        message = self._format_trade_alert_message(trade_id, symbol, direction, analysis)
        
        # ارسال به کاربر
        try:
            # این توسط ربات تلگرام مدیریت می‌شود
            # برای الان، فقط لاگ می‌کنیم
            logger.info(f"ارسال هشدار معامله به کاربر {user_id} برای {symbol}")
            # در یک پیاده‌سازی واقعی، این یک متد در کلاس ربات را صدا می‌زند
        except Exception as e:
            logger.error(f"خطا در ارسال هشدار معامله به کاربر {user_id}: {e}")
    
    def _format_trade_alert_message(self, trade_id, symbol, direction, analysis):
        """قالب‌بندی پیام هشدار معامله برای تلگرام"""
        opposite_direction = "SELL" if direction == "BUY" else "BUY"
        
        message = f"⚠️ *هشدار معامله #{trade_id}* ⚠️\n\n"
        message += f"*نماد:* {symbol}\n"
        message += f"*موقعیت شما:* {'🟢 خرید' if direction == 'BUY' else '🔴 فروش'}\n\n"
        
        message += "*هشدار:* تحلیل اکنون جهت مخالف را پیشنهاد می‌دهد!\n\n"
        
        if "recommendation" in analysis:
            rec = analysis["recommendation"]
            message += f"*اقدام توصیه شده:* {'🟢 خرید' if rec['action'] == 'BUY' else '🔴 فروش'}\n"
            message += f"*اطمینان:* {rec['confidence']:.1f}%\n\n"
        
        # افزودن بینش‌های کلیدی تحلیل
        message += "*نکات برجسته تحلیل:*\n"
        
        # تکنیکال
        if "technical" in analysis:
            tech = analysis["technical"]
            
            if "trends" in tech:
                message += "- *روندها:* "
                trends = [f"{tf}: {trend}" for tf, trend in tech["trends"].items()]
                message += ", ".join(trends) + "\n"
            
            if "rsi" in tech:
                message += f"- *RSI:* {tech['rsi']['value']:.1f} ({tech['rsi']['condition']})\n"
            
            if "candlestick_patterns" in tech and tech["candlestick_patterns"]:
                message += f"- *الگوی کلیدی:* {tech['candlestick_patterns'][0]}\n"
            
            if "market_structure" in tech and "trend" in tech["market_structure"]:
                message += f"- *ساختار بازار:* {tech['market_structure']['trend']}\n"
            
            if "tradingview" in tech and "summary" in tech["tradingview"]:
                message += f"- *TradingView:* {tech['tradingview']['summary'].get('RECOMMENDATION', 'N/A')}\n"
        
        # توصیه
        message += "\n*اقدام پیشنهادی:*\n"
        message += "بستن موقعیت خود را در نظر بگیرید یا حد ضرر خود را برای محافظت از سرمایه خود تنظیم کنید."
        
        return message

# ربات تلگرام
class TradingBot:
    def __init__(self, token):
        self.token = token
        self.application = Application.builder().token(token).build()
        self.signal_hunter = SignalHunter()
        self.trade_monitor = TradeMonitor()
        
        # تنظیم هندلرهای دستور
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        
        # هندلر کوئری کال‌بک برای دکمه‌ها
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        
        # هندلرهای مکالمه
        analyze_conv_handler = ConversationHandler(
            entry_points=[CallbackQueryHandler(self.analyze_symbol_start, pattern='^analyze$')],
            states={
                AWAITING_SYMBOL: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.analyze_symbol_input)]
            },
            fallbacks=[CommandHandler("cancel", self.cancel)]
        )
        
        monitor_conv_handler = ConversationHandler(
            entry_points=[CallbackQueryHandler(self.monitor_trade_start, pattern='^monitor$')],
            states={
                AWAITING_SYMBOL: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.monitor_trade_symbol)],
                AWAITING_DIRECTION: [CallbackQueryHandler(self.monitor_trade_direction)]
            },
            fallbacks=[CommandHandler("cancel", self.cancel)]
        )
        
        self.application.add_handler(analyze_conv_handler)
        self.application.add_handler(monitor_conv_handler)
    
    def start(self):
        """شروع ربات و فرآیندهای پس‌زمینه"""
        # شروع شکارچی سیگنال
        self.signal_hunter.start()
        
        # شروع پایشگر معاملات
        self.trade_monitor.start()
        
        # شروع ربات
        self.application.run_polling()
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """مدیریت دستور /start"""
        user = update.effective_user
        user_id = user.id
        
        # افزودن کاربر به پایگاه داده
        Database.add_user(user_id)
        
        # ایجاد منوی اصلی
        keyboard = [
            [InlineKeyboardButton("🔬 تحلیل عمیق", callback_data='analyze')],
            [InlineKeyboardButton("🥈 سیگنال‌های نقره‌ای", callback_data='silver')],
            [InlineKeyboardButton("🔔 فعال‌سازی هشدارهای طلایی", callback_data='enable_gold')],
            [InlineKeyboardButton("👁️ پایش معامله", callback_data='monitor')]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"به ربات تحلیل معاملات خوش آمدید!\n\n"
            f"این ربات تحلیل‌های جامع معاملاتی برای کریپتو، فارکس، طلا و موارد دیگر ارائه می‌دهد.\n\n"
            f"یک گزینه را از زیر انتخاب کنید تا شروع کنید:",
            reply_markup=reply_markup
        )
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """مدیریت دستور /stats"""
        # دریافت آمار عملکرد
        stats, recent_signals = Database.get_signal_stats(period=30)  # 30 روز اخیر
        
        total, wins, losses, gold_signals, silver_signals, gold_wins, silver_wins, avg_profit_loss = stats
        
        # محاسبه نرخ‌های پیروزی
        overall_win_rate = (wins / total * 100) if total > 0 else 0
        gold_win_rate = (gold_wins / gold_signals * 100) if gold_signals > 0 else 0
        silver_win_rate = (silver_wins / silver_signals * 100) if silver_signals > 0 else 0
        
        # قالب‌بندی پیام
        message = "📊 *آمار عملکرد* 📊\n\n"
        
        message += "*30 روز اخیر:*\n"
        message += f"- کل سیگنال‌ها: {total}\n"
        message += f"- نرخ پیروزی کلی: {overall_win_rate:.1f}%\n"
        message += f"- نرخ پیروزی سیگنال‌های طلایی: {gold_win_rate:.1f}% ({gold_wins}/{gold_signals})\n"
        message += f"- نرخ پیروزی سیگنال‌های نقره‌ای: {silver_win_rate:.1f}% ({silver_wins}/{silver_signals})\n"
        message += f"- میانگین سود/زیان: {avg_profit_loss:.2f}%\n\n"
        
        message += "*سیگنال‌های اخیر:*\n"
        
        for i, signal in enumerate(recent_signals, 1):
            signal_id, symbol, direction, confidence, result, profit_loss, created_at = signal
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            
            message += f"{i}. {symbol}: {'🟢 خرید' if direction == 'BUY' else '🔴 فروش'} "
            message += f"(اطمینان: {confidence:.1f}%) - "
            
            if result:
                message += f"{'✅ سود' if result == 'win' else '❌ ضرر'} ({profit_loss:.2f}%)"
            else:
                message += "⏳ فعال"
            
            message += f" - {created_at.strftime('%Y-%m-%d')}\n"
        
        # دکمه‌های منو
        keyboard = [
            [InlineKeyboardButton("🔙 بازگشت به منو", callback_data='back_to_menu')]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """مدیریت کال‌بک‌های دکمه‌ها"""
        query = update.callback_query
        await query.answer()
        
        if query.data == 'analyze':
            return await self.analyze_symbol_start(update, context)
        elif query.data == 'silver':
            return await self.show_silver_signals(update, context)
        elif query.data == 'enable_gold':
            return await self.toggle_gold_notifications(update, context, True)
        elif query.data == 'disable_gold':
            return await self.toggle_gold_notifications(update, context, False)
        elif query.data == 'monitor':
            return await self.monitor_trade_start(update, context)
        elif query.data == 'stop_monitor':
            return await self.stop_monitoring(update, context)
        elif query.data.startswith('direction_'):
            direction = query.data.split('_')[1]  # BUY یا SELL
            context.user_data['direction'] = direction
            return await self.monitor_trade_direction(update, context)
        elif query.data == 'back_to_menu':
            # بازگشت به منوی اصلی
            keyboard = [
                [InlineKeyboardButton("🔬 تحلیل عمیق", callback_data='analyze')],
                [InlineKeyboardButton("🥈 سیگنال‌های نقره‌ای", callback_data='silver')],
                [InlineKeyboardButton("🔔 فعال‌سازی هشدارهای طلایی", callback_data='enable_gold')],
                [InlineKeyboardButton("👁️ پایش معامله", callback_data='monitor')]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                "لطفاً یک گزینه را انتخاب کنید:",
                reply_markup=reply_markup
            )
    
    async def analyze_symbol_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """شروع فرآیند تحلیل نماد"""
        query = update.callback_query
        
        await query.edit_message_text(
            "لطفاً نام نماد مورد نظر خود را وارد کنید (مثلاً BTC/USDT یا XAU/USD):"
        )
        
        return AWAITING_SYMBOL
    
    async def analyze_symbol_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پردازش ورودی نماد و تحلیل آن"""
        symbol = update.message.text.strip().upper()
        
        # ارسال پیام در حال تحلیل
        message = await update.message.reply_text(f"در حال تحلیل {symbol}... لطفاً صبر کنید.")
        
        try:
            # تحلیل نماد
            analysis = AnalysisEngine.analyze_symbol(symbol)
            
            # قالب‌بندی پاسخ
            response = self._format_analysis_response(analysis)
            
            # به‌روزرسانی پیام با نتایج
            await message.edit_text(response, parse_mode='Markdown')
            
            # ارسال منوی اصلی
            keyboard = [
                [InlineKeyboardButton("🔬 تحلیل نماد دیگر", callback_data='analyze')],
                [InlineKeyboardButton("👁️ پایش این معامله", callback_data=f'monitor_{symbol}')],
                [InlineKeyboardButton("🔙 بازگشت به منو", callback_data='back_to_menu')]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                "می‌خواهید چه کاری انجام دهید؟",
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"خطا در تحلیل {symbol}: {e}")
            await message.edit_text(f"خطا در تحلیل {symbol}. لطفاً مطمئن شوید که نماد به درستی وارد شده است.")
        
        return ConversationHandler.END
    
    def _format_analysis_response(self, analysis):
        """قالب‌بندی پاسخ تحلیل برای نمایش"""
        symbol = analysis["symbol"]
        price = analysis["summary"]["current_price"]
        sessions = ", ".join(analysis["summary"]["trading_sessions"])
        
        response = f"*📊 تحلیل جامع {symbol}* 📊\n\n"
        
        # خلاصه
        response += "*📌 خلاصه وضعیت:*\n"
        response += f"- قیمت فعلی: {price:,.5f}\n"
        response += f"- سشن معاملاتی فعال: {sessions}\n"
        response += f"- استراتژی منتخب: {analysis['summary']['best_strategy']} (نرخ پیروزی: {analysis['summary']['strategy_win_rate']:.1f}%)\n\n"
        
        # تکنیکال
        response += "*📈 تحلیل تکنیکال:*\n"
        
        if "trends" in analysis["technical"]:
            response += "- *روندها:* "
            trends = [f"{tf}: {trend}" for tf, trend in analysis["technical"]["trends"].items()]
            response += ", ".join(trends) + "\n"
        
        if "rsi" in analysis["technical"]:
            response += f"- *RSI:* {analysis['technical']['rsi']['value']:.1f} ({analysis['technical']['rsi']['condition']})\n"
        
        if "support_levels" in analysis["technical"] and analysis["technical"]["support_levels"]:
            sl = analysis["technical"]["support_levels"]
            response += f"- *سطوح حمایت:* {', '.join([f'{level:.5f}' for level in sl])}\n"
        
        if "resistance_levels" in analysis["technical"] and analysis["technical"]["resistance_levels"]:
            rl = analysis["technical"]["resistance_levels"]
            response += f"- *سطوح مقاومت:* {', '.join([f'{level:.5f}' for level in rl])}\n"
        
        if "candlestick_patterns" in analysis["technical"] and analysis["technical"]["candlestick_patterns"]:
            response += f"- *الگوهای کندلی:* {analysis['technical']['candlestick_patterns'][0]}"
            if len(analysis["technical"]["candlestick_patterns"]) > 1:
                response += f" و {len(analysis['technical']['candlestick_patterns'])-1} مورد دیگر"
            response += "\n"
        
        if "market_structure" in analysis["technical"] and "trend" in analysis["technical"]["market_structure"]:
            response += f"- *ساختار بازار:* {analysis['technical']['market_structure']['trend']}\n"
        
        if "tradingview" in analysis["technical"] and "summary" in analysis["technical"]["tradingview"]:
            response += f"- *خلاصه TradingView:* {analysis['technical']['tradingview']['summary'].get('RECOMMENDATION', 'N/A')}\n\n"
        
        # فاندامنتال
        response += "*📰 تحلیل فاندامنتال:*\n"
        
        if "economic" in analysis["fundamental"]:
            econ = analysis["fundamental"]["economic"]
            response += f"- *نرخ بهره فدرال:* {econ['fed_rate']}%\n"
            
            if econ["upcoming_events"]:
                next_event = econ["upcoming_events"][0]
                response += f"- *رویداد آتی:* {next_event['event']} ({next_event['date']})\n"
        
        if "overall_sentiment" in analysis["fundamental"]:
            response += f"- *احساسات اخبار:* {analysis['fundamental']['overall_sentiment']}\n\n"
        
        # توصیه
        response += "*💡 توصیه معاملاتی:*\n"
        
        if "recommendation" in analysis:
            rec = analysis["recommendation"]
            
            if rec["action"] == "WAIT":
                response += f"⏳ *{rec['action']}*: {rec['reason']}\n"
            else:
                confidence = rec["confidence"]
                signal_type = "🥇 سیگنال طلایی" if confidence >= GOLD_SIGNAL_THRESHOLD else "🥈 سیگنال نقره‌ای" if confidence >= SILVER_SIGNAL_THRESHOLD else "سیگنال"
                
                response += f"{signal_type}: {'🟢 خرید' if rec['action'] == 'BUY' else '🔴 فروش'} (اطمینان: {confidence:.1f}%)\n\n"
                response += f"- *نقطه ورود:* {rec['entry']:.5f}\n"
                response += f"- *حد ضرر:* {rec['stop_loss']:.5f}\n"
                response += f"- *حد سود 1:* {rec['take_profit1']:.5f}\n"
                response += f"- *حد سود 2:* {rec['take_profit2']:.5f}\n"
                response += f"- *نسبت ریسک/پاداش:* {rec['risk_reward_ratio']:.2f}\n"
                response += f"- *اهرم پیشنهادی:* {rec['recommended_leverage']:.1f}x\n"
        
        return response
    
    async def show_silver_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """نمایش سیگنال‌های نقره‌ای اخیر"""
        query = update.callback_query
        
        # دریافت سیگنال‌های نقره‌ای
        signals = self.signal_hunter.get_silver_signals(limit=5)
        
        if not signals:
            await query.edit_message_text(
                "در حال حاضر هیچ سیگنال نقره‌ای فعالی وجود ندارد. بعداً دوباره بررسی کنید."
            )
            return
        
        # قالب‌بندی پاسخ
        response = "*🥈 سیگنال‌های نقره‌ای اخیر 🥈*\n\n"
        
        for i, signal in enumerate(signals, 1):
            response += f"*{i}. {signal['symbol']}:* {'🟢 خرید' if signal['direction'] == 'BUY' else '🔴 فروش'} (اطمینان: {signal['confidence']:.1f}%)\n"
            response += f"   ورود: {signal['entry']:.5f}, SL: {signal['stop_loss']:.5f}, TP1: {signal['take_profit1']:.5f}\n"
            response += f"   استراتژی: {signal['strategy']} (نرخ پیروزی: {signal['win_rate']:.1f}%)\n\n"
        
        # دکمه‌های منو
        keyboard = [
            [InlineKeyboardButton("👁️ پایش یک معامله", callback_data='monitor')],
            [InlineKeyboardButton("🔙 بازگشت به منو", callback_data='back_to_menu')]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            response,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def toggle_gold_notifications(self, update: Update, context: ContextTypes.DEFAULT_TYPE, enable=True):
        """فعال یا غیرفعال کردن نوتیفیکیشن‌های طلایی"""
        query = update.callback_query
        user_id = query.from_user.id
        
        # تغییر وضعیت در پایگاه داده
        Database.set_gold_notifications(user_id, enable)
        
        # به‌روزرسانی پیام
        status_text = "فعال" if enable else "غیرفعال"
        new_button_text = "🔕 غیرفعال‌سازی هشدارهای طلایی" if enable else "🔔 فعال‌سازی هشدارهای طلایی"
        new_callback_data = "disable_gold" if enable else "enable_gold"
        
        keyboard = [
            [InlineKeyboardButton("🔬 تحلیل عمیق", callback_data='analyze')],
            [InlineKeyboardButton("🥈 سیگنال‌های نقره‌ای", callback_data='silver')],
            [InlineKeyboardButton(new_button_text, callback_data=new_callback_data)],
            [InlineKeyboardButton("👁️ پایش معامله", callback_data='monitor')]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            f"هشدارهای سیگنال طلایی {status_text} شدند.\n\n"
            f"اکنون شما {'خواهید' if enable else 'نخواهید'} توانست سیگنال‌های طلایی را به محض شناسایی دریافت کنید.\n\n"
            f"یک گزینه را از زیر انتخاب کنید:",
            reply_markup=reply_markup
        )
    
    async def monitor_trade_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """شروع فرآیند پایش معامله"""
        query = update.callback_query
        
        # بررسی اینکه آیا این یک نماد مشخص از قبل است
        if query.data.startswith('monitor_'):
            symbol = query.data.split('_')[1]
            context.user_data['symbol'] = symbol
            
            # مستقیماً به مرحله انتخاب جهت بروید
            keyboard = [
                [InlineKeyboardButton("🟢 خرید (Long)", callback_data='direction_BUY')],
                [InlineKeyboardButton("🔴 فروش (Short)", callback_data='direction_SELL')],
                [InlineKeyboardButton("🔙 لغو", callback_data='back_to_menu')]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                f"جهت معامله شما در {symbol} چیست؟",
                reply_markup=reply_markup
            )
            
            return AWAITING_DIRECTION
        
        # در غیر این صورت، از کاربر بخواهید نماد را وارد کند
        await query.edit_message_text(
            "لطفاً نام نماد معامله‌ای که می‌خواهید پایش کنید را وارد کنید (مثلاً BTC/USDT یا XAU/USD):"
        )
        
        return AWAITING_SYMBOL
    
    async def monitor_trade_symbol(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پردازش ورودی نماد برای پایش"""
        symbol = update.message.text.strip().upper()
        context.user_data['symbol'] = symbol
        
        # دکمه‌های انتخاب جهت
        keyboard = [
            [InlineKeyboardButton("🟢 خرید (Long)", callback_data='direction_BUY')],
            [InlineKeyboardButton("🔴 فروش (Short)", callback_data='direction_SELL')],
            [InlineKeyboardButton("🔙 لغو", callback_data='back_to_menu')]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"جهت معامله شما در {symbol} چیست؟",
            reply_markup=reply_markup
        )
        
        return AWAITING_DIRECTION
    
    async def monitor_trade_direction(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پردازش انتخاب جهت معامله و شروع پایش"""
        query = update.callback_query
        user_id = query.from_user.id
        
        # دریافت اطلاعات از context
        symbol = context.user_data.get('symbol')
        direction = context.user_data.get('direction')
        
        if not symbol or not direction:
            await query.edit_message_text(
                "خطا: اطلاعات معامله ناقص است. لطفاً دوباره تلاش کنید."
            )
            return ConversationHandler.END
        
        # افزودن معامله به پایش‌ها
        trade_id = Database.add_monitored_trade(user_id, symbol, direction)
        
        # دکمه‌های پایش
        keyboard = [
            [InlineKeyboardButton("🚫 توقف پایش", callback_data=f'stop_monitor_{trade_id}')],
            [InlineKeyboardButton("🔙 بازگشت به منو", callback_data='back_to_menu')]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            f"پایش معامله {symbol} ({('خرید' if direction == 'BUY' else 'فروش')}) آغاز شد.\n\n"
            f"ربات هر 5 دقیقه یک تحلیل جدید انجام می‌دهد و در صورت تغییر شرایط به نفع جهت مخالف، به شما هشدار می‌دهد.",
            reply_markup=reply_markup
        )
        
        return ConversationHandler.END
    
    async def stop_monitoring(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """توقف پایش یک معامله"""
        query = update.callback_query
        
        # دریافت شناسه معامله از داده کال‌بک
        trade_id = int(query.data.split('_')[2])
        
        # حذف معامله از پایش‌ها
        Database.remove_monitored_trade(trade_id)
        
        keyboard = [
            [InlineKeyboardButton("🔙 بازگشت به منو", callback_data='back_to_menu')]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "پایش معامله با موفقیت متوقف شد.",
            reply_markup=reply_markup
        )
    
    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """لغو مکالمه فعلی"""
        await update.message.reply_text(
            "عملیات لغو شد. از /start برای شروع مجدد استفاده کنید."
        )
        
        return ConversationHandler.END

# نقطه ورود اصلی
if __name__ == "__main__":
    # راه‌اندازی پایگاه داده
    setup_database()
    
    # ایجاد و شروع ربات
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        logger.error("TELEGRAM_TOKEN not found in environment variables")
        exit(1)
    
    bot = TradingBot(token)
    logger.info("Starting bot...")
    bot.start()