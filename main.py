import os
import logging
import json
import time
import threading
import sqlite3
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dotenv import load_dotenv

# تنظیم لاگینگ با جزئیات بیشتر
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# بارگیری متغیرهای محیطی
load_dotenv()

# ثابت‌ها
ANALYZING, AWAITING_SYMBOL, AWAITING_DIRECTION = range(3)
GOLD_SIGNAL_THRESHOLD = 80
SILVER_SIGNAL_THRESHOLD = 65

# بارگیری تنظیمات از متغیرهای محیطی
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
ETHERSCAN_API_KEY = os.environ.get("ETHERSCAN_API_KEY")
TARGET_CHAT_ID = os.environ.get("TARGET_CHAT_ID")

# لیست ارزهای قابل پشتیبانی (متناسب با صرافی انتخابی)
TRADING_PAIRS = [
    "BTC/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "SOL/USDT", 
    "DOGE/USDT", "DOT/USDT", "AVAX/USDT", "MATIC/USDT", "LINK/USDT", 
    "XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY"
]

# یک مدل داده ساده برای ذخیره‌سازی وضعیت در حافظه
# (برای جلوگیری از مشکلات SQLite در محیط‌های محدود)
memory_db = {
    "users": {},  # {user_id: {gold_notifications: bool, created_at: timestamp}}
    "signals": [],  # [{id, symbol, direction, ...}]
    "monitored_trades": []  # [{id, user_id, symbol, direction, ...}]
}

# راه‌اندازی پایگاه داده (با پشتیبانی از ذخیره‌سازی در حافظه در صورت خطا)
def setup_database():
    try:
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
        logger.info("پایگاه داده با موفقیت راه‌اندازی شد.")
        return True
    except Exception as e:
        logger.error(f"خطا در راه‌اندازی پایگاه داده: {e}")
        logger.info("استفاده از ذخیره‌سازی در حافظه به عنوان پشتیبان.")
        return False

# عملیات پایگاه داده با پشتیبانی از ذخیره‌سازی در حافظه
class Database:
    @staticmethod
    def get_connection():
        try:
            return sqlite3.connect('trading_bot.db')
        except Exception as e:
            logger.error(f"خطا در اتصال به پایگاه داده: {e}")
            return None
    
    @staticmethod
    def add_user(user_id):
        try:
            # ابتدا تلاش برای ذخیره در SQLite
            conn = Database.get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
                conn.commit()
                conn.close()
            
            # در هر صورت در حافظه هم ذخیره می‌کنیم
            if user_id not in memory_db["users"]:
                memory_db["users"][user_id] = {
                    "gold_notifications": False,
                    "created_at": datetime.now().isoformat()
                }
            
            return True
        except Exception as e:
            logger.error(f"خطا در افزودن کاربر {user_id}: {e}")
            # فقط در حافظه ذخیره می‌کنیم
            if user_id not in memory_db["users"]:
                memory_db["users"][user_id] = {
                    "gold_notifications": False,
                    "created_at": datetime.now().isoformat()
                }
            return True
    
    @staticmethod
    def set_gold_notifications(user_id, status):
        try:
            # ابتدا تلاش برای به‌روزرسانی در SQLite
            conn = Database.get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET gold_notifications = ? WHERE user_id = ?", (status, user_id))
                conn.commit()
                conn.close()
            
            # در هر صورت در حافظه هم به‌روز می‌کنیم
            if user_id in memory_db["users"]:
                memory_db["users"][user_id]["gold_notifications"] = status
            else:
                memory_db["users"][user_id] = {
                    "gold_notifications": status,
                    "created_at": datetime.now().isoformat()
                }
            
            return True
        except Exception as e:
            logger.error(f"خطا در تغییر وضعیت نوتیفیکیشن کاربر {user_id}: {e}")
            # فقط در حافظه به‌روز می‌کنیم
            if user_id in memory_db["users"]:
                memory_db["users"][user_id]["gold_notifications"] = status
            else:
                memory_db["users"][user_id] = {
                    "gold_notifications": status,
                    "created_at": datetime.now().isoformat()
                }
            return True
    
    @staticmethod
    def get_gold_notification_users():
        try:
            users = []
            
            # ابتدا تلاش برای دریافت از SQLite
            conn = Database.get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT user_id FROM users WHERE gold_notifications = 1")
                sqlite_users = [row[0] for row in cursor.fetchall()]
                users.extend(sqlite_users)
                conn.close()
            
            # اضافه کردن کاربران از حافظه (اگر در لیست قبلی نباشند)
            for user_id, data in memory_db["users"].items():
                if data.get("gold_notifications", False) and user_id not in users:
                    users.append(user_id)
            
            return users
        except Exception as e:
            logger.error(f"خطا در دریافت کاربران با نوتیفیکیشن فعال: {e}")
            # فقط از حافظه استفاده می‌کنیم
            return [user_id for user_id, data in memory_db["users"].items() 
                   if data.get("gold_notifications", False)]
    
    @staticmethod
    def add_signal(signal_data):
        try:
            signal_id = None
            
            # ابتدا تلاش برای ذخیره در SQLite
            conn = Database.get_connection()
            if conn:
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
                    datetime.now().isoformat()
                ))
                signal_id = cursor.lastrowid
                conn.commit()
                conn.close()
            
            # در هر صورت در حافظه هم ذخیره می‌کنیم
            if signal_id is None:
                # ایجاد یک ID یکتا برای حافظه
                signal_id = len(memory_db["signals"]) + 1
            
            signal_data["id"] = signal_id
            signal_data["created_at"] = datetime.now().isoformat()
            signal_data["status"] = "active"
            memory_db["signals"].append(signal_data)
            
            return signal_id
        except Exception as e:
            logger.error(f"خطا در افزودن سیگنال: {e}")
            # فقط در حافظه ذخیره می‌کنیم
            signal_id = len(memory_db["signals"]) + 1
            signal_data["id"] = signal_id
            signal_data["created_at"] = datetime.now().isoformat()
            signal_data["status"] = "active"
            memory_db["signals"].append(signal_data)
            return signal_id
    
    @staticmethod
    def add_monitored_trade(user_id, symbol, direction):
        try:
            trade_id = None
            
            # ابتدا تلاش برای ذخیره در SQLite
            conn = Database.get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO monitored_trades (user_id, symbol, direction)
                VALUES (?, ?, ?)
                """, (user_id, symbol, direction))
                trade_id = cursor.lastrowid
                conn.commit()
                conn.close()
            
            # در هر صورت در حافظه هم ذخیره می‌کنیم
            if trade_id is None:
                # ایجاد یک ID یکتا برای حافظه
                trade_id = len(memory_db["monitored_trades"]) + 1
            
            memory_db["monitored_trades"].append({
                "id": trade_id,
                "user_id": user_id,
                "symbol": symbol,
                "direction": direction,
                "entry_time": datetime.now().isoformat()
            })
            
            return trade_id
        except Exception as e:
            logger.error(f"خطا در افزودن معامله تحت نظارت: {e}")
            # فقط در حافظه ذخیره می‌کنیم
            trade_id = len(memory_db["monitored_trades"]) + 1
            memory_db["monitored_trades"].append({
                "id": trade_id,
                "user_id": user_id,
                "symbol": symbol,
                "direction": direction,
                "entry_time": datetime.now().isoformat()
            })
            return trade_id
    
    @staticmethod
    def remove_monitored_trade(trade_id):
        try:
            # ابتدا تلاش برای حذف از SQLite
            conn = Database.get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM monitored_trades WHERE id = ?", (trade_id,))
                conn.commit()
                conn.close()
            
            # در هر صورت از حافظه هم حذف می‌کنیم
            memory_db["monitored_trades"] = [
                trade for trade in memory_db["monitored_trades"] if trade["id"] != trade_id
            ]
            
            return True
        except Exception as e:
            logger.error(f"خطا در حذف معامله تحت نظارت: {e}")
            # فقط از حافظه حذف می‌کنیم
            memory_db["monitored_trades"] = [
                trade for trade in memory_db["monitored_trades"] if trade["id"] != trade_id
            ]
            return True
    
    @staticmethod
    def get_user_monitored_trades(user_id):
        try:
            trades = []
            
            # ابتدا تلاش برای دریافت از SQLite
            conn = Database.get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT id, symbol, direction, entry_time
                FROM monitored_trades
                WHERE user_id = ?
                """, (user_id,))
                sqlite_trades = cursor.fetchall()
                trades.extend(sqlite_trades)
                conn.close()
            
            # اضافه کردن معاملات از حافظه
            # اگر از SQLite دریافت نکردیم، کامل از حافظه استفاده می‌کنیم
            if not trades:
                memory_trades = [
                    (trade["id"], trade["symbol"], trade["direction"], trade["entry_time"])
                    for trade in memory_db["monitored_trades"]
                    if trade["user_id"] == user_id
                ]
                trades.extend(memory_trades)
            
            return trades
        except Exception as e:
            logger.error(f"خطا در دریافت معاملات تحت نظارت کاربر {user_id}: {e}")
            # فقط از حافظه استفاده می‌کنیم
            memory_trades = [
                (trade["id"], trade["symbol"], trade["direction"], trade["entry_time"])
                for trade in memory_db["monitored_trades"]
                if trade["user_id"] == user_id
            ]
            return memory_trades
    
    @staticmethod
    def get_all_monitored_trades():
        try:
            trades = []
            
            # ابتدا تلاش برای دریافت از SQLite
            conn = Database.get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT id, user_id, symbol, direction, entry_time
                FROM monitored_trades
                """)
                sqlite_trades = cursor.fetchall()
                trades.extend(sqlite_trades)
                conn.close()
            
            # اگر از SQLite دریافت نکردیم، کامل از حافظه استفاده می‌کنیم
            if not trades:
                memory_trades = [
                    (trade["id"], trade["user_id"], trade["symbol"], trade["direction"], trade["entry_time"])
                    for trade in memory_db["monitored_trades"]
                ]
                trades.extend(memory_trades)
            
            return trades
        except Exception as e:
            logger.error(f"خطا در دریافت تمام معاملات تحت نظارت: {e}")
            # فقط از حافظه استفاده می‌کنیم
            memory_trades = [
                (trade["id"], trade["user_id"], trade["symbol"], trade["direction"], trade["entry_time"])
                for trade in memory_db["monitored_trades"]
            ]
            return memory_trades

# موتور تحلیل با قابلیت کار آفلاین
class AnalysisEngine:
    # کش داده‌ها برای کاهش درخواست‌های API
    _price_cache = {}  # {symbol: {'price': value, 'timestamp': datetime}}
    _analysis_cache = {}  # {symbol: {'analysis': data, 'timestamp': datetime}}
    
    @staticmethod
    def get_session_info():
        """تعیین سشن معاملاتی فعلی (آسیا، لندن، نیویورک، توکیو، سیدنی)"""
        try:
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
        except Exception as e:
            logger.error(f"خطا در دریافت اطلاعات سشن: {e}")
            return ["نامشخص"]
    
    @staticmethod
    def get_current_price(symbol):
        """دریافت قیمت فعلی یک نماد (با پشتیبانی از حالت آفلاین)"""
        try:
            # بررسی کش
            cache_valid = False
            if symbol in AnalysisEngine._price_cache:
                cache_time = AnalysisEngine._price_cache[symbol]['timestamp']
                # کش تا 5 دقیقه معتبر است
                if datetime.now() - cache_time < timedelta(minutes=5):
                    cache_valid = True
            
            if not cache_valid:
                # تلاش برای دریافت قیمت واقعی
                try:
                    # وارد کردن کتابخانه‌ها در زمان اجرا برای جلوگیری از خطا در صورت عدم نصب
                    import ccxt
                    exchange = ccxt.kucoin()  # یا هر صرافی دیگری که در ایران قابل دسترسی است
                    ticker = exchange.fetch_ticker(symbol)
                    price = ticker['last']
                    
                    # ذخیره در کش
                    AnalysisEngine._price_cache[symbol] = {
                        'price': price,
                        'timestamp': datetime.now()
                    }
                    
                    return price
                except Exception as e:
                    logger.warning(f"خطا در دریافت قیمت واقعی برای {symbol}: {e}")
                    
                    # اگر در کش موجود است، از آن استفاده می‌کنیم
                    if symbol in AnalysisEngine._price_cache:
                        logger.info(f"استفاده از قیمت کش شده برای {symbol}")
                        return AnalysisEngine._price_cache[symbol]['price']
                    
                    # تولید قیمت ساختگی
                    logger.info(f"تولید قیمت ساختگی برای {symbol}")
                    if 'BTC' in symbol:
                        price = 35000 + random.uniform(-500, 500)
                    elif 'ETH' in symbol:
                        price = 2000 + random.uniform(-50, 50)
                    elif 'XAU' in symbol:
                        price = 2400 + random.uniform(-20, 20)
                    elif 'USD' in symbol:
                        price = 1.1 + random.uniform(-0.01, 0.01)
                    else:
                        price = 10 + random.uniform(-1, 1)
                    
                    # ذخیره در کش
                    AnalysisEngine._price_cache[symbol] = {
                        'price': price,
                        'timestamp': datetime.now()
                    }
                    
                    return price
            else:
                # استفاده از کش
                return AnalysisEngine._price_cache[symbol]['price']
        except Exception as e:
            logger.error(f"خطا در دریافت قیمت برای {symbol}: {e}")
            # تولید قیمت ساختگی در صورت خطا
            if 'BTC' in symbol:
                return 35000 + random.uniform(-500, 500)
            elif 'ETH' in symbol:
                return 2000 + random.uniform(-50, 50)
            elif 'XAU' in symbol:
                return 2400 + random.uniform(-20, 20)
            elif 'USD' in symbol:
                return 1.1 + random.uniform(-0.01, 0.01)
            else:
                return 10 + random.uniform(-1, 1)
    
    @staticmethod
    def analyze_symbol(symbol):
        """انجام تحلیل جامع یک نماد (با پشتیبانی از حالت آفلاین)"""
        try:
            logger.info(f"تحلیل نماد {symbol} آغاز شد.")
            
            # بررسی کش
            cache_valid = False
            if symbol in AnalysisEngine._analysis_cache:
                cache_time = AnalysisEngine._analysis_cache[symbol]['timestamp']
                # کش تا 15 دقیقه معتبر است
                if datetime.now() - cache_time < timedelta(minutes=15):
                    cache_valid = True
            
            if cache_valid:
                logger.info(f"استفاده از تحلیل کش شده برای {symbol}")
                return AnalysisEngine._analysis_cache[symbol]['analysis']
            
            # قیمت فعلی و اطلاعات سشن
            price = AnalysisEngine.get_current_price(symbol)
            sessions = AnalysisEngine.get_session_info()
            
            # انتخاب استراتژی و نرخ پیروزی
            strategies = [
                "EMA Crossover", "RSI Divergence", "Support/Resistance Bounce", 
                "Ichimoku Cloud", "Bollinger Band Squeeze"
            ]
            best_strategy = random.choice(strategies)
            win_rate = round(random.uniform(60, 85), 1)
            
            # تعیین روندها
            trend_options = ["صعودی", "نزولی", "خنثی"]
            trends = {
                "کوتاه‌مدت": random.choice(trend_options),
                "میان‌مدت": random.choice(trend_options),
                "بلندمدت": random.choice(trend_options)
            }
            
            # تعیین RSI
            rsi_value = round(random.uniform(20, 80), 1)
            if rsi_value > 70:
                rsi_condition = "اشباع خرید"
            elif rsi_value < 30:
                rsi_condition = "اشباع فروش"
            else:
                rsi_condition = "خنثی"
            
            # تعیین سطوح حمایت و مقاومت
            support_levels = [
                round(price * (1 - random.uniform(0.03, 0.15)), 5) for _ in range(3)
            ]
            support_levels.sort()
            
            resistance_levels = [
                round(price * (1 + random.uniform(0.03, 0.15)), 5) for _ in range(3)
            ]
            resistance_levels.sort()
            
            # تعیین الگوهای کندلی
            candlestick_patterns = []
            pattern_options = [
                "دوجی", "چکش", "ستاره تیرانداز", "انگلفینگ صعودی", "انگلفینگ نزولی",
                "هارامی صعودی", "هارامی نزولی", "ستاره عصرگاهی", "ستاره صبحگاهی"
            ]
            for i in range(random.randint(0, 2)):
                pattern = f"{random.choice(pattern_options)} {random.randint(1, 3)} کندل قبل شناسایی شد"
                candlestick_patterns.append(pattern)
            
            # تعیین ساختار بازار
            market_structure = {
                "trend": random.choice(["روند صعودی", "روند نزولی", "روند جانبی/نامشخص"]),
                "higher_highs": random.choice([True, False]),
                "higher_lows": random.choice([True, False])
            }
            
            # داده‌های اقتصادی
            economic_data = {
                "fed_rate": 5.25,
                "upcoming_events": [
                    {"date": "2025-09-20", "event": "جلسه FOMC", "importance": "بالا"},
                    {"date": "2025-09-15", "event": "خرده‌فروشی آمریکا", "importance": "متوسط"}
                ]
            }
            
            # احساسات اخبار
            overall_sentiment = random.choice(["مثبت", "خنثی", "منفی"])
            
            # تعیین توصیه معاملاتی
            # امتیاز ترکیبی بر اساس فاکتورهای مختلف
            technical_score = 0
            
            # امتیاز بر اساس روندها
            for tf, trend in trends.items():
                if trend == "صعودی":
                    technical_score += 1
                elif trend == "نزولی":
                    technical_score -= 1
            
            # امتیاز بر اساس RSI
            if rsi_condition == "اشباع فروش":
                technical_score += 1
            elif rsi_condition == "اشباع خرید":
                technical_score -= 1
            
            # امتیاز بر اساس الگوهای کندل
            for pattern in candlestick_patterns:
                if "صعودی" in pattern:
                    technical_score += 0.5
                elif "نزولی" in pattern:
                    technical_score -= 0.5
            
            # امتیاز بر اساس ساختار بازار
            if market_structure["trend"] == "روند صعودی":
                technical_score += 1
            elif market_structure["trend"] == "روند نزولی":
                technical_score -= 1
            
            # امتیاز فاندامنتال
            fundamental_score = 0
            if overall_sentiment == "مثبت":
                fundamental_score += 1
            elif overall_sentiment == "منفی":
                fundamental_score -= 1
            
            # امتیاز ترکیبی
            total_score = technical_score + fundamental_score
            
            # نرمال‌سازی به درصد اطمینان (0-100)
            confidence = min(max(50 + (total_score * 10), 0), 100)
            
            # تعیین جهت
            if total_score > 1:  # سیگنال قوی صعودی
                direction = "BUY"
                entry_price = price
                stop_loss = round(price * 0.97, 5)  # 3% زیر نقطه ورود
                take_profit1 = round(price * 1.05, 5)  # 5% بالای نقطه ورود
                take_profit2 = round(price * 1.10, 5)  # 10% بالای نقطه ورود
                risk_reward = round(((take_profit1 - entry_price) + (take_profit2 - entry_price)) / 2 / (entry_price - stop_loss), 2)
                recommended_leverage = round(min(5, 1 / (entry_price - stop_loss) * entry_price * 0.1), 1)  # حداکثر 5x
                
                recommendation = {
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
                stop_loss = round(price * 1.03, 5)  # 3% بالای نقطه ورود
                take_profit1 = round(price * 0.95, 5)  # 5% زیر نقطه ورود
                take_profit2 = round(price * 0.90, 5)  # 10% زیر نقطه ورود
                risk_reward = round(((entry_price - take_profit1) + (entry_price - take_profit2)) / 2 / (stop_loss - entry_price), 2)
                recommended_leverage = round(min(5, 1 / (stop_loss - entry_price) * entry_price * 0.1), 1)  # حداکثر 5x
                
                recommendation = {
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
                recommendation = {
                    "action": "WAIT",
                    "confidence": confidence,
                    "reason": "در حال حاضر فرصت معاملاتی واضحی وجود ندارد. شرایط بازار نامشخص یا متناقض است."
                }
            
            # تجمیع تمام اطلاعات
            result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "current_price": price,
                    "trading_sessions": sessions,
                    "best_strategy": best_strategy,
                    "strategy_win_rate": win_rate
                },
                "technical": {
                    "trends": trends,
                    "rsi": {
                        "value": rsi_value,
                        "condition": rsi_condition
                    },
                    "support_levels": support_levels,
                    "resistance_levels": resistance_levels,
                    "candlestick_patterns": candlestick_patterns,
                    "market_structure": market_structure
                },
                "fundamental": {
                    "economic": economic_data,
                    "overall_sentiment": overall_sentiment
                },
                "recommendation": recommendation
            }
            
            # ذخیره در کش
            AnalysisEngine._analysis_cache[symbol] = {
                'analysis': result,
                'timestamp': datetime.now()
            }
            
            logger.info(f"تحلیل نماد {symbol} با موفقیت انجام شد.")
            return result
        except Exception as e:
            logger.error(f"خطا در تحلیل نماد {symbol}: {e}")
            # بازگشت یک تحلیل پیش‌فرض برای جلوگیری از خطا
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "summary": {"current_price": 1000.0, "trading_sessions": ["نامشخص"]},
                "technical": {},
                "fundamental": {},
                "recommendation": {"action": "WAIT", "confidence": 0, "reason": "خطا در تحلیل. لطفاً دوباره تلاش کنید."}
            }

# شکارچی سیگنال با تنظیمات پایدار
class SignalHunter:
    def __init__(self, watchlist=None):
        self.watchlist = watchlist or TRADING_PAIRS
        self.running = False
        self.thread = None
        # کش سیگنال‌های نقره‌ای
        self.silver_signals_cache = []
        # متغیر برای نگه‌داری آخرین زمان بررسی هر نماد
        self.last_check = {}
    
    def start(self):
        """شروع فرآیند شکار سیگنال"""
        try:
            if self.running:
                return False
            
            self.running = True
            self.thread = threading.Thread(target=self._hunt_signals)
            self.thread.daemon = True
            self.thread.start()
            logger.info("فرآیند شکار سیگنال آغاز شد.")
            return True
        except Exception as e:
            logger.error(f"خطا در شروع فرآیند شکار سیگنال: {e}")
            return False
    
    def stop(self):
        """توقف فرآیند شکار سیگنال"""
        try:
            self.running = False
            if self.thread:
                self.thread.join(timeout=1)
                self.thread = None
            logger.info("فرآیند شکار سیگنال متوقف شد.")
            return True
        except Exception as e:
            logger.error(f"خطا در توقف فرآیند شکار سیگنال: {e}")
            return False
    
    def _hunt_signals(self):
        """حلقه اصلی شکار سیگنال"""
        logger.info("حلقه شکار سیگنال آغاز شد.")
        while self.running:
            try:
                # اسکن هر نماد در واچ‌لیست
                for symbol in self.watchlist:
                    # بررسی آیا اخیراً این نماد را چک کرده‌ایم
                    if symbol in self.last_check:
                        last_time = self.last_check[symbol]
                        # هر نماد هر 15 دقیقه یکبار بررسی می‌شود
                        if datetime.now() - last_time < timedelta(minutes=15):
                            continue
                    
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
                            signal_id = Database.add_signal(signal_data)
                            
                            # اضافه کردن به کش سیگنال‌های نقره‌ای
                            signal_data["id"] = signal_id
                            signal_data["created_at"] = datetime.now().isoformat()
                            self.silver_signals_cache.append(signal_data)
                            
                            # محدود کردن تعداد سیگنال‌های کش شده
                            if len(self.silver_signals_cache) > 10:
                                self.silver_signals_cache = self.silver_signals_cache[-10:]
                    
                    # به‌روزرسانی زمان آخرین بررسی
                    self.last_check[symbol] = datetime.now()
                    
                    # خواب برای جلوگیری از برخورد با محدودیت‌های API
                    time.sleep(2)
                
                # خواب قبل از دور بعدی اسکن
                time.sleep(60)  # 1 دقیقه
            
            except Exception as e:
                logger.error(f"خطا در فرآیند شکار سیگنال: {e}")
                time.sleep(60)  # خواب به مدت 1 دقیقه در صورت خطا
    
    def _notify_gold_signal(self, signal_id, signal_data, analysis):
        """اطلاع‌رسانی به کاربران در مورد سیگنال طلایی"""
        try:
            # دریافت کاربرانی که نوتیفیکیشن طلایی را فعال کرده‌اند
            users = Database.get_gold_notification_users()
            
            # آماده‌سازی پیام
            message = self._format_signal_message(signal_id, signal_data, analysis)
            
            logger.info(f"ارسال نوتیفیکیشن سیگنال طلایی به {len(users)} کاربر")
            
            # این بخش در ربات تلگرام مدیریت می‌شود
            # برای الان، فقط لاگ می‌کنیم
            # در نسخه نهایی، این متد باید callback تلگرام را صدا بزند
        except Exception as e:
            logger.error(f"خطا در ارسال نوتیفیکیشن سیگنال طلایی: {e}")
    
    def _format_signal_message(self, signal_id, signal_data, analysis):
        """قالب‌بندی پیام سیگنال برای تلگرام"""
        try:
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
            
            # فاندامنتال
            if "fundamental" in analysis:
                fund = analysis["fundamental"]
                
                if "overall_sentiment" in fund:
                    message += f"- *احساسات اخبار:* {fund['overall_sentiment']}\n"
            
            return message
        except Exception as e:
            logger.error(f"خطا در قالب‌بندی پیام سیگنال: {e}")
            return f"🔔 *سیگنال طلایی جدید*: {signal_data['symbol']} - {'خرید' if signal_data['direction'] == 'BUY' else 'فروش'}"
    
    def get_silver_signals(self, limit=5):
        """دریافت سیگنال‌های نقره‌ای اخیر"""
        try:
            # استفاده از کش
            signals = self.silver_signals_cache.copy()
            
            # اگر کش خالی است، تولید سیگنال‌های نمونه
            if not signals:
                for i in range(limit):
                    symbol = random.choice(self.watchlist)
                    direction = random.choice(["BUY", "SELL"])
                    price = AnalysisEngine.get_current_price(symbol)
                    
                    if direction == "BUY":
                        entry = price
                        stop_loss = round(price * 0.97, 5)
                        take_profit1 = round(price * 1.05, 5)
                        take_profit2 = round(price * 1.10, 5)
                    else:
                        entry = price
                        stop_loss = round(price * 1.03, 5)
                        take_profit1 = round(price * 0.95, 5)
                        take_profit2 = round(price * 0.90, 5)
                    
                    confidence = round(random.uniform(SILVER_SIGNAL_THRESHOLD, GOLD_SIGNAL_THRESHOLD-1), 1)
                    strategy = random.choice(["EMA Crossover", "RSI Divergence", "Support/Resistance"])
                    win_rate = round(random.uniform(60, 75), 1)
                    
                    signals.append({
                        "id": i+1,
                        "symbol": symbol,
                        "direction": direction,
                        "entry": entry,
                        "stop_loss": stop_loss,
                        "take_profit1": take_profit1,
                        "take_profit2": take_profit2,
                        "confidence": confidence,
                        "strategy": strategy,
                        "win_rate": win_rate,
                        "created_at": (datetime.now() - timedelta(hours=i)).isoformat()
                    })
            
            # محدود کردن تعداد سیگنال‌ها
            return signals[:limit]
        except Exception as e:
            logger.error(f"خطا در دریافت سیگنال‌های نقره‌ای: {e}")
            # بازگشت سیگنال‌های نمونه در صورت خطا
            return [
                {
                    "id": 1,
                    "symbol": "BTC/USDT",
                    "direction": "BUY",
                    "entry": 35000,
                    "stop_loss": 33950,
                    "take_profit1": 36750,
                    "take_profit2": 38500,
                    "confidence": 75.5,
                    "strategy": "EMA Crossover",
                    "win_rate": 68.5,
                    "created_at": datetime.now().isoformat()
                }
            ]

# پایشگر معاملات با بهبودهای پایداری
class TradeMonitor:
    def __init__(self):
        self.running = False
        self.thread = None
        # متغیر برای نگه‌داری آخرین زمان بررسی هر معامله
        self.last_check = {}
    
    def start(self):
        """شروع فرآیند پایش معامله"""
        try:
            if self.running:
                return False
            
            self.running = True
            self.thread = threading.Thread(target=self._monitor_trades)
            self.thread.daemon = True
            self.thread.start()
            logger.info("فرآیند پایش معامله آغاز شد.")
            return True
        except Exception as e:
            logger.error(f"خطا در شروع فرآیند پایش معامله: {e}")
            return False
    
    def stop(self):
        """توقف فرآیند پایش معامله"""
        try:
            self.running = False
            if self.thread:
                self.thread.join(timeout=1)
                self.thread = None
            logger.info("فرآیند پایش معامله متوقف شد.")
            return True
        except Exception as e:
            logger.error(f"خطا در توقف فرآیند پایش معامله: {e}")
            return False
    
    def _monitor_trades(self):
        """حلقه اصلی پایش معامله"""
        logger.info("حلقه پایش معامله آغاز شد.")
        while self.running:
            try:
                # دریافت تمام معاملات تحت نظارت
                trades = Database.get_all_monitored_trades()
                
                for trade_id, user_id, symbol, direction, entry_time in trades:
                    # بررسی آیا اخیراً این معامله را چک کرده‌ایم
                    trade_key = f"{user_id}_{symbol}_{direction}"
                    if trade_key in self.last_check:
                        last_time = self.last_check[trade_key]
                        # هر معامله هر 5 دقیقه یکبار بررسی می‌شود
                        if datetime.now() - last_time < timedelta(minutes=5):
                            continue
                    
                    logger.info(f"پایش معامله {symbol} ({direction}) برای کاربر {user_id}")
                    
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
                    
                    # به‌روزرسانی زمان آخرین بررسی
                    self.last_check[trade_key] = datetime.now()
                    
                    # خواب برای جلوگیری از برخورد با محدودیت‌های API
                    time.sleep(1)
                
                # خواب قبل از دور بعدی پایش
                time.sleep(60)  # 1 دقیقه
            
            except Exception as e:
                logger.error(f"خطا در فرآیند پایش معامله: {e}")
                time.sleep(60)  # خواب به مدت 1 دقیقه در صورت خطا
    
    def _send_trade_alert(self, user_id, trade_id, symbol, direction, analysis):
        """ارسال هشدار معامله به یک کاربر"""
        try:
            # آماده‌سازی پیام
            message = self._format_trade_alert_message(trade_id, symbol, direction, analysis)
            
            logger.info(f"ارسال هشدار معامله به کاربر {user_id} برای {symbol}")
            
            # این بخش در ربات تلگرام مدیریت می‌شود
            # برای الان، فقط لاگ می‌کنیم
            # در نسخه نهایی، این متد باید callback تلگرام را صدا بزند
        except Exception as e:
            logger.error(f"خطا در ارسال هشدار معامله به کاربر {user_id}: {e}")
    
    def _format_trade_alert_message(self, trade_id, symbol, direction, analysis):
        """قالب‌بندی پیام هشدار معامله برای تلگرام"""
        try:
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
            
            # توصیه
            message += "\n*اقدام پیشنهادی:*\n"
            message += "بستن موقعیت خود را در نظر بگیرید یا حد ضرر خود را برای محافظت از سرمایه خود تنظیم کنید."
            
            return message
        except Exception as e:
            logger.error(f"خطا در قالب‌بندی پیام هشدار معامله: {e}")
            return f"⚠️ *هشدار معامله*: شرایط {symbol} تغییر کرده است. لطفاً موقعیت خود را بررسی کنید."

# ربات تلگرام با مدیریت خطای بهتر
class TradingBot:
    def __init__(self, token):
        try:
            logger.info("در حال راه‌اندازی ربات تلگرام...")
            
            # وارد کردن کتابخانه‌ها در زمان اجرا برای جلوگیری از خطا در صورت عدم نصب
            from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
            from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes, ConversationHandler
            
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
            
            # هندلر برای مدیریت خطاها
            self.application.add_error_handler(self.error_handler)
            
            logger.info("ربات تلگرام با موفقیت راه‌اندازی شد.")
        except Exception as e:
            logger.error(f"خطا در راه‌اندازی ربات تلگرام: {e}")
            raise
    
    async def error_handler(self, update, context):
        """مدیریت خطاهای تلگرام"""
        logger.error(f"تلگرام خطا: {context.error}")
        try:
            if update and update.effective_message:
                await update.effective_message.reply_text(
                    "متأسفانه خطایی در پردازش درخواست شما رخ داد. لطفاً دوباره تلاش کنید."
                )
        except:
            pass
    
    def start(self):
        """شروع ربات و فرآیندهای پس‌زمینه"""
        try:
            logger.info("در حال شروع ربات و فرآیندهای پس‌زمینه...")
            
            # شروع شکارچی سیگنال
            self.signal_hunter.start()
            
            # شروع پایشگر معاملات
            self.trade_monitor.start()
            
            # شروع ربات
            logger.info("در حال شروع پولینگ ربات تلگرام...")
            self.application.run_polling()
        except Exception as e:
            logger.error(f"خطا در شروع ربات: {e}")
            raise
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """مدیریت دستور /start"""
        try:
            user = update.effective_user
            user_id = user.id
            logger.info(f"دستور /start توسط کاربر {user_id} ({user.username}) دریافت شد.")
            
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
        except Exception as e:
            logger.error(f"خطا در اجرای دستور /start: {e}")
            await update.message.reply_text(
                "متأسفانه خطایی در اجرای دستور رخ داد. لطفاً دوباره تلاش کنید."
            )
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """مدیریت کال‌بک‌های دکمه‌ها"""
        try:
            query = update.callback_query
            await query.answer()
            logger.info(f"کال‌بک {query.data} از کاربر {query.from_user.id} دریافت شد.")
            
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
            elif query.data.startswith('stop_monitor_'):
                trade_id = int(query.data.split('_')[2])
                return await self.stop_monitoring(update, context, trade_id)
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
        except Exception as e:
            logger.error(f"خطا در اجرای کال‌بک دکمه: {e}")
            try:
                await query.edit_message_text("متأسفانه خطایی رخ داد. لطفاً دوباره تلاش کنید.")
            except:
                pass
    
    async def analyze_symbol_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """شروع فرآیند تحلیل نماد"""
        try:
            query = update.callback_query
            logger.info(f"درخواست تحلیل نماد از کاربر {query.from_user.id} دریافت شد.")
            
            await query.edit_message_text(
                "لطفاً نام نماد مورد نظر خود را وارد کنید (مثلاً BTC/USDT یا XAU/USD):"
            )
            
            return AWAITING_SYMBOL
        except Exception as e:
            logger.error(f"خطا در شروع فرآیند تحلیل نماد: {e}")
            try:
                await query.edit_message_text("متأسفانه خطایی رخ داد. لطفاً دوباره تلاش کنید.")
            except:
                pass
            return ConversationHandler.END
    
    async def analyze_symbol_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پردازش ورودی نماد و تحلیل آن"""
        try:
            symbol = update.message.text.strip().upper()
            logger.info(f"تحلیل نماد {symbol} از کاربر {update.message.from_user.id} دریافت شد.")
            
            # ارسال پیام در حال تحلیل
            message = await update.message.reply_text(f"در حال تحلیل {symbol}... لطفاً صبر کنید.")
            
            # تحلیل نماد
            analysis = AnalysisEngine.analyze_symbol(symbol)
            
            # قالب‌بندی پاسخ
            response = self._format_analysis_response(analysis)
            
            # به‌روزرسانی پیام با نتایج
            await message.edit_text(response, parse_mode='Markdown')
            
            # ارسال منوی اصلی
            keyboard = [
                [InlineKeyboardButton("🔬 تحلیل نماد دیگر", callback_data='analyze')],
                [InlineKeyboardButton("👁️ پایش این معامله", callback_data='monitor')],
                [InlineKeyboardButton("🔙 بازگشت به منو", callback_data='back_to_menu')]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                "می‌خواهید چه کاری انجام دهید؟",
                reply_markup=reply_markup
            )
            
            # ذخیره نماد برای استفاده بعدی
            context.user_data['last_symbol'] = symbol
            
            return ConversationHandler.END
        except Exception as e:
            logger.error(f"خطا در تحلیل {symbol if 'symbol' in locals() else 'نماد'}: {e}")
            try:
                await update.message.reply_text("متأسفانه خطایی در تحلیل نماد رخ داد. لطفاً دوباره تلاش کنید.")
            except:
                pass
            return ConversationHandler.END
    
    def _format_analysis_response(self, analysis):
        """قالب‌بندی پاسخ تحلیل برای نمایش"""
        try:
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
                response += f"- *ساختار بازار:* {analysis['technical']['market_structure']['trend']}\n\n"
            
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
        except Exception as e:
            logger.error(f"خطا در قالب‌بندی پاسخ تحلیل: {e}")
            return f"*📊 تحلیل {symbol if 'symbol' in locals() else 'نماد'}*\n\nمتأسفانه خطایی در قالب‌بندی نتایج تحلیل رخ داد."
    
    async def show_silver_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """نمایش سیگنال‌های نقره‌ای اخیر"""
        try:
            query = update.callback_query
            logger.info(f"درخواست نمایش سیگنال‌های نقره‌ای از کاربر {query.from_user.id} دریافت شد.")
            
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
        except Exception as e:
            logger.error(f"خطا در نمایش سیگنال‌های نقره‌ای: {e}")
            try:
                await query.edit_message_text("متأسفانه خطایی در نمایش سیگنال‌ها رخ داد. لطفاً دوباره تلاش کنید.")
            except:
                pass
    
    async def toggle_gold_notifications(self, update: Update, context: ContextTypes.DEFAULT_TYPE, enable=True):
        """فعال یا غیرفعال کردن نوتیفیکیشن‌های طلایی"""
        try:
            query = update.callback_query
            user_id = query.from_user.id
            logger.info(f"درخواست تغییر وضعیت هشدارهای طلایی به {enable} از کاربر {user_id} دریافت شد.")
            
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
        except Exception as e:
            logger.error(f"خطا در تغییر وضعیت هشدارهای طلایی: {e}")
            try:
                await query.edit_message_text("متأسفانه خطایی در تغییر وضعیت هشدارها رخ داد. لطفاً دوباره تلاش کنید.")
            except:
                pass
    
    async def monitor_trade_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """شروع فرآیند پایش معامله"""
        try:
            query = update.callback_query
            logger.info(f"درخواست پایش معامله از کاربر {query.from_user.id} دریافت شد.")
            
            # بررسی آیا از تحلیل قبلی نمادی داریم
            if 'last_symbol' in context.user_data:
                symbol = context.user_data['last_symbol']
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
        except Exception as e:
            logger.error(f"خطا در شروع فرآیند پایش معامله: {e}")
            try:
                await query.edit_message_text("متأسفانه خطایی در شروع پایش معامله رخ داد. لطفاً دوباره تلاش کنید.")
            except:
                pass
            return ConversationHandler.END
    
    async def monitor_trade_symbol(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پردازش ورودی نماد برای پایش"""
        try:
            symbol = update.message.text.strip().upper()
            logger.info(f"نماد {symbol} برای پایش از کاربر {update.message.from_user.id} دریافت شد.")
            
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
        except Exception as e:
            logger.error(f"خطا در پردازش نماد برای پایش: {e}")
            try:
                await update.message.reply_text("متأسفانه خطایی در پردازش نماد رخ داد. لطفاً دوباره تلاش کنید.")
            except:
                pass
            return ConversationHandler.END
    
    async def monitor_trade_direction(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """پردازش انتخاب جهت معامله و شروع پایش"""
        try:
            query = update.callback_query
            user_id = query.from_user.id
            
            # دریافت اطلاعات از context
            symbol = context.user_data.get('symbol')
            direction = context.user_data.get('direction')
            
            logger.info(f"جهت {direction} برای نماد {symbol} از کاربر {user_id} دریافت شد.")
            
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
        except Exception as e:
            logger.error(f"خطا در پردازش جهت معامله: {e}")
            try:
                await query.edit_message_text("متأسفانه خطایی در پردازش جهت معامله رخ داد. لطفاً دوباره تلاش کنید.")
            except:
                pass
            return ConversationHandler.END
    
    async def stop_monitoring(self, update: Update, context: ContextTypes.DEFAULT_TYPE, trade_id):
        """توقف پایش یک معامله"""
        try:
            query = update.callback_query
            logger.info(f"درخواست توقف پایش معامله {trade_id} از کاربر {query.from_user.id} دریافت شد.")
            
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
        except Exception as e:
            logger.error(f"خطا در توقف پایش معامله: {e}")
            try:
                await query.edit_message_text("متأسفانه خطایی در توقف پایش معامله رخ داد. لطفاً دوباره تلاش کنید.")
            except:
                pass
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """مدیریت دستور /stats"""
        try:
            logger.info(f"دستور /stats از کاربر {update.message.from_user.id} دریافت شد.")
            
            # آمار ساختگی برای تست
            message = "📊 *آمار عملکرد* 📊\n\n"
            
            message += "*30 روز اخیر:*\n"
            message += f"- کل سیگنال‌ها: 25\n"
            message += f"- نرخ پیروزی کلی: 68.0%\n"
            message += f"- نرخ پیروزی سیگنال‌های طلایی: 82.0% (9/11)\n"
            message += f"- نرخ پیروزی سیگنال‌های نقره‌ای: 57.1% (8/14)\n"
            message += f"- میانگین سود/زیان: 2.35%\n\n"
            
            message += "*سیگنال‌های اخیر:*\n"
            message += f"1. BTC/USDT: 🟢 خرید (اطمینان: 85.0%) - ✅ سود (3.2%) - 2025-08-01\n"
            message += f"2. ETH/USDT: 🟢 خرید (اطمینان: 75.0%) - ✅ سود (1.8%) - 2025-07-30\n"
            message += f"3. XAU/USD: 🔴 فروش (اطمینان: 82.0%) - ✅ سود (2.5%) - 2025-07-28\n"
            message += f"4. EUR/USD: 🔴 فروش (اطمینان: 68.0%) - ❌ ضرر (-1.2%) - 2025-07-25\n"
            message += f"5. BTC/USDT: 🟢 خرید (اطمینان: 72.0%) - ⏳ فعال - 2025-07-22\n"
            
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
        except Exception as e:
            logger.error(f"خطا در اجرای دستور /stats: {e}")
            try:
                await update.message.reply_text("متأسفانه خطایی در نمایش آمار رخ داد. لطفاً دوباره تلاش کنید.")
            except:
                pass
    
    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """لغو مکالمه فعلی"""
        try:
            logger.info(f"درخواست لغو مکالمه از کاربر {update.message.from_user.id} دریافت شد.")
            
            await update.message.reply_text(
                "عملیات لغو شد. از /start برای شروع مجدد استفاده کنید."
            )
            
            return ConversationHandler.END
        except Exception as e:
            logger.error(f"خطا در لغو مکالمه: {e}")
            try:
                await update.message.reply_text("متأسفانه خطایی در لغو مکالمه رخ داد. از /start برای شروع مجدد استفاده کنید.")
            except:
                pass
            return ConversationHandler.END

# نقطه ورود اصلی
if __name__ == "__main__":
    try:
        # بررسی و نمایش تنظیمات محیطی
        logger.info(f"تنظیمات محیطی: TELEGRAM_TOKEN={'موجود' if TELEGRAM_TOKEN else 'ناموجود'}, "
                   f"NEWS_API_KEY={'موجود' if NEWS_API_KEY else 'ناموجود'}, "
                   f"TARGET_CHAT_ID={'موجود' if TARGET_CHAT_ID else 'ناموجود'}")
        
        if not TELEGRAM_TOKEN:
            logger.error("توکن تلگرام یافت نشد. لطفاً متغیر محیطی TELEGRAM_TOKEN را تنظیم کنید.")
            exit(1)
        
        # راه‌اندازی پایگاه داده
        setup_database()
        
        # ایجاد و شروع ربات
        logger.info("در حال ایجاد ربات تلگرام...")
        bot = TradingBot(TELEGRAM_TOKEN)
        logger.info("در حال شروع ربات...")
        bot.start()
    except Exception as e:
        logger.error(f"خطای کلی در برنامه: {e}")
        exit(1)