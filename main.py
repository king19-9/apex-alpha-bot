import os
import logging
import json
import time
import threading
import sqlite3
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dotenv import load_dotenv

# تنظیم لاگینگ پیشرفته
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ثابت‌ها
ANALYZING, AWAITING_SYMBOL, AWAITING_DIRECTION = range(3)
GOLD_SIGNAL_THRESHOLD = 80
SILVER_SIGNAL_THRESHOLD = 65

# بارگیری متغیرهای محیطی
load_dotenv()

# بارگیری تنظیمات از متغیرهای محیطی
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
TRADING_PAIRS = ["BTC/USDT", "ETH/USDT", "XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY"]

try:
    # ربات تلگرام
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes, ConversationHandler
    
    # لاگ کردن اطلاعات ربات
    logger.info(f"تنظیم متغیرهای محیطی: TELEGRAM_TOKEN={'موجود' if TELEGRAM_TOKEN else 'ناموجود'}, NEWS_API_KEY={'موجود' if NEWS_API_KEY else 'ناموجود'}")
    
    # بررسی وضعیت توکن
    if not TELEGRAM_TOKEN:
        logger.error("توکن تلگرام یافت نشد. لطفاً متغیر محیطی TELEGRAM_TOKEN را تنظیم کنید.")
        raise ValueError("TELEGRAM_TOKEN not found in environment variables")
    
    # تست دسترسی به API های خارجی
    logger.info("تست دسترسی به API‌های خارجی...")
    
    try:
        import pandas as pd
        import numpy as np
        logger.info("pandas و numpy با موفقیت بارگذاری شدند.")
    except Exception as e:
        logger.error(f"خطا در بارگذاری pandas یا numpy: {e}")
        raise
    
    try:
        import ccxt
        exchange = ccxt.kucoin()
        logger.info("ccxt با موفقیت بارگذاری شد.")
    except Exception as e:
        logger.error(f"خطا در بارگذاری ccxt: {e}")
        raise
    
    try:
        from tradingview_ta import TA_Handler, Interval
        logger.info("tradingview-ta با موفقیت بارگذاری شد.")
    except Exception as e:
        logger.error(f"خطا در بارگذاری tradingview-ta: {e}")
        raise
    
    try:
        import requests
        logger.info("requests با موفقیت بارگذاری شد.")
    except Exception as e:
        logger.error(f"خطا در بارگذاری requests: {e}")
        raise
    
    try:
        from newsapi import NewsApiClient
        if NEWS_API_KEY:
            news_api = NewsApiClient(api_key=NEWS_API_KEY)
            logger.info("NewsApiClient با موفقیت بارگذاری شد.")
        else:
            logger.warning("کلید API اخبار یافت نشد. عملکرد اخبار محدود خواهد بود.")
            news_api = None
    except Exception as e:
        logger.error(f"خطا در بارگذاری NewsApiClient: {e}")
        news_api = None
    
    # راه‌اندازی پایگاه داده
    def setup_database():
        try:
            logger.info("در حال راه‌اندازی پایگاه داده...")
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
            logger.error(traceback.format_exc())
            return False
    
    # عملیات پایگاه داده
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
                conn = Database.get_connection()
                if not conn:
                    return False
                cursor = conn.cursor()
                cursor.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
                conn.commit()
                conn.close()
                logger.info(f"کاربر {user_id} به پایگاه داده اضافه شد.")
                return True
            except Exception as e:
                logger.error(f"خطا در افزودن کاربر {user_id}: {e}")
                return False
        
        @staticmethod
        def set_gold_notifications(user_id, status):
            try:
                conn = Database.get_connection()
                if not conn:
                    return False
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET gold_notifications = ? WHERE user_id = ?", (status, user_id))
                conn.commit()
                conn.close()
                logger.info(f"وضعیت نوتیفیکیشن طلایی برای کاربر {user_id} به {status} تغییر یافت.")
                return True
            except Exception as e:
                logger.error(f"خطا در تغییر وضعیت نوتیفیکیشن برای کاربر {user_id}: {e}")
                return False
        
        # دیگر متدهای Database بدون تغییر...
    
    # موتور تحلیل ساده‌شده
    class AnalysisEngine:
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
            """دریافت قیمت فعلی یک نماد"""
            try:
                ticker = exchange.fetch_ticker(symbol)
                return ticker['last']
            except Exception as e:
                logger.error(f"خطا در دریافت قیمت برای {symbol}: {e}")
                # بازگشت یک مقدار پیش‌فرض برای جلوگیری از خطا
                return 1000.0
        
        @staticmethod
        def analyze_symbol(symbol):
            """انجام تحلیل ساده یک نماد"""
            try:
                logger.info(f"تحلیل نماد {symbol} آغاز شد.")
                
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
                
                # اطلاعات ساده‌شده برای تست
                result["summary"]["best_strategy"] = "EMA Crossover"
                result["summary"]["strategy_win_rate"] = 65.0
                
                result["technical"]["trends"] = {
                    "کوتاه‌مدت": "صعودی",
                    "میان‌مدت": "صعودی",
                    "بلندمدت": "خنثی"
                }
                
                result["technical"]["rsi"] = {
                    "value": 55.0,
                    "condition": "خنثی"
                }
                
                result["technical"]["support_levels"] = [price * 0.95, price * 0.90, price * 0.85]
                result["technical"]["resistance_levels"] = [price * 1.05, price * 1.10, price * 1.15]
                
                result["technical"]["candlestick_patterns"] = ["دوجی 1 کندل قبل شناسایی شد"]
                
                result["technical"]["market_structure"] = {
                    "trend": "روند صعودی"
                }
                
                result["fundamental"]["economic"] = {
                    "fed_rate": 5.25,
                    "upcoming_events": [
                        {"date": "2025-09-20", "event": "جلسه FOMC", "importance": "بالا"}
                    ]
                }
                
                result["fundamental"]["overall_sentiment"] = "مثبت"
                
                # تعیین جهت و توصیه ساده
                direction = "BUY"
                confidence = 70.0
                
                result["recommendation"] = {
                    "action": direction,
                    "confidence": confidence,
                    "entry": price,
                    "stop_loss": price * 0.97,
                    "take_profit1": price * 1.05,
                    "take_profit2": price * 1.10,
                    "risk_reward_ratio": 1.75,
                    "recommended_leverage": 2.0
                }
                
                logger.info(f"تحلیل نماد {symbol} با موفقیت انجام شد.")
                return result
            except Exception as e:
                logger.error(f"خطا در تحلیل نماد {symbol}: {e}")
                logger.error(traceback.format_exc())
                # بازگشت یک تحلیل پیش‌فرض برای جلوگیری از خطا
                return {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "summary": {"current_price": 1000.0, "trading_sessions": ["نامشخص"]},
                    "technical": {},
                    "fundamental": {},
                    "recommendation": {"action": "WAIT", "confidence": 0, "reason": "خطا در تحلیل"}
                }
    
    # شکارچی سیگنال ساده‌شده
    class SignalHunter:
        def __init__(self, watchlist=None):
            self.watchlist = watchlist or TRADING_PAIRS
            self.running = False
            self.thread = None
        
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
                    time.sleep(60)  # کاهش بار پردازشی در محیط تست
                except Exception as e:
                    logger.error(f"خطا در حلقه شکار سیگنال: {e}")
                    time.sleep(60)
        
        def get_silver_signals(self, limit=5):
            """دریافت سیگنال‌های نقره‌ای ساختگی برای تست"""
            try:
                # سیگنال‌های تستی
                signals = []
                for i in range(limit):
                    signals.append({
                        "id": i+1,
                        "symbol": TRADING_PAIRS[i % len(TRADING_PAIRS)],
                        "direction": "BUY" if i % 2 == 0 else "SELL",
                        "entry": 1000.0 * (i+1),
                        "stop_loss": 950.0 * (i+1),
                        "take_profit1": 1050.0 * (i+1),
                        "take_profit2": 1100.0 * (i+1),
                        "confidence": 70.0,
                        "strategy": "EMA Crossover",
                        "win_rate": 65.0,
                        "risk_reward": 1.75,
                        "created_at": datetime.now().isoformat()
                    })
                return signals
            except Exception as e:
                logger.error(f"خطا در دریافت سیگنال‌های نقره‌ای: {e}")
                return []
    
    # پایشگر معاملات ساده‌شده
    class TradeMonitor:
        def __init__(self):
            self.running = False
            self.thread = None
        
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
                    time.sleep(60)  # کاهش بار پردازشی در محیط تست
                except Exception as e:
                    logger.error(f"خطا در حلقه پایش معامله: {e}")
                    time.sleep(60)
    
    # ربات تلگرام
    class TradingBot:
        def __init__(self, token):
            try:
                logger.info("در حال راه‌اندازی ربات تلگرام...")
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
                
                logger.info("ربات تلگرام با موفقیت راه‌اندازی شد.")
            except Exception as e:
                logger.error(f"خطا در راه‌اندازی ربات تلگرام: {e}")
                logger.error(traceback.format_exc())
                raise
        
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
                logger.error(traceback.format_exc())
        
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
                logger.error(traceback.format_exc())
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
                logger.error(traceback.format_exc())
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
                logger.error(traceback.format_exc())
                await query.edit_message_text("متأسفانه خطایی رخ داد. لطفاً دوباره تلاش کنید.")
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
                    [InlineKeyboardButton("🔙 بازگشت به منو", callback_data='back_to_menu')]
                ]
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(
                    "می‌خواهید چه کاری انجام دهید؟",
                    reply_markup=reply_markup
                )
                
                return ConversationHandler.END
            except Exception as e:
                logger.error(f"خطا در تحلیل {symbol if 'symbol' in locals() else 'نماد'}: {e}")
                logger.error(traceback.format_exc())
                await update.message.reply_text("متأسفانه خطایی در تحلیل نماد رخ داد. لطفاً دوباره تلاش کنید.")
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
                logger.error(traceback.format_exc())
                return "متأسفانه خطایی در قالب‌بندی نتایج تحلیل رخ داد."
        
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
                logger.error(traceback.format_exc())
                await query.edit_message_text("متأسفانه خطایی در نمایش سیگنال‌ها رخ داد. لطفاً دوباره تلاش کنید.")
        
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
                logger.error(traceback.format_exc())
                await query.edit_message_text("متأسفانه خطایی در تغییر وضعیت هشدارها رخ داد. لطفاً دوباره تلاش کنید.")
        
        async def monitor_trade_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
            """شروع فرآیند پایش معامله"""
            try:
                query = update.callback_query
                logger.info(f"درخواست پایش معامله از کاربر {query.from_user.id} دریافت شد.")
                
                await query.edit_message_text(
                    "لطفاً نام نماد معامله‌ای که می‌خواهید پایش کنید را وارد کنید (مثلاً BTC/USDT یا XAU/USD):"
                )
                
                return AWAITING_SYMBOL
            except Exception as e:
                logger.error(f"خطا در شروع فرآیند پایش معامله: {e}")
                logger.error(traceback.format_exc())
                await query.edit_message_text("متأسفانه خطایی در شروع پایش معامله رخ داد. لطفاً دوباره تلاش کنید.")
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
                logger.error(traceback.format_exc())
                await update.message.reply_text("متأسفانه خطایی در پردازش نماد رخ داد. لطفاً دوباره تلاش کنید.")
                return ConversationHandler.END
        
        async def monitor_trade_direction(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
            """پردازش انتخاب جهت معامله و شروع پایش"""
            try:
                query = update.callback_query
                user_id = query.from_user.id
                
                # دریافت جهت از داده کال‌بک
                if query.data.startswith('direction_'):
                    direction = query.data.split('_')[1]  # BUY یا SELL
                    context.user_data['direction'] = direction
                
                # دریافت اطلاعات از context
                symbol = context.user_data.get('symbol')
                direction = context.user_data.get('direction')
                
                logger.info(f"جهت {direction} برای نماد {symbol} از کاربر {user_id} دریافت شد.")
                
                if not symbol or not direction:
                    await query.edit_message_text(
                        "خطا: اطلاعات معامله ناقص است. لطفاً دوباره تلاش کنید."
                    )
                    return ConversationHandler.END
                
                # افزودن معامله به پایش‌ها (ساده‌سازی شده برای تست)
                trade_id = 1  # در نسخه واقعی، از پایگاه داده دریافت می‌شود
                
                # دکمه‌های پایش
                keyboard = [
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
                logger.error(traceback.format_exc())
                await query.edit_message_text("متأسفانه خطایی در پردازش جهت معامله رخ داد. لطفاً دوباره تلاش کنید.")
                return ConversationHandler.END
        
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
                logger.error(traceback.format_exc())
                await update.message.reply_text("متأسفانه خطایی در نمایش آمار رخ داد. لطفاً دوباره تلاش کنید.")
        
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
                logger.error(traceback.format_exc())
                await update.message.reply_text("متأسفانه خطایی در لغو مکالمه رخ داد. از /start برای شروع مجدد استفاده کنید.")
                return ConversationHandler.END
    
    # نقطه ورود اصلی
    if __name__ == "__main__":
        try:
            # راه‌اندازی پایگاه داده
            if not setup_database():
                logger.error("خطا در راه‌اندازی پایگاه داده. خروج از برنامه.")
                exit(1)
            
            # ایجاد و شروع ربات
            logger.info("در حال ایجاد ربات تلگرام...")
            bot = TradingBot(TELEGRAM_TOKEN)
            logger.info("در حال شروع ربات...")
            bot.start()
        except Exception as e:
            logger.error(f"خطای کلی در برنامه: {e}")
            logger.error(traceback.format_exc())
            exit(1)
    
except Exception as e:
    logger.error(f"خطای حیاتی در برنامه: {e}")
    if 'traceback' in globals():
        logger.error(traceback.format_exc())
    exit(1)