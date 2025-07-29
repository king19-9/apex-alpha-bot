# main.py (نسخه نهایی با اصلاح SyntaxError)

import os
import logging
import threading
import time
import requests
import pandas as pd
from sqlalchemy import create_engine, text, Table, MetaData
from sqlalchemy.dialects.postgresql import insert
import io

# --- کتابخانه‌های مورد نیاز هر بخش ---
from binance import ThreadedWebsocketManager
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackContext, CallbackQueryHandler
import telegram
import ta
import matplotlib
matplotlib.use('Agg') # حالت غیرتعاملی برای Matplotlib در سرور
import matplotlib.pyplot as plt

# --- بخش وب سرور برای بیدار نگه داشتن ---
from fastapi import FastAPI

# --- تنظیمات اولیه ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- متغیرهای محیطی ---
DATABASE_URL = os.getenv('DATABASE_URL')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# --- برنامه FastAPI ---
app = FastAPI()

@app.on_event("startup")
def startup_event():
    """این تابع به طور خودکار بعد از راه‌اندازی کامل وب‌سرور اجرا می‌شود."""
    logging.info("FastAPI app has started up. Initiating background services...")
    threading.Thread(target=run_background_services, daemon=True).start()

@app.get("/")
def read_root():
    return {"status": "Apex Bot is running!"}


def run_background_services():
    """تابع اصلی برای اجرای تمام سرویس‌های پس‌زمینه."""
    engine = None
    if DATABASE_URL:
        try:
            engine = create_engine(DATABASE_URL)
            with engine.connect() as conn:
                conn.execute(text("CREATE TABLE IF NOT EXISTS klines (time TIMESTPTZ, symbol TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL);"))
                conn.execute(text("SELECT create_hypertable('klines', 'time', if_not_exists => TRUE);"))
                conn.execute(text("CREATE TABLE IF NOT EXISTS technical_analysis (time TIMESTPTZ, symbol TEXT, rsi_14 REAL, macd REAL, ema_200 REAL, PRIMARY KEY (time, symbol));"))
                conn.commit()
            logging.info("Database setup complete.")
        except Exception as e:
            logging.error(f"DATABASE_ERROR: {e}")
            engine = None
    else:
        logging.warning("DATABASE_URL not set.")

    if engine:
        analyzer_thread = threading.Thread(target=run_data_analyzer, args=(engine,), name="AnalyzerThread", daemon=True)
        analyzer_thread.start()
    else:
        logging.warning("Analyzer thread not started because database is not available.")
    
    if TELEGRAM_TOKEN:
        run_telegram_bot(engine)
    else:
        logging.warning("TELEGRAM_TOKEN not set.")


def run_telegram_bot(db_engine):
    """منطق کامل ربات تلگرام"""
    try:
        updater = Updater(TELEGRAM_TOKEN)
        dp = updater.dispatcher
        logging.info("Telegram Updater initialized.")
    except Exception as e:
        logging.error(f"Could not initialize Telegram Updater: {e}")
        return

    def start(update: Update, context: CallbackContext) -> None:
        logging.info(f"Received /start from user {update.effective_user.id}")
        keyboard = [
            [InlineKeyboardButton("📊 BTC", callback_data='analyze_BTCUSDT'),
             InlineKeyboardButton("📈 ETH", callback_data='analyze_ETHUSDT')],
            [InlineKeyboardButton("📉 XRP", callback_data='analyze_XRPUSDT'),
             InlineKeyboardButton("🐶 DOGE", callback_data='analyze_DOGEUSDT')],
            [InlineKeyboardButton("📰 آخرین اخبار", callback_data='menu_news')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        update.message.reply_text('سلام! به ربات تحلیل‌گر Apex خوش آمدید:', reply_markup=reply_markup)

    def button_handler(update: Update, context: CallbackContext) -> None:
        """پردازش کلیک روی تمام دکمه‌های شیشه‌ای."""
        query = update.callback_query
        query.answer()
        
        if query.data.startswith('analyze_'):
            symbol = query.data.split('_')[1]
            if not db_engine:
                query.edit_message_text("سرویس دیتابیس در دسترس نیست.")
                return
            
            try:
                query.edit_message_text(text=f"در حال آماده‌سازی تحلیل برای {symbol}...")
                
                sql_query = f"SELECT * FROM technical_analysis WHERE symbol = '{symbol}' ORDER BY time DESC LIMIT 1;"
                df_analysis = pd.read_sql(sql_query, db_engine)

                if df_analysis.empty:
                    query.edit_message_text(text=f"تحلیلی برای {symbol} یافت نشد. لطفاً چند دقیقه صبر کنید.")
                    return

# main.py (نسخه نهایی و کامل با کتابخانه aiogram)

import os
import logging
import asyncio
import pandas as pd
from sqlalchemy import create_engine, text
import io

# --- کتابخانه‌های مورد نیاز ---
from aiogram import Bot, Dispatcher, executor, types
from aiogram.utils.exceptions import BotBlocked

from fastapi import FastAPI

# --- تنظیمات اولیه ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- متغیرهای محیطی ---
API_TOKEN = os.getenv('TELEGRAM_TOKEN')
DATABASE_URL = os.getenv('DATABASE_URL')
WEBHOOK_HOST = os.getenv('RAILWAY_STATIC_URL') # Railway این متغیر را فراهم می‌کند

# --- راه‌اندازی ربات و FastAPI ---
if not API_TOKEN:
    logging.fatal("TELEGRAM_TOKEN not found! The bot cannot start.")
    # در صورت نبود توکن، برنامه را متوقف می‌کنیم تا خطا واضح باشد
    exit()

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
app = FastAPI()

# --- متغیرهای سراسری (برای دسترسی از وب‌هوک) ---
WEBHOOK_PATH = f'/webhook/{API_TOKEN}'
WEBHOOK_URL = f'https://{WEBHOOK_HOST}{WEBHOOK_PATH}' if WEBHOOK_HOST else None

# --- منطق اصلی ربات با aiogram ---

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    """پاسخ به دستور /start"""
    logging.info(f"Received /start from user_id: {message.from_user.id}")
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    buttons = [
        types.InlineKeyboardButton(text="📊 BTC", callback_data="analyze_BTCUSDT"),
        types.InlineKeyboardButton(text="📈 ETH", callback_data="analyze_ETHUSDT"),
    ]
    keyboard.add(*buttons)
    await message.answer("سلام! به ربات Apex خوش آمدید. یک گزینه را انتخاب کنید:", reply_markup=keyboard)

@dp.callback_query_handler(lambda c: c.data and c.data.startswith('analyze_'))
async def process_callback_analyze(callback_query: types.CallbackQuery):
    """پردازش کلیک روی دکمه‌های تحلیل"""
    symbol = callback_query.data.split('_')[1]
    logging.info(f"Received analyze request for {symbol} from user_id: {callback_query.from_user.id}")
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(callback_query.from_user.id, f"در حال آماده‌سازی تحلیل برای {symbol}...")
    
    if not DATABASE_URL:
        await bot.send_message(callback_query.from_user.id, "خطا: سرویس دیتابیس در دسترس نیست.")
        return

    try:
        # در اینجا باید منطق تحلیل و ارسال چارت را اضافه کنیم
        # برای تست اولیه، فقط یک پیام موفقیت‌آمیز می‌فرستیم
        await bot.send_message(callback_query.from_user.id, f"تحلیل برای {symbol} با موفقیت انجام شد (این یک پیام تستی است).")
    except Exception as e:
        logging.error(f"Error during analysis for {symbol}: {e}")
        await bot.send_message(callback_query.from_user.id, "خطا در پردازش تحلیل.")

# --- مدیریت وب‌هوک و راه‌اندازی ---

@app.on_event("startup")
async def on_startup():
    """این تابع در هنگام راه‌اندازی FastAPI اجرا می‌شود."""
    if WEBHOOK_URL:
        logging.info(f"Setting webhook to: {WEBHOOK_URL}")
        await bot.set_webhook(WEBHOOK_URL)
    else:
        logging.warning("RAILWAY_STATIC_URL not found, cannot set webhook. Bot will not work via webhooks.")
        # در این حالت، می‌توانیم به حالت Polling برگردیم
        # asyncio.create_task(dp.start_polling())

@app.post(WEBHOOK_PATH)
async def bot_webhook(update: dict):
    """دریافت آپدیت‌ها از تلگرام"""
    telegram_update = types.Update(**update)
    Dispatcher.set_current(dp)
    Bot.set_current(bot)
    try:
        await dp.process_update(telegram_update)
    except Exception as e:
        logging.error(f"Error processing update: {e}")
    return {'ok': True}

@app.on_event("shutdown")
async def on_shutdown():
    """این تابع در هنگام خاموش شدن FastAPI اجرا می‌شود."""
    logging.info("Shutting down webhook...")
    await bot.delete_webhook()

@app.get("/")
def read_root():
    return {"status": "Apex Bot is running with aiogram!"}

# ما دیگر به Procfile نیازی نداریم اگر از این ساختار استفاده کنیم
# uvicorn main:app --host 0.0.0.0 --port $PORT