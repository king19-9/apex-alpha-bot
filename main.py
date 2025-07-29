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
        