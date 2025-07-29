# main.py (نسخه نهایی و کامل با تمام توابع پر شده)

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
    # اجرای منطق اصلی ربات در یک نخ جداگانه تا وب‌سرور مسدود نشود
    threading.Thread(target=run_background_services, daemon=True).start()

@app.get("/")
def read_root():
    return {"status": "Apex Bot is running!"}


def run_background_services():
    """تابع اصلی برای اجرای تمام سرویس‌های پس‌زمینه."""
    # --- اتصال به دیتابیس ---
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

    # --- راه‌اندازی نخ‌های دیگر ---
    # نخ تحلیلگر (فقط اگر دیتابیس وصل باشد)
    if engine:
        analyzer_thread = threading.Thread(target=run_data_analyzer, args=(engine,), name="AnalyzerThread", daemon=True)
        analyzer_thread.start()
    else:
        logging.warning("Analyzer thread not started because database is not available.")
    
    # نخ ربات تلگرام در نخ اصلی این تابع اجرا می‌شود چون updater.idle() آن را مسدود می‌کند
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

                analysis = df_analysis.iloc[0]
                message = (f"🔎 **تحلیل تکنیکال برای #{symbol}**\n\n"
                           f"**RSI (14):** `{analysis['rsi_14']:.2f}`\n"
                           f"**MACD:** `{analysis['macd']:.2f}`\n"
                           f"**قیمت نزدیک به EMA (200):** `${analysis['ema_200']:,.2f}`\n\n"
                           f"_بروزرسانی در: {analysis['time'].strftime('%H:%M:%S UTC')}_")

                sql_history = f"SELECT time, close FROM klines WHERE symbol = '{symbol}' AND time > NOW() - INTERVAL '1 day' ORDER BY time;"
                df_history = pd.read_sql(sql_history, db_engine, index_col='time')
                
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_history.index, df_history['close'], color='cyan')
                ax.set_title(f"نمودار قیمت ۲۴ ساعت گذشته {symbol}", color='white')
                ax.grid(True, linestyle='--', alpha=0.3)
                fig.autofmt_xdate()

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plt.close(fig)
                
                context.bot.send_photo(chat_id=query.message.chat_id, photo=buf, caption=message, parse_mode='Markdown')
                query.delete_message()
            except Exception as e:
                logging.error(f"Error in button_handler for {symbol}: {e}")
                query.edit_message_text("خطا در پردازش تحلیل. لطفاً بعداً تلاش کنید.")
        
        elif query.data == 'menu_news':
            if not NEWS_API_KEY:
                query.edit_message_text("سرویس اخبار در دسترس نیست.")
                return
            
            try:
                query.edit_message_text("در حال دریافت آخرین اخبار...")
                url = f"https://newsapi.org/v2/everything?q=crypto&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
                response = requests.get(url)
                articles = response.json().get('articles', [])
                
                if not articles:
                    query.edit_message_text("خبری یافت نشد.")
                    return
                
                message = "📰 **آخرین اخبار مهم دنیای کریپتو:**\n\n"
                for article in articles:
                    message += f"🔹 {article['title']}\n"
                
                query.edit_message_text(message, disable_web_page_preview=True)
            except Exception as e:
                logging.error(f"Error fetching news: {e}")
                query.edit_message_text("خطا در دریافت اخبار.")

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(button_handler))
    
    logging.info("Starting Telegram bot polling...")
    updater.start_polling()
    updater.idle()


def run_data_analyzer(db_engine):
    """منطق کامل دریافت داده و تحلیل"""
    logging.info("Data Analyzer thread started.")
    metadata_analyzer = MetaData()
    technical_analysis_table = Table('technical_analysis', metadata_analyzer, autoload_with=db_engine)
    
    def process_kline(kline_data):
        try:
            symbol = kline_data['s']
            df_kline = pd.DataFrame([{'time': pd.to_datetime(kline_data['t'], unit='ms'), 'symbol': symbol,
                                      'open': float(kline_data['o']), 'high': float(kline_data['h']),
                                      'low': float(kline_data['l']), 'close': float(kline_data['c']),
                                      'volume': float(kline_data['v'])}])
            df_kline.to_sql('klines', db_engine, if_exists='append', index=False)

            query = f"SELECT time, close FROM klines WHERE symbol = '{symbol}' ORDER BY time DESC LIMIT 250;"
            df_history = pd.read_sql(query, db_engine, index_col='time').sort_index()

            if len(df_history) < 200: return

            rsi = ta.momentum.rsi(df_history['close'], window=14).iloc[-1]
            macd = ta.trend.macd(df_history['close']).iloc[-1]
            ema_200 = ta.trend.ema_indicator(df_history['close'], window=200).iloc[-1]
            
            analysis_data = {'time': pd.to_datetime(kline_data['T'], unit='ms'), 'symbol': symbol,
                             'rsi_14': rsi, 'macd': macd, 'ema_200': ema_200}
            
            stmt = 