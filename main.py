# main.py (نسخه نهایی و کامل با لاگ‌گیری دقیق)

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# --- متغیرهای محیطی ---
DATABASE_URL = os.getenv('DATABASE_URL')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
# آیدی چت برای تست و نوتیفیکیشن
TARGET_CHAT_ID = os.getenv('TARGET_CHAT_ID') 

# --- برنامه FastAPI (برای وب سرور) ---
app = FastAPI()
@app.get("/")
def read_root():
    return {"status": "Apex Bot is running and healthy!"}

# --- بخش‌های اصلی ربات که در نخ‌های جداگانه اجرا می‌شوند ---

def setup_and_run_bot_logic():
    """این تابع تمام منطق سنگین ربات را بعد از راه‌اندازی وب‌سرور اجرا می‌کند."""
    logging.info("BACKGROUND_SERVICES: Starting setup...")
    time.sleep(5)

    engine = None
    if DATABASE_URL:
        try:
            logging.info("DATABASE: Attempting to create engine...")
            engine = create_engine(DATABASE_URL)
            logging.info("DATABASE: Engine created. Attempting to connect...")
            with engine.connect() as conn:
                logging.info("DATABASE: Connection successful. Setting up tables...")
                conn.execute(text("CREATE TABLE IF NOT EXISTS klines (time TIMESTAMPTZ, symbol TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL);"))
                conn.execute(text("SELECT create_hypertable('klines', 'time', if_not_exists => TRUE);"))
                conn.execute(text("CREATE TABLE IF NOT EXISTS technical_analysis (time TIMESTAMPTZ, symbol TEXT, rsi_14 REAL, macd REAL, ema_200 REAL, PRIMARY KEY (time, symbol));"))
                conn.commit()
            logging.info("DATABASE: Setup complete.")
        except Exception as e:
            logging.error(f"FATAL_DATABASE_ERROR: {e}")
            engine = None
    else:
        logging.warning("DATABASE_URL not set. Database features will be disabled.")

    if TELEGRAM_TOKEN:
        logging.info("TELEGRAM: Starting Telegram bot thread...")
        threading.Thread(target=run_telegram_bot, args=(engine,), name="TelegramThread", daemon=True).start()
    else:
        logging.warning("TELEGRAM_TOKEN not set. Telegram bot will not start.")

    if engine:
        logging.info("ANALYZER: Starting data analyzer thread...")
        threading.Thread(target=run_data_analyzer, args=(engine,), name="AnalyzerThread", daemon=True).start()
    else:
        logging.warning("ANALYZER: Not started because database is not available.")
    
    logging.info("BACKGROUND_SERVICES: All background services have been initiated.")

def run_telegram_bot(db_engine):
    """منطق کامل ربات تلگرام"""
    if not TELEGRAM_TOKEN: return
    
    try:
        logging.info("TELEGRAM_BOT: Initializing Updater...")
        updater = Updater(TELEGRAM_TOKEN)
        dp = updater.dispatcher
        logging.info("TELEGRAM_BOT: Updater initialized successfully.")
    except Exception as e:
        logging.error(f"FATAL_TELEGRAM_ERROR: Could not initialize Updater: {e}")
        return

    def start(update: Update, context: CallbackContext) -> None:
        logging.info(f"Received /start command from user {update.effective_user.id}")
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
                
                query.edit_message_text(message)
            except Exception as e:
                logging.error(f"Error fetching news: {e}")
                query.edit_message_text("خطا در دریافت اخبار.")

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(button_handler))
    
    logging.info("TELEGRAM_BOT: Starting polling...")
    updater.start_polling()
    updater.idle()


def run_data_analyzer(db_engine):
    """منطق کامل دریافت داده و تحلیل"""
    logging.info("ANALYZER: Starting...")
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

            if len(df_history) < 200:
                return

            rsi = ta.momentum.rsi(df_history['close'], window=14).iloc[-1]
            macd = ta.trend.macd(df_history['close']).iloc[-1]
            ema_200 = ta.trend.ema_indicator(df_history['close'], window=200).iloc[-1]
            
            analysis_data = {'time': pd.to_datetime(kline_data['T'], unit='ms'), 'symbol': symbol,
                             'rsi_14': rsi, 'macd': macd, 'ema_200': ema_200}
            
            stmt = insert(technical_analysis_table).values(analysis_data)
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=['time', 'symbol'],
                set_={col: getattr(stmt.excluded, col) for col in analysis_data}
            )
            with db_engine.connect() as conn:
                conn.execute(on_conflict_stmt)
                conn.commit()
            logging.info(f"ANALYZER: Analyzed and saved for {symbol}: RSI={rsi:.2f}")
        except Exception as e:
            logging.error(f"ANALYZER_ERROR: Error in process_kline for {kline_data.get('s')}: {e}")

    def handle_kline_message(msg):
        if msg.get('e') == 'kline' and msg.get('k', {}).get('x'):
            process_kline(msg['k'])
            
    logging.info("ANALYZER: Starting websocket manager...")
    twm = ThreadedWebsocketManager()
    twm.start()
    streams = ['btcusdt@kline_1m', 'ethusdt@kline_1m', 'xrpusdt@kline_1m', 'dogeusdt@kline_1m']
    twm.start_multiplex_socket(callback=handle_kline_message, streams=streams)
    twm.join()

# --- نقطه شروع برنامه ---
# یک نخ جدا برای منطق اصلی ربات ایجاد می‌کنیم تا وب‌سرور را مسدود نکند
threading.Thread(target=setup_and_run_bot_logic, daemon=True).start()

# نخ اصلی برنامه، وب سرور را برای بیدار ماندن اجرا می‌کند
logging.info("MAIN_THREAD: Starting web server to keep the service alive...")