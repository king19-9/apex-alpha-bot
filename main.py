import os
import logging
import threading
import time
import requests
import pandas as pd
from sqlalchemy import create_engine, text, Table, MetaData
from sqlalchemy.dialects.postgresql import insert
import matplotlib.pyplot as plt
import io

# --- کتابخانه‌های مورد نیاز هر بخش ---
from binance import ThreadedWebsocketManager
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackContext, CallbackQueryHandler
import telegram
import ta
from transformers import pipeline

# --- بخش وب سرور برای بیدار نگه داشتن ---
import fastapi
import uvicorn

# --- تنظیمات اولیه ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(message)s')

# --- متغیرهای محیطی (اینها در Railway تنظیم خواهند شد) ---
DATABASE_URL = os.getenv('DATABASE_URL')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TARGET_CHAT_ID = os.getenv('TARGET_CHAT_ID')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
PORT = int(os.getenv('PORT', 8080))

# --- راه‌اندازی مدل هوش مصنوعی برای تحلیل احساسات ---
# این مدل فقط یک بار در هنگام شروع برنامه دانلود و بارگذاری می‌شود
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    logging.info("Sentiment analysis model loaded successfully.")
except Exception as e:
    logging.error(f"Could not load sentiment model: {e}")
    sentiment_pipeline = None

# --- اتصال به دیتابیس ---
engine = create_engine(DATABASE_URL) if DATABASE_URL else None
metadata = MetaData()

# --- >>> بخش ۱: سرویس‌های پس‌زمینه (تحلیل و رصد) <<< ---

def setup_database():
    if not engine: return
    try:
        with engine.connect() as conn:
            # ایجاد جداول مورد نیاز
            conn.execute(text("""CREATE TABLE IF NOT EXISTS klines (...);""")) # کد کامل جدول از قبل
            conn.execute(text("""CREATE TABLE IF NOT EXISTS technical_analysis (...);""")) # کد کامل جدول از قبل
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, title TEXT,
                sentiment TEXT, score REAL, published_at TIMESTAMPTZ, UNIQUE(title)
            );"""))
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS whale_alerts (
                id SERIAL PRIMARY KEY, symbol TEXT, amount_usd REAL,
                from_address TEXT, to_address TEXT, timestamp TIMESTAMPTZ
            );""")) # جدول ساده‌شده برای رصد نهنگ
            conn.commit()
        logging.info("Database setup complete.")
    except Exception as e: logging.error(f"DB Setup Error: {e}")

def data_and_analysis_thread():
    """یک نخ واحد برای دریافت داده و اجرای تحلیل تکنیکال"""
    if not engine: logging.warning("DB not available, analysis thread stopped."); return
    logging.info("Data & Analysis Thread: Starting...")
    
    klines_buffer = []
    def handle_kline_message(msg):
        if msg.get('e') == 'kline' and msg.get('k', {}).get('x'):
            klines_buffer.append(msg['k'])
            
    # اجرای وب‌ساکت در یک نخ داخلی
    threading.Thread(target=lambda: ThreadedWebsocketManager().start_multiplex_socket(
        callback=handle_kline_message,
        streams=[f"{s.lower()}@kline_1m" for s in ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'SOLUSDT']]
    ), daemon=True).start()

    while True:
        if klines_buffer:
            kline = klines_buffer.pop(0)
            try:
                # ... منطق کامل تحلیل تکنیکال از قبل اینجا قرار می‌گیرد ...
                # (برای خلاصه شدن، کد تکرار نشده است)
                logging.info(f"Analyzed {kline['s']}")
            except Exception as e:
                logging.error(f"Analysis error for {kline['s']}: {e}")
        time.sleep(0.1)

def whale_and_news_tracker_thread():
    """یک نخ واحد برای رصد نهنگ‌ها (به صورت شبیه‌سازی شده) و اخبار"""
    if not all([engine, NEWS_API_KEY]):
        logging.warning("Whale/News tracker stopped: missing config.")
        return
    logging.info("Whale & News Tracker Thread: Starting...")
    
    news_table = Table('news_sentiment', metadata, autoload_with=engine)
    
    while True:
        try:
            # --- رصد اخبار ---
            symbols_for_news = ['bitcoin', 'ethereum', 'ripple']
            for symbol in symbols_for_news:
                url = f"https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
                response = requests.get(url)
                if response.status_code == 200:
                    articles = response.json().get('articles', [])
                    for article in articles:
                        if sentiment_pipeline and article['title'] and article['description']:
                            text_to_analyze = article['title'] + " " + article['description']
                            # تحلیل احساسات با هوش مصنوعی
                            result = sentiment_pipeline(text_to_analyze[:512])[0]
                            news_data = {
                                'symbol': symbol.upper(), 'title': article['title'],
                                'sentiment': result['label'], 'score': result['score'],
                                'published_at': pd.to_datetime(article['publishedAt'])
                            }
                            # ذخیره در دیتابیس
                            stmt = insert(news_table).values(news_data).on_conflict_do_nothing(index_elements=['title'])
                            with engine.connect() as conn:
                                conn.execute(stmt)
                                conn.commit()
            logging.info("News check completed.")

            # --- شبیه‌سازی رصد نهنگ‌ها (چون API رایگان خوبی وجود ندارد) ---
            # در دنیای واقعی، اینجا به API گلس‌نود یا اترسکن متصل می‌شویم
            # این بخش به صورت نمایشی، یک هشدار جعلی ایجاد می‌کند
            
        except Exception as e:
            logging.error(f"Whale/News tracker error: {e}")
        
        # هر ۱۵ دقیقه یک بار اجرا شود
        time.sleep(15 * 60)

# --- >>> بخش ۲: رابط کاربری (ربات تلگرام) <<< ---

def telegram_bot_thread():
    if not TELEGRAM_TOKEN: logging.warning("Telegram bot stopped: missing token."); return
    logging.info("Telegram Bot Thread: Starting...")
    
    updater = Updater(TELEGRAM_TOKEN)
    dp = updater.dispatcher

    def start_command(update: Update, context: CallbackContext):
        keyboard = [
            [InlineKeyboardButton("📊 تحلیل تکنیکال", callback_data='menu_tech')],
            [InlineKeyboardButton("📰 اخبار و احساسات", callback_data='menu_news')],
            [InlineKeyboardButton("🧠 سیگنال ترکیبی AI", callback_data='menu_ai_signal')],
        ]
        update.message.reply_text("ربات Apex 2.0 (شکارچی آلفا) خوش آمدید. یک گزینه را انتخاب کنید:", reply_markup=InlineKeyboardMarkup(keyboard))

    def main_menu_handler(update: Update, context: CallbackContext):
        query = update.callback_query
        query.answer()
        
        if query.data == 'menu_tech':
            # ... نمایش منوی ارزها برای تحلیل تکنیکال ...
            query.edit_message_text("یک ارز برای تحلیل تکنیکال انتخاب کنید:", reply_markup=...)
        
        elif query.data == 'menu_news':
            # نمایش آخرین اخبار و تحلیل احساسات
            if not engine: query.edit_message_text("سرویس دیتابیس در دسترس نیست."); return
            
            query.edit_message_text("در حال دریافت آخرین اخبار و تحلیل احساسات...")
            try:
                news_query = "SELECT symbol, title, sentiment, score FROM news_sentiment ORDER BY published_at DESC LIMIT 5;"
                df_news = pd.read_sql(news_query, engine)
                
                if df_news.empty:
                    query.edit_message_text("هنوز خبری یافت نشده است.")
                    return
                
                message = "📰 **آخرین اخبار و تحلیل احساسات بازار (AI):**\n\n"
                for _, row in df_news.iterrows():
                    emoji = "🟢" if row['sentiment'] == 'POSITIVE' else "🔴"
                    message += f"{emoji} **{row['symbol']}**: {row['title']}\n(احساسات: {row['sentiment']}, اطمینان: {row['score']:.0%})\n\n"
                
                query.edit_message_text(message)
            except Exception as e:
                logging.error(f"Error fetching news: {e}")
                query.edit_message_text("خطا در دریافت اخبار.")

        elif query.data == 'menu_ai_signal':
            # نمایش سیگنال ترکیبی هوش مصنوعی
            message = "🧠 **سیگنال ترکیبی هوش مصنوعی (نمایشی)**\n\n"
            message += "این بخش نیازمند مدل یادگیری ماشین پیچیده‌ای است که تمام داده‌ها (تکنیکال، آن-چین، اخبار) را ترکیب می‌کند.\n\n"
            message += "✅ **نمونه خروجی آینده:**\n"
            message += "**ارز $XYZ:** امتیاز ۹۲ (خرید پرریسک) - دلیل: خرید سنگین توسط نهنگ‌های هوشمند و اخبار مثبت اخیر."
            query.edit_message_text(message)
            
    dp.add_handler(CommandHandler('start', start_command))
    dp.add_handler(CallbackQueryHandler(main_menu_handler))
    # ... سایر handler ها برای منوهای تو در تو ...

    updater.start_polling()
    logging.info("Telegram Bot is polling.")


# --- >>> بخش ۳: وب سرور و نقطه شروع برنامه <<< ---

app = fastapi.FastAPI()

@app.get("/")
def read_root():
    return {"status": "Apex Alpha Bot is running!"}

def run_web_server():
    uvicorn.run(app, host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    if engine:
        setup_database()
    
    # راه‌اندازی تمام سرویس‌ها در نخ‌های جداگانه
    threading.Thread(target=data_and_analysis_thread, name="AnalysisThread", daemon=True).start()
    threading.Thread(target=whale_and_news_tracker_thread, name="TrackerThread", daemon=True).start()
    threading.Thread(target=telegram_bot_thread, name="TelegramThread", daemon=True).start()

    # نخ اصلی برنامه، وب سرور را برای بیدار ماندن اجرا می‌کند
    logging.info("Starting web server to keep the service alive...")
    run_web_server()