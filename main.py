# main.py (نسخه نهایی با تمام قابلیت‌ها و تحلیل‌های واقعی)

import os
import logging
import time
import telepot
from telepot.loop import MessageLoop
from telepot.namedtuple import InlineKeyboardMarkup, InlineKeyboardButton
import pandas as pd
from sqlalchemy import create_engine
import io
from fastapi import FastAPI
import uvicorn
import threading
import requests
from binance.client import Client
import ta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- تنظیمات اولیه ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- متغیرهای محیطی ---
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
DATABASE_URL = os.getenv('DATABASE_URL')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
# کلیدهای API بایننس (فعلا برای داده‌های عمومی نیاز نیست)
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')

# --- کلاینت‌ها و سرویس‌ها ---
app = FastAPI()
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
engine = create_engine(DATABASE_URL) if DATABASE_URL else None
bot = telepot.Bot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
user_states = {}

# --- توابع سازنده کیبوردها ---
def get_main_menu_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='📊 تحلیل تکنیکال', callback_data='menu_tech_analysis')],
        [InlineKeyboardButton(text='📰 اخبار و احساسات', callback_data='menu_news')],
        [InlineKeyboardButton(text='🐳 رصد نهنگ‌ها', callback_data='menu_whales')],
        [InlineKeyboardButton(text='🧠 سیگنال‌های AI', callback_data='menu_ai')]
    ])

def get_symbol_analysis_keyboard(symbol):
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='📈 نمایش چارت', callback_data=f'action_chart_{symbol}')],
        [InlineKeyboardButton(text='📉 اندیکاتورها', callback_data=f'action_indicators_{symbol}')],
        [InlineKeyboardButton(text='🗞 اخبار مرتبط', callback_data=f'action_news_{symbol}')],
        [InlineKeyboardButton(text='🔙 بازگشت (ورود نماد جدید)', callback_data='menu_tech_analysis')]
    ])

def get_back_to_main_menu_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='🔙 بازگشت به منوی اصلی', callback_data='main_menu')]
    ])

# --- توابع اجرایی (منطق اصلی برنامه) ---

def get_price_chart(symbol):
    """داده‌های تاریخی را گرفته و یک نمودار تصویری برمی‌گرداند."""
    try:
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "1 day ago UTC")
        df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['close'] = df['close'].astype(float)

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['time'], df['close'], color='cyan')
        ax.set_title(f"نمودار قیمت ۲۴ ساعت گذشته {symbol}", color='white')
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.autofmt_xdate()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf
    except Exception as e:
        logging.error(f"Error creating chart for {symbol}: {e}")
        return None

def get_technical_indicators(symbol):
    """اندیکاتورهای اصلی را محاسبه و به صورت متن برمی‌گرداند."""
    try:
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_4HOUR, "90 day ago UTC")
        df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        df['close'] = df['close'].astype(float)
        
        if len(df) < 200:
            return "داده کافی برای تحلیل بلندمدت وجود ندارد."

        rsi = ta.momentum.rsi(df['close'], window=14).iloc[-1]
        macd = ta.trend.macd_diff(df['close']).iloc[-1] # هیستوگرام MACD
        ema200 = ta.trend.ema_indicator(df['close'], window=200).iloc[-1]
        last_price = df['close'].iloc[-1]
        
        message = (f"🔎 **تحلیل تکنیکال برای #{symbol}**\n\n"
                   f"**قیمت فعلی:** `${last_price:,.2f}`\n"
                   f"**RSI (14):** `{rsi:.2f}`\n"
                   f"**MACD Histogram:** `{macd:.2f}`\n"
                   f"**EMA (200):** `${ema200:,.2f}`")
        return message
    except Exception as e:
        logging.error(f"Error getting indicators for {symbol}: {e}")
        return "خطا در محاسبه اندیکاتورها."

def get_symbol_news(symbol):
    """اخبار مرتبط با یک نماد خاص را واکشی می‌کند."""
    if not NEWS_API_KEY:
        return "سرویس اخبار در دسترس نیست."
    
    # حذف USDT از انتهای نماد برای جستجوی بهتر
    query = symbol.replace('USDT', '')
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        articles = response.json().get('articles', [])
        
        if not articles:
            return f"خبری برای {query} یافت نشد."
        
        message = f"📰 **آخرین اخبار مرتبط با #{query}**\n\n"
        for article in articles:
            message += f"🔹 {article['title']}\n\n"
        return message
    except Exception as e:
        logging.error(f"Error fetching news for {symbol}: {e}")
        return "خطا در دریافت اخبار."

# --- کنترل‌کننده‌های ربات ---

def handle_chat(msg):
    """پردازش پیام‌های متنی"""
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type != 'text': return

    text = msg['text']
    
    if user_states.get(chat_id) == 'awaiting_symbol':
        handle_symbol_input(chat_id, text)
        return
        
    if text == '/start':
        user_states[chat_id] = 'main_menu'
        bot.sendMessage(chat_id, 'به ربات هوشمند Apex خوش آمدید. چه بخشی را می‌خواهید بررسی کنید؟',
                        reply_markup=get_main_menu_keyboard())

def handle_callback_query(msg):
    """پردازش کلیک روی دکمه‌ها"""
    query_id, from_id, query_data = telepot.glance(msg, flavor='callback_query')
    chat_id = from_id
    bot.answerCallbackQuery(query_id)
    
    # منوی اصلی
    if query_data == 'main_menu':
        user_states[chat_id] = 'main_menu'
        bot.sendMessage(chat_id, 'منوی اصلی:', reply_markup=get_main_menu_keyboard())

    elif query_data == 'menu_tech_analysis':
        user_states[chat_id] = 'awaiting_symbol'
        bot.sendMessage(chat_id, 'لطفاً نماد ارز مورد نظر خود را با فرمت صحیح وارد کنید (مثلاً: BTCUSDT).',
                        reply_markup=get_back_to_main_menu_keyboard())
    
    # بخش‌های نمایشی
    elif query_data in ['menu_news', 'menu_whales', 'menu_ai']:
        user_states[chat_id] = 'main_menu'
        message = "این بخش در حال توسعه است و به زودی به طور کامل فعال خواهد شد."
        bot.sendMessage(chat_id, message, reply_markup=get_back_to_main_menu_keyboard())

    # اقدامات مربوط به یک نماد
    elif query_data.startswith('action_'):
        action, symbol = query_data.split('_', 2)[1:]
        
        if action == 'chart':
            bot.sendMessage(chat_id, f'در حال آماده‌سازی چارت برای {symbol}...')
            chart_image = get_price_chart(symbol)
            if chart_image:
                bot.sendPhoto(chat_id, chart_image)
            else:
                bot.sendMessage(chat_id, "خطا در ساخت نمودار.")

        elif action == 'indicators':
            bot.sendMessage(chat_id, f'در حال واکشی اندیکاتورها برای {symbol}...')
            indicators_text = get_technical_indicators(symbol)
            bot.sendMessage(chat_id, indicators_text, parse_mode='Markdown')

        elif action == 'news':
            bot.sendMessage(chat_id, f'در حال جستجوی اخبار برای {symbol}...')
            news_text = get_symbol_news(symbol)
            bot.sendMessage(chat_id, news_text)

def handle_symbol_input(chat_id, text):
    """پردازش نماد ورودی کاربر"""
    symbol = text.strip().upper()
    try:
        # اعتبارسنجی نماد با یک درخواست ساده به بایننس
        client.get_symbol_ticker(symbol=symbol)
        is_valid = True
    except Exception:
        is_valid = False
    
    if is_valid:
        user_states[chat_id] = f'symbol_menu_{symbol}'
        bot.sendMessage(chat_id, f'نماد {symbol} تایید شد. کدام تحلیل را نیاز دارید؟',
                        reply_markup=get_symbol_analysis_keyboard(symbol))
    else:
        bot.sendMessage(chat_id, 'خطا: نماد وارد شده معتبر نیست یا در بایننس وجود ندارد.')

# --- راه‌اندازی ربات و وب‌سرور ---
def run_web_server():
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == '__main__':
    if not TELEGRAM_TOKEN:
        logging.fatal("TELEGRAM_TOKEN not found!")
    else:
        MessageLoop(bot, {'chat': handle_chat,
                          'callback_query': handle_callback_query}).run_as_thread()
        logging.info('Telepot bot is listening...')

        if os.getenv('RAILWAY_ENVIRONMENT'):
            threading.Thread(target=run_web_server, daemon=True).start()

        logging.info("Bot is running. Press Ctrl+C to exit.")
        while 1:
            time.sleep(10)