# main.py (نسخه نهایی با KuCoin و ccxt)

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
import ccxt # <--- کتابخانه جدید
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

# --- کلاینت‌ها و سرویس‌ها ---
app = FastAPI()
# اتصال به صرافی KuCoin با استفاده از ccxt
exchange = ccxt.kucoin()
engine = create_engine(DATABASE_URL) if DATABASE_URL else None
bot = telepot.Bot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
user_states = {}

# --- توابع سازنده کیبوردها (بدون تغییر) ---
def get_main_menu_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='📊 تحلیل تکنیکال', callback_data='menu_tech_analysis')],
        [InlineKeyboardButton(text='📰 اخبار', callback_data='menu_news')],
    ])
def get_symbol_analysis_keyboard(symbol):
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='📈 نمایش چارت', callback_data=f'action_chart_{symbol}')],
        [InlineKeyboardButton(text='📉 اندیکاتورها', callback_data=f'action_indicators_{symbol}')],
        [InlineKeyboardButton(text='🔙 بازگشت', callback_data='menu_tech_analysis')]
    ])
def get_back_to_main_menu_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='🔙 بازگشت به منوی اصلی', callback_data='main_menu')]
    ])

# --- توابع اجرایی (بازنویسی شده برای KuCoin) ---

def get_price_chart(symbol):
    try:
        # ccxt نمادها را با / می‌خواهد (مثال: BTC/USDT)
        kucoin_symbol = symbol.replace('USDT', '/USDT')
        # دریافت کندل‌های ۱ ساعته
        ohlcv = exchange.fetch_ohlcv(kucoin_symbol, timeframe='1h', limit=24)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        
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
    try:
        kucoin_symbol = symbol.replace('USDT', '/USDT')
        # دریافت کندل‌های ۴ ساعته
        ohlcv = exchange.fetch_ohlcv(kucoin_symbol, timeframe='4h', limit=250)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        if len(df) < 200:
            return "داده کافی برای تحلیل بلندمدت وجود ندارد."

        rsi = ta.momentum.rsi(df['close'], window=14).iloc[-1]
        macd = ta.trend.macd_diff(df['close']).iloc[-1]
        ema200 = ta.trend.ema_indicator(df['close'], window=200).iloc[-1]
        last_price = df['close'].iloc[-1]
        
        message = (f"🔎 **تحلیل تکنیکال برای #{symbol} (از KuCoin)**\n\n"
                   f"**قیمت فعلی:** `${last_price:,.2f}`\n"
                   f"**RSI (14):** `{rsi:.2f}`\n"
                   f"**MACD Histogram:** `{macd:.2f}`\n"
                   f"**EMA (200):** `${ema200:,.2f}`")
        return message
    except Exception as e:
        logging.error(f"Error getting indicators for {symbol}: {e}")
        return "خطا در محاسبه اندیکاتورها."

# --- کنترل‌کننده‌های ربات (تقریباً بدون تغییر) ---
# ... (کدهای handle_chat, handle_callback_query, handle_symbol_input اینجا قرار دارند) ...
# من فقط handle_symbol_input را برای اعتبارسنجی با ccxt تغییر می‌دهم
def handle_chat(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type != 'text': return
    text = msg['text']
    if user_states.get(chat_id) == 'awaiting_symbol':
        handle_symbol_input(chat_id, text)
        return
    if text == '/start':
        user_states[chat_id] = 'main_menu'
        bot.sendMessage(chat_id, 'سلام! به ربات Apex خوش آمدید.', reply_markup=get_main_menu_keyboard())

def handle_callback_query(msg):
    query_id, from_id, query_data = telepot.glance(msg, flavor='callback_query')
    chat_id = from_id
    bot.answerCallbackQuery(query_id)
    if query_data == 'main_menu':
        user_states[chat_id] = 'main_menu'
        bot.sendMessage(chat_id, 'منوی اصلی:', reply_markup=get_main_menu_keyboard())
    elif query_data == 'menu_tech_analysis':
        user_states[chat_id] = 'awaiting_symbol'
        bot.sendMessage(chat_id, 'لطفاً نماد ارز را وارد کنید (مثلاً: BTCUSDT).', reply_markup=get_back_to_main_menu_keyboard())
    elif query_data.startswith('action_'):
        action, symbol = query_data.split('_', 2)[1:]
        if action == 'chart':
            bot.sendMessage(chat_id, f'در حال آماده‌سازی چارت برای {symbol} از KuCoin...')
            chart_image = get_price_chart(symbol)
            if chart_image: bot.sendPhoto(chat_id, chart_image)
            else: bot.sendMessage(chat_id, "خطا در ساخت نمودار.")
        elif action == 'indicators':
            bot.sendMessage(chat_id, f'در حال واکشی اندیکاتورها برای {symbol} از KuCoin...')
            indicators_text = get_technical_indicators(symbol)
            bot.sendMessage(chat_id, indicators_text, parse_mode='Markdown')

def handle_symbol_input(chat_id, text):
    symbol = text.strip().upper()
    kucoin_symbol = symbol.replace('USDT', '/USDT')
    try:
        # اعتبارسنجی نماد با ccxt
        exchange.load_markets()
        if kucoin_symbol in exchange.markets:
            is_valid = True
        else:
            is_valid = False
    except Exception:
        is_valid = False
    
    if is_valid:
        user_states[chat_id] = f'symbol_menu_{symbol}'
        bot.sendMessage(chat_id, f'نماد {symbol} تایید شد. کدام تحلیل را نیاز دارید؟', reply_markup=get_symbol_analysis_keyboard(symbol))
    else:
        bot.sendMessage(chat_id, f'خطا: نماد {symbol} در صرافی KuCoin یافت نشد.')


# --- راه‌اندازی ربات و وب‌سرور (بدون تغییر) ---
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