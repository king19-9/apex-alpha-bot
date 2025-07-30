# main.py (نسخه نهایی، بهینه‌سازی شده و مقاوم)

import os
import logging
import time
import telepot
from telepot.loop import MessageLoop
from telepot.namedtuple import InlineKeyboardMarkup, InlineKeyboardButton
import pandas as pd
from fastapi import FastAPI
import uvicorn
import threading
import ccxt
import ta
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
import io

# --- تنظیمات اولیه و متغیرهای محیطی ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

# --- کلاینت‌ها و سرویس‌ها ---
app = FastAPI()
exchange = ccxt.kucoin()
bot = telepot.Bot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
user_states = {}

# --- توابع سازنده کیبورد ---
def get_main_menu_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='📊 تحلیل جامع ارز', callback_data='menu_full_analysis')]
    ])

def get_back_to_main_menu_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='🔙 بازگشت به منوی اصلی', callback_data='main_menu')]
    ])

# --- توابع تحلیل ---

def fetch_data(symbol, timeframe, limit):
    """تابع متمرکز برای دریافت داده از صرافی"""
    try:
        kucoin_symbol = f"{symbol.upper()}/USDT"
        ohlcv = exchange.fetch_ohlcv(kucoin_symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logging.error(f"Failed to fetch data for {symbol}: {e}")
        return None

def create_chart(df, symbol):
    """ساخت چارت حرفه‌ای"""
    try:
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
        ap = [
            mpf.make_addplot(df['ema_50'], panel=0, color='blue', width=0.7),
            mpf.make_addplot(df['ema_200'], panel=0, color='orange', width=1.5),
        ]
        style = mpf.make_marketcolors(up='#26a69a', down='#ef5350', wick='inherit')
        mpf_style = mpf.make_mpf_style(marketcolors=style, base_mpf_style='nightclouds')
        buf = io.BytesIO()
        mpf.plot(df, type='candle', style=mpf_style,
                 title=f'\nتحلیل {symbol} (4H)',
                 volume=True, addplot=ap, panel_ratios=(4,1),
                 savefig=dict(fname=buf, dpi=100))
        buf.seek(0)
        return buf
    except Exception as e:
        logging.error(f"Failed to create chart for {symbol}: {e}")
        return None

def generate_analysis_text(df, symbol):
    """تولید تحلیل متنی"""
    try:
        # محاسبه اندیکاتورهای لازم
        df['rsi'] = ta.momentum.rsi(df['close'])
        df['macd_diff'] = ta.trend.macd_diff(df['close'])
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
        last_candle = df.iloc[-1]

        # بخش تحلیل کمی
        analysis = f"🧠 **تحلیل جامع برای #{symbol}**\n\n**--- تحلیل کمی (اندیکاتورها) ---**\n"
        if last_candle['ema_50'] > last_candle['ema_200']: analysis += "📈 **روند:** صعودی\n"
        else: analysis += "📉 **روند:** نزولی\n"
        if last_candle['rsi'] > 70: analysis += "🥵 **هیجان:** اشباع خرید\n"
        elif last_candle['rsi'] < 30: analysis += "🥶 **هیجان:** اشباع فروش\n"
        else: analysis += "😐 **هیجان:** خنثی\n"
        if last_candle['macd_diff'] > 0: analysis += "🟢 **قدرت:** خریداران غالب هستند\n"
        else: analysis += "🔴 **قدرت:** فروشندگان غالب هستند\n"
        
        # بخش پرایس اکشن
        analysis += "\n**--- تحلیل پرایس اکشن (ال بروکس) ---**\n"
        body_sizes = abs(df['close'] - df['open'])
        if body_sizes.iloc[-1] > body_sizes.rolling(20).mean().iloc[-1] * 1.5:
            analysis += "⚡️ **وضعیت:** بازار در یک روند قوی (Impulse Move) قرار دارد.\n"
        else:
            analysis += "⏸️ **وضعیت:** بازار در یک محدوده ترید (Trading Range) قرار دارد.\n"
        
        return analysis
    except Exception as e:
        logging.error(f"Failed to generate analysis text for {symbol}: {e}")
        return "خطا در تولید تحلیل متنی."


# --- کنترل‌کننده‌های ربات ---
def handle_chat(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type != 'text': return

    text = msg['text']
    
    if user_states.get(chat_id) == 'awaiting_symbol':
        handle_symbol_input(chat_id, text)
        return
        
    if text == '/start':
        user_states[chat_id] = 'main_menu'
        bot.sendMessage(chat_id, 'به ربات هوشمند Apex Pro خوش آمدید.',
                        reply_markup=get_main_menu_keyboard())

def handle_callback_query(msg):
    query_id, from_id, query_data = telepot.glance(msg, flavor='callback_query')
    chat_id = from_id
    bot.answerCallbackQuery(query_id)
    
    if query_data == 'main_menu':
        user_states[chat_id] = 'main_menu'
        bot.sendMessage(chat_id, 'منوی اصلی:', reply_markup=get_main_menu_keyboard())

    elif query_data == 'menu_full_analysis':
        user_states[chat_id] = 'awaiting_symbol'
        bot.sendMessage(chat_id, 'لطفاً نماد ارز را وارد کنید (مثلاً: BTC).',
                        reply_markup=get_back_to_main_menu_keyboard())

def handle_symbol_input(chat_id, text):
    symbol = text.strip().upper()
    
    # ۱. بازخورد فوری به کاربر
    bot.sendMessage(chat_id, f"✅ نماد {symbol} دریافت شد. در حال واکشی داده از صرافی...")
    
    # ۲. دریافت داده
    df = fetch_data(symbol, timeframe='4h', limit=200)
    if df is None or df.empty:
        bot.sendMessage(chat_id, f"خطا: داده‌ای برای نماد {symbol} یافت نشد. لطفاً از صحت نام نماد مطمئن شوید.", reply_markup=get_back_to_main_menu_keyboard())
        return

    # ۳. تولید تحلیل متنی
    bot.sendMessage(chat_id, "📊 داده‌ها دریافت شد. در حال تولید تحلیل متنی...")
    analysis_text = generate_analysis_text(df, symbol)
    bot.sendMessage(chat_id, analysis_text, parse_mode='Markdown')

    # ۴. ساخت و ارسال چارت
    bot.sendMessage(chat_id, "🖼 در حال ساخت نمودار حرفه‌ای...")
    chart_buffer = create_chart(df.tail(70), symbol)
    if chart_buffer:
        bot.sendPhoto(chat_id, chart_buffer)
    else:
        bot.sendMessage(chat_id, "خطا در ساخت نمودار.")

    # ۵. بازگرداندن کاربر به منوی اصلی
    user_states[chat_id] = 'main_menu'
    bot.sendMessage(chat_id, "تحلیل کامل شد. برای تحلیل جدید، از منوی اصلی شروع کنید.", reply_markup=get_main_menu_keyboard())

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