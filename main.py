# main.py (نسخه نهایی: ترکیب تحلیل اندیکاتورها و پرایس اکشن ال بروکس)

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
import requests
import ccxt
import ta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import io

# --- تنظیمات اولیه و متغیرهای محیطی ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

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

# --- توابع تحلیل پیشرفته ---

def get_comprehensive_analysis(symbol):
    """تابع اصلی که تمام تحلیل‌ها را جمع‌آوری و بسته‌بندی می‌کند."""
    try:
        kucoin_symbol = f"{symbol.upper()}/USDT"
        
        # ۱. دریافت داده‌های کندل (تایم‌فریم ۴ ساعته برای تحلیل جامع)
        ohlcv = exchange.fetch_ohlcv(kucoin_symbol, timeframe='4h', limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # ۲. محاسبه اندیکاتورها برای هر دو تحلیل
        df = calculate_indicators(df)

        # ۳. ساخت چارت حرفه‌ای با mplfinance
        chart_buffer = create_professional_chart(df.tail(70), symbol)

        # ۴. تولید تحلیل هوش مصنوعی (ترکیبی)
        ai_analysis_text = generate_combined_analysis(df, symbol)

        return chart_buffer, ai_analysis_text

    except Exception as e:
        logging.error(f"Error in comprehensive analysis for {symbol}: {e}")
        if isinstance(e, ccxt.BadSymbol):
            return None, "خطا: نماد وارد شده در صرافی یافت نشد."
        return None, "خطا در پردازش تحلیل. لطفاً بعداً تلاش کنید."

def calculate_indicators(df):
    """یکجا تمام اندیکاتورهای لازم را محاسبه می‌کند."""
    df['rsi'] = ta.momentum.rsi(df['close'])
    df['macd_diff'] = ta.trend.macd_diff(df['close'])
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
    return df

def create_professional_chart(df, symbol):
    """ساخت چارت حرفه‌ای با اندیکاتورهای کلیدی."""
    ap = [
        mpf.make_addplot(df['ema_50'], panel=0, color='blue', width=0.7, ylabel='Price'),
        mpf.make_addplot(df['ema_200'], panel=0, color='orange', width=1.5),
        mpf.make_addplot(df['rsi'], panel=1, color='purple', ylabel='RSI'),
        mpf.make_addplot(df['macd_diff'], type='bar', panel=2, color='green', ylabel='MACD Hist')
    ]
    
    style = mpf.make_marketcolors(up='#26a69a', down='#ef5350', wick='inherit')
    mpf_style = mpf.make_mpf_style(marketcolors=style, base_mpf_style='nightclouds')
    
    buf = io.BytesIO()
    mpf.plot(df, type='candle', style=mpf_style,
             title=f'\nتحلیل جامع {symbol} (4H)',
             volume=True, addplot=ap, panel_ratios=(6,2,2),
             savefig=dict(fname=buf, dpi=120),
             datetime_format='%b %d, %H:%M')
    buf.seek(0)
    return buf


def generate_combined_analysis(df, symbol):
    """تفسیر ترکیبی اندیکاتورها و پرایس اکشن به سبک ال بروکس"""
    analysis = f"🧠 **تحلیل جامع برای #{symbol}**\n\n"
    last_candle = df.iloc[-1]
    
    # --- بخش ۱: تحلیل مبتنی بر اندیکاتور ---
    analysis += "**--- تحلیل کمی (اندیکاتورها) ---**\n"
    # روند
    if last_candle['ema_50'] > last_candle['ema_200']:
        analysis += "📈 **روند:** صعودی (میانگین متحرک کوتاه مدت بالاتر از بلند مدت است).\n"
    else:
        analysis += "📉 **روند:** نزولی (میانگین متحرک کوتاه مدت پایین‌تر از بلند مدت است).\n"
    # هیجان بازار
    if last_candle['rsi'] > 70:
        analysis += "🥵 **هیجان:** بازار در حالت اشباع خرید است (RSI > 70).\n"
    elif last_candle['rsi'] < 30:
        analysis += "🥶 **هیجان:** بازار در حالت اشباع فروش است (RSI < 30).\n"
    else:
        analysis += "😐 **هیجان:** خنثی (RSI در محدوده نرمال).\n"
    # قدرت حرکت
    if last_candle['macd_diff'] > 0:
        analysis += "🟢 **قدرت:** قدرت خریداران در حال افزایش است (MACD مثبت).\n"
    else:
        analysis += "🔴 **قدرت:** قدرت فروشندگان در حال افزایش است (MACD منفی).\n"
        
    # --- بخش ۲: تحلیل پرایس اکشن (ال بروکس) ---
    analysis += "\n**--- تحلیل پرایس اکشن (ال بروکس) ---**\n"
    # تشخیص روند یا رنج
    body_sizes = abs(df['close'] - df['open'])
    avg_body_size = body_sizes.rolling(20).mean().iloc[-1]
    if body_sizes.iloc[-1] > avg_body_size * 1.5:
        analysis += "📈 **وضعیت:** بازار در یک روند قوی (Strong Trend) قرار دارد.\n"
    else:
        analysis += "↔️ **وضعیت:** بازار در یک محدوده ترید (Trading Range) قرار دارد.\n"
    
    # شناسایی کندل سیگنال
    body_size = abs(last_candle['close'] - last_candle['open'])
    candle_range = last_candle['high'] - last_candle['low']
    upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
    lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
    if lower_wick > body_size * 2 and upper_wick < body_size:
        analysis += "🐂 **کندل سیگنال:** یک پین‌بار صعودی قوی (Bullish Pin Bar) شناسایی شد.\n"
    elif upper_wick > body_size * 2 and lower_wick < body_size:
        analysis += "🐻 **کندل سیگنال:** یک پین‌بار نزولی قوی (Bearish Pin Bar) شناسایی شد.\n"
    else:
        analysis += "캔 **کندل سیگنال:** کندل آخر یک سیگنال پرایس اکشن واضح نیست.\n"
        
    # --- بخش ۳: جمع‌بندی استراتژیک (ترکیب دو تحلیل) ---
    analysis += "\n**--- جمع‌بندی استراتژیک ---**\n"
    # منطق ترکیبی (شبیه‌سازی AI)
    if (last_candle['ema_50'] > last_candle['ema_200']) and (lower_wick > body_size * 2) and (last_candle['rsi'] < 50):
        analysis += "✅ **نتیجه:** **سیگنال خرید با احتمال بالا.** ترکیب روند صعودی کلی با یک کندل سیگنال پرایس اکشن قوی و عدم اشباع خرید، یک فرصت مناسب را نشان می‌دهد."
    elif (last_candle['ema_50'] < last_candle['ema_200']) and (upper_wick > body_size * 2) and (last_candle['rsi'] > 50):
        analysis += "❌ **نتیجه:** **سیگنال فروش با احتمال بالا.** ترکیب روند نزولی کلی با یک کندل سیگنال پرایس اکشن قوی و عدم اشباع فروش، یک فرصت مناسب را نشان می‌دهد."
    else:
        analysis += "⚠️ **نتیجه:** **شرایط نامشخص.** سیگنال‌های اندیکاتورها و پرایس اکشن در حال حاضر با یکدیگر همخوانی ندارند. بهترین استراتژی، صبر کردن برای یک سیگنال واضح‌تر است."
        
    return analysis

# --- کنترل‌کننده‌های ربات ---
def handle_chat(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type != 'text': return
    text = msg['text']
    
    if user_states.get(chat_id) == 'awaiting_symbol':
        # حذف پیام‌های قبلی برای تمیز ماندن چت
        if 'last_message_id' in user_states.get(chat_id, {}):
            try: bot.deleteMessage((chat_id, user_states[chat_id]['last_message_id']))
            except: pass
        try: bot.deleteMessage((chat_id, msg['message_id']))
        except: pass
        
        processing_message = bot.sendMessage(chat_id, f"در حال پردازش نماد {text.upper()}... لطفاً صبر کنید.")
        handle_symbol_input(chat_id, text, processing_message['message_id'])
        return
        
    if text == '/start':
        user_states[chat_id] = {'state': 'main_menu'}
        bot.sendMessage(chat_id, 'به ربات هوشمند Apex Pro خوش آمدید. برای دریافت تحلیل، لطفاً گزینه زیر را انتخاب کنید:',
                        reply_markup=get_main_menu_keyboard())

def handle_callback_query(msg):
    query_id, from_id, query_data = telepot.glance(msg, flavor='callback_query')
    chat_id = from_id
    bot.answerCallbackQuery(query_id)
    
    if query_data == 'main_menu':
        user_states[chat_id] = {'state': 'main_menu'}
        bot.editMessageText((chat_id, msg['message']['message_id']), 'منوی اصلی:', reply_markup=get_main_menu_keyboard())

    elif query_data == 'menu_full_analysis':
        sent_msg = bot.editMessageText((chat_id, msg['message']['message_id']),
                                       'لطفاً نماد ارز مورد نظر خود را با فرمت صحیح وارد کنید (مثلاً: BTC).',
                                       reply_markup=get_back_to_main_menu_keyboard())
        user_states[chat_id] = {'state': 'awaiting_symbol', 'last_message_id': sent_msg['message_id']}

def handle_symbol_input(chat_id, text, processing_message_id):
    symbol = text.strip().replace('/', '').replace('-', '').upper() # تمیز کردن ورودی کاربر
    chart_buffer, analysis_text = get_comprehensive_analysis(symbol)
    
    try:
        bot.deleteMessage((chat_id, processing_message_id))
    except:
        pass # اگر پیام قبلا حذف شده باشد
    
    if chart_buffer and analysis_text:
        bot.sendPhoto(chat_id, chart_buffer, caption=analysis_text, parse_mode='Markdown')
        user_states[chat_id] = {'state': 'main_menu'}
    else:
        bot.sendMessage(chat_id, analysis_text, reply_markup=get_back_to_main_menu_keyboard())

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