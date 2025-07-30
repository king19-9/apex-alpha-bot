# main.py (نسخه نهایی با تحلیل پرایس اکشن ال بروکس)

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
        [InlineKeyboardButton(text='📊 تحلیل جامع پرایس اکشن', callback_data='menu_full_analysis')]
    ])

def get_back_to_main_menu_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='🔙 بازگشت به منوی اصلی', callback_data='main_menu')]
    ])

# --- توابع تحلیل پیشرفته ---

def get_comprehensive_analysis(symbol):
    try:
        kucoin_symbol = f"{symbol.upper()}/USDT"
        ohlcv = exchange.fetch_ohlcv(kucoin_symbol, timeframe='4h', limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        chart_buffer = create_price_action_chart(df.tail(70), symbol)
        ai_analysis_text = generate_al_brooks_analysis(df, symbol)

        return chart_buffer, ai_analysis_text
    except Exception as e:
        logging.error(f"Error in comprehensive analysis for {symbol}: {e}")
        if isinstance(e, ccxt.BadSymbol): return None, "خطا: نماد وارد شده در صرافی یافت نشد."
        return None, "خطا در پردازش تحلیل. لطفاً بعداً تلاش کنید."

def create_price_action_chart(df, symbol):
    # این تابع مشابه قبل است و نیازی به تغییر ندارد
    ap = [mpf.make_addplot(df['close'].rolling(20).mean(), panel=0, color='blue', width=0.7)]
    style = mpf.make_marketcolors(up='green', down='red', wick={'up':'green','down':'red'})
    mpf_style = mpf.make_mpf_style(marketcolors=style, base_mpf_style='nightclouds')
    buf = io.BytesIO()
    mpf.plot(df, type='candle', style=mpf_style,
             title=f'\nتحلیل پرایس اکشن {symbol} (4H)',
             volume=True, addplot=ap, panel_ratios=(4,1),
             savefig=dict(fname=buf, dpi=120))
    buf.seek(0)
    return buf

def generate_al_brooks_analysis(df, symbol):
    """(شبیه‌سازی هوشمند) تفسیر پرایس اکشن به سبک ال بروکس"""
    analysis = f"🧠 **تحلیل پرایس اکشن برای #{symbol} (سبک ال بروکس)**\n\n"
    
    # استخراج ۵ کندل آخر برای تحلیل دقیق
    last_5_candles = df.iloc[-5:]
    last_candle = last_5_candles.iloc[-1]
    
    # ۱. تشخیص حالت کلی بازار (روند یا رنج)
    body_sizes = abs(df['close'] - df['open'])
    avg_body_size = body_sizes.rolling(20).mean().iloc[-1]
    is_trending = body_sizes.iloc[-1] > avg_body_size * 1.5 # اگر کندل آخر بزرگ باشد، نشانه روند است

    if is_trending:
        analysis += "📈 **وضعیت بازار:** در یک روند قوی (Strong Trend) قرار داریم. کندل‌های اخیر بدنه‌های بزرگی دارند.\n"
    else:
        analysis += "↔️ **وضعیت بازار:** در یک محدوده ترید (Trading Range) قرار داریم. بازار در حال حاضر بلاتکلیف است.\n"
        
    # ۲. شناسایی کندل سیگنال (Signal Bar) در کندل آخر
    body_size = abs(last_candle['close'] - last_candle['open'])
    candle_range = last_candle['high'] - last_candle['low']
    upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
    lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']

    is_bullish_pin_bar = lower_wick > body_size * 2 and upper_wick < body_size
    is_bearish_pin_bar = upper_wick > body_size * 2 and lower_wick < body_size
    
    if is_bullish_pin_bar:
        analysis += "🐂 **کندل سیگنال:** یک پین‌بار صعودی (Bullish Pin Bar) شناسایی شد. این کندل نشان‌دهنده رد شدن قیمت‌های پایین‌تر توسط خریداران و احتمال صعود است.\n"
    elif is_bearish_pin_bar:
        analysis += "🐻 **کندل سیگنال:** یک پین‌بار نزولی (Bearish Pin Bar) شناسایی شد. این کندل نشان‌دهنده رد شدن قیمت‌های بالاتر توسط فروشندگان و احتمال نزول است.\n"

    # ۳. تحلیل فشار خرید و فروش
    recent_closes = last_5_candles['close'].values
    if all(recent_closes[i] <= recent_closes[i+1] for i in range(len(recent_closes)-1)):
        analysis += "🟢 **فشار بازار:** فشار خرید در ۵ کندل اخیر غالب بوده است (Consecutive Bull Bars).\n"
    elif all(recent_closes[i] >= recent_closes[i+1] for i in range(len(recent_closes)-1)):
        analysis += "🔴 **فشار بازار:** فشار فروش در ۵ کندل اخیر غالب بوده است (Consecutive Bear Bars).\n"

    # ۴. جمع‌بندی استراتژیک به سبک ال بروکس
    analysis += "\n**جمع‌بندی استراتژیک:**\n"
    if is_trending and is_bullish_pin_bar:
        analysis += "در یک روند صعودی، یک کندل سیگنال خرید ظاهر شده. استراتژی مناسب، خرید در بالای این کندل سیگنال با هدف ادامه روند است (Trend Continuation)."
    elif is_trending and is_bearish_pin_bar:
        analysis += "در یک روند نزولی، یک کندل سیگنال فروش ظاهر شده. استراتژی مناسب، فروش در پایین این کندل سیگنال با هدف ادامه روند است."
    elif not is_trending and is_bullish_pin_bar:
        analysis += "در یک محدوده ترید، یک پین‌بار صعودی در کف محدوده می‌تواند نشانه خوبی برای خرید با هدف رسیدن به سقف محدوده باشد (Range Trading)."
    elif not is_trending and is_bearish_pin_bar:
        analysis += "در یک محدوده ترید، یک پین‌بار نزولی در سقف محدوده می‌تواند نشانه خوبی برای فروش با هدف رسیدن به کف محدوده باشد."
    else:
        analysis += "در حال حاضر هیچ سیگنال واضحی (High Probability Setup) وجود ندارد. بهترین استراتژی صبر کردن است. بازار همیشه فرصت دیگری خواهد داد."
        
    return analysis

# --- کنترل‌کننده‌های ربات (اصلاح شده) ---

def handle_chat(msg):
    """پردازش پیام‌های متنی"""
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type != 'text': return

    text = msg['text']
    
    if user_states.get(chat_id) == 'awaiting_symbol':
        # پاک کردن پیام "لطفا نماد را وارد کنید"
        if 'last_message_id' in user_states.get(chat_id, {}):
            try: bot.deleteMessage((chat_id, user_states[chat_id]['last_message_id']))
            except: pass
        # پاک کردن پیامی که کاربر فرستاده (نام نماد)
        try: bot.deleteMessage((chat_id, msg['message_id']))
        except: pass
        
        processing_message = bot.sendMessage(chat_id, f"در حال پردازش نماد {text.upper()}...")
        handle_symbol_input(chat_id, text, processing_message['message_id'])
        return
        
    if text == '/start':
        user_states[chat_id] = {'state': 'main_menu'}
        bot.sendMessage(chat_id, 'به ربات هوشمند Apex Pro خوش آمدید. برای دریافت تحلیل، لطفاً گزینه زیر را انتخاب کنید:',
                        reply_markup=get_main_menu_keyboard())

def handle_callback_query(msg):
    """پردازش کلیک روی دکمه‌ها"""
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
    """پردازش نماد ورودی کاربر"""
    symbol = text.strip().upper()
    chart_buffer, analysis_text = get_comprehensive_analysis(symbol)
    
    bot.deleteMessage((chat_id, processing_message_id))
    
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