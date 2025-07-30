# main.py (نسخه Apex Pro: دستیار هوشمند بازار)

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
# برای این نسخه، دیتابیس فعلا استفاده نمی‌شود تا کد ساده‌تر باشد
# DATABASE_URL = os.getenv('DATABASE_URL')

# --- کلاینت‌ها و سرویس‌ها ---
app = FastAPI()
exchange = ccxt.kucoin()
bot = telepot.Bot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
user_states = {}

# --- توابع سازنده کیبورد ---
def get_main_menu_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='📊 تحلیل جامع ارز', callback_data='menu_full_analysis')],
        [InlineKeyboardButton(text='🐳 رادار نهنگ‌ها (آزمایشی)', callback_data='menu_whales')]
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

        # ۲. محاسبه اندیکاتورهای کلیدی
        df['rsi'] = ta.momentum.rsi(df['close'])
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['bb_high'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_low'] = ta.volatility.bollinger_lband(df['close'])
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)

        # ۳. ساخت چارت حرفه‌ای با mplfinance
        chart_buffer = create_professional_chart(df.tail(70), symbol) # نمایش ۷۰ کندل آخر

        # ۴. تولید تحلیل هوش مصنوعی
        ai_analysis_text = generate_ai_analysis(df, symbol)

        return chart_buffer, ai_analysis_text

    except Exception as e:
        logging.error(f"Error in comprehensive analysis for {symbol}: {e}")
        if isinstance(e, ccxt.BadSymbol):
            return None, "خطا: نماد وارد شده در صرافی یافت نشد."
        return None, "خطا در پردازش تحلیل. لطفاً بعداً تلاش کنید."


def create_professional_chart(df, symbol):
    """ساخت چارت کندل استیک حرفه‌ای با اندیکاتورها."""
    # آماده‌سازی داده برای mplfinance
    df_plot = df.copy()
    
    # اضافه کردن اندیکاتورها به پنل‌های جداگانه
    ap = [
        mpf.make_addplot(df_plot['ema_50'], panel=0, color='blue', width=0.7),
        mpf.make_addplot(df_plot['ema_200'], panel=0, color='orange', width=1.5),
        mpf.make_addplot(df_plot['bb_high'], panel=0, color='gray', linestyle='--'),
        mpf.make_addplot(df_plot['bb_low'], panel=0, color='gray', linestyle='--'),
        mpf.make_addplot(df_plot['rsi'], panel=1, color='purple', ylabel='RSI'),
        mpf.make_addplot(df_plot['macd'], type='bar', panel=2, color='green', ylabel='MACD')
    ]
    
    # استایل چارت
    style = mpf.make_marketcolors(up='green', down='red', wick={'up':'green','down':'red'})
    mpf_style = mpf.make_mpf_style(marketcolors=style, base_mpf_style='nightclouds')
    
    buf = io.BytesIO()
    mpf.plot(df_plot, type='candle', style=mpf_style,
             title=f'\nتحلیل تکنیکال {symbol} (4H)',
             volume=True, addplot=ap, panel_ratios=(6,2,2),
             savefig=dict(fname=buf, dpi=120))
    buf.seek(0)
    return buf


def generate_ai_analysis(df, symbol):
    """
    (شبیه‌سازی هوش مصنوعی)
    تفسیر داده‌های تکنیکال و اخبار برای ارائه یک تحلیل جامع.
    """
    # استخراج آخرین مقادیر
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    analysis = f"🧠 **تحلیل هوشمند برای #{symbol}**\n\n"
    
    # ۱. تحلیل روند کلی (بر اساس EMA)
    if last_row['ema_50'] > last_row['ema_200'] and prev_row['ema_50'] <= prev_row['ema_200']:
        analysis += "📈 **تغییر روند:** روند به تازگی صعودی شده (Golden Cross کوتاه‌مدت).\n"
    elif last_row['ema_50'] > last_row['ema_200']:
        analysis += "📈 **روند فعلی:** صعودی (EMA-50 بالاتر از EMA-200).\n"
    else:
        analysis += "📉 **روند فعلی:** نزولی (EMA-50 پایین‌تر از EMA-200).\n"
        
    # ۲. تحلیل نوسان و هیجان بازار (بر اساس RSI و Bollinger Bands)
    if last_row['rsi'] > 70:
        analysis += "🥵 **هیجان خرید:** بازار در حالت اشباع خرید است (RSI > 70). احتمال اصلاح قیمت وجود دارد.\n"
    elif last_row['rsi'] < 30:
        analysis += "🥶 **هیجان فروش:** بازار در حالت اشباع فروش است (RSI < 30). احتمال بازگشت قیمت به بالا وجود دارد.\n"
    
    if last_row['close'] > last_row['bb_high']:
        analysis += "💥 **شکست نوسان:** قیمت باند بالای بولینگر را شکسته. این نشانه قدرت زیاد خریداران است اما می‌تواند ناپایدار باشد (Offering Fair Value).\n"
    elif last_row['close'] < last_row['bb_low']:
        analysis += "💧 **تست نقدینگی:** قیمت به باند پایینی بولینگر برخورد کرده. این می‌تواند نشانه جستجو برای نقدینگی (Liquidity Run) در سطوح پایین‌تر باشد.\n"
        
    # ۳. تحلیل قدرت حرکت (بر اساس MACD)
    if last_row['macd'] > 0 and prev_row['macd'] < 0:
        analysis += "🟢 **قدرت حرکت:** MACD به تازگی مثبت شده، نشان‌دهنده افزایش قدرت خریداران است.\n"
    elif last_row['macd'] < 0 and prev_row['macd'] > 0:
        analysis += "🔴 **قدرت حرکت:** MACD به تازگی منفی شده، نشان‌دهنده افزایش قدرت فروشندگان است.\n"
        
    # ۴. تحلیل سشن‌ها و تله‌ها (مفهومی و شبیه‌سازی شده)
    # برای این بخش به داده‌های دقیق‌تر و ساعت فعلی نیاز است
    current_hour_utc = pd.Timestamp.utcnow().hour
    session_info = ""
    if 4 <= current_hour_utc < 12: # سشن لندن (تقریبی)
        session_info = "در سشن لندن هستیم که معمولاً نقدینگی و نوسان بالایی دارد. مراقب حرکات فیک برای شکار حد ضرر (Trapping) باشید.\n"
    elif 13 <= current_hour_utc < 21: # همپوشانی لندن و نیویورک
        session_info = "در همپوشانی سشن‌های لندن و نیویورک هستیم، پرنوسان‌ترین زمان بازار. احتمال Liquidity Sweep بالاست.\n"
    analysis += f"🕰️ **تحلیل سشن:** {session_info}"
    
    # ۵. تحلیل اخبار (شبیه‌سازی شده)
    # news_sentiment = get_news_sentiment(symbol) # این تابع باید به NewsAPI وصل شود
    news_sentiment = "خنثی" # مقدار نمایشی
    analysis += f"📰 **احساسات اخبار:** {news_sentiment}.\n"
    
    # ۶. جمع‌بندی نهایی
    analysis += "\n**جمع‌بندی استراتژیک:**\n"
    # این بخش می‌تواند یک مدل AI واقعی باشد که تمام متغیرها را ترکیب می‌کند
    # در اینجا یک منطق ساده پیاده‌سازی شده است
    bullish_score = 0
    if 'صعودی' in analysis: bullish_score += 2
    if 'اشباع فروش' in analysis: bullish_score += 1.5
    if 'MACD به تازگی مثبت' in analysis: bullish_score += 1
    if 'شکست نوسان' in analysis and 'صعودی' in analysis: bullish_score += 1
    
    bearish_score = 0
    if 'نزولی' in analysis: bearish_score += 2
    if 'اشباع خرید' in analysis: bearish_score += 1.5
    if 'MACD به تازگی منفی' in analysis: bearish_score += 1
    
    if bullish_score > bearish_score + 1:
        analysis += "با توجه به نشانه‌های متعدد، سناریوی صعودی محتمل‌تر به نظر می‌رسد. سطوح حمایتی کلیدی را زیر نظر داشته باشید."
    elif bearish_score > bullish_score + 1:
        analysis += "با توجه به نشانه‌های متعدد، سناریوی نزولی محتمل‌تر به نظر می‌رسد. سطوح مقاومتی کلیدی را زیر نظر داشته باشید."
    else:
        analysis += "بازار در حال حاضر در شرایط خنثی یا نامشخص قرار دارد. بهتر است منتظر یک سیگنال واضح‌تر بمانید."
        
    return analysis

# --- کنترل‌کننده‌های ربات ---

def handle_chat(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type != 'text': return

    text = msg['text']
    
    if user_states.get(chat_id) == 'awaiting_symbol':
        # پاک کردن پیام قبلی ربات (درخواست ورود نماد)
        if 'last_message' in user_states.get(chat_id, {}):
            try:
                bot.deleteMessage((chat_id, user_states[chat_id]['last_message']))
            except: pass
        
        # نمایش پیام "در حال پردازش"
        sent_message = bot.sendMessage(chat_id, f"در حال پردازش نماد {text.upper()}...")
        
        # شروع تحلیل
        handle_symbol_input(chat_id, text, sent_message['message_id'])
        return
        
    if text == '/start':
        user_states[chat_id] = {'state': 'main_menu'}
        bot.sendMessage(chat_id, 'به ربات هوشمند Apex Pro خوش آمدید. چه بخشی را می‌خواهید بررسی کنید؟',
                        reply_markup=get_main_menu_keyboard())

def handle_callback_query(msg):
    query_id, from_id, query_data = telepot.glance(msg, flavor='callback_query')
    chat_id = from_id
    bot.answerCallbackQuery(query_id)
    
    if query_data == 'main_menu':
        user_states[chat_id] = {'state': 'main_menu'}
        bot.editMessageText((chat_id, msg['message']['message_id']), 'منوی اصلی:', reply_markup=get_main_menu_keyboard())

    elif query_data == 'menu_full_analysis':
        sent_message = bot.sendMessage(chat_id, 'لطفاً نماد ارز مورد نظر خود را با فرمت صحیح وارد کنید (مثلاً: BTC).',
                        reply_markup=get_back_to_main_menu_keyboard())
        user_states[chat_id] = {'state': 'awaiting_symbol', 'last_message': sent_message['message_id']}

    elif query_data == 'menu_whales':
        message = "🐳 **رادار نهنگ‌ها (نسخه آزمایشی)**\nاین بخش در حال حاضر در حال جمع‌آوری داده است و به زودی نتایج اولیه را نمایش خواهد داد."
        bot.editMessageText((chat_id, msg['message']['message_id']), message, reply_markup=get_back_to_main_menu_keyboard())


def handle_symbol_input(chat_id, text, processing_message_id):
    symbol = text.strip().upper()
    
    chart, analysis = get_comprehensive_analysis(symbol)
    
    bot.deleteMessage((chat_id, processing_message_id)) # حذف پیام "در حال پردازش"
    
    if chart and analysis:
        bot.sendPhoto(chat_id, chart, caption=analysis, parse_mode='Markdown')
        # بعد از ارسال تحلیل، کاربر را به منوی اصلی برمی‌گردانیم
        bot.sendMessage(chat_id, "برای تحلیل جدید، لطفاً دوباره از منوی اصلی شروع کنید.", reply_markup=get_main_menu_keyboard())
        user_states[chat_id] = {'state': 'main_menu'}
    else:
        bot.sendMessage(chat_id, analysis, reply_markup=get_back_to_main_menu_keyboard())


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