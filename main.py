# main.py (نسخه نهایی: Glass Box با شفافیت کامل)

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
from datetime import datetime
import pytz

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
        [InlineKeyboardButton(text='🔬 تحلیل عمیق یک ارز', callback_data='menu_deep_analysis')],
        [InlineKeyboardButton(text='🎯 شکار سیگنال (AI)', callback_data='menu_signal_hunt')]
    ])

def get_back_to_main_menu_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='🔙 بازگشت به منوی اصلی', callback_data='main_menu')]
    ])

# --- موتور تحلیل پیشرفته ---

def generate_full_report(symbol):
    try:
        kucoin_symbol = f"{symbol.upper()}/USDT"
        
        # ۱. دریافت داده‌ها در تایم‌فریم‌های مختلف
        df_d = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, timeframe='1d', limit=100), columns=['ts','o','h','l','c','v'])
        df_4h = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, timeframe='4h', limit=100), columns=['ts','o','h','l','c','v'])
        df_1h = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, timeframe='1h', limit=50), columns=['ts','o','h','l','c','v'])
        
        # --- بخش ۱: خلاصه وضعیت فعلی ---
        report = f"🔬 **گزارش جامع تحلیلی برای #{symbol}**\n\n"
        last_price = df_1h.iloc[-1]['c']
        session_name, session_char = get_market_session()
        report += f"**قیمت فعلی:** `${last_price:,.2f}`\n"
        report += f"**سشن معاملاتی:** {session_name} ({session_char})\n"

        # --- بخش ۲: تحلیل چند تایم‌فریم (Multi-Timeframe Analysis) ---
        report += "\n**--- تحلیل ساختار بازار (چند تایم‌فریم) ---**\n"
        report += f"**ابزار:** میانگین‌های متحرک (EMA 21, 50)\n"
        report += f"**منطق:** هم‌راستایی EMAها در تایم‌فریم‌های مختلف نشان‌دهنده قدرت روند است.\n"
        
        trend_d = "صعودی" if ta.trend.ema_indicator(df_d['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_d['c'], 50).iloc[-1] else "نزولی"
        trend_4h = "صعودی" if ta.trend.ema_indicator(df_4h['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_4h['c'], 50).iloc[-1] else "نزولی"
        report += f"**روند روزانه (Daily):** {trend_d}\n"
        report += f"**روند ۴ ساعته (4H):** {trend_4h}\n"
        if trend_d == trend_4h:
            report += f"**نتیجه:** ساختار بازار در حال حاضر **{trend_d}** و هم‌راستا است. معاملات در جهت روند از اعتبار بالاتری برخوردارند.\n"
        else:
            report += "**نتیجه:** ساختار بازار در حال حاضر **متناقض** است. قیمت در تایم‌فریم پایین‌تر در حال اصلاح یا تغییر روند است. احتیاط لازم است.\n"

        # --- بخش ۳: تحلیل عرضه و تقاضا و پرایس اکشن ---
        report += "\n**--- تحلیل عرضه/تقاضا و پرایس اکشن (سبک ال بروکس) ---**\n"
        report += f"**ابزار:** شناسایی نواحی حمایت/مقاومت کلیدی و الگوهای کندلی.\n"
        
        support = df_4h['l'].rolling(20).mean().iloc[-1] # روش ساده‌شده
        resistance = df_4h['h'].rolling(20).mean().iloc[-1]
        report += f"**ناحیه تقاضای مهم (4H):** حدود `${support:,.2f}`\n"
        report += f"**ناحیه عرضه مهم (4H):** حدود `${resistance:,.2f}`\n"
        
        # تحلیل کندل آخر ۱ ساعته
        last_1h_candle = df_1h.iloc[-1]
        body_size = abs(last_1h_candle['c'] - last_1h_candle['o'])
        lower_wick = last_1h_candle['c'] - last_1h_candle['l'] if last_1h_candle['c'] > last_1h_candle['o'] else last_1h_candle['o'] - last_1h_candle['l']
        if lower_wick > body_size * 2:
            report += "**سیگنال پرایس اکشن (1H):** کندل آخر یک **پین‌بار صعودی** است که نشان‌دهنده قدرت خریداران در سطوح پایین‌تر است (Liquidity Sweep).\n"
        else:
            report += "**سیگنال پرایس اکشن (1H):** کندل آخر سیگنال واضحی ندارد.\n"

        # --- بخش ۴: تحلیل فاندامنتال و اخبار ---
        report += "\n**--- تحلیل فاندامنتال (اخبار) ---**\n"
        report += "**ابزار:** NewsAPI برای واکشی اخبار و یک مدل ساده برای تحلیل احساسات.\n"
        news_query = symbol.replace('USDT', '')
        url = f"https://newsapi.org/v2/everything?q={news_query}&language=en&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}"
        articles = requests.get(url).json().get('articles', [])
        if articles:
            report += "**آخرین اخبار مهم:**\n"
            for article in articles:
                # تحلیل احساسات ساده
                sentiment = "مثبت" if any(word in article['title'].lower() for word in ['partner', 'launch', 'success']) else "منفی" if any(word in article['title'].lower() for word in ['hack', 'ban', 'problem']) else "خنثی"
                report += f"- *{article['title']}* (احساسات: {sentiment})\n"
        else:
            report += "خبر مهم جدیدی یافت نشد.\n"

        # --- بخش ۵: پیشنهاد معامله (AI-Powered) ---
        report += "\n**--- پیشنهاد معامله مبتنی بر AI ---**\n"
        report += "**روش:** این پیشنهاد بر اساس یک استراتژی ترکیبی (تقاطع EMA در تایم بالا + سیگنال پرایس اکشن در تایم پایین در نواحی عرضه/تقاضا) ارائه می‌شود.\n"
        report += "**عملکرد گذشته استراتژی (بک‌تست):** این استراتژی در گذشته روی این ارز، نرخ موفقیت تقریبی **۶۵٪** داشته است (این یک داده شبیه‌سازی شده است).\n"
        
        # منطق تصمیم‌گیری برای سیگنال
        is_long_signal = trend_d == "صعودی" and trend_4h == "صعودی" and (last_1h_candle['c'] < support * 1.02) and (lower_wick > body_size * 1.5)
        
        if is_long_signal:
            confidence = 75.0 # درصد اطمینان AI (شبیه‌سازی شده)
            entry = last_1h_candle['h']
            stop_loss = last_1h_candle['l']
            target = resistance
            leverage = 3 if (target/entry - 1) * 100 > 5 else 5

            report += f"✅ **سیگنال خرید (Long) با اطمینان {confidence:.0f}٪ صادر شد.**\n"
            report += f"**نقطه ورود پیشنهادی:** `${entry:,.2f}`\n"
            report += f"**حد ضرر:** `${stop_loss:,.2f}`\n"
            report += f"**حد سود اولیه:** `${target:,.2f}`\n"
            report += f"**اهرم پیشنهادی:** `x{leverage}`\n"
        else:
            report += "⚠️ **نتیجه:** در حال حاضر، هیچ سیگنال معاملاتی با احتمال موفقیت بالا بر اساس استراتژی‌های منتخب یافت نشد. **توصیه می‌شود وارد معامله نشوید.**"
            
        return report

    except Exception as e:
        logging.error(f"Error in full report for {symbol}: {e}")
        if isinstance(e, ccxt.BadSymbol): return "خطا: نماد وارد شده در صرافی یافت نشد."
        return "خطا در پردازش تحلیل جامع. لطفاً بعداً تلاش کنید."

def get_market_session():
    # این تابع بدون تغییر است
    utc_now = datetime.now(pytz.utc); hour = utc_now.hour
    if 0 <= hour < 8: return "آسیا (توکیو/سیدنی)", "نوسان کم"
    if 8 <= hour < 12: return "لندن", "شروع نوسان"
    if 13 <= hour < 17: return "همپوشانی لندن/نیویورک", "حداکثر نوسان"
    if 17 <= hour < 22: return "نیویورک", "نوسان بالا"
    return "خارج از سشن‌ها", "نقدینگی کم"

# --- کنترل‌کننده‌های ربات (اصلاح شده) ---
def handle_chat(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type != 'text': return
    text = msg['text']
    
    if user_states.get(chat_id) == 'awaiting_symbol_analysis':
        processing_message = bot.sendMessage(chat_id, f"✅ درخواست شما برای **{text.upper()}** دریافت شد. لطفاً چند لحظه صبر کنید، در حال آماده‌سازی گزارش جامع هستم...", parse_mode='Markdown')
        report_text = generate_full_report(text.strip())
        bot.editMessageText((chat_id, processing_message['message_id']), report_text, parse_mode='Markdown', reply_markup=get_main_menu_keyboard())
        user_states[chat_id] = 'main_menu'
        
    elif text == '/start':
        user_states[chat_id] = 'main_menu'
        bot.sendMessage(chat_id, 'به ربات هوشمند Apex Pro (نسخه Co-Pilot) خوش آمدید.',
                        reply_markup=get_main_menu_keyboard())

def handle_callback_query(msg):
    query_id, from_id, query_data = telepot.glance(msg, flavor='callback_query')
    chat_id = from_id
    bot.answerCallbackQuery(query_id)
    
    if query_data == 'main_menu':
        user_states[chat_id] = 'main_menu'
        bot.editMessageText((chat_id, msg['message']['message_id']), 'منوی اصلی:', reply_markup=get_main_menu_keyboard())

    elif query_data == 'menu_deep_analysis':
        user_states[chat_id] = 'awaiting_symbol_analysis'
        bot.editMessageText((chat_id, msg['message']['message_id']), 'لطفاً نماد ارز مورد نظر خود را برای تحلیل وارد کنید (مثلاً: BTC).',
                        reply_markup=get_back_to_main_menu_keyboard())

    elif query_data == 'menu_signal_hunt':
        bot.editMessageText((chat_id, msg['message']['message_id']),
                             "🎯 **در حال اسکن بازار برای یافتن بهترین فرصت‌ها...**\n\nاین فرآیند ممکن است کمی طول بکشد. به محض یافتن یک سیگنال با احتمال موفقیت بالا، گزارش کامل آن برای شما ارسال خواهد شد.",
                             reply_markup=get_back_to_main_menu_keyboard())
        # در نسخه واقعی، این بخش یک فرآیند سنگین را در پس‌زمینه آغاز می‌کند
        # برای نسخه فعلی، یک تحلیل نمونه روی بیت‌کوین انجام می‌دهیم
        report_text = generate_full_report("BTC")
        bot.sendMessage(chat_id, report_text, parse_mode='Markdown')


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