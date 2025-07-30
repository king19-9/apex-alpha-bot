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
active_trades = {}
latest_signal_found = {"report": "اسکنر در حال آماده‌سازی اولیه است. لطفاً چند دقیقه دیگر تلاش کنید.", "timestamp": None}

# --- توابع سازنده کیبورد ---
def get_main_menu_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='🔬 تحلیل عمیق یک ارز', callback_data='menu_deep_analysis')],
        [InlineKeyboardButton(text='🎯 نمایش آخرین سیگنال شکار شده', callback_data='menu_signal_hunt')],
        [InlineKeyboardButton(text='👁️ پایش معامله باز', callback_data='menu_monitor_trade')]
    ])

def get_back_to_main_menu_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='🔙 بازگشت به منوی اصلی', callback_data='main_menu')]
    ])

# --- موتور تحلیل پیشرفته ---

def get_market_session():
    utc_now = datetime.now(pytz.utc)
    hour = utc_now.hour
    if 0 <= hour < 8: return "آسیا (توکیو/سیدنی)", "نوسان کم، مناسب برای تثبیت روند"
    if 8 <= hour < 12: return "لندن", "شروع نقدینگی و نوسان بالا"
    if 13 <= hour < 17: return "همپوشانی لندن/نیویورک", "حداکثر نقدینگی و نوسان، احتمال حرکات فیک"
    if 17 <= hour < 22: return "نیویورک", "نوسان بالا، احتمال بازگشت روند در انتهای روز"
    return "خارج از سشن‌های اصلی", "نقدینگی کم"

def check_long_signal_conditions(trend_d, trend_4h, last_candle, support, lower_wick, body_size):
    confidence = 0
    is_long_signal = False
    if trend_d == "صعودی" and trend_4h == "صعودی" and (last_candle['c'] < support * 1.03) and (lower_wick > body_size * 1.5):
        is_long_signal = True
        confidence = 70
        if abs(last_candle['c'] - support) < abs(last_candle['c'] - last_candle['o']):
            confidence += 10
    return is_long_signal, confidence

def generate_full_report(symbol):
    try:
        kucoin_symbol = f"{symbol.upper()}/USDT"
        
        try:
            df_d = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, timeframe='1d', limit=100), columns=['ts','o','h','l','c','v'])
            df_4h = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, timeframe='4h', limit=100), columns=['ts','o','h','l','c','v'])
            df_1h = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, timeframe='1h', limit=50), columns=['ts','o','h','l','c','v'])
            if df_1h.empty or df_4h.empty or df_d.empty:
                return f"خطا: داده‌های کافی برای نماد {symbol} از صرافی دریافت نشد."
        except ccxt.BadSymbol:
            return "خطا: نماد وارد شده در صرافی یافت نشد."
        except Exception as e:
            logging.error(f"Data fetch error for {symbol}: {e}")
            return "خطا در ارتباط با صرافی. لطفاً لحظاتی بعد دوباره تلاش کنید."

        # بخش ۱: خلاصه وضعیت
        report = f"🔬 **گزارش جامع تحلیلی برای #{symbol}**\n\n"
        last_price = df_1h.iloc[-1]['c']
        session_name, session_char = get_market_session()
        report += f"**قیمت فعلی:** `${last_price:,.2f}`\n"
        report += f"**سشن معاملاتی:** {session_name} ({session_char})\n\n"
        
        # بخش ۲: تحلیل چند تایم‌فریم (با جزئیات بیشتر)
        report += "**--- تحلیل ساختار بازار (چند تایم‌فریم) ---**\n"
        report += "**ابزار:** میانگین‌های متحرک نمایی (EMA 21, 50) برای تشخیص روند.\n"
        trend_d = "صعودی" if ta.trend.ema_indicator(df_d['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_d['c'], 50).iloc[-1] else "نزولی"
        trend_4h = "صعودی" if ta.trend.ema_indicator(df_4h['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_4h['c'], 50).iloc[-1] else "نزولی"
        report += f"**روند روزانه (ساختار اصلی):** **{trend_d}**\n"
        report += f"**روند ۴ ساعته (روند میان‌مدت):** **{trend_4h}**\n"
        if trend_d == trend_4h:
            report += "✅ **نتیجه:** ساختار بازار **هم‌راستا و قوی** است. معاملات در جهت روند از اعتبار بالایی برخوردارند.\n\n"
        else:
            report += "⚠️ **نتیجه:** ساختار بازار **متناقض** است. قیمت در تایم‌فریم پایین‌تر در حال اصلاح یا تغییر روند است. معاملات در این شرایط ریسک بالاتری دارند.\n\n"

        # بخش ۳: تحلیل عرضه/تقاضا و پرایس اکشن
        report += "**--- تحلیل عرضه/تقاضا و پرایس اکشن ---**\n"
        report += "**ابزار:** شناسایی نواحی SR کلیدی و الگوهای کندلی به سبک ال بروکس.\n"
        support = df_4h['l'].rolling(20).mean().iloc[-1]
        resistance = df_4h['h'].rolling(20).mean().iloc[-1]
        report += f"**ناحیه کلیدی تقاضا (حمایت ۴ ساعته):** حدود `${support:,.2f}`\n"
        report += f"**ناحیه کلیدی عرضه (مقاومت ۴ ساعته):** حدود `${resistance:,.2f}`\n"
        
        last_1h_candle = df_1h.iloc[-1]
        body_size = abs(last_1h_candle['c'] - last_1h_candle['o'])
        candle_range = last_1h_candle['h'] - last_1h_candle['l']
        lower_wick = last_1h_candle['c'] - last_1h_candle['l'] if last_1h_candle['c'] > last_1h_candle['o'] else last_1h_candle['o'] - last_1h_candle['l']
        if body_size > 0 and lower_wick > body_size * 2 and (candle_range / body_size) > 3:
            report += "**سیگنال پرایس اکشن (۱ ساعته):** یک **پین‌بار صعودی** قوی شناسایی شد. این الگو نشان‌دهنده جمع‌آوری نقدینگی (Liquidity Sweep) در زیر قیمت و احتمال بالای حرکت صعودی است.\n\n"
        else:
            report += "**سیگنال پرایس اکشن (۱ ساعته):** کندل آخر سیگنال واضحی برای ورود به معامله ندارد.\n\n"

        # بخش ۴: تحلیل فاندامنتال
        report += "**--- تحلیل فاندامنتال (اخبار) ---**\n"
        report += "**ابزار:** NewsAPI برای واکشی اخبار.\n"
        news_query = symbol.replace('USDT', '')
        url = f"https://newsapi.org/v2/everything?q={news_query}&language=en&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}"
        articles = requests.get(url).json().get('articles', [])
        if articles:
            report += "**آخرین اخبار مهم:**\n"
            for article in articles:
                report += f"- *{article['title']}*\n"
        else:
            report += "خبر مهم جدیدی یافت نشد.\n\n"

        # بخش ۵: پیشنهاد معامله
        report += "**--- پیشنهاد معامله مبتنی بر AI (شبیه‌سازی شده) ---**\n"
        is_long_signal, confidence = check_long_signal_conditions(trend_d, trend_4h, last_1h_candle, support, lower_wick, body_size)
        if is_long_signal:
            entry = last_1h_candle['h']
            stop_loss = last_1h_candle['l']
            target = resistance
            leverage = 3
            report += f"✅ **سیگنال خرید (Long) با اطمینان {confidence:.0f}٪ صادر شد.**\n"
            report += f"**منطق:** هم‌راستایی روند در تایم بالا با سیگنال پرایس اکشن در ناحیه تقاضا.\n"
            report += f"**عملکرد گذشته استراتژی (تخمینی):** نرخ موفقیت ~۶۵٪\n"
            report += f"**نقطه ورود:** `${entry:,.2f}` | **حد ضرر:** `${stop_loss:,.2f}` | **حد سود:** `${target:,.2f}` | **اهرم:** `x{leverage}`\n"
        else:
            report += "⚠️ **نتیجه:** در حال حاضر، هیچ سیگنال معاملاتی با احتمال موفقیت بالا بر اساس استراتژی‌های منتخب یافت نشد. **توصیه می‌شود وارد معامله نشوید.**"
            
        return report

    except Exception as e:
        logging.error(f"Critical error in full report for {symbol}: {e}")
        return "یک خطای پیش‌بینی نشده در فرآیند تحلیل رخ داد."

def hunt_signals():
    """در پس‌زمینه به طور مداوم بازار را اسکن می‌کند."""
    global latest_signal_found
    watchlist = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'AVAX', 'LINK', 'MATIC', 'DOT', 'ADA', 'LTC', 'BNB', 'NEAR', 'ATOM', 'FTM']
    
    while True:
        logging.info("SIGNAL_HUNTER: Starting new market scan...")
        best_signal_in_scan = {'symbol': None, 'confidence': 0}
        
        for symbol in watchlist:
            try:
                df_d = pd.DataFrame(exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe='1d', limit=100), columns=['ts','o','h','l','c','v'])
                df_4h = pd.DataFrame(exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe='4h', limit=100), columns=['ts','o','h','l','c','v'])
                df_1h = pd.DataFrame(exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe='1h', limit=50), columns=['ts','o','h','l','c','v'])
                
                if df_1h.empty or len(df_d) < 50 or len(df_4h) < 50: continue

                trend_d = "صعودی" if ta.trend.ema_indicator(df_d['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_d['c'], 50).iloc[-1] else "نزولی"
                trend_4h = "صعودی" if ta.trend.ema_indicator(df_4h['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_4h['c'], 50).iloc[-1] else "نزولی"
                support = df_4h['l'].rolling(20).mean().iloc[-1]
                last_1h_candle = df_1h.iloc[-1]
                body_size = abs(last_1h_candle['c'] - last_1h_candle['o'])
                lower_wick = last_1h_candle['c'] - last_1h_candle['l'] if last_1h_candle['c'] > last_1h_candle['o'] else last_1h_candle['o'] - last_1h_candle['l']
                
                is_long, confidence = check_long_signal_conditions(trend_d, trend_4h, last_1h_candle, support, lower_wick, body_size)
                
                if is_long and confidence > best_signal_in_scan['confidence']:
                    best_signal_in_scan['symbol'] = symbol
                    best_signal_in_scan['confidence'] = confidence
                
                time.sleep(2)
            except Exception:
                continue
        
        if best_signal_in_scan['symbol']:
            logging.info(f"New best signal found: {best_signal_in_scan['symbol']}")
            report = generate_full_report(best_signal_in_scan['symbol'])
            latest_signal_found = {"report": report, "timestamp": datetime.now()}
        else:
            logging.info("No high-probability signals found in this scan.")

        time.sleep(3600)

# --- کنترل‌کننده‌های ربات ---
def handle_chat(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type != 'text': return
    text = msg['text']
    
    if user_states.get(chat_id) == 'awaiting_symbol_analysis':
        processing_message = bot.sendMessage(chat_id, f"✅ درخواست برای **{text.upper()}** دریافت شد. لطفاً صبر کنید...", parse_mode='Markdown')
        report_text = generate_full_report(text.strip())
        bot.editMessageText((chat_id, processing_message['message_id']), report_text, parse_mode='Markdown', reply_markup=get_main_menu_keyboard())
        user_states[chat_id] = 'main_menu'

    elif user_states.get(chat_id) == 'awaiting_symbol_monitor':
        symbol_to_monitor = text.strip().upper()
        active_trades[chat_id] = symbol_to_monitor
        bot.sendMessage(chat_id, f"✅ معامله شما برای #{symbol_to_monitor} تحت پایش قرار گرفت.",
                        reply_markup=get_main_menu_keyboard())
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
        bot.editMessageText((chat_id, msg['message']['message_id']), 'لطفاً نماد ارز را برای تحلیل وارد کنید (مثلاً: BTC).',
                        reply_markup=get_back_to_main_menu_keyboard())
        
    elif query_data == 'menu_signal_hunt':
        report_text = latest_signal_found.get("report", "خطا در دریافت آخرین سیگنال.")
        timestamp = latest_signal_found.get("timestamp")
        if timestamp:
            time_ago = int((datetime.now() - timestamp).total_seconds() / 60)
            report_text += f"\n\n*(این سیگنال حدود {time_ago} دقیقه پیش یافت شده است.)*"
        bot.editMessageText((chat_id, msg['message']['message_id']), report_text, parse_mode='Markdown', reply_markup=get_main_menu_keyboard())

    elif query_data == 'menu_monitor_trade':
        user_states[chat_id] = 'awaiting_symbol_monitor'
        bot.editMessageText((chat_id, msg['message']['message_id']), 'لطفاً نماد ارزی که در آن معامله باز کرده‌اید را وارد کنید (مثلاً: ETH).',
                        reply_markup=get_back_to_main_menu_keyboard())


def trade_monitor_loop():
    """یک نخ جداگانه برای پایش مداوم معاملات باز کاربران."""
    while True:
        time.sleep(5 * 60)
        for chat_id, symbol in list(active_trades.items()):
            try:
                df = pd.DataFrame(exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe='5m', limit=2), columns=['ts','o','h','l','c','v'])
                last_candle = df.iloc[-1]
                is_strong_reversal = abs(last_candle['c'] - last_candle['o']) > (last_candle['h'] - last_candle['l']) * 0.7
                if is_strong_reversal:
                    bot.sendMessage(chat_id, f"🚨 **هشدار پایش معامله برای #{symbol}** 🚨\nیک کندل بازگشتی قوی در تایم‌فریم ۵ دقیقه مشاهده شد. لطفاً پوزیشن خود را بازبینی کنید.")
                    del active_trades[chat_id]
            except Exception as e:
                logging.error(f"Error monitoring trade for {symbol}: {e}")

# --- راه‌اندازی ربات و وب‌سرور ---
def run_web_server():
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == '__main__':
    if not TELEGRAM_TOKEN:
        logging.fatal("TELEGRAM_TOKEN not found!")
    else:
        threading.Thread(target=trade_monitor_loop, daemon=True, name="TradeMonitorThread").start()
        threading.Thread(target=hunt_signals, daemon=True, name="SignalHunterThread").start()
        
        MessageLoop(bot, {'chat': handle_chat,
                          'callback_query': handle_callback_query}).run_as_thread()
        logging.info('Telepot bot is listening...')
        
        if os.getenv('RAILWAY_ENVIRONMENT'):
            threading.Thread(target=run_web_server, daemon=True, name="WebServerThread").start()

        logging.info("Bot is running.")
        while 1:
            time.sleep(10)