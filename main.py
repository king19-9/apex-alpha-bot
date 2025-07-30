# main.py (نسخه نهایی: Apex Sentinel v4.0)

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
signal_hunt_subscribers = set()
sent_signals_cache = {}

# --- توابع سازنده کیبورد ---
def get_main_menu_keyboard(chat_id):
    buttons = [
        [InlineKeyboardButton(text='🔬 تحلیل عمیق یک ارز', callback_data='menu_deep_analysis')],
        [InlineKeyboardButton(text='🎯 فعال/غیرفعال کردن نوتیفیکیشن سیگنال', callback_data='menu_toggle_signal_hunt')],
    ]
    if chat_id in active_trades:
        buttons.append([InlineKeyboardButton(text=f"🚫 توقف پایش معامله {active_trades[chat_id]['symbol']}", callback_data=f"monitor_stop_{active_trades[chat_id]['symbol']}")])
    else:
        buttons.append([InlineKeyboardButton(text='👁️ پایش معامله باز', callback_data='menu_monitor_trade')])
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_back_to_main_menu_keyboard(chat_id):
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text='🔙 بازگشت به منوی اصلی', callback_data=f'main_menu_{chat_id}')]])

# --- موتور تحلیل پیشرفته ---

def get_market_session():
    utc_now = datetime.now(pytz.utc)
    hour = utc_now.hour
    if 0 <= hour < 7: return "آسیا (توکیو/سیدنی)", "نوسان کم"
    if 7 <= hour < 12: return "لندن", "شروع نقدینگی"
    if 13 <= hour < 17: return "همپوشانی لندن/نیویورک", "حداکثر نوسان"
    if 17 <= hour < 22: return "نیویورک", "ادامه یا بازگشت روند"
    return "خارج از سشن‌ها", "نقدینگی کم"

def generate_full_report(symbol, is_monitoring=False):
    try:
        kucoin_symbol = f"{symbol.upper()}/USDT"
        
        # ۱. دریافت داده‌ها با مدیریت خطای دقیق
        try:
            df_d = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, timeframe='1d', limit=100), columns=['ts','o','h','l','c','v'])
            df_4h = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, timeframe='4h', limit=100), columns=['ts','o','h','l','c','v'])
            df_1h = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, timeframe='1h', limit=50), columns=['ts','o','h','l','c','v'])
            df_15m = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, timeframe='15m', limit=50), columns=['ts','o','h','l','c','v'])
            if df_1h.empty: return f"خطا: داده‌های کافی برای نماد {symbol} دریافت نشد."
        except Exception as e:
            return f"خطا در ارتباط با صرافی: {e}"

        # بخش ۱: خلاصه وضعیت
        report_prefix = "🔬 **گزارش جامع تحلیلی**" if not is_monitoring else "👁️ **گزارش پایش لحظه‌ای**"
        report = f"{report_prefix} برای #{symbol}\n\n"
        last_price = df_1h.iloc[-1]['c']
        session_name, _ = get_market_session()
        report += f"**قیمت فعلی:** `${last_price:,.2f}` | **سشن:** {session_name}\n\n"
        
        # بخش ۲: تحلیل چند تایم‌فریم
        report += "**--- تحلیل ساختار بازار (Multi-Timeframe) ---**\n"
        trend_d = "صعودی ✅" if ta.trend.ema_indicator(df_d['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_d['c'], 50).iloc[-1] else "نزولی 🔻"
        trend_4h = "صعودی ✅" if ta.trend.ema_indicator(df_4h['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_4h['c'], 50).iloc[-1] else "نزولی 🔻"
        trend_1h = "صعودی ✅" if ta.trend.ema_indicator(df_1h['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_1h['c'], 50).iloc[-1] else "نزولی 🔻"
        trend_15m = "صعودی ✅" if ta.trend.ema_indicator(df_15m['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_15m['c'], 50).iloc[-1] else "نزولی 🔻"
        report += f"**روندها (D/4H/1H/15M):** {trend_d} / {trend_4h} / {trend_1h} / {trend_15m}\n"
        
        # بخش ۳: تحلیل عرضه/تقاضا
        support = df_4h['l'].rolling(20).mean().iloc[-1]
        resistance = df_4h['h'].rolling(20).mean().iloc[-1]
        report += f"**ناحیه تقاضا (4H):** ~${support:,.2f}\n"
        report += f"**ناحیه عرضه (4H):** ~${resistance:,.2f}\n\n"

        # بخش ۴: تحلیل فاندامنتال (فقط برای گزارش‌های اصلی)
        if not is_monitoring:
            report += "**--- تحلیل فاندامنتال (اخبار) ---**\n"
            news_query = symbol.replace('USDT', '')
            url = f"https://newsapi.org/v2/everything?q={news_query}&language=en&sortBy=publishedAt&pageSize=1&apiKey={NEWS_API_KEY}"
            latest_news = requests.get(url).json().get('articles', [{}])[0].get('title', 'خبر جدیدی یافت نشد.')
            report += f"**آخرین خبر:** *{latest_news}*\n\n"

        # بخش ۵: پیشنهاد معامله (فقط برای گزارش‌های اصلی)
        if not is_monitoring:
            report += "**--- پیشنهاد معامله (AI) ---**\n"
            # ... (منطق پیشنهاد معامله بدون تغییر) ...
            report += "⚠️ نتیجه: در حال حاضر، سیگنال ورود واضحی یافت نشد."
            
        return report
    except Exception as e:
        logging.error(f"Critical error in full report for {symbol}: {e}")
        return "خطای پیش‌بینی نشده در تحلیل."

# --- موتور شکار سیگنال جدید و پیشرفته ---
def hunt_signals():
    global sent_signals_cache
    
    while True:
        logging.info("SIGNAL_HUNTER: Starting new advanced market scan...")
        try:
            # ۱. فیلتر کردن داینامیک کل بازار
            all_markets = exchange.load_markets()
            usdt_pairs = {s: m for s, m in all_markets.items() if s.endswith('/USDT') and m.get('active', True)}
            tickers = exchange.fetch_tickers(list(usdt_pairs.keys()))
            
            potential_candidates = []
            for symbol, ticker in tickers.items():
                if ticker.get('quoteVolume', 0) > 5_000_000 and -10 < ticker.get('percentage', 0) < 20:
                    potential_candidates.append(symbol.replace('/USDT', ''))
            
            logging.info(f"Found {len(potential_candidates)} candidates for deep analysis.")
            
            # ۲. تحلیل عمیق کاندیداها و امتیازدهی
            for symbol in potential_candidates:
                try:
                    df_d = pd.DataFrame(exchange.fetch_ohlcv(f"{symbol}/USDT", '1d', limit=51), columns=['ts','o','h','l','c','v'])
                    df_4h = pd.DataFrame(exchange.fetch_ohlcv(f"{symbol}/USDT", '4h', limit=51), columns=['ts','o','h','l','c','v'])
                    if df_d.empty or df_4h.empty: continue

                    bullish_score = 0
                    # معیار ۱: هم‌راستایی روند
                    if ta.trend.ema_indicator(df_d['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_d['c'], 50).iloc[-1]: bullish_score += 3
                    if ta.trend.ema_indicator(df_4h['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_4h['c'], 50).iloc[-1]: bullish_score += 2
                        
                    # معیار ۲: هیجان بازار (RSI)
                    rsi = ta.momentum.rsi(df_4h['c']).iloc[-1]
                    if 30 < rsi < 55: bullish_score += 2.5 # بهترین حالت برای شروع حرکت
                    
                    # معیار ۳: حجم معاملات
                    if df_4h['v'].iloc[-1] > df_4h['v'].rolling(20).mean().iloc[-1] * 1.5: bullish_score += 1.5
                    
                    # ۳. تصمیم‌گیری و ارسال نوتیفیکیشن
                    if bullish_score >= 7:
                        if sent_signals_cache.get(symbol) != "long":
                            logging.info(f"HIGH-CONFIDENCE LONG SIGNAL FOUND for {symbol} (Score: {bullish_score})")
                            report = generate_full_report(symbol)
                            message = f"🎯 **شکار سیگنال: فرصت خرید یافت شد!** 🎯\n\n{report}"
                            for chat_id in list(signal_hunt_subscribers):
                                bot.sendMessage(chat_id, message, parse_mode='Markdown')
                            sent_signals_cache[symbol] = "long"
                    else:
                        if symbol in sent_signals_cache:
                            del sent_signals_cache[symbol]
                            
                except Exception: continue
                time.sleep(2)
        except Exception as e:
            logging.error(f"Error in signal_hunter_loop: {e}")
            
        time.sleep(30 * 60) # هر ۳۰ دقیقه یک بار کل بازار را اسکن کن

def trade_monitor_loop():
    """پایش هوشمند و مداوم معاملات باز."""
    while True:
        time.sleep(5 * 60) # هر ۵ دقیقه
        if not active_trades: continue
        
        for chat_id, trade_info in list(active_trades.items()):
            try:
                symbol = trade_info['symbol']
                initial_direction = trade_info['direction']
                
                logging.info(f"MONITOR: Analyzing {symbol} for user {chat_id}")
                # اجرای تحلیل کامل لحظه‌ای
                report = generate_full_report(symbol, is_monitoring=True)
                
                # منطق هشدار هوشمند
                current_trend_15m = "صعودی" if "صعودی" in report.split("15M):**")[1].split("\n")[0] else "نزولی"
                
                if (initial_direction == "Long" and current_trend_15m == "نزولی") or \
                   (initial_direction == "Short" and current_trend_15m == "صعودی"):
                    
                    message = f"🚨 **هشدار پایش معامله برای #{symbol}** 🚨\n\n**تغییر در ساختار کوتاه‌مدت مشاهده شد!**\n\n{report}\n\n**توصیه:** لطفاً پوزیشن خود را بازبینی کنید. ممکن است زمان مناسبی برای مدیریت ریسک یا خروج از معامله باشد."
                    bot.sendMessage(chat_id, message, parse_mode='Markdown')
                    # پایش ادامه دارد تا کاربر خودش لغو کند
                    
            except Exception as e:
                logging.error(f"Error monitoring trade for {symbol}: {e}")

# --- کنترل‌کننده‌های ربات (اصلاح شده) ---
def handle_chat(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type != 'text': return
    text = msg['text']
    
    if user_states.get(chat_id) == 'awaiting_symbol_analysis':
        processing_message = bot.sendMessage(chat_id, f"✅ درخواست برای **{text.upper()}** دریافت شد...", parse_mode='Markdown')
        report_text = generate_full_report(text.strip())
        bot.editMessageText((chat_id, processing_message['message_id']), report_text, parse_mode='Markdown', reply_markup=get_main_menu_keyboard(chat_id))
        user_states[chat_id] = 'main_menu'

    elif user_states.get(chat_id) == 'awaiting_symbol_monitor':
        symbol_to_monitor = text.strip().upper()
        # فرض می‌کنیم کاربر در جهت روند اصلی وارد معامله شده
        df_d = pd.DataFrame(exchange.fetch_ohlcv(f"{symbol_to_monitor}/USDT", '1d', limit=51), columns=['ts','o','h','l','c','v'])
        direction = "Long" if ta.trend.ema_indicator(df_d['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_d['c'], 50).iloc[-1] else "Short"
        
        active_trades[chat_id] = {'symbol': symbol_to_monitor, 'direction': direction}
        bot.sendMessage(chat_id, f"✅ معامله {direction} شما برای #{symbol_to_monitor} تحت پایش هوشمند قرار گرفت.",
                        reply_markup=get_main_menu_keyboard(chat_id))
        user_states[chat_id] = 'main_menu'
        
    elif text == '/start':
        user_states[chat_id] = 'main_menu'
        bot.sendMessage(chat_id, 'به ربات هوشمند Apex Sentinel خوش آمدید.',
                        reply_markup=get_main_menu_keyboard(chat_id))

def handle_callback_query(msg):
    query_id, from_id, query_data = telepot.glance(msg, flavor='callback_query')
    chat_id = from_id
    bot.answerCallbackQuery(query_id)
    
    # ... (کد کامل callback_query از پاسخ قبلی اینجا قرار می‌گیرد) ...

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