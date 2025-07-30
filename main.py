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
    ]
    if chat_id in signal_hunt_subscribers:
        buttons.append([InlineKeyboardButton(text='🎯 غیرفعال کردن نوتیفیکیشن سیگنال', callback_data='menu_toggle_signal_hunt')])
    else:
        buttons.append([InlineKeyboardButton(text='🎯 فعال کردن نوتیفیکیشن سیگنال', callback_data='menu_toggle_signal_hunt')])
        
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
    if 0 <= hour < 7: return "آسیا (توکیو/سیدنی)", "نوسان کم و ساخت ساختار"
    if 7 <= hour < 12: return "لندن", "شروع نقدینگی و احتمال حرکات فیک اولیه"
    if 13 <= hour < 17: return "همپوشانی لندن/نیویورک", "حداکثر حجم و نوسان، بهترین زمان برای معامله"
    if 17 <= hour < 22: return "نیویورک", "ادامه روند یا بازگشت در انتهای روز"
    return "خارج از سشن‌های اصلی", "نقدینگی بسیار کم"

def check_long_signal_conditions(trend_d, trend_4h, last_candle, support, lower_wick, body_size):
    confidence = 0
    is_long_signal = False
    if trend_d == "صعودی" and trend_4h == "صعودی" and (last_candle['c'] > support) and (last_candle['c'] < support * 1.03) and (lower_wick > body_size * 1.5):
        is_long_signal = True
        confidence = 70
        if body_size > 0 and abs(last_candle['c'] - support) < abs(last_candle['c'] - last_candle['o']):
            confidence += 10
    return is_long_signal, confidence

def generate_full_report(symbol, is_monitoring=False):
    try:
        kucoin_symbol = f"{symbol.upper()}/USDT"
        
        try:
            df_d = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, timeframe='1d', limit=100), columns=['ts','o','h','l','c','v'])
            df_4h = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, timeframe='4h', limit=100), columns=['ts','o','h','l','c','v'])
            df_1h = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, timeframe='1h', limit=50), columns=['ts','o','h','l','c','v'])
            df_15m = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, timeframe='15m', limit=50), columns=['ts','o','h','l','c','v'])
            if df_1h.empty: return f"خطا: داده‌های کافی برای نماد {symbol} دریافت نشد."
        except Exception as e:
            return f"خطا در ارتباط با صرافی: {e}"

        report_prefix = "🔬 **گزارش جامع تحلیلی**" if not is_monitoring else "👁️ **گزارش پایش لحظه‌ای**"
        report = f"{report_prefix} برای #{symbol}\n\n"
        last_price = df_1h.iloc[-1]['c']
        session_name, _ = get_market_session()
        report += f"**قیمت فعلی:** `${last_price:,.2f}` | **سشن:** {session_name}\n\n"
        
        report += "**--- تحلیل ساختار بازار (Multi-Timeframe) ---**\n"
        trend_d = "صعودی ✅" if ta.trend.ema_indicator(df_d['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_d['c'], 50).iloc[-1] else "نزولی 🔻"
        trend_4h = "صعودی ✅" if ta.trend.ema_indicator(df_4h['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_4h['c'], 50).iloc[-1] else "نزولی 🔻"
        trend_1h = "صعودی ✅" if ta.trend.ema_indicator(df_1h['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_1h['c'], 50).iloc[-1] else "نزولی 🔻"
        trend_15m = "صعودی ✅" if ta.trend.ema_indicator(df_15m['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_15m['c'], 50).iloc[-1] else "نزولی 🔻"
        report += f"**روندها (D/4H/1H/15M):** {trend_d} / {trend_4h} / {trend_1h} / {trend_15m}\n"
        
        support = df_4h['l'].rolling(20).mean().iloc[-1]
        resistance = df_4h['h'].rolling(20).mean().iloc[-1]
        report += f"**ناحیه تقاضا (4H):** ~${support:,.2f}\n"
        report += f"**ناحیه عرضه (4H):** ~${resistance:,.2f}\n\n"

        if not is_monitoring:
            report += "**--- تحلیل فاندامنتال (اخبار) ---**\n"
            news_query = symbol.replace('USDT', '')
            url = f"https://newsapi.org/v2/everything?q={news_query}&language=en&sortBy=publishedAt&pageSize=1&apiKey={NEWS_API_KEY}"
            latest_news = requests.get(url).json().get('articles', [{}])[0].get('title', 'خبر جدیدی یافت نشد.')
            report += f"**آخرین خبر:** *{latest_news}*\n\n"
            report += "**--- پیشنهاد معامله (AI) ---**\n"
            last_1h_candle = df_1h.iloc[-1]
            body_size = abs(last_1h_candle['c'] - last_1h_candle['o'])
            lower_wick = last_1h_candle['c'] - last_1h_candle['l'] if last_1h_candle['c'] > last_1h_candle['o'] else last_1h_candle['o'] - last_1h_candle['l']
            is_long_signal, confidence = check_long_signal_conditions(trend_d.split(" ")[0], trend_4h.split(" ")[0], last_1h_candle, support, lower_wick, body_size)
            if is_long_signal:
                entry = last_1h_candle['h']
                stop_loss = last_1h_candle['l']
                target = resistance
                report += f"✅ **سیگنال خرید (Long) با اطمینان {confidence:.0f}٪ صادر شد.**\n"
                report += f"**نقطه ورود:** `${entry:,.2f}` | **حد ضرر:** `${stop_loss:,.2f}` | **حد سود:** `${target:,.2f}`"
            else:
                report += "⚠️ نتیجه: در حال حاضر، سیگنال ورود واضحی یافت نشد."
            
        return report
    except Exception as e:
        logging.error(f"Critical error in full report for {symbol}: {e}")
        return "خطای پیش‌بینی نشده در تحلیل."

def hunt_signals():
    global sent_signals_cache
    watchlist = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'AVAX', 'LINK', 'MATIC', 'DOT', 'ADA', 'LTC', 'BNB', 'NEAR', 'ATOM', 'FTM']
    while True:
        logging.info("SIGNAL_HUNTER: Starting new market scan...")
        for symbol in watchlist:
            try:
                df_d = pd.DataFrame(exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe='1d', limit=100), columns=['ts','o','h','l','c','v'])
                df_4h = pd.DataFrame(exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe='4h', limit=100), columns=['ts','o','h','l','c','v'])
                df_1h = pd.DataFrame(exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe='1h', limit=50), columns=['ts','o','h','l','c','v'])
                if df_1h.empty or len(df_d) < 51 or len(df_4h) < 51: continue
                trend_d = "صعودی" if ta.trend.ema_indicator(df_d['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_d['c'], 50).iloc[-1] else "نزولی"
                trend_4h = "صعودی" if ta.trend.ema_indicator(df_4h['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_4h['c'], 50).iloc[-1] else "نزولی"
                support = df_4h['l'].rolling(20).mean().iloc[-1]
                last_1h_candle = df_1h.iloc[-1]
                body_size = abs(last_1h_candle['c'] - last_1h_candle['o'])
                lower_wick = last_1h_candle['c'] - last_1h_candle['l'] if last_1h_candle['c'] > last_1h_candle['o'] else last_1h_candle['o'] - last_1h_candle['l']
                is_long, confidence = check_long_signal_conditions(trend_d, trend_4h, last_1h_candle, support, lower_wick, body_size)
                if is_long and confidence > 85:
                    if sent_signals_cache.get(symbol) != "long":
                        report = generate_full_report(symbol)
                        message = f"🎯 **شکار سیگنال: فرصت خرید یافت شد!** 🎯\n\n{report}"
                        for chat_id in list(signal_hunt_subscribers):
                            try: bot.sendMessage(chat_id, message, parse_mode='Markdown')
                            except Exception as e:
                                if 'Forbidden' in str(e): signal_hunt_subscribers.remove(chat_id)
                        sent_signals_cache[symbol] = "long"
                else:
                    if symbol in sent_signals_cache: del sent_signals_cache[symbol]
            except Exception as e:
                logging.warning(f"Could not scan symbol {symbol}: {e}")
                continue
            time.sleep(5)
        time.sleep(15 * 60)

def trade_monitor_loop():
    """پایش هوشمند و مداوم معاملات باز."""
    while True:
        time.sleep(5 * 60)
        if not active_trades: continue
        for chat_id, trade_info in list(active_trades.items()):
            try:
                symbol = trade_info['symbol']
                initial_direction = trade_info['direction']
                report = generate_full_report(symbol, is_monitoring=True)
                current_trend_15m = "صعودی" if "صعودی" in report.split("15M):**")[1].split("\n")[0] else "نزولی"
                if (initial_direction == "Long" and current_trend_15m == "نزولی") or \
                   (initial_direction == "Short" and current_trend_15m == "صعودی"):
                    message = f"🚨 **هشدار پایش معامله برای #{symbol}** 🚨\n\n**تغییر در ساختار کوتاه‌مدت مشاهده شد!**\n\n{report}\n\n**توصیه:** لطفاً پوزیشن خود را بازبینی کنید."
                    bot.sendMessage(chat_id, message, parse_mode='Markdown')
            except Exception as e:
                logging.error(f"Error monitoring trade for {symbol}: {e}")

# --- کنترل‌کننده‌های ربات ---
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
    elif text == '/stats':
        bot.sendMessage(chat_id, "📊 **آمار عملکرد سیگنال‌ها (آزمایشی)**\n\nاین قابلیت در حال توسعه است.")

def handle_callback_query(msg):
    query_id, from_id, query_data = telepot.glance(msg, flavor='callback_query')
    chat_id = from_id
    bot.answerCallbackQuery(query_id)
    
    if query_data.startswith('main_menu'):
        user_states[chat_id] = 'main_menu'
        bot.editMessageText((chat_id, msg['message']['message_id']), 'منوی اصلی:', reply_markup=get_main_menu_keyboard(chat_id))
    elif query_data == 'menu_deep_analysis':
        user_states[chat_id] = 'awaiting_symbol_analysis'
        bot.editMessageText((chat_id, msg['message']['message_id']), 'لطفاً نماد ارز را برای تحلیل وارد کنید (مثلاً: BTC).',
                        reply_markup=get_back_to_main_menu_keyboard(chat_id))
    elif query_data == 'menu_toggle_signal_hunt':
        if chat_id in signal_hunt_subscribers:
            signal_hunt_subscribers.remove(chat_id)
            bot.editMessageText((chat_id, msg['message']['message_id']),
                                "✅ **شکار سیگنال غیرفعال شد.**",
                                reply_markup=get_main_menu_keyboard(chat_id))
        else:
            signal_hunt_subscribers.add(chat_id)
            bot.editMessageText((chat_id, msg['message']['message_id']),
                                "✅ **شکار سیگنال فعال شد.**",
                                reply_markup=get_main_menu_keyboard(chat_id))
    elif query_data == 'menu_monitor_trade':
        user_states[chat_id] = 'awaiting_symbol_monitor'
        bot.editMessageText((chat_id, msg['message']['message_id']), 'لطفاً نماد ارزی که در آن معامله باز کرده‌اید را وارد کنید (مثلاً: ETH).',
                        reply_markup=get_back_to_main_menu_keyboard(chat_id))
    elif query_data.startswith('monitor_stop_'):
        symbol_to_stop = query_data.split('_')[2]
        if chat_id in active_trades and active_trades[chat_id]['symbol'] == symbol_to_stop:
            del active_trades[chat_id]
            bot.editMessageText((chat_id, msg['message']['message_id']),
                              f"پایش برای معامله #{symbol_to_stop} متوقف شد.",
                              reply_markup=get_main_menu_keyboard(chat_id))

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