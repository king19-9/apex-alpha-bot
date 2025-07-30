# main.py (نسخه نهایی: شامل تمام قابلیت‌ها + اسکنر داینامیک)

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
latest_signal_found = {"report": "هنوز هیچ سیگنال قابل اعتمادی در اسکن اخیر بازار یافت نشده است. لطفاً بعداً دوباره تلاش کنید.", "timestamp": None}

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

def generate_full_report(symbol):
    try:
        kucoin_symbol = f"{symbol.upper()}/USDT"
        
        # ۱. دریافت داده‌ها با مدیریت خطای دقیق
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
        session_name, _ = get_market_session()
        report += f"**قیمت فعلی:** `${last_price:,.2f}` | **سشن:** {session_name}\n\n"

        # بخش ۲: تحلیل ساختار بازار
        trend_d = "صعودی" if ta.trend.ema_indicator(df_d['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_d['c'], 50).iloc[-1] else "نزولی"
        trend_4h = "صعودی" if ta.trend.ema_indicator(df_4h['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_4h['c'], 50).iloc[-1] else "نزولی"
        report += f"**روند روزانه:** {trend_d} | **روند ۴ ساعته:** {trend_4h}\n"
        
        # بخش ۳: تحلیل عرضه/تقاضا
        support = df_4h['l'].rolling(20).mean().iloc[-1]
        resistance = df_4h['h'].rolling(20).mean().iloc[-1]
        report += f"**ناحیه تقاضا (4H):** ~${support:,.2f}\n"
        report += f"**ناحیه عرضه (4H):** ~${resistance:,.2f}\n"
        
        # بخش ۴: تحلیل فاندامنتال (اخبار)
        news_query = symbol.replace('USDT', '')
        url = f"https://newsapi.org/v2/everything?q={news_query}&language=en&sortBy=publishedAt&pageSize=1&apiKey={NEWS_API_KEY}"
        latest_news = requests.get(url).json().get('articles', [{}])[0].get('title', 'خبر جدیدی یافت نشد.')
        report += f"**آخرین خبر:** *{latest_news}*\n\n"

        # بخش ۵: پیشنهاد معامله (AI-Powered)
        report += "**--- پیشنهاد معامله مبتنی بر AI ---**\n"
        
        last_1h_candle = df_1h.iloc[-1]
        body_size = abs(last_1h_candle['c'] - last_1h_candle['o'])
        lower_wick = last_1h_candle['c'] - last_1h_candle['l'] if last_1h_candle['c'] > last_1h_candle['o'] else last_1h_candle['o'] - last_1h_candle['l']
        
        is_long_signal, confidence = check_long_signal_conditions(trend_d, trend_4h, last_1h_candle, support, lower_wick, body_size)
        
        if is_long_signal:
            entry = last_1h_candle['h']
            stop_loss = last_1h_candle['l']
            target = resistance
            
            report += f"✅ **سیگنال خرید (Long) با اطمینان {confidence:.0f}٪ صادر شد.**\n"
            report += f"**نقطه ورود:** `${entry:,.2f}` | **حد ضرر:** `${stop_loss:,.2f}` | **حد سود:** `${target:,.2f}`"
        else:
            report += "⚠️ **نتیجه:** در حال حاضر، هیچ سیگنال معاملاتی با احتمال موفقیت بالا یافت نشد."
            
        return report

    except Exception as e:
        logging.error(f"Critical error in full report for {symbol}: {e}")
        return "یک خطای پیش‌بینی نشده در فرآیند تحلیل رخ داد."

def hunt_signals():
    """بازار را برای یافتن بهترین فرصت معاملاتی اسکن می‌کند."""
    global latest_signal_found
    
    while True:
        logging.info("SIGNAL_HUNTER: Starting new DYNAMIC market scan...")
        
        try:
            all_markets = exchange.load_markets()
            usdt_pairs = {symbol: market for symbol, market in all_markets.items() if symbol.endswith('/USDT') and market.get('active', True)}
            logging.info(f"Found {len(usdt_pairs)} active USDT pairs.")

            potential_candidates = []
            tickers = exchange.fetch_tickers(list(usdt_pairs.keys()))
            
            for symbol, ticker in tickers.items():
                volume_usd = ticker.get('quoteVolume', 0)
                if volume_usd < 1_000_000:
                    continue
                price_change_percent = ticker.get('percentage', 0)
                if not (-15 < price_change_percent < 30):
                    continue
                potential_candidates.append(symbol.replace('/USDT', ''))

            logging.info(f"Found {len(potential_candidates)} potential candidates after filtering.")
            
            best_signal_in_scan = {'symbol': None, 'confidence': 0}
            
            for symbol in potential_candidates:
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
                    
                    time.sleep(1.5)
                except Exception:
                    continue
            
            if best_signal_in_scan['symbol']:
                logging.info(f"New best signal found: {best_signal_in_scan['symbol']} with confidence {best_signal_in_scan['confidence']}%")
                report = generate_full_report(best_signal_in_scan['symbol'])
                latest_signal_found = {"report": report, "timestamp": datetime.now()}
            else:
                logging.info("No high-probability signals found in this scan.")
        
        except Exception as e:
            logging.error(f"Error in signal_hunter_loop: {e}")

        time.sleep(2 * 3600)

def check_long_signal_conditions(trend_d, trend_4h, last_candle, support, lower_wick, body_size):
    confidence = 0
    is_long_signal = False
    if trend_d == "صعودی" and trend_4h == "صعودی" and (last_candle['c'] < support * 1.03) and (lower_wick > body_size * 1.5):
        is_long_signal = True
        confidence = 70
        if abs(last_candle['c'] - support) < abs(last_candle['c'] - last_candle['o']):
            confidence += 10
    return is_long_signal, confidence

def get_market_session():
    utc_now = datetime.now(pytz.utc); hour = utc_now.hour
    if 0 <= hour < 8: return "آسیا (توکیو/سیدنی)", "نوسان کم"
    if 8 <= hour < 12: return "لندن", "شروع نوسان"
    if 13 <= hour < 17: return "همپوشانی لندن/نیویورک", "حداکثر نوسان"
    if 17 <= hour < 22: return "نیویورک", "نوسان بالا"
    return "خارج از سشن‌ها", "نقدینگی کم"

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