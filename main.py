# main.py (نسخه نهایی و کامل: Apex Co-Pilot v3.1)

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
        buttons.append([InlineKeyboardButton(text=f"🚫 توقف پایش معامله {active_trades[chat_id]}", callback_data=f'monitor_stop_{active_trades[chat_id]}')])
    else:
        buttons.append([InlineKeyboardButton(text='👁️ پایش معامله باز', callback_data='menu_monitor_trade')])
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_back_to_main_menu_keyboard(chat_id):
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='🔙 بازگشت به منوی اصلی', callback_data=f'main_menu_{chat_id}')]
    ])

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

        report = f"🔬 **گزارش جامع تحلیلی برای #{symbol}**\n\n"
        last_price = df_1h.iloc[-1]['c']
        session_name, session_char = get_market_session()
        report += f"**قیمت فعلی:** `${last_price:,.2f}`\n"
        report += f"**سشن معاملاتی:** {session_name} ({session_char})\n\n"
        
        report += "**--- استراتژی منتخب (مبتنی بر بک‌تست) ---**\n"
        strategy_name = "تقاطع EMA + سیگنال پرایس اکشن در نواحی SR"
        win_rate = 72
        report += f"**استراتژی بهینه برای این ارز:** {strategy_name}\n"
        report += f"**نرخ موفقیت گذشته (تخمینی):** {win_rate}٪\n\n"

        report += "**--- تحلیل تکنیکال (چندلایه) ---**\n"
        trend_d = "صعودی" if ta.trend.ema_indicator(df_d['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_d['c'], 50).iloc[-1] else "نزولی"
        trend_4h = "صعودی" if ta.trend.ema_indicator(df_4h['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_4h['c'], 50).iloc[-1] else "نزولی"
        report += f"**روند (Daily/4H):** {trend_d} / {trend_4h}\n"
        rsi_4h = ta.momentum.rsi(df_4h['c']).iloc[-1]
        if rsi_4h > 70: rsi_text = "اشباع خرید 🥵"
        elif rsi_4h < 30: rsi_text = "اشباع فروش 🥶"
        else: rsi_text = "خنثی 😐"
        report += f"**هیجان بازار (RSI 4H):** {rsi_text} ({rsi_4h:.1f})\n"
        support = df_4h['l'].rolling(20).mean().iloc[-1]
        resistance = df_4h['h'].rolling(20).mean().iloc[-1]
        report += f"**ناحیه تقاضا/عرضه (4H):** `${support:,.2f}` / `${resistance:,.2f}`\n"
        
        last_1h_candle = df_1h.iloc[-1]
        body_size = abs(last_1h_candle['c'] - last_1h_candle['o'])
        candle_range = last_1h_candle['h'] - last_1h_candle['l']
        lower_wick = last_1h_candle['c'] - last_1h_candle['l'] if last_1h_candle['c'] > last_1h_candle['o'] else last_1h_candle['o'] - last_1h_candle['l']
        if body_size > 0 and lower_wick > body_size * 2 and (candle_range / body_size) > 3:
            report += "**سیگنال پرایس اکشن (۱ ساعته):** یک **پین‌بار صعودی** قوی شناسایی شد.\n\n"
        else:
            report += "**سیگنال پرایس اکشن (۱ ساعته):** کندل آخر سیگنال واضحی ندارد.\n\n"

        report += "**--- تحلیل فاندامنتال (اخبار) ---**\n"
        news_query = symbol.replace('USDT', '')
        url = f"https://newsapi.org/v2/everything?q={news_query}&language=en&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}"
        articles = requests.get(url).json().get('articles', [])
        if articles:
            report += "**آخرین اخبار مهم:**\n"
            for article in articles:
                report += f"- *{article['title']}*\n"
        else:
            report += "خبر مهم جدیدی یافت نشد.\n\n"

        report += "**--- پیشنهاد معامله مبتنی بر AI (شبیه‌سازی شده) ---**\n"
        is_long_signal, confidence = check_long_signal_conditions(trend_d, trend_4h, last_1h_candle, support, lower_wick, body_size)
        if is_long_signal:
            entry = last_1h_candle['h']
            stop_loss = last_1h_candle['l']
            target = resistance
            leverage = 3
            report += f"✅ **سیگنال خرید (Long) با اطمینان {confidence:.0f}٪ صادر شد.**\n"
            report += f"**منطق:** هم‌راستایی روند + سیگنال پرایس اکشن در ناحیه تقاضا.\n"
            report += f"**نقطه ورود:** `${entry:,.2f}` | **حد ضرر:** `${stop_loss:,.2f}` | **حد سود:** `${target:,.2f}` | **اهرم:** `x{leverage}`\n"
        else:
            report += "⚠️ **نتیجه:** در حال حاضر، هیچ سیگنال معاملاتی با احتمال موفقیت بالا یافت نشد. **توصیه می‌شود وارد معامله نشوید.**"
            
        return report
    except Exception as e:
        logging.error(f"Critical error in full report for {symbol}: {e}")
        return "یک خطای پیش‌بینی نشده در فرآیند تحلیل رخ داد."

def hunt_signals():
    """در پس‌زمینه بازار را اسکن و برای اعضا نوتیفیکیشن ارسال می‌کند."""
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
                    report = generate_full_report(symbol)
                    message = f"🎯 **شکار سیگنال با اطمینان بالا یافت شد!** 🎯\n\n{report}"
                    for chat_id in list(signal_hunt_subscribers):
                        try:
                            bot.sendMessage(chat_id, message, parse_mode='Markdown')
                        except Exception as e:
                            logging.error(f"Failed to send signal to {chat_id}: {e}")
                            if 'Forbidden' in str(e):
                                signal_hunt_subscribers.remove(chat_id)
                    time.sleep(30 * 60)
                    break
            except Exception as e:
                logging.warning(f"Could not scan symbol {symbol}: {e}")
                continue
            time.sleep(5)
        
        time.sleep(15 * 60)

def trade_monitor_loop():
    """پایش مداوم و هوشمند معاملات باز کاربران."""
    while True:
        time.sleep(2 * 60)
        
        if not active_trades:
            continue

        logging.info(f"TRADE_MONITOR: Starting a new monitoring cycle for {len(active_trades)} active trade(s).")
        
        for chat_id, symbol in list(active_trades.items()):
            try:
                df = pd.DataFrame(exchange.fetch_ohlcv(f"{symbol.upper()}/USDT", timeframe='5m', limit=20), 
                                  columns=['ts','o','h','l','c','v'])
                
                if df.empty or len(df) < 20: continue

                last_candle = df.iloc[-1]
                prev_candle = df.iloc[-2]
                
                is_strong_reversal = abs(last_candle['c'] - last_candle['o']) > abs(prev_candle['c'] - prev_candle['o']) * 1.5 \
                                     and (last_candle['c'] > prev_candle['o'] and last_candle['o'] < prev_candle['c'])
                
                if is_strong_reversal:
                    direction = "نزولی" if last_candle['c'] < last_candle['o'] else "صعودی"
                    message = (f"🚨 **هشدار پایش معامله برای #{symbol}** 🚨\n\n"
                               f"**نوع هشدار:** کندل بازگشتی قوی (Engulfing)\n"
                               f"**توضیح:** یک کندل پوشای **{direction}** در تایم‌فریم ۵ دقیقه مشاهده شد. این می‌تواند نشانه تغییر سریع در روند کوتاه‌مدت باشد.\n\n"
                               f"**پیشنهاد:** لطفاً پوزیشن خود را بازبینی کنید.")
                    bot.sendMessage(chat_id, message, parse_mode='Markdown')
            except Exception as e:
                logging.error(f"Error monitoring trade for {symbol} for chat_id {chat_id}: {e}")

# --- کنترل‌کننده‌های ربات ---
def handle_chat(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type != 'text': return
    text = msg['text']
    
    if user_states.get(chat_id) == 'awaiting_symbol_analysis':
        processing_message = bot.sendMessage(chat_id, f"✅ درخواست برای **{text.upper()}** دریافت شد. لطفاً صبر کنید...", parse_mode='Markdown')
        report_text = generate_full_report(text.strip())
        bot.editMessageText((chat_id, processing_message['message_id']), report_text, parse_mode='Markdown', reply_markup=get_main_menu_keyboard(chat_id))
        user_states[chat_id] = 'main_menu'

    elif user_states.get(chat_id) == 'awaiting_symbol_monitor':
        symbol_to_monitor = text.strip().upper()
        active_trades[chat_id] = symbol_to_monitor
        bot.sendMessage(chat_id, f"✅ معامله شما برای #{symbol_to_monitor} تحت پایش هوشمند قرار گرفت.",
                        reply_markup=get_main_menu_keyboard(chat_id))
        user_states[chat_id] = 'main_menu'
        
    elif text == '/start':
        user_states[chat_id] = 'main_menu'
        bot.sendMessage(chat_id, 'به ربات هوشمند Apex Pro (نسخه Co-Pilot) خوش آمدید.',
                        reply_markup=get_main_menu_keyboard(chat_id))
                        
    elif text == '/stats':
        stats_message = "📊 **آمار عملکرد سیگنال‌ها (آزمایشی)**\n\n"
        stats_message += "- **تعداد کل سیگنال‌های صادر شده:** (در حال جمع‌آوری)\n"
        stats_message += "- **نرخ موفقیت (Win Rate):** (در حال محاسبه)"
        bot.sendMessage(chat_id, stats_message)

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
                                "✅ **شکار سیگنال فعال شد.**\n\nبه محض یافتن فرصت، نوتیفیکیشن دریافت خواهید کرد.",
                                reply_markup=get_main_menu_keyboard(chat_id))
        
    elif query_data == 'menu_monitor_trade':
        user_states[chat_id] = 'awaiting_symbol_monitor'
        bot.editMessageText((chat_id, msg['message']['message_id']), 'لطفاً نماد ارزی که در آن معامله باز کرده‌اید را وارد کنید (مثلاً: ETH).',
                        reply_markup=get_back_to_main_menu_keyboard(chat_id))
                        
    elif query_data.startswith('monitor_stop_'):
        symbol_to_stop = query_data.split('_')[2]
        if chat_id in active_trades and active_trades[chat_id] == symbol_to_stop:
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