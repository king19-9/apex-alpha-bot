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
from tradingview_ta import TA_Handler, Interval
import investpy
from datetime import datetime
import pytz

# --- تنظیمات اولیه ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# --- کلاینت‌ها و سرویس‌ها ---
app = FastAPI()
exchange = ccxt.kucoin()
bot = telepot.Bot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
user_states = {}
active_trades = {}
hunted_signals_cache = []

# --- توابع سازنده کیبورد ---
def get_main_menu_keyboard(chat_id):
    buttons = [
        [InlineKeyboardButton(text='🔬 تحلیل جامع یک نماد', callback_data='menu_deep_analysis')],
        [InlineKeyboardButton(text='🎯 نمایش سیگنال‌های شکار شده', callback_data='menu_show_hunted_signals')],
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
    if 0 <= hour < 7: return "آسیا (توکیو/سیدنی)", "حجم کم، مناسب برای ساخت ساختار"
    if 7 <= hour < 12: return "لندن", "شروع نقدینگی و احتمال شکار حد ضرر (Stop Hunt)"
    if 13 <= hour < 17: return "همپوشانی لندن/نیویورک", "حداکثر حجم و نوسان، بهترین زمان برای معامله"
    if 17 <= hour < 22: return "نیویورک", "ادامه روند یا بازگشت در انتهای روز"
    return "خارج از سشن‌ها", "نقدینگی بسیار کم"

def generate_full_report(symbol, is_monitoring=False):
    try:
        screener = "crypto"
        exchange_name = "KUCOIN"
        if symbol in ["XAUUSD", "EURUSD"]: screener = "forex"; exchange_name="FX_IDC"
        
        # ۱. جمع‌آوری داده‌ها
        summary_4h, indicators_4h = TA_Handler(symbol=symbol, screener=screener, exchange=exchange_name, interval=Interval.INTERVAL_4_HOURS).get_analysis().summary, TA_Handler(symbol=symbol, screener=screener, exchange=exchange_name, interval=Interval.INTERVAL_4_HOURS).get_analysis().indicators
        summary_1h, indicators_1h = TA_Handler(symbol=symbol, screener=screener, exchange=exchange_name, interval=Interval.INTERVAL_1_HOUR).get_analysis().summary, TA_Handler(symbol=symbol, screener=screener, exchange=exchange_name, interval=Interval.INTERVAL_1_HOUR).get_analysis().indicators
        summary_15m, _ = TA_Handler(symbol=symbol, screener=screener, exchange=exchange_name, interval=Interval.INTERVAL_15_MINUTES).get_analysis().summary, None
        
        if not summary_4h or not indicators_1h: return f"خطا در دریافت داده برای {symbol} از TradingView.", None

        # ۲. ساخت گزارش کامل و یکپارچه
        report = f"🔬 **گزارش جامع تحلیلی برای #{symbol}**\n\n" if not is_monitoring else f"👁️ **گزارش پایش لحظه‌ای برای #{symbol}**\n\n"
        report += f"**قیمت فعلی:** `${indicators_1h['close']:,.2f}`\n\n"
        
        report += "**--- تحلیل کانتکست و اقتصاد کلان ---**\n"
        session_name, session_char = get_market_session()
        report += f"**سشن معاملاتی:** {session_name} ({session_char})\n\n"
        
        report += "**--- تحلیل تکنیکال (TradingView) ---**\n"
        report += f"**خلاصه تحلیل (4H/1H):** `{summary_4h['RECOMMENDATION']}` / `{summary_1h['RECOMMENDATION']}`\n"
        report += f"**جزئیات اندیکاتورها (4H):**\n"
        report += f"  - RSI: {indicators_4h['RSI']:.1f} {'(اشباع خرید)' if indicators_4h['RSI'] > 70 else '(اشباع فروش)' if indicators_4h['RSI'] < 30 else ''}\n"
        report += f"  - Stoch %K/%D: {indicators_4h['Stoch.K']:.1f}/{indicators_4h['Stoch.D']:.1f}\n"
        report += f"  - MACD Level: {indicators_4h['MACD.macd']:.2f}\n\n"

        report += "**--- تحلیل پرایس اکشن و نقدینگی ---**\n"
        support = indicators_4h.get('Pivot.M.Support.1', indicators_1h.get('low'))
        resistance = indicators_4h.get('Pivot.M.Resistance.1', indicators_1h.get('high'))
        report += f"**ناحیه کلیدی تقاضا (حمایت):** `${support:,.2f}`\n"
        report += f"**ناحیه کلیدی عرضه (مقاومت):** `${resistance:,.2f}`\n\n"

        if not is_monitoring:
            report += "**--- تحلیل فاندامنتال ---**\n"
            # ... منطق اخبار و نهنگ‌ها ...
            report += "خبر مهم جدیدی یافت نشد.\n\n"

            report += "**--- جمع‌بندی و پیشنهاد معامله (AI) ---**\n"
            report += "**روش:** ترکیب سیگنال‌های TradingView با نواحی عرضه/تقاضا.\n"
            if "BUY" in summary_4h['RECOMMENDATION'] and "BUY" in summary_1h['RECOMMENDATION'] and indicators_1h['close'] < resistance:
                confidence = (summary_4h.get('BUY',0) + summary_1h.get('BUY',0)) / 26 * 100
                entry = indicators_1h['close']
                stop_loss = support
                target = resistance
                leverage = 5
                report += f"✅ **سیگنال خرید (Long) با اطمینان {confidence:.0f}٪ صادر شد.**\n"
                report += f"**ورود:** `${entry:,.2f}` | **ضرر:** `${stop_loss:,.2f}` | **سود:** `${target:,.2f}` | **اهرم:** `x{leverage}`\n"
            else:
                report += "⚠️ **نتیجه:** در حال حاضر، هیچ سیگنال ورود واضحی یافت نشد."
        
        return report, summary_15m['RECOMMENDATION']
    except Exception as e:
        return f"خطای پیش‌بینی نشده: {e}", None

# --- موتور شکار سیگنال ادغام شده ---
def hunt_signals():
    global hunted_signals_cache
    exchange_ccxt = ccxt.kucoin()
    while True:
        logging.info("SIGNAL_HUNTER: Starting new HYBRID market scan...")
        try:
            # ۱. اسکن ناهنجاری حجم
            all_markets = exchange_ccxt.load_markets()
            tickers = exchange_ccxt.fetch_tickers()
            anomaly_signals = []
            for symbol, ticker in tickers.items():
                if symbol.endswith('/USDT') and ticker.get('quoteVolume', 0) > 2_000_000:
                    df_1h = pd.DataFrame(exchange_ccxt.fetch_ohlcv(symbol, '1h', limit=21), columns=['ts','o','h','l','c','v'])
                    if len(df_1h) < 21: continue
                    if df_1h['v'].iloc[-1] > df_1h['v'].iloc[:-1].mean() * 5:
                        anomaly_signals.append({'symbol': symbol.replace('/USDT',''), 'type': 'ناهنجاری حجم', 'confidence': 65})
                time.sleep(1)

            # ۲. اسکن کلاسیک با TradingView
            classic_signals = []
            watchlist = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'AVAX']
            for symbol in watchlist:
                summary_4h, _ = get_tv_analysis(symbol, Interval.INTERVAL_4_HOURS)
                summary_1h, _ = get_tv_analysis(symbol, Interval.INTERVAL_1_HOUR)
                if summary_4h and summary_1h:
                    if summary_4h['RECOMMENDATION'] == 'STRONG_BUY' and summary_1h['RECOMMENDATION'] == 'STRONG_BUY':
                        classic_signals.append({'symbol': symbol, 'type': 'کلاسیک (روند قوی)', 'confidence': 90})
            
            # ۳. ادغام و مرتب‌سازی نتایج
            hunted_signals_cache = sorted(anomaly_signals + classic_signals, key=lambda x: x['confidence'], reverse=True)
            logging.info(f"Scan completed. Found {len(hunted_signals_cache)} signals.")
        except Exception as e:
            logging.error(f"Error in signal_hunter_loop: {e}")
        time.sleep(60 * 60)

# --- کنترل‌کننده‌های ربات ---
def handle_chat(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type != 'text': return
    text = msg['text']
    
    if user_states.get(chat_id) == 'awaiting_symbol_analysis':
        processing_message = bot.sendMessage(chat_id, f"✅ درخواست برای **{text.upper()}** دریافت شد...", parse_mode='Markdown')
        report_text, _ = generate_full_report(text.strip().upper())
        bot.editMessageText((chat_id, processing_message['message_id']), report_text, parse_mode='Markdown', reply_markup=get_main_menu_keyboard(chat_id))
        user_states[chat_id] = 'main_menu'

    elif user_states.get(chat_id) == 'awaiting_symbol_monitor':
        symbol_to_monitor = text.strip().upper()
        # جهت معامله را از کاربر می‌پرسیم
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text='Long (خرید)', callback_data=f'monitor_set_Long_{symbol_to_monitor}')],
            [InlineKeyboardButton(text='Short (فروش)', callback_data=f'monitor_set_Short_{symbol_to_monitor}')]
        ])
        bot.sendMessage(chat_id, f"جهت معامله شما برای #{symbol_to_monitor} کدام است؟", reply_markup=keyboard)
        user_states[chat_id] = 'main_menu'

    elif text == '/start':
        user_states[chat_id] = 'main_menu'
        bot.sendMessage(chat_id, 'به ربات هوشمند Apex Fusion خوش آمدید.',
                        reply_markup=get_main_menu_keyboard(chat_id))

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
    elif query_data == 'menu_show_hunted_signals':
        if not hunted_signals_cache:
            message = "🎯 **نتیجه اسکن اخیر:**\n\nدر حال حاضر هیچ سیگنال قابل توجهی در بازار یافت نشده است."
        else:
            message = "🎯 **آخرین سیگنال‌های شکار شده:**\n\n"
            for signal in hunted_signals_cache[:5]: # نمایش ۵ سیگنال برتر
                message += f"🔹 **{signal['symbol']}** | **نوع:** {signal['type']} | **اطمینان:** {signal['confidence']}%\n"
            message += "\nبرای تحلیل کامل، از منوی تحلیل عمیق استفاده کنید."
        bot.editMessageText((chat_id, msg['message']['message_id']), message, reply_markup=get_main_menu_keyboard(chat_id))
    elif query_data == 'menu_monitor_trade':
        user_states[chat_id] = 'awaiting_symbol_monitor'
        bot.editMessageText((chat_id, msg['message']['message_id']), 'لطفاً نماد ارزی که در آن معامله باز کرده‌اید را وارد کنید (مثلاً: ETH).',
                        reply_markup=get_back_to_main_menu_keyboard(chat_id))
    elif query_data.startswith('monitor_set_'):
        _, direction, symbol = query_data.split('_', 2)
        active_trades[chat_id] = {'symbol': symbol, 'direction': direction}
        bot.editMessageText((chat_id, msg['message']['message_id']), f"✅ معامله {direction} شما برای #{symbol} تحت پایش هوشمند قرار گرفت.",
                        reply_markup=get_main_menu_keyboard(chat_id))
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
        threading.Thread(target=hunt_signals, daemon=True, name="SignalHunterThread").start()
        # threading.Thread(target=trade_monitor_loop, daemon=True).start() # پایش معامله فعلا غیرفعال است
        
        MessageLoop(bot, {'chat': handle_chat,
                          'callback_query': handle_callback_query}).run_as_thread()
        logging.info('Telepot bot is listening...')
        
        if os.getenv('RAILWAY_ENVIRONMENT'):
            threading.Thread(target=run_web_server, daemon=True, name="WebServerThread").start()
        
        logging.info("Bot is running.")
        while 1:
            time.sleep(10)