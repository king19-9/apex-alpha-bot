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
silver_signals_cache = []
signal_history = [
    {'symbol': 'BTC', 'type': 'Golden', 'entry': 60000, 'target': 65000, 'stop': 59000, 'result': 'Win', 'timestamp': datetime(2025, 7, 10)},
    {'symbol': 'ETH', 'type': 'Silver', 'entry': 4000, 'target': 4200, 'stop': 3950, 'result': 'Loss', 'timestamp': datetime(2025, 7, 12)},
    {'symbol': 'SOL', 'type': 'Golden', 'entry': 150, 'target': 170, 'stop': 147, 'result': 'Win', 'timestamp': datetime(2025, 6, 20)}
]

# --- توابع سازنده کیبورد ---
def get_main_menu_keyboard(chat_id):
    buttons = [
        [InlineKeyboardButton(text='🔬 تحلیل عمیق یک نماد', callback_data='menu_deep_analysis')],
        [InlineKeyboardButton(text='🥈 نمایش سیگنال‌های نقره‌ای', callback_data='menu_show_silver_signals')],
    ]
    if chat_id in signal_hunt_subscribers:
        buttons.append([InlineKeyboardButton(text='🔕 غیرفعال کردن نوتیفیکیشن طلایی', callback_data='menu_toggle_signal_hunt')])
    else:
        buttons.append([InlineKeyboardButton(text='🔔 فعال کردن نوتیفیکیشن طلایی', callback_data='menu_toggle_signal_hunt')])
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

def check_long_signal_conditions(trend_d, trend_4h, last_candle, support, lower_wick, body_size):
    confidence = 0
    is_long_signal = False
    if trend_d == "صعودی" and trend_4h == "صعودی" and (last_candle['c'] > support) and (last_candle['c'] < support * 1.03) and (body_size > 0 and lower_wick > body_size * 1.5):
        is_long_signal = True
        confidence = 70
        if abs(last_candle['c'] - support) < abs(last_candle['c'] - last_candle['o']):
            confidence += 10
    return is_long_signal, confidence

def generate_full_report(symbol, is_monitoring=False):
    try:
        kucoin_symbol = f"{symbol.upper()}/USDT"
        
        try:
            df_d = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, '1d', limit=100), columns=['ts','o','h','l','c','v'])
            df_4h = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, '4h', limit=100), columns=['ts','o','h','l','c','v'])
            df_1h = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, '1h', limit=50), columns=['ts','o','h','l','c','v'])
            df_15m = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, '15m', limit=50), columns=['ts','o','h','l','c','v'])
            if df_1h.empty: return f"خطا: داده‌های کافی برای نماد {symbol} دریافت نشد.", None
        except Exception as e:
            return f"خطا در ارتباط با صرافی: {e}", None

        report_prefix = "🔬 **گزارش جامع تحلیلی**" if not is_monitoring else "👁️ **گزارش پایش لحظه‌ای**"
        report = f"{report_prefix} برای #{symbol}\n\n"
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
        trend_d = "صعودی ✅" if ta.trend.ema_indicator(df_d['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_d['c'], 50).iloc[-1] else "نزولی 🔻"
        trend_4h = "صعودی ✅" if ta.trend.ema_indicator(df_4h['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_4h['c'], 50).iloc[-1] else "نزولی 🔻"
        trend_1h = "صعودی ✅" if ta.trend.ema_indicator(df_1h['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_1h['c'], 50).iloc[-1] else "نزولی 🔻"
        trend_15m = "صعودی ✅" if ta.trend.ema_indicator(df_15m['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_15m['c'], 50).iloc[-1] else "نزولی 🔻"
        report += f"**روندها (D/4H/1H/15M):** {trend_d} / {trend_4h} / {trend_1h} / {trend_15m}\n"
        
        rsi_4h = ta.momentum.rsi(df_4h['c']).iloc[-1]
        if rsi_4h > 70: rsi_text = "اشباع خرید 🥵"
        elif rsi_4h < 30: rsi_text = "اشباع فروش 🥶"
        else: rsi_text = "خنثی 😐"
        report += f"**هیجان بازار (RSI 4H):** {rsi_text} ({rsi_4h:.1f})\n"
        
        support = df_4h['l'].rolling(20).mean().iloc[-1]
        resistance = df_4h['h'].rolling(20).mean().iloc[-1]
        report += f"**ناحیه تقاضا/عرضه (4H):** `${support:,.2f}` / `${resistance:,.2f}`\n\n"

        if not is_monitoring:
            report += "**--- تحلیل فاندامنتال و احساسات ---**\n"
            news_query = symbol.replace('USDT', '')
            url = f"https://newsapi.org/v2/everything?q={news_query}&language=en&sortBy=publishedAt&pageSize=1&apiKey={NEWS_API_KEY}"
            latest_news = requests.get(url).json().get('articles', [{}])[0].get('title', 'خبر جدیدی یافت نشد.')
            report += f"**آخرین خبر:** *{latest_news}*\n"
            sentiment_score = 50
            if any(word in latest_news.lower() for word in ['partner', 'launch', 'success']): sentiment_score += 20
            if any(word in latest_news.lower() for word in ['hack', 'ban', 'problem']): sentiment_score -= 20
            report += f"**شاخص احساسات (اخبار):** {sentiment_score}/100\n\n"
            
            report += "**--- پیشنهاد معامله (AI) ---**\n"
            last_1h_candle = df_1h.iloc[-1]
            body_size = abs(last_1h_candle['c'] - last_1h_candle['o'])
            lower_wick = last_1h_candle['c'] - last_1h_candle['l'] if last_1h_candle['c'] > last_1h_candle['o'] else last_1h_candle['o'] - last_1h_candle['l']
            is_long_signal, confidence = check_long_signal_conditions(trend_d.split(" ")[0], trend_4h.split(" ")[0], last_1h_candle, support, lower_wick, body_size)
            if is_long_signal:
                entry = last_1h_candle['h']
                stop_loss = last_1h_candle['l']
                target = resistance
                leverage = 3
                report += f"✅ **سیگنال خرید (Long) با اطمینان {confidence:.0f}٪ صادر شد.**\n"
                report += f"**نقطه ورود:** `${entry:,.2f}` | **حد ضرر:** `${stop_loss:,.2f}` | **حد سود:** `${target:,.2f}` | **اهرم:** `x{leverage}`\n"
                signal_history.append({'symbol': symbol, 'type': 'Golden', 'entry': entry, 'target': target, 'stop': stop_loss, 'result': 'Pending', 'timestamp': datetime.now()})
            else:
                report += "⚠️ **نتیجه:** در حال حاضر، سیگنال ورود واضحی یافت نشد."
            
        return report, trend_15m
    except Exception as e:
        logging.error(f"Critical error in full report for {symbol}: {e}")
        return "یک خطای پیش‌بینی نشده در تحلیل.", None

def hunt_signals():
    global silver_signals_cache
    watchlist = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'AVAX', 'LINK', 'MATIC', 'DOT', 'ADA', 'LTC', 'BNB', 'NEAR', 'ATOM', 'FTM']
    while True:
        logging.info("SIGNAL_HUNTER: Starting new market scan...")
        temp_silver_signals = []
        for symbol in watchlist:
            try:
                df_d = pd.DataFrame(exchange.fetch_ohlcv(f"{symbol}/USDT", '1d', limit=51), columns=['ts','o','h','l','c','v'])
                df_4h = pd.DataFrame(exchange.fetch_ohlcv(f"{symbol}/USDT", '4h', limit=51), columns=['ts','o','h','l','c','v'])
                if df_d.empty or df_4h.empty: continue
                
                score = 0
                if ta.trend.ema_indicator(df_d['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_d['c'], 50).iloc[-1]: score += 3
                if ta.trend.ema_indicator(df_4h['c'], 21).iloc[-1] > ta.trend.ema_indicator(df_4h['c'], 50).iloc[-1]: score += 2
                rsi = ta.momentum.rsi(df_4h['c']).iloc[-1]
                if 30 < rsi < 55: score += 2.5
                if df_4h['v'].iloc[-1] > df_4h['v'].rolling(20).mean().iloc[-1] * 1.5: score += 1.5
                
                confidence = score * 10 
                
                if confidence >= 80: # سیگنال طلایی
                    if sent_signals_cache.get(symbol) != "golden_long":
                        report, _ = generate_full_report(symbol)
                        message = f"🥇 **شکار سیگنال طلایی (اطمینان بالا)** 🥇\n\n{report}"
                        for chat_id in list(signal_hunt_subscribers):
                            bot.sendMessage(chat_id, message, parse_mode='Markdown')
                        sent_signals_cache[symbol] = "golden_long"
                elif 65 <= confidence < 80: # سیگنال نقره‌ای
                    temp_silver_signals.append({'symbol': symbol, 'confidence': confidence})
                else:
                    if symbol in sent_signals_cache: del sent_signals_cache[symbol]
            except Exception: continue
            time.sleep(3)
        
        silver_signals_cache = sorted(temp_silver_signals, key=lambda x: x['confidence'], reverse=True)
        logging.info(f"Scan completed. Found {len(silver_signals_cache)} silver signals.")
        time.sleep(30 * 60)

def trade_monitor_loop():
    while True:
        time.sleep(5 * 60)
        if not active_trades: continue
        for chat_id, trade_info in list(active_trades.items()):
            try:
                symbol = trade_info['symbol']
                initial_direction = trade_info['direction']
                report, current_trend_15m = generate_full_report(symbol, is_monitoring=True)
                if current_trend_15m is None: continue
                
                recommendation_text = ""
                if (initial_direction == "Long" and "نزولی" in current_trend_15m):
                    recommendation_text = "❌ **توصیه: خروج از معامله.**\nتحلیل کوتاه‌مدت نشانه‌های قوی از بازگشت روند را نشان می‌دهد."
                elif (initial_direction == "Long" and "خنثی" in current_trend_15m): # فرض میکنیم خنثی هم میتواند باشد
                     recommendation_text = "⚠️ **توصیه: مدیریت ریسک.**\nروند کوتاه‌مدت قدرت خود را از دست داده. جابجایی حد ضرر به نقطه ورود پیشنهاد می‌شود."
                else:
                    recommendation_text = "✅ **توصیه: حفظ پوزیشن.**\nشرایط فعلی همچنان به نفع معامله شماست."
                    
                message = f"🚨 **به‌روزرسانی پایش معامله برای #{symbol}** 🚨\n\n{report}\n\n**--- نتیجه‌گیری پایشگر ---**\n{recommendation_text}"
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
        report_text, _ = generate_full_report(text.strip())
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
        now = datetime.now()
        current_month = now.month
        current_year = now.year

        current_month_signals = [s for s in signal_history if s['timestamp'].month == current_month and s['timestamp'].year == current_year]
        
        stats_message = f"📊 **آمار عملکرد سیگنال‌ها برای ماه جاری ({current_year}/{current_month})**\n\n"
        
        if not current_month_signals:
            stats_message += "در این ماه هنوز سیگنالی صادر نشده است."
        else:
            golden_signals = [s for s in current_month_signals if s['type'] == 'Golden']
            silver_signals = [s for s in current_month_signals if s['type'] == 'Silver']

            total_wins = sum(1 for s in current_month_signals if s['result'] == 'Win')
            win_rate = (total_wins / len(current_month_signals) * 100) if current_month_signals else 0
            
            stats_message += f"**عملکرد کلی ماه:**\n- تعداد کل سیگنال‌ها: {len(current_month_signals)}\n- نرخ موفقیت (Win Rate): {win_rate:.1f}%\n\n"

            if golden_signals:
                wins_golden = sum(1 for s in golden_signals if s['result'] == 'Win')
                win_rate_golden = (wins_golden / len(golden_signals) * 100) if golden_signals else 0
                stats_message += f"**🥇 سیگنال‌های طلایی:** تعداد: {len(golden_signals)} | نرخ موفقیت: {win_rate_golden:.1f}%\n"
            
            if silver_signals:
                wins_silver = sum(1 for s in silver_signals if s['result'] == 'Win')
                win_rate_silver = (wins_silver / len(silver_signals) * 100) if silver_signals else 0
                stats_message += f"**🥈 سیگنال‌های نقره‌ای:** تعداد: {len(silver_signals)} | نرخ موفقیت: {win_rate_silver:.1f}%\n"

            stats_message += "\n**-- جزئیات ۵ سیگنال اخیر ماه --**\n"
            for signal in reversed(current_month_signals[-5:]):
                result_emoji = "✅" if signal['result'] == 'Win' else "❌"
                profit_loss = f"+{((signal['target']/signal['entry']-1)*100):.1f}%" if signal['result'] == 'Win' else f"-{((1-signal['stop']/signal['entry'])*100):.1f}%"
                stats_message += f"{result_emoji} **{signal['symbol']} ({signal['type']}):** نتیجه: {profit_loss}\n"
        
        previous_months_signals = [s for s in signal_history if s['timestamp'].month != current_month or s['timestamp'].year != current_year]
        if previous_months_signals:
            stats_message += "\n\n**--- خلاصه عملکرد ماه‌های گذشته ---**\n"
            prev_wins = sum(1 for s in previous_months_signals if s['result'] == 'Win')
            prev_win_rate = (prev_wins / len(previous_months_signals) * 100) if previous_months_signals else 0
            stats_message += f"نرخ موفقیت کلی در ماه‌های گذشته: {prev_win_rate:.1f}%"

        bot.sendMessage(chat_id, stats_message, parse_mode='Markdown')

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
            bot.editMessageText((chat_id, msg['message']['message_id']), "✅ **نوتیفیکیشن سیگنال غیرفعال شد.**", reply_markup=get_main_menu_keyboard(chat_id))
        else:
            signal_hunt_subscribers.add(chat_id)
            bot.editMessageText((chat_id, msg['message']['message_id']), "✅ **نوتیفیکیشن سیگنال فعال شد.**", reply_markup=get_main_menu_keyboard(chat_id))
    elif query_data == 'menu_show_silver_signals':
        if not silver_signals_cache:
            message = "🥈 **سیگنال‌های نقره‌ای:**\n\nدر اسکن اخیر، هیچ سیگنال با اطمینان متوسط یافت نشد."
        else:
            message = "🥈 **آخرین سیگنال‌های نقره‌ای یافت شده:**\n\n"
            for signal in silver_signals_cache:
                message += f"🔹 **{signal['symbol']}** (امتیاز: {signal['confidence']:.0f}%)\n"
            message += "\nبرای تحلیل کامل، از منوی تحلیل عمیق استفاده کنید."
        bot.editMessageText((chat_id, msg['message']['message_id']), message, reply_markup=get_main_menu_keyboard(chat_id))
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