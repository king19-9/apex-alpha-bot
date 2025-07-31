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
from tradingview_ta import TA_Handler, Interval
import investpy
from datetime import datetime
import pytz

# --- تنظیمات اولیه ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY')

# --- کلاینت‌ها و سرویس‌ها ---
app = FastAPI()
bot = telepot.Bot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
user_states = {}
active_trades = {}
signal_hunt_subscribers = set()
silver_signals_cache = []
trade_journal = {}
backtest_results_cache = {}

# --- توابع سازنده کیبورد ---
def get_main_menu_keyboard(chat_id):
    buttons = [
        [InlineKeyboardButton(text='🔬 تحلیل عمیق تعاملی', callback_data='menu_interactive_analysis')],
        [InlineKeyboardButton(text='🌋 نمایش سیگنال‌های ناهنجاری', callback_data='menu_anomaly_hunt')],
        [InlineKeyboardButton(text='🐳 رصد نهنگ‌های USDT', callback_data='menu_whale_watch')],
    ]
    if chat_id in signal_hunt_subscribers:
        buttons.append([InlineKeyboardButton(text='𔕕 غیرفعال کردن نوتیفیکیشن طلایی', callback_data='menu_toggle_signal_hunt')])
    else:
        buttons.append([InlineKeyboardButton(text='🔔 فعال کردن نوتیفیکیشن طلایی', callback_data='menu_toggle_signal_hunt')])
    if chat_id in active_trades:
        buttons.append([InlineKeyboardButton(text=f"🚫 توقف پایش معامله {active_trades[chat_id]['symbol']}", callback_data=f"monitor_stop_{active_trades[chat_id]['symbol']}")])
    else:
        buttons.append([InlineKeyboardButton(text='👁️ پایش معامله باز', callback_data='menu_monitor_trade')])
    buttons.append([InlineKeyboardButton(text='✍️ ثبت در ژورنال معاملاتی', callback_data='menu_journal')])
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_back_to_main_menu_keyboard(chat_id):
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text='🔙 بازگشت به منوی اصلی', callback_data=f'main_menu_{chat_id}')]])

def get_interactive_report_keyboard(symbol):
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='ساختار بازار', callback_data=f'show_report_structure_{symbol}'),
         InlineKeyboardButton(text='عرضه/تقاضا', callback_data=f'show_report_liquidity_{symbol}')],
        [InlineKeyboardButton(text='فاندامنتال', callback_data=f'show_report_fundamental_{symbol}'),
         InlineKeyboardButton(text='پیشنهاد AI', callback_data=f'show_report_ai_proposal_{symbol}')],
    ])

# --- موتور بک‌تستینگ واقعی (ساده شده) ---
def run_backtest_simulation(symbol):
    if symbol in backtest_results_cache: return backtest_results_cache[symbol]
    try:
        kucoin_exchange = ccxt.kucoin()
        df = pd.DataFrame(kucoin_exchange.fetch_ohlcv(f"{symbol}/USDT", '4h', limit=500), columns=['ts','o','h','l','c','v'])
        if len(df) < 100: return {"name": "N/A", "details": {"win_rate": 0, "description": "داده کافی برای بک‌تست نیست."}}
        
        df['ema_fast'] = ta.trend.ema_indicator(df['c'], 20)
        df['ema_slow'] = ta.trend.ema_indicator(df['c'], 50)
        df['signal'] = 0
        df.loc[df['ema_fast'] > df['ema_slow'], 'signal'] = 1
        df['position'] = df['signal'].diff()
        
        wins, trades = 0, 0
        for i, row in df.iterrows():
            if row['position'] in [2, -2] and i + 5 < len(df):
                trades += 1
                if (row['position'] == 2 and df.iloc[i+5]['c'] > row['c']) or \
                   (row['position'] == -2 and df.iloc[i+5]['c'] < row['c']):
                    wins += 1
        
        win_rate = (wins / trades * 100) if trades > 0 else 0
        result = {"name": "EMA_Cross", "details": {"win_rate": win_rate, "description": "تقاطع میانگین متحرک ۲۰ و ۵۰ در تایم ۴ ساعته"}}
        backtest_results_cache[symbol] = result
        return result
    except Exception:
        return {"name": "N/A", "details": {"win_rate": 0, "description": "خطا در اجرای بک‌تست."}}

# --- موتور تحلیل پیشرفته ---
def get_whale_transactions():
    if not ETHERSCAN_API_KEY: return "سرویس رصد نهنگ‌ها پیکربندی نشده است. (نیاز به کلید API از Etherscan)"
    contract_address = "0xdac17f958d2ee523a2206206994597c13d831ec7" 
    try:
        url = f"https://api.etherscan.io/api?module=account&action=tokentx&contractaddress={contract_address}&page=1&offset=100&sort=desc&apikey={ETHERSCAN_API_KEY}"
        response = requests.get(url).json()
        if response['status'] == '1':
            transactions = response['result']
            report = "🐳 **آخرین تراکنش‌های بزرگ USDT در شبکه اتریوم:**\n\n"
            count = 0
            for tx in transactions:
                value = int(tx['value']) / (10**int(tx['tokenDecimal']))
                if value > 500_000:
                    to_address = tx['to']
                    tx_type = "🔥 **به صرافی (احتمال فروش)**" if "Binance" in to_address or "KuCoin" in to_address else "❄️ **به کیف پول (احتمال نگهداری)**"
                    report += f"- **مقدار:** `{value:,.0f} USDT`\n  **به:** `...{to_address[-8:]}` ({tx_type})\n"
                    count += 1
                    if count >= 5: break
            return report if count > 0 else "تراکنش بزرگ جدیدی یافت نشد."
        else:
            return "خطا در دریافت اطلاعات از Etherscan."
    except Exception as e:
        return f"خطا در سرویس رصد نهنگ: {e}"

def generate_full_report_data(symbol):
    screener = "crypto"
    if symbol in ["XAUUSD", "EURUSD"]: screener = "forex"
    try:
        handler_4h = TA_Handler(symbol=symbol, screener=screener, exchange="KUCOIN", interval=Interval.INTERVAL_4_HOURS)
        handler_1h = TA_Handler(symbol=symbol, screener=screener, exchange="KUCOIN", interval=Interval.INTERVAL_1_HOUR)
        data = { "summary_4h": handler_4h.get_analysis().summary, "indicators_4h": handler_4h.get_analysis().indicators,
                 "summary_1h": handler_1h.get_analysis().summary, "indicators_1h": handler_1h.get_analysis().indicators }
        return data
    except Exception:
        return None

# --- موتور شکار سیگنال (ترکیبی) ---
def hunt_signals():
    global anomaly_signals_cache
    exchange_ccxt = ccxt.kucoin()
    while True:
        logging.info("SIGNAL_HUNTER: Starting new HYBRID market scan...")
        try:
            all_markets = exchange_ccxt.load_markets()
            usdt_pairs = {s: m for s, m in all_markets.items() if s.endswith('/USDT')}
            tickers = exchange_ccxt.fetch_tickers(list(usdt_pairs.keys()))
            temp_anomaly_signals = []
            for symbol, ticker in tickers.items():
                volume_usd = ticker.get('quoteVolume', 0)
                if volume_usd > 1_000_000:
                    df_1h = pd.DataFrame(exchange_ccxt.fetch_ohlcv(symbol, '1h', limit=21), columns=['ts','o','h','l','c','v'])
                    if len(df_1h) < 21: continue
                    avg_volume = df_1h['v'].iloc[:-1].mean()
                    last_volume = df_1h['v'].iloc[-1]
                    if last_volume > avg_volume * 5:
                        temp_anomaly_signals.append({'symbol': symbol.replace('/USDT',''), 'reason': f"افزایش ناگهانی حجم ({last_volume/avg_volume:.1f}x)"})
                time.sleep(1)
            anomaly_signals_cache = temp_anomaly_signals
            logging.info(f"ANOMALY_SCAN: Found {len(anomaly_signals_cache)} volume anomalies.")
        except Exception as e:
            logging.error(f"Error in anomaly hunter: {e}")
        time.sleep(60 * 60)

# --- کنترل‌کننده‌های ربات ---
def handle_chat(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type != 'text': return
    text = msg['text']
    
    if user_states.get(chat_id) == 'awaiting_interactive_analysis':
        symbol = text.strip().upper()
        processing_message = bot.sendMessage(chat_id, f"✅ درخواست برای **{symbol}** دریافت شد...", parse_mode='Markdown')
        message = f"🔬 **تحلیل تعاملی برای #{symbol}**\n\nلطفاً بخش مورد نظر خود را برای مشاهده جزئیات انتخاب کنید:"
        keyboard = get_interactive_report_keyboard(symbol)
        bot.editMessageText((chat_id, processing_message['message_id']), message, parse_mode='Markdown', reply_markup=keyboard)
        user_states[chat_id] = 'main_menu'

    elif user_states.get(chat_id) == 'awaiting_journal_entry':
        try:
            parts = text.split(',')
            symbol, entry, result = parts[0].strip().upper(), float(parts[1]), parts[2].strip().title()
            if chat_id not in trade_journal: trade_journal[chat_id] = []
            trade_journal[chat_id].append({'symbol': symbol, 'entry': entry, 'result': result})
            bot.sendMessage(chat_id, "✅ معامله شما با موفقیت در ژورنال ثبت شد.", reply_markup=get_main_menu_keyboard(chat_id))
        except:
            bot.sendMessage(chat_id, "❌ فرمت ورودی اشتباه است. لطفاً دوباره تلاش کنید.")
        user_states[chat_id] = 'main_menu'
        
    elif text == '/start':
        user_states[chat_id] = 'main_menu'
        bot.sendMessage(chat_id, 'به ربات هوشمند Apex Singularity خوش آمدید.',
                        reply_markup=get_main_menu_keyboard(chat_id))

    elif text == '/stats':
        stats_message = "📊 **آمار عملکرد سیگنال‌ها (آزمایشی)**\n\nاین قابلیت در حال توسعه است."
        if chat_id in trade_journal:
            journal = trade_journal[chat_id]
            wins = sum(1 for t in journal if t['result'] == 'Win')
            win_rate = (wins / len(journal) * 100) if journal else 0
            stats_message += f"\n\n**--- آمار ژورنال شخصی شما ---**\n"
            stats_message += f"- **تعداد کل معاملات ثبت شده:** {len(journal)}\n"
            stats_message += f"- **نرخ موفقیت شخصی:** {win_rate:.1f}%"
        bot.sendMessage(chat_id, stats_message, parse_mode='Markdown')

def handle_callback_query(msg):
    query_id, from_id, query_data = telepot.glance(msg, flavor='callback_query')
    chat_id = from_id
    bot.answerCallbackQuery(query_id)
    
    if query_data.startswith('main_menu'):
        user_states[chat_id] = 'main_menu'
        bot.editMessageText((chat_id, msg['message']['message_id']), 'منوی اصلی:', reply_markup=get_main_menu_keyboard(chat_id))
    elif query_data == 'menu_interactive_analysis':
        user_states[chat_id] = 'awaiting_interactive_analysis'
        bot.editMessageText((chat_id, msg['message']['message_id']), 'لطفاً نماد ارز را برای تحلیل وارد کنید (مثلاً: BTC).',
                        reply_markup=get_back_to_main_menu_keyboard(chat_id))
    elif query_data == 'menu_anomaly_hunt':
        if not anomaly_signals_cache:
            message = "🌋 **رادار ناهنجاری:**\n\nدر اسکن اخیر، هیچ ارز با افزایش حجم ناگهانی یافت نشد."
        else:
            message = "🌋 **رادار ناهنجاری (افزایش حجم):**\n\n"
            for signal in anomaly_signals_cache[:5]:
                message += f"🔹 **{signal['symbol']}** ({signal['reason']})\n"
        bot.editMessageText((chat_id, msg['message']['message_id']), message, reply_markup=get_main_menu_keyboard(chat_id))
    elif query_data == 'menu_whale_watch':
        processing_message = bot.editMessageText((chat_id, msg['message']['message_id']), "در حال رصد شبکه برای یافتن تراکنش‌های نهنگ‌ها...")
        report = get_whale_transactions()
        bot.editMessageText((chat_id, processing_message['message_id']), report, parse_mode='Markdown', reply_markup=get_main_menu_keyboard(chat_id))
    elif query_data == 'menu_journal':
        user_states[chat_id] = 'awaiting_journal_entry'
        message = "✍️ **ثبت معامله در ژورنال:**\n\nلطفاً جزئیات معامله خود را با فرمت زیر ارسال کنید:\n`نماد, قیمت ورود, نتیجه`\n\n**مثال:**\n`BTC, 65000, Win`"
        bot.editMessageText((chat_id, msg['message']['message_id']), message, parse_mode='Markdown', reply_markup=get_back_to_main_menu_keyboard(chat_id))
    elif query_data.startswith('show_report_'):
        parts = query_data.split('_')
        section, symbol = parts[2], parts[3]
        
        bot.sendMessage(chat_id, f"در حال آماده‌سازی بخش '{section}' برای #{symbol}...")
        
        data = generate_full_report_data(symbol)
        if not data:
            bot.sendMessage(chat_id, "خطا در دریافت داده برای این بخش.")
            return

        report_section = f"**--- {section.replace('_', ' ')} ---**\n\n"
        if section == 'structure':
            report_section += f"**تحلیل TradingView (4H):** `{data['summary_4h']['RECOMMENDATION']}`\n"
            report_section += f"**تحلیل TradingView (1H):** `{data['summary_1h']['RECOMMENDATION']}`\n"
        elif section == 'liquidity':
             report_section += f"**حمایت:** `${data['indicators_4h'].get('Pivot.M.Support.1', 'N/A'):,.2f}`\n"
             report_section += f"**مقاومت:** `${data['indicators_4h'].get('Pivot.M.Resistance.1', 'N/A'):,.2f}`\n"
        elif section == 'fundamental':
            report_section += "منطق تحلیل فاندامنتال در اینجا نمایش داده می‌شود..."
        elif section == 'ai_proposal':
            report_section += "منطق پیشنهاد AI در اینجا نمایش داده می‌شود..."
        bot.sendMessage(chat_id, report_section, parse_mode='Markdown')

# --- راه‌اندازی ربات ---
def run_web_server():
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == '__main__':
    if not TELEGRAM_TOKEN:
        logging.fatal("TELEGRAM_TOKEN not found!")
    else:
        threading.Thread(target=hunt_signals, daemon=True, name="SignalHunterThread").start()
        
        MessageLoop(bot, {'chat': handle_chat,
                          'callback_query': handle_callback_query}).run_as_thread()
        logging.info('Telepot bot is listening...')
        
        if os.getenv('RAILWAY_ENVIRONMENT'):
            threading.Thread(target=run_web_server, daemon=True, name="WebServerThread").start()
        
        logging.info("Bot is running.")
        while 1:
            time.sleep(10)