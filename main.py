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
from datetime import datetime, timedelta
import pytz
import random

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
anomaly_signals_cache = []
trade_journal = {}
backtest_results_cache = {}
silver_signals_cache = []
signal_history = []

# --- توابع سازنده کیبورد ---
def get_main_menu_keyboard(chat_id):
    buttons = [
        [InlineKeyboardButton(text='🔬 تحلیل عمیق یکپارچه', callback_data='menu_deep_analysis_unified')],
        [InlineKeyboardButton(text='✨ تحلیل عمیق تعاملی', callback_data='menu_deep_analysis_interactive')],
        [InlineKeyboardButton(text='🥈 نمایش سیگنال‌های نقره‌ای', callback_data='menu_show_silver_signals')],
        [InlineKeyboardButton(text='🌋 نمایش سیگنال‌های ناهنجاری', callback_data='menu_anomaly_hunt')],
        [InlineKeyboardButton(text='🐳 رصد نهنگ‌های USDT', callback_data='menu_whale_watch')],
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

def get_interactive_report_keyboard(symbol):
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='ساختار بازار', callback_data=f'show_report_structure_{symbol}'),
         InlineKeyboardButton(text='عرضه/تقاضا', callback_data=f'show_report_liquidity_{symbol}')],
        [InlineKeyboardButton(text='فاندامنتال', callback_data=f'show_report_fundamental_{symbol}'),
         InlineKeyboardButton(text='پیشنهاد AI', callback_data=f'show_report_ai_proposal_{symbol}')],
    ])

# --- موتور بک‌تستینگ واقعی ---
def run_backtest_simulation(symbol):
    if symbol in backtest_results_cache: return backtest_results_cache[symbol]
    try:
        kucoin_exchange = ccxt.kucoin()
        df = pd.DataFrame(kucoin_exchange.fetch_ohlcv(f"{symbol.upper()}/USDT", '4h', limit=500), columns=['ts','o','h','l','c','v'])
        if len(df) < 100: return {"name": "N/A", "details": {"win_rate": 0, "description": "داده ناکافی برای بک‌تست."}}
        
        df.rename(columns={'c': 'close'}, inplace=True)
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], 20)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], 50)
        df['signal'] = 0
        df.loc[df['ema_fast'] > df['ema_slow'], 'signal'] = 1
        df['position'] = df['signal'].diff()
        
        wins, trades = 0, 0
        for i, row in df.iterrows():
            if row['position'] in [2, -2] and i + 5 < len(df):
                trades += 1
                if (row['position'] == 2 and df.iloc[i + 5]['close'] > row['close']) or \
                   (row['position'] == -2 and df.iloc[i + 5]['close'] < row['close']):
                    wins += 1
        
        win_rate = (wins / trades * 100) if trades > 0 else 0
        result = {"name": "EMA_Cross_4H", "details": {"win_rate": win_rate, "description": "تقاطع EMA (20, 50) در تایم ۴ ساعته"}}
        backtest_results_cache[symbol] = result
        return result
    except Exception as e:
        logging.error(f"Error in backtest for {symbol}: {e}")
        return {"name": "N/A", "details": {"win_rate": 0, "description": "خطا در اجرای بک‌تست."}}

# --- موتور تحلیل پیشرفته ---
def get_whale_transactions():
    if not ETHERSCAN_API_KEY: return "سرویس رصد نهنگ‌ها پیکربندی نشده است."
    contract_address = "0xdac17f958d2ee523a2206206994597c13d831ec7" 
    try:
        url = f"https://api.etherscan.io/api?module=account&action=tokentx&contractaddress={contract_address}&page=1&offset=100&sort=desc&apikey={ETHERSCAN_API_KEY}"
        response = requests.get(url).json()
        if response['status'] == '1':
            transactions = response['result']
            report = "🐳 **آخرین تراکنش‌های بزرگ USDT:**\n\n"
            count = 0
            for tx in transactions:
                value = int(tx['value']) / (10**int(tx['tokenDecimal']))
                if value > 500_000:
                    to_address = tx['to']
                    tx_type = "🔥 **به صرافی**" if "binance" in to_address or "kucoin" in to_address else "❄️ **به کیف پول**"
                    report += f"- **مقدار:** `{value:,.0f} USDT` ({tx_type})\n"
                    count += 1
                    if count >= 5: break
            return report if count > 0 else "تراکنش بزرگ جدیدی یافت نشد."
        else:
            return "خطا در دریافت اطلاعات از Etherscan."
    except Exception as e:
        return f"خطا در سرویس رصد نهنگ: {e}"

def generate_full_report_data(symbol):
    # ... (کد کامل این تابع از پاسخ قبلی) ...
    pass

def generate_full_report(symbol):
    # ... (کد کامل این تابع با جزئیات کامل از پاسخ قبلی) ...
    pass

# --- موتور شکار سیگنال (ادغام شده) ---
def hunt_signals():
    # ... (کد کامل این تابع با دو موتور ادغام شده) ...
    pass

# --- پایش معامله ---
def trade_monitor_loop():
    # ... (کد کامل این تابع) ...
    pass

# --- کنترل‌کننده‌های ربات ---
def handle_chat(msg):
    # ... (کد کامل handle_chat با تمام منطق‌ها) ...
    pass

def handle_callback_query(msg):
    # ... (کد کامل handle_callback_query با تمام دکمه‌ها) ...
    pass

# --- راه‌اندازی ربات ---
def run_web_server():
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == '__main__':
    if not TELEGRAM_TOKEN:
        logging.fatal("TELEGRAM_TOKEN not found!")
    else:
        threading.Thread(target=hunt_signals, daemon=True).start()
        threading.Thread(target=trade_monitor_loop, daemon=True).start()
        
        MessageLoop(bot, {'chat': handle_chat, 'callback_query': handle_callback_query}).run_as_thread()
        logging.info('Telepot bot is listening...')
        
        if os.getenv('RAILWAY_ENVIRONMENT'):
            threading.Thread(target=run_web_server, daemon=True).start()
        
        logging.info("Bot is running.")
        while 1:
            time.sleep(10)