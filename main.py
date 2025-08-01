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
exchange = ccxt.kucoin()
bot = telepot.Bot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
user_states = {}
active_trades = {}
signal_hunt_subscribers = set()
anomaly_signals_cache = []
trade_journal = {}
backtest_results_cache = {}
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
    buttons.append([InlineKeyboardButton(text='✍️ ثبت در ژورنال معاملاتی', callback_data='menu_journal')])
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_back_to_main_menu_keyboard(chat_id):
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text='🔙 بازگشت به منوی اصلی', callback_data=f'main_menu_{chat_id}')]])

def get_interactive_report_keyboard(symbol):
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='ساختار بازار', callback_data=f'show_report_structure_{symbol}'),
         InlineKeyboardButton(text='عرضه/تقاضا و نقدینگی', callback_data=f'show_report_liquidity_{symbol}')],
        [InlineKeyboardButton(text='فاندامنتال و اقتصاد کلان', callback_data=f'show_report_fundamental_{symbol}'),
         InlineKeyboardButton(text='پیشنهاد AI', callback_data=f'show_report_ai_proposal_{symbol}')],
    ])

# --- موتور بک‌تستینگ و تحلیل پیشرفته ---
def run_backtest_simulation(symbol):
    symbol_upper = symbol.upper()
    if symbol_upper in backtest_results_cache: return backtest_results_cache[symbol_upper]
    try:
        df = pd.DataFrame(exchange.fetch_ohlcv(f"{symbol_upper}/USDT", '4h', limit=500), columns=['ts','o','h','l','c','v'])
        if len(df) < 100: return {"name": "N/A", "details": {"win_rate": 0, "description": "داده ناکافی برای بک‌تست."}}
        
        df['ema_fast'] = ta.trend.ema_indicator(df['c'], 20); df['ema_slow'] = ta.trend.ema_indicator(df['c'], 50)
        df['signal'] = 0; df.loc[df['ema_fast'] > df['ema_slow'], 'signal'] = 1; df['position'] = df['signal'].diff()
        
        wins, trades = 0, 0
        for i, row in df.iterrows():
            if row['position'] in [2, -2] and i + 5 < len(df):
                trades += 1
                if (row['position'] == 2 and df.iloc[i + 5]['c'] > row['c']) or (row['position'] == -2 and df.iloc[i + 5]['c'] < row['c']):
                    wins += 1
        
        win_rate = (wins / trades * 100) if trades > 0 else 0
        result = {"name": "EMA_Cross_4H", "details": {"win_rate": win_rate, "description": "تقاطع EMA (20, 50) در تایم ۴ ساعته"}}
        backtest_results_cache[symbol_upper] = result
        return result
    except Exception as e:
        logging.error(f"Error in backtest for {symbol}: {e}")
        return {"name": "N/A", "details": {"win_rate": 0, "description": "خطا در اجرای بک‌تست."}}

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

def generate_full_report(symbol, is_monitoring=False):
    try:
        kucoin_symbol = f"{symbol.upper()}/USDT"
        
        # ۱. جمع‌آوری داده‌ها
        df_d = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, '1d', limit=100), columns=['ts','o','h','l','c','v'])
        df_4h = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, '4h', limit=100), columns=['ts','o','h','l','c','v'])
        df_1h = pd.DataFrame(exchange.fetch_ohlcv(kucoin_symbol, '1h', limit=50), columns=['ts','o','h','l','c','v'])
        
        summary_4h, indicators_4h = TA_Handler(symbol=symbol, screener="crypto", exchange="KUCOIN", interval=Interval.INTERVAL_4_HOURS).get_analysis().summary, TA_Handler(symbol=symbol, screener="crypto", exchange="KUCOIN", interval=Interval.INTERVAL_4_HOURS).get_analysis().indicators
        summary_1h, indicators_1h = TA_Handler(symbol=symbol, screener="crypto", exchange="KUCOIN", interval=Interval.INTERVAL_1_HOUR).get_analysis().summary, TA_Handler(symbol=symbol, screener="crypto", exchange="KUCOIN", interval=Interval.INTERVAL_1_HOUR).get_analysis().indicators
        
        if df_1h.empty or not summary_1h: return f"خطا در دریافت داده برای {symbol}.", None

        # ۲. ساخت گزارش
        report = f"🔬 **گزارش جامع برای #{symbol}**\n\n"
        last_price = df_1h.iloc[-1]['c']
        report += f"**قیمت فعلی:** `${last_price:,.2f}`\n\n"
        
        # ۳. پیشنهاد معامله فوق هوشمند (بخش اصلی جدید)
        report += "**--- جمع‌بندی و پیشنهاد معامله (Hyper-Intelligent AI) ---**\n"
        
        # فاکتورها
        tv_buy_score = summary_4h.get('BUY', 0) + summary_1h.get('BUY', 0)
        tv_sell_score = summary_4h.get('SELL', 0) + summary_1h.get('SELL', 0)
        is_uptrend_aligned = "BUY" in summary_4h['RECOMMENDATION']
        support = indicators_4h.get('Pivot.M.Support.1', df_4h['l'].rolling(20).mean().iloc[-1])
        is_near_support = last_price < support * 1.02
        is_liquidity_sweep = (df_1h.iloc[-1]['l'] < df_1h.iloc[-2]['l']) and (df_1h.iloc[-1]['c'] > df_1h.iloc[-2]['l'])
        events_df = investpy.economic_calendar()
        has_high_impact_news = not events_df[events_df['importance'] == 'high'].empty

        # سیستم امتیازدهی
        long_confidence = 0
        if tv_buy_score > tv_sell_score * 1.5: long_confidence += 30
        if is_uptrend_aligned: long_confidence += 25
        if is_near_support: long_confidence += 20
        if is_liquidity_sweep: long_confidence += 25
        if has_high_impact_news: long_confidence -= 40 # جریمه سنگین برای اخبار پرخطر

        # صدور سیگنال
        if long_confidence >= 75:
            resistance = indicators_4h.get('Pivot.M.Resistance.1', df_4h['h'].rolling(20).mean().iloc[-1])
            entry = last_price
            stop_loss = support * 0.99
            target1 = resistance
            risk = entry - stop_loss
            target2 = entry + (risk * 2.5)
            leverage = 5

            report += f"✅ **سیگنال خرید (Long) با اطمینان {long_confidence:.0f}٪ صادر شد.**\n"
            report += f"**منطق:** هم‌راستایی قوی بین تحلیل TV، ساختار روند و تاییدیه پرایس اکشن.\n"
            report += f"**نقطه ورود:** `${entry:,.2f}`\n"
            report += f"**حد ضرر:** `${stop_loss:,.2f}`\n"
            report += f"**حد سود ۱:** `${target1:,.2f}`\n"
            report += f"**حد سود ۲:** `${target2:,.2f}`\n"
            report += f"**اهرم بهینه:** `x{leverage}`\n"
        else:
            report += f"⚠️ **نتیجه (اطمینان: {long_confidence:.0f}٪):** در حال حاضر، هیچ سیگنال با هم‌راستایی کافی یافت نشد."

        return report, summary_1h['RECOMMENDATION']
        
    except Exception as e:
        logging.error(f"Critical error in full report for {symbol}: {e}")
        return "یک خطای پیش‌بینی نشده در فرآیند تحلیل رخ داد.", None

# --- موتور شکار سیگنال و پایش معامله ---
def hunt_signals():
    pass
def trade_monitor_loop():
    pass

# --- کنترل‌کننده‌های ربات ---
def handle_chat(msg):
    # این تابع کامل است
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
    # این تابع کامل است
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

        report_section = f"**--- {section.replace('_', ' ').title()} ---**\n\n"
        if section == 'structure':
            best_strategy = run_backtest_simulation(symbol)
            report_section += f"**استراتژی منتخب:** {best_strategy['details']['description']}\n"
            report_section += f"**نرخ موفقیت گذشته:** {best_strategy['details']['win_rate']:.1f}%\n\n"
            report_section += f"**تحلیل TradingView (4H):** `{data['summary_4h']['RECOMMENDATION']}`\n"
            report_section += f"**تحلیل TradingView (1H):** `{data['summary_1h']['RECOMMENDATION']}`\n"
        elif section == 'liquidity':
             support = data['indicators_4h'].get('Pivot.M.Support.1', data['indicators_1h']['low'])
             resistance = data['indicators_4h'].get('Pivot.M.Resistance.1', data['indicators_1h']['high'])
             report_section += f"**حمایت:** `${support:,.2f}`\n"
             report_section += f"**مقاومت:** `${resistance:,.2f}`\n"
        elif section == 'fundamental':
            report_section += "در حال دریافت داده‌های اقتصادی...\n(این بخش در حال توسعه است)"
        elif section == 'ai_proposal':
            report_section += "منطق پیشنهاد AI در اینجا نمایش داده می‌شود...\n(این بخش در حال توسعه است)"
        bot.sendMessage(chat_id, report_section, parse_mode='Markdown')

# --- راه‌اندازی ربات و وب‌سرور ---
def run_web_server():
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == '__main__':
    if not TELEGRAM_TOKEN:
        logging.fatal("TELEGRAM_TOKEN not found!")
    else:
        threading.Thread(target=hunt_signals, daemon=True, name="SignalHunterThread").start()
        # threading.Thread(target=trade_monitor_loop, daemon=True).start()
        
        MessageLoop(bot, {'chat': handle_chat,
                          'callback_query': handle_callback_query}).run_as_thread()
        logging.info('Telepot bot is listening...')
        
        if os.getenv('RAILWAY_ENVIRONMENT'):
            threading.Thread(target=run_web_server, daemon=True, name="WebServerThread").start()
        
        logging.info("Bot is running.")
        while 1:
            time.sleep(10)