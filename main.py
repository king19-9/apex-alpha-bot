import os
import logging
import time
import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
)
from sqlalchemy import create_engine, Column, Integer, String, Boolean, JSON, Float
from sqlalchemy.orm import declarative_base, sessionmaker
import redis
from apscheduler.schedulers.background import BackgroundScheduler

# تحلیل و ML (نمونه اولیه)
import yfinance as yf
import ccxt
import tradingview_ta
import investpy
import nltk

# --- تنظیمات ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
POSTGRES_URL = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")  # این خط اصلاح شد
REDIS_URL = os.getenv("REDIS_URL")
DEFAULT_LANG = "fa"

# --- لاگ ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- دیتابیس ---
Base = declarative_base()
engine = create_engine(POSTGRES_URL)
Session = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True)
    language = Column(String, default=DEFAULT_LANG)
    watchlist = Column(JSON, default=[])
    notifications_enabled = Column(Boolean, default=True)
    state = Column(String, default=None)

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    symbol = Column(String)
    direction = Column(String)  # Long/Short
    active = Column(Boolean, default=True)

Base.metadata.create_all(engine)

# --- Redis ---
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# --- Scheduler ---
scheduler = BackgroundScheduler()
scheduler.start()

# --- Helper: زبان ---
def t(msg, lang):
    # پیام‌های چندزبانه (نمونه)
    fa = {
        "main_menu": "منوی اصلی:\n1. تحلیل عمیق\n2. سیگنال نقره‌ای\n3. نوتیف طلایی\n4. پایش معامله\n5. واچ‌لیست\n6. تنظیمات",
        "send_symbol": "نماد را ارسال کنید (مثلاً BTC-USD):",
        "invalid_symbol": "نماد نامعتبر است.",
        "analysis_report": "گزارش تحلیل {symbol}:\nقیمت: {price}\nاستراتژی: {strategy} (WinRate: {winrate}%)\nتکنیکال: {tech}\nفاندامنتال: {fund}\nسیگنال: {signal}\nاحساسات: {sentiment}",
        "done": "انجام شد.",
        "add_watchlist": "نماد به واچ‌لیست افزوده شد.",
        "remove_watchlist": "نماد حذف شد.",
        "your_watchlist": "واچ‌لیست شما:\n{list}",
        "choose_lang": "زبان را انتخاب کنید: /lang fa یا /lang en",
        "stats": "آمار سیگنال‌ها:\nکل: {total}\nطلایی: {gold}\nنقره‌ای: {silver}\nWinRate: {winrate}%",
    }
    en = {
        "main_menu": "Main Menu:\n1. Deep Analysis\n2. Silver Signals\n3. Golden Notif\n4. Trade Monitor\n5. Watchlist\n6. Settings",
        "send_symbol": "Send symbol (e.g. BTC-USD):",
        "invalid_symbol": "Invalid symbol.",
        "analysis_report": "Analysis for {symbol}:\nPrice: {price}\nStrategy: {strategy} (WinRate: {winrate}%)\nTechnical: {tech}\nFundamental: {fund}\nSignal: {signal}\nSentiment: {sentiment}",
        "done": "Done.",
        "add_watchlist": "Symbol added to watchlist.",
        "remove_watchlist": "Symbol removed.",
        "your_watchlist": "Your watchlist:\n{list}",
        "choose_lang": "Choose language: /lang fa or /lang en",
        "stats": "Signal stats:\nTotal: {total}\nGold: {gold}\nSilver: {silver}\nWinRate: {winrate}%",
    }
    return fa[msg] if lang == "fa" else en[msg]

# --- تحلیل عمیق (نمونه اولیه) ---
def get_deep_analysis(symbol):
    # داده قیمت
    try:
        data = yf.Ticker(symbol).history(period="1mo")
        price = data['Close'][-1]
    except Exception:
        return None
    # تحلیل تکنیکال (نمونه)
    try:
        ta = tradingview_ta.TA_Handler(
            symbol=symbol.split('-')[0],
            screener="crypto",
            exchange="BINANCE",
            interval=tradingview_ta.Interval.INTERVAL_1_HOUR
        )
        tech = ta.get_analysis().summary
    except Exception:
        tech = {}
    # تحلیل فاندامنتال (نمونه)
    try:
        fund = investpy.get_crypto_historical_data(crypto=symbol.split('-')[0], from_date="01/01/2023", to_date="01/02/2023")
        fund = "OK"
    except Exception:
        fund = "N/A"
    # تحلیل احساسات (نمونه)
    sentiment = "Neutral"
    # مدل ML (نمونه اولیه)
    strategy = "RF+LSTM"
    winrate = 87
    signal = "Buy" if price % 2 == 0 else "Sell"
    return {
        "symbol": symbol,
        "price": price,
        "strategy": strategy,
        "winrate": winrate,
        "tech": tech,
        "fund": fund,
        "signal": signal,
        "sentiment": sentiment
    }

# --- سیگنال نقره‌ای (نمونه) ---
def scan_signals(watchlist):
    signals = []
    for symbol in watchlist:
        analysis = get_deep_analysis(symbol)
        if analysis and 65 <= analysis['winrate'] < 80:
            signals.append(analysis)
    return signals

# --- سیگنال طلایی (نمونه) ---
def scan_golden_signals(watchlist):
    signals = []
    for symbol in watchlist:
        analysis = get_deep_analysis(symbol)
        if analysis and analysis['winrate'] >= 80:
            signals.append(analysis)
    return signals

# --- پایش معاملات ---
def monitor_trades():
    session = Session()
    trades = session.query(Trade).filter_by(active=True).all()
    for trade in trades:
        analysis = get_deep_analysis(trade.symbol)
        if analysis:
            # اگر سیگنال مخالف جهت معامله بود، هشدار (نمونه)
            if (trade.direction == "Long" and analysis['signal'] == "Sell") or \
               (trade.direction == "Short" and analysis['signal'] == "Buy"):
                logger.warning(f"Trade alert for {trade.symbol} ({trade.direction})!")
    session.close()

scheduler.add_job(monitor_trades, 'interval', minutes=5)

# --- هندلرها ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = Session()
    user = session.query(User).filter_by(user_id=update.effective_user.id).first()
    if not user:
        user = User(user_id=update.effective_user.id)
        session.add(user)
        session.commit()
    lang = user.language
    keyboard = [
        [InlineKeyboardButton("1", callback_data="analyze"),
         InlineKeyboardButton("2", callback_data="silver"),
         InlineKeyboardButton("3", callback_data="golden")],
        [InlineKeyboardButton("4", callback_data="monitor"),
         InlineKeyboardButton("5", callback_data="watchlist"),
         InlineKeyboardButton("6", callback_data="settings")]
    ]
    await update.message.reply_text(t("main_menu", lang), reply_markup=InlineKeyboardMarkup(keyboard))
    session.close()

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    session = Session()
    user = session.query(User).filter_by(user_id=query.from_user.id).first()
    lang = user.language
    if query.data == "analyze":
        user.state = "analyze"
        session.commit()
        await query.message.reply_text(t("send_symbol", lang))
    elif query.data == "silver":
        signals = scan_signals(user.watchlist)
        msg = "\n".join([f"{s['symbol']}: {s['signal']} ({s['winrate']}%)" for s in signals]) or "No signals."
        await query.message.reply_text(msg)
    elif query.data == "golden":
        user.notifications_enabled = not user.notifications_enabled
        session.commit()
        await query.message.reply_text(t("done", lang))
    elif query.data == "monitor":
        user.state = "monitor"
        session.commit()
        await query.message.reply_text("Send symbol and direction (e.g. BTC-USD Long):")
    elif query.data == "watchlist":
        user.state = "watchlist"
        session.commit()
        await query.message.reply_text("Send 'add SYMBOL', 'remove SYMBOL' or 'list'")
    elif query.data == "settings":
        user.state = "settings"
        session.commit()
        await query.message.reply_text(t("choose_lang", lang))
    session.close()

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = Session()
    user = session.query(User).filter_by(user_id=update.effective_user.id).first()
    lang = user.language
    if user.state == "analyze":
        symbol = update.message.text.strip().upper()
        analysis = get_deep_analysis(symbol)
        if not analysis:
            await update.message.reply_text(t("invalid_symbol", lang))
        else:
            await update.message.reply_text(
                t("analysis_report", lang).format(**analysis)
            )
        user.state = None
        session.commit()
    elif user.state == "monitor":
        parts = update.message.text.strip().split()
        if len(parts) == 2:
            symbol, direction = parts
            trade = Trade(user_id=user.user_id, symbol=symbol.upper(), direction=direction.capitalize())
            session.add(trade)
            session.commit()
            await update.message.reply_text(t("done", lang))
        user.state = None
        session.commit()
    elif user.state == "watchlist":
        txt = update.message.text.strip().lower()
        if txt.startswith("add "):
            symbol = txt.split()[1].upper()
            wl = user.watchlist or []
            if symbol not in wl:
                wl.append(symbol)
                user.watchlist = wl
                session.commit()
            await update.message.reply_text(t("add_watchlist", lang))
        elif txt.startswith("remove "):
            symbol = txt.split()[1].upper()
            wl = user.watchlist or []
            if symbol in wl:
                wl.remove(symbol)
                user.watchlist = wl
                session.commit()
            await update.message.reply_text(t("remove_watchlist", lang))
        elif txt == "list":
            wl = user.watchlist or []
            await update.message.reply_text(t("your_watchlist", lang).format(list="\n".join(wl)))
        user.state = None
        session.commit()
    elif user.state == "settings":
        txt = update.message.text.strip().lower()
        if txt.startswith("lang "):
            lang = txt.split()[1]
            user.language = lang
            session.commit()
            await update.message.reply_text(t("done", lang))
        user.state = None
        session.commit()
    session.close()

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # نمونه آماری
    total = int(redis_client.get("signals_total") or 0)
    gold = int(redis_client.get("signals_gold") or 0)
    silver = int(redis_client.get("signals_silver") or 0)
    winrate = 85
    session = Session()
    user = session.query(User).filter_by(user_id=update.effective_user.id).first()
    lang = user.language
    await update.message.reply_text(
        t("stats", lang).format(total=total, gold=gold, silver=silver, winrate=winrate)
    )
    session.close()

# --- راه‌اندازی ربات ---
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CallbackQueryHandler(button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.run_polling()

if __name__ == "__main__":
    main()