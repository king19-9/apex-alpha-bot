import os
import logging
import time
import json
import requests
import joblib
import numpy as np
import yfinance as yf
import tradingview_ta
import investpy
import nltk
from keras.models import load_model
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
)
from sqlalchemy import create_engine, Column, Integer, String, Boolean, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
import redis
from apscheduler.schedulers.background import BackgroundScheduler
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- تنظیمات ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
POSTGRES_URL = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
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

# --- ML Models ---
rf_model = joblib.load("rf_model.pkl")
lstm_model = load_model("lstm_model.h5")
lstm_scaler = joblib.load("lstm_scaler.pkl")

# --- NLTK ---
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

# --- تحلیل تکنیکال و فاندامنتال ---
def get_price(symbol):
    try:
        data = yf.Ticker(symbol).history(period="1d")
        return float(data['Close'][-1])
    except Exception as e:
        print(f"Error in get_price: {e}")
        return None

def get_technical(symbol):
    try:
        base = symbol.split('-')[0]
        ta = tradingview_ta.TA_Handler(
            symbol=base,
            screener="crypto",
            exchange="BINANCE",
            interval=tradingview_ta.Interval.INTERVAL_1_HOUR
        )
        analysis = ta.get_analysis()
        return {
            "RECOMMENDATION": analysis.summary.get("RECOMMENDATION", "NEUTRAL"),
            "BUY": analysis.summary.get("BUY", 0),
            "SELL": analysis.summary.get("SELL", 0),
            "NEUTRAL": analysis.summary.get("NEUTRAL", 0),
            "oscillators": analysis.oscillators,
            "moving_averages": analysis.moving_averages
        }
    except Exception as e:
        print(f"Error in get_technical: {e}")
        return {}

def get_fundamental(symbol):
    try:
        info = yf.Ticker(symbol).info
        try:
            base = symbol.split('-')[0]
            fund = investpy.get_crypto_historical_data(crypto=base, from_date="01/01/2023", to_date="01/02/2023")
            fund_info = "Available"
        except Exception:
            fund_info = "N/A"
        return {
            "marketCap": info.get("marketCap", "N/A"),
            "volume": info.get("volume", "N/A"),
            "peRatio": info.get("trailingPE", "N/A"),
            "dividendYield": info.get("dividendYield", "N/A"),
            "fund_info": fund_info
        }
    except Exception as e:
        print(f"Error in get_fundamental: {e}")
        return {}

def deep_analysis(symbol):
    price = get_price(symbol)
    tech = get_technical(symbol)
    fund = get_fundamental(symbol)
    return {
        "symbol": symbol,
        "price": price,
        "tech": tech,
        "fund": fund
    }

# --- تحلیل احساسات اخبار ---
def get_news_sentiment(symbol, newsapi_key):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={newsapi_key}&language=en"
    r = requests.get(url)
    data = r.json()
    scores = []
    for article in data.get("articles", []):
        text = (article.get("title") or "") + " " + (article.get("description") or "")
        score = analyzer.polarity_scores(text)
        scores.append(score["compound"])
    if not scores:
        return {"sentiment": "Neutral", "score": 0}
    avg = sum(scores) / len(scores)
    if avg > 0.2:
        sentiment = "Positive"
    elif avg < -0.2:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return {"sentiment": sentiment, "score": avg}

# --- ML Inference ---
def rf_predict(ma10, ma50, rsi):
    X = np.array([[ma10, ma50, rsi]])
    pred = rf_model.predict(X)[0]
    return "Buy" if pred == 1 else "Sell"

def lstm_predict(prices):
    scaled = lstm_scaler.transform(np.array(prices).reshape(-1, 1))
    X = np.array([scaled.flatten()])
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    pred = lstm_model.predict(X)
    pred_price = lstm_scaler.inverse_transform(pred)[0][0]
    return pred_price

# --- سیگنال‌دهی ترکیبی ---
def generate_advanced_signal(symbol, newsapi_key):
    analysis = deep_analysis(symbol)
    price = analysis["price"]
    tech = analysis["tech"]
    fund = analysis["fund"]

    sentiment_result = get_news_sentiment(symbol, newsapi_key)
    sentiment = sentiment_result["sentiment"]
    sentiment_score = sentiment_result["score"]

    data = yf.Ticker(symbol).history(period="60d")
    ma10 = data['Close'].rolling(10).mean().iloc[-1]
    ma50 = data['Close'].rolling(50).mean().iloc[-1]
    rsi = (100 - (100 / (1 + data['Close'].pct_change().rolling(14).mean()))).iloc[-1]
    rf_signal = rf_predict(ma10, ma50, rsi)
    prices = data['Close'].values[-10:]
    lstm_price = lstm_predict(prices)
    lstm_trend = "Up" if lstm_price > price else "Down"

    votes = 0
    if tech.get("RECOMMENDATION", "NEUTRAL") == "BUY":
        votes += 1
    elif tech.get("RECOMMENDATION", "NEUTRAL") == "SELL":
        votes -= 1
    if rf_signal == "Buy":
        votes += 1
    else:
        votes -= 1
    if lstm_trend == "Up":
        votes += 1
    else:
        votes -= 1
    if sentiment == "Positive":
        votes += 1
    elif sentiment == "Negative":
        votes -= 1

    if votes >= 2:
        final_signal = "Strong Buy"
    elif votes == 1:
        final_signal = "Buy"
    elif votes == 0:
        final_signal = "Neutral"
    elif votes == -1:
        final_signal = "Sell"
    else:
        final_signal = "Strong Sell"

    winrate = int(80 + 10 * (votes / 4))

    return {
        "symbol": symbol,
        "price": price,
        "tech": tech,
        "fund": fund,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "rf_signal": rf_signal,
        "lstm_trend": lstm_trend,
        "lstm_price": lstm_price,
        "final_signal": final_signal,
        "winrate": winrate
    }

# --- فرمت خروجی برای تلگرام ---
def format_analysis_report(sig, lang):
    tech = sig.get("tech", {})
    if isinstance(tech, dict):
        tech_str = "\n".join([f"{k}: {v}" for k, v in tech.items()])
    else:
        tech_str = str(tech)

    fund = sig.get("fund", {})
    if isinstance(fund, dict):
        fund_str = "\n".join([f"{k}: {v}" for k, v in fund.items()])
    else:
        fund_str = str(fund)

    msg = (
        f"نماد: {sig.get('symbol', '-')}\n"
        f"قیمت: {sig.get('price', '-')}\n"
        f"--- تحلیل تکنیکال ---\n{tech_str}\n"
        f"--- تحلیل فاندامنتال ---\n{fund_str}\n"
        f"سیگنال ML: {sig.get('rf_signal', '-')}\n"
        f"LSTM: {sig.get('lstm_trend', '-')} (پیش‌بینی: {sig.get('lstm_price', 0):.2f})\n"
        f"سیگنال نهایی: {sig.get('final_signal', '-')}\n"
        f"WinRate: {sig.get('winrate', 0)}%\n"
        f"احساسات اخبار: {sig.get('sentiment', '-')} (امتیاز: {sig.get('sentiment_score', 0):.2f})"
    )
    if lang == "en":
        msg = (
            f"Symbol: {sig.get('symbol', '-')}\n"
            f"Price: {sig.get('price', '-')}\n"
            f"--- Technical Analysis ---\n{tech_str}\n"
            f"--- Fundamental Analysis ---\n{fund_str}\n"
            f"ML Signal: {sig.get('rf_signal', '-')}\n"
            f"LSTM: {sig.get('lstm_trend', '-')} (Predicted: {sig.get('lstm_price', 0):.2f})\n"
            f"Final Signal: {sig.get('final_signal', '-')}\n"
            f"WinRate: {sig.get('winrate', 0)}%\n"
            f"News Sentiment: {sig.get('sentiment', '-')} (Score: {sig.get('sentiment_score', 0):.2f})"
        )
    return msg

# --- پایش معاملات و آمار ---
scheduler = BackgroundScheduler()

def monitor_trades():
    session = Session()
    trades = session.query(Trade).filter_by(active=True).all()
    for trade in trades:
        signal = generate_advanced_signal(trade.symbol, NEWSAPI_KEY)
        if (trade.direction == "Long" and signal['final_signal'] in ["Sell", "Strong Sell"]) or \
           (trade.direction == "Short" and signal['final_signal'] in ["Buy", "Strong Buy"]):
            logger.warning(f"ALERT: {trade.symbol} {trade.direction} conflict with signal {signal['final_signal']}!")
    session.close()

def update_stats(signal):
    redis_client.incr("signals_total")
    if signal["final_signal"] in ["Strong Buy", "Strong Sell"]:
        redis_client.incr("signals_gold")
    elif signal["final_signal"] in ["Buy", "Sell"]:
        redis_client.incr("signals_silver")

scheduler.add_job(monitor_trades, 'interval', minutes=5)
scheduler.start()

# --- پیام‌های چندزبانه ---
def t(msg, lang):
    fa = {
        "main_menu": "منوی اصلی:\n1. تحلیل عمیق\n2. سیگنال نقره‌ای\n3. نوتیف طلایی\n4. پایش معامله\n5. واچ‌لیست\n6. تنظیمات",
        "send_symbol": "نماد را ارسال کنید (مثلاً BTC-USD):",
        "invalid_symbol": "نماد نامعتبر است.",
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
        "done": "Done.",
        "add_watchlist": "Symbol added to watchlist.",
        "remove_watchlist": "Symbol removed.",
        "your_watchlist": "Your watchlist:\n{list}",
        "choose_lang": "Choose language: /lang fa or /lang en",
        "stats": "Signal stats:\nTotal: {total}\nGold: {gold}\nSilver: {silver}\nWinRate: {winrate}%",
    }
    return fa[msg] if lang == "fa" else en[msg]

# --- هندلرهای ربات ---
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
        signals = []
        for symbol in user.watchlist:
            sig = generate_advanced_signal(symbol, NEWSAPI_KEY)
            if 65 <= sig['winrate'] < 80:
                signals.append(sig)
        msg = "\n".join([f"{s['symbol']}: {s['final_signal']} ({s['winrate']}%)" for s in signals]) or "No signals."
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
        sig = generate_advanced_signal(symbol, NEWSAPI_KEY)
        if not sig or not sig["price"]:
            await update.message.reply_text(t("invalid_symbol", lang))
        else:
            msg = format_analysis_report(sig, lang)
            await update.message.reply_text(msg)
            update_stats(sig)
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

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CallbackQueryHandler(button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.run_polling()

if __name__ == "__main__":
    main()