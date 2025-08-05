import os
import logging
import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import joblib
import numpy as np
from keras.models import load_model
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
)

# --- تنظیمات ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
DEFAULT_LANG = "fa"

# --- لاگ ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- NLTK و ترجمه ---
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

# --- مدل‌های ML ---
rf_model = joblib.load("rf_model.pkl")
lstm_model = load_model("lstm_model.h5")
lstm_scaler = joblib.load("lstm_scaler.pkl")

# --- دریافت داده کندل ---
def detect_symbol_type(symbol):
    if symbol.endswith("USDT") or symbol.endswith("BTC") or symbol.endswith("ETH"):
        return "crypto"
    elif symbol.endswith("=X"):
        return "forex"
    else:
        return "stock"

def get_crypto_candles(symbol, interval="1h", limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data, columns=["open_time","open","high","low","close","volume","close_time","qav","trades","tbbav","tbqav","ignore"])
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df[["open","high","low","close","volume"]]

def get_stock_candles(symbol, interval="1h", period="60d"):
    data = yf.Ticker(symbol).history(interval=interval, period=period)
    return data[["Open","High","Low","Close","Volume"]].rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})

def get_forex_candles(symbol, interval="1h", period="60d"):
    data = yf.Ticker(symbol).history(interval=interval, period=period)
    return data[["Open","High","Low","Close","Volume"]].rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})

def get_candles(symbol, interval="1h", limit=200):
    t = detect_symbol_type(symbol)
    if t == "crypto":
        return get_crypto_candles(symbol, interval, limit)
    elif t == "forex":
        return get_forex_candles(symbol, interval)
    else:
        return get_stock_candles(symbol, interval)

# --- اندیکاتورها ---
def calc_indicators(df):
    df = df.copy()
    df["rsi"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"])
    df["macd"] = macd["MACD_12_26_9"]
    df["ema10"] = ta.ema(df["close"], length=10)
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    ichi = ta.ichimoku(df["high"], df["low"], df["close"])
    df["ichimoku_base"] = ichi["ISA_9"]
    df["ichimoku_conv"] = ichi["ISB_26"]
    return df

# --- تحلیل پرایس‌اکشن و عرضه/تقاضا ---
def albrooks_analysis(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last["close"] > prev["close"]:
        trend = "روند صعودی"
    elif last["close"] < prev["close"]:
        trend = "روند نزولی"
    else:
        trend = "خنثی"
    return trend

def detect_supply_demand(df):
    lows = df["low"].rolling(10).min()
    highs = df["high"].rolling(10).max()
    supply = highs.iloc[-1]
    demand = lows.iloc[-1]
    return f"عرضه: {supply:.2f} | تقاضا: {demand:.2f}"

# --- تحلیل فاندامنتال ---
def get_fundamental(symbol):
    try:
        info = yf.Ticker(symbol).info
        return {
            "marketCap": info.get("marketCap", "N/A"),
            "volume": info.get("volume", "N/A"),
            "peRatio": info.get("trailingPE", "N/A"),
            "dividendYield": info.get("dividendYield", "N/A")
        }
    except Exception as e:
        return {"error": f"تحلیل فاندامنتال برای این نماد ممکن نشد: {e}"}

# --- تحلیل احساسات اخبار ---
def get_news_sentiment(symbol, newsapi_key):
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={newsapi_key}&language=en"
        r = requests.get(url)
        data = r.json()
        scores = []
        fa_news = []
        for article in data.get("articles", [])[:5]:
            text = (article.get("title") or "") + " " + (article.get("description") or "")
            score = analyzer.polarity_scores(text)
            scores.append(score["compound"])
            try:
                fa_title = GoogleTranslator(source='auto', target='fa').translate(article.get("title", ""))
                fa_desc = GoogleTranslator(source='auto', target='fa').translate(article.get("description", ""))
                fa_news.append(f"عنوان: {fa_title}\nتوضیح: {fa_desc}")
            except:
                pass
        if not scores:
            return {"sentiment": "خنثی", "score": 0, "fa_news": ["اخباری یافت نشد."]}
        avg = sum(scores) / len(scores)
        if avg > 0.2:
            sentiment = "مثبت"
        elif avg < -0.2:
            sentiment = "منفی"
        else:
            sentiment = "خنثی"
        return {"sentiment": sentiment, "score": avg, "fa_news": fa_news}
    except Exception as e:
        return {"sentiment": "خطا", "score": 0, "fa_news": [f"خطا در دریافت اخبار: {e}"]}

# --- مدل‌های ML ---
def rf_predict(ma10, ma50, rsi):
    X = np.array([[ma10, ma50, rsi]])
    pred = rf_model.predict(X)[0]
    return "خرید" if pred == 1 else "فروش"

def lstm_predict(prices):
    scaled = lstm_scaler.transform(np.array(prices).reshape(-1, 1))
    X = np.array([scaled.flatten()])
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    pred = lstm_model.predict(X)
    pred_price = lstm_scaler.inverse_transform(pred)[0][0]
    return pred_price

# --- سیگنال‌دهی ترکیبی ---
def generate_ai_signal(df, tech, fund, news):
    ma10 = df["ema10"].iloc[-1]
    ma50 = df["ema50"].iloc[-1]
    rsi = df["rsi"].iloc[-1]
    rf_signal = rf_predict(ma10, ma50, rsi)
    prices = df["close"].values[-10:]
    lstm_price = lstm_predict(prices)
    lstm_trend = "صعودی" if lstm_price > prices[-1] else "نزولی"

    votes = 0
    if tech.get("RECOMMENDATION", "NEUTRAL") == "BUY":
        votes += 1
    elif tech.get("RECOMMENDATION", "NEUTRAL") == "SELL":
        votes -= 1
    if rf_signal == "خرید":
        votes += 1
    else:
        votes -= 1
    if lstm_trend == "صعودی":
        votes += 1
    else:
        votes -= 1
    if news["sentiment"] == "مثبت":
        votes += 1
    elif news["sentiment"] == "منفی":
        votes -= 1

    if votes >= 2:
        final_signal = "خرید قوی"
    elif votes == 1:
        final_signal = "خرید"
    elif votes == 0:
        final_signal = "خنثی"
    elif votes == -1:
        final_signal = "فروش"
    else:
        final_signal = "فروش قوی"

    return {
        "rf_signal": rf_signal,
        "lstm_trend": lstm_trend,
        "lstm_price": lstm_price,
        "final_signal": final_signal
    }

# --- فرمت خروجی فارسی و کامل ---
def format_farsi_report(symbol, df, tech, fund, albrooks, supply_demand, news, ai_signal):
    tech_str = "\n".join([f"{k}: {v}" for k, v in tech.items()]) if "error" not in tech else tech["error"]
    fund_str = "\n".join([f"{k}: {v}" for k, v in fund.items()]) if "error" not in fund else fund["error"]
    news_str = f"احساسات اخبار: {news['sentiment']} (امتیاز: {news['score']:.2f})\n"
    news_str += "\n".join(news['fa_news'])
    msg = (
        f"نماد: {symbol}\n"
        f"قیمت آخرین کندل: {df['close'].iloc[-1]:.2f}\n"
        f"--- تحلیل تکنیکال ---\n{tech_str}\n"
        f"--- اندیکاتورها ---\n"
        f"RSI: {df['rsi'].iloc[-1]:.2f}\n"
        f"MACD: {df['macd'].iloc[-1]:.2f}\n"
        f"EMA10: {df['ema10'].iloc[-1]:.2f}\n"
        f"EMA50: {df['ema50'].iloc[-1]:.2f}\n"
        f"ایچیموکو: Base {df['ichimoku_base'].iloc[-1]:.2f} | Conv {df['ichimoku_conv'].iloc[-1]:.2f}\n"
        f"--- تحلیل فاندامنتال ---\n{fund_str}\n"
        f"--- پرایس‌اکشن (سبک البروکس) ---\n{albrooks}\n"
        f"--- سطوح عرضه و تقاضا ---\n{supply_demand}\n"
        f"--- تحلیل احساسات اخبار ---\n{news_str}\n"
        f"--- سیگنال هوشمند AI ---\n"
        f"سیگنال RandomForest: {ai_signal['rf_signal']}\n"
        f"LSTM: {ai_signal['lstm_trend']} (پیش‌بینی: {ai_signal['lstm_price']:.2f})\n"
        f"سیگنال نهایی: {ai_signal['final_signal']}\n"
    )
    return msg

# --- هندلرهای ربات ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("تحلیل عمیق", callback_data="analyze")]
    ]
    await update.message.reply_text("سلام! برای تحلیل، روی دکمه زیر بزنید.", reply_markup=InlineKeyboardMarkup(keyboard))

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "analyze":
        await query.message.reply_text("نماد را وارد کنید (مثلاً BTCUSDT یا AAPL یا EURUSD=X):")

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.message.text.strip().upper()
    try:
        df = get_candles(symbol)
        df = calc_indicators(df)
        tech = {"RECOMMENDATION": "N/A"}
        try:
            import tradingview_ta
            tv_symbol = symbol
            if detect_symbol_type(symbol) == "crypto" and not symbol.endswith("USDT"):
                tv_symbol = symbol.replace("USD", "USDT")
            ta_handler = tradingview_ta.TA_Handler(
                symbol=tv_symbol,
                screener="crypto" if detect_symbol_type(symbol) == "crypto" else "america",
                exchange="BINANCE" if detect_symbol_type(symbol) == "crypto" else "NASDAQ",
                interval=tradingview_ta.Interval.INTERVAL_1_HOUR
            )
            analysis = ta_handler.get_analysis()
            tech = {
                "RECOMMENDATION": analysis.summary.get("RECOMMENDATION", "NEUTRAL"),
                "BUY": analysis.summary.get("BUY", 0),
                "SELL": analysis.summary.get("SELL", 0),
                "NEUTRAL": analysis.summary.get("NEUTRAL", 0)
            }
        except Exception as e:
            tech = {"error": f"تحلیل تکنیکال TradingView ممکن نشد: {e}"}
        fund = get_fundamental(symbol)
        albrooks = albrooks_analysis(df)
        supply_demand = detect_supply_demand(df)
        news = get_news_sentiment(symbol, NEWSAPI_KEY)
        ai_signal = generate_ai_signal(df, tech, fund, news)
        msg = format_farsi_report(symbol, df, tech, fund, albrooks, supply_demand, news, ai_signal)
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"خطا در تحلیل: {e}")

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.run_polling()

if __name__ == "__main__":
    main()