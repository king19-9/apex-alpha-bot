import os
import logging
import yfinance as yf
import tradingview_ta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
)

# --- تنظیمات ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DEFAULT_LANG = "fa"

# --- لاگ ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- تبدیل نماد به فرمت TradingView ---
def symbol_to_tradingview(symbol):
    # فقط برای کریپتو: BTC-USD -> BTCUSDT
    if symbol.endswith("-USD"):
        return symbol.replace("-USD", "USDT")
    return symbol

# --- تحلیل تکنیکال ---
def get_technical(symbol):
    try:
        tv_symbol = symbol_to_tradingview(symbol)
        ta = tradingview_ta.TA_Handler(
            symbol=tv_symbol,
            screener="crypto",
            exchange="BINANCE",
            interval=tradingview_ta.Interval.INTERVAL_1_HOUR
        )
        analysis = ta.get_analysis()
        return {
            "RECOMMENDATION": analysis.summary.get("RECOMMENDATION", "NEUTRAL"),
            "BUY": analysis.summary.get("BUY", 0),
            "SELL": analysis.summary.get("SELL", 0),
            "NEUTRAL": analysis.summary.get("NEUTRAL", 0)
        }
    except Exception as e:
        return {"error": "تحلیل تکنیکال برای این نماد در TradingView/Binance پیدا نشد."}

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
        return {"error": "تحلیل فاندامنتال برای این نماد یافت نشد."}

# --- قیمت ---
def get_price(symbol):
    try:
        data = yf.Ticker(symbol).history(period="1d")
        return float(data['Close'][-1])
    except Exception as e:
        return "N/A"

# --- فرمت خروجی ---
def format_analysis_report(symbol, price, tech, fund):
    tech_str = ""
    if "error" in tech:
        tech_str = tech["error"]
    else:
        tech_str = "\n".join([f"{k}: {v}" for k, v in tech.items()])

    fund_str = ""
    if "error" in fund:
        fund_str = fund["error"]
    else:
        fund_str = "\n".join([f"{k}: {v}" for k, v in fund.items()])

    msg = (
        f"نماد: {symbol}\n"
        f"قیمت: {price}\n"
        f"--- تحلیل تکنیکال ---\n{tech_str}\n"
        f"--- تحلیل فاندامنتال ---\n{fund_str}\n"
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
        await query.message.reply_text("نماد را وارد کنید (مثلاً BTC-USD):")

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.message.text.strip().upper()
    price = get_price(symbol)
    tech = get_technical(symbol)
    fund = get_fundamental(symbol)
    msg = format_analysis_report(symbol, price, tech, fund)
    await update.message.reply_text(msg)

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.run_polling()

if __name__ == "__main__":
    main()