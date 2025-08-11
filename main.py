# main.py
import os
import logging
import asyncio
import time
import pytz
import re
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest
from dotenv import load_dotenv

from config import Config
from bot import AdvancedTradingBot

# --- تنظیمات لاگینگ ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger("TradingBot")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler('bot_activity.log', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

# --- نمونه‌سازی از ربات ---
try:
    bot_instance = AdvancedTradingBot()
except Exception as e:
    logger.critical(f"Could not initialize the bot. Shutting down. Error: {e}", exc_info=True)
    exit()

# --- توابع مدیریت دستورات تلگرام ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("سلام! برای تحلیل، نماد ارز (مانند BTC) را ارسال کنید.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("این ربات تحلیل‌های تکنیکال و فاندامنتال جامعی ارائه می‌دهد.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.message.text.strip().upper()
    if re.match("^[A-Z0-9]{2,10}$", symbol):
        await process_analysis_request(update, symbol)
    else:
        await update.message.reply_text("لطفاً یک نماد معتبر ارسال کنید.")

async def process_analysis_request(update: Update, symbol: str):
    msg = await update.message.reply_text(f"⏳ در حال انجام تحلیل جامع برای *{symbol}*...", parse_mode='Markdown')
    
    try:
        start_time = time.time()
        result = await bot_instance.perform_full_analysis(symbol)
        duration = time.time() - start_time
        
        if "error" in result:
            await msg.edit_text(f"❌ خطا در تحلیل: {result['error']}")
            return

        response_text = format_response(result, duration)
        await msg.edit_text(response_text, parse_mode='Markdown', disable_web_page_preview=True)

    except Exception as e:
        logger.error(f"Critical error in process_analysis_request for {symbol}: {e}", exc_info=True)
        await msg.edit_text("یک خطای داخلی رخ داد. لطفاً بعداً دوباره تلاش کنید.")

def format_response(result, duration):
    symbol = result.get('symbol', 'N/A')
    market = result.get('market_data', {})
    signal_info = result.get('final_signal', {})
    tech = result.get('analysis', {}).get('technical', {})

    signal = signal_info.get('signal', 'N/A')
    score = signal_info.get('score', 0.5)
    
    response = f"📊 *تحلیل برای {symbol}*\n\n"
    response += f"🚨 *سیگنال نهایی: {signal}* (امتیاز: {score:.2f})\n\n"
    
    if market:
        response += f"📈 قیمت: `${market.get('price', 0):,.2f}`\n"
        response += f"📊 تغییر ۲۴ ساعته: `{market.get('percent_change_24h', 0):.2f}%`\n\n"
    
    if tech:
        response += "🛠️ *تحلیل تکنیکال:*\n"
        trend = tech.get('trend', {})
        oscillators = tech.get('oscillators', {})
        response += f"  - روند (EMA 50/200): `{trend.get('direction', 'N/A')}`\n"
        response += f"  - RSI: `{oscillators.get('rsi', 0):.2f}`\n"
        
        patterns = tech.get('candlestick_patterns', {})
        if patterns:
            response += f"  - الگوهای شمعی اخیر: `{', '.join(patterns.keys())}`\n"
    
    response += f"\n⏱️ (زمان تحلیل: {duration:.2f} ثانیه)"
    return response

async def on_shutdown(application: Application):
    await bot_instance.close_connections()

def main():
    logger.info("Starting Telegram bot...")
    
    if not Config.TELEGRAM_BOT_TOKEN:
        logger.critical("FATAL: TELEGRAM_BOT_TOKEN is not set!")
        return

    app_builder = Application.builder().token(Config.TELEGRAM_BOT_TOKEN)
    if Config.AIOHTTP_PROXY_URL:
        request = HTTPXRequest(proxy_url=Config.AIOHTTP_PROXY_URL)
        app_builder.request(request)
    
    application = app_builder.post_shutdown(on_shutdown).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    try:
        application.run_polling()
    except Exception as e:
        logger.critical(f"Bot polling failed critically: {e}", exc_info=True)

if __name__ == '__main__':
    main()