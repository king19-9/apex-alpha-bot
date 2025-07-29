# main.py (نسخه فوق‌العاده ساده برای تست اتصال)
import os
import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from fastapi import FastAPI
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Minimal test bot is running!"}
    
def run_bot():
    if not TELEGRAM_TOKEN:
        logging.error("TELEGRAM_TOKEN not found!")
        return
    try:
        updater = Updater(TELEGRAM_TOKEN)
        dispatcher = updater.dispatcher

        def start(update: Update, context: CallbackContext) -> None:
            logging.info("SUCCESS: Received /start command!")
            update.message.reply_text('تست اتصال موفقیت آمیز بود! ربات شما آنلاین است.')

        dispatcher.add_handler(CommandHandler("start", start))
        
        logging.info("Minimal test: Telegram bot started polling.")
        updater.start_polling()
        updater.idle()
    except Exception as e:
        logging.error(f"Minimal test FAILED: {e}")

@app.on_event("startup")
def startup_event():
    threading.Thread(target=run_bot, daemon=True).start()