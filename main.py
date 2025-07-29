# main.py (نسخه نهایی با اصلاح SyntaxError در aiogram)

import os
import logging
import asyncio
import pandas as pd
from sqlalchemy import create_engine, text
import io

# --- کتابخانه‌های مورد نیاز ---
from aiogram import Bot, Dispatcher, executor, types
from aiogram.utils.exceptions import BotBlocked

from fastapi import FastAPI

# --- تنظیمات اولیه ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- متغیرهای محیطی ---
API_TOKEN = os.getenv('TELEGRAM_TOKEN')
DATABASE_URL = os.getenv('DATABASE_URL')
WEBHOOK_HOST = os.getenv('RAILWAY_STATIC_URL')

# --- راه‌اندازی ربات و FastAPI ---
if not API_TOKEN:
    logging.fatal("TELEGRAM_TOKEN not found! The bot cannot start.")
    exit()

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
app = FastAPI()

# --- متغیرهای سراسری ---
WEBHOOK_PATH = f'/webhook/{API_TOKEN}'
WEBHOOK_URL = f'https://{WEBHOOK_HOST}{WEBHOOK_PATH}' if WEBHOOK_HOST else None

# --- منطق اصلی ربات با aiogram ---

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    """پاسخ به دستور /start"""
    logging.info(f"Received /start from user_id: {message.from_user.id}")
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    buttons = [
        types.InlineKeyboardButton(text="📊 BTC", callback_data="analyze_BTCUSDT"),
        types.InlineKeyboardButton(text="📈 ETH", callback_data="analyze_ETHUSDT"),
    ]
    keyboard.add(*buttons)
    await message.answer("سلام! به ربات Apex خوش آمدید. یک گزینه را انتخاب کنید:", reply_markup=keyboard)

@dp.callback_query_handler(lambda c: c.data and c.data.startswith('analyze_'))
async def process_callback_analyze(callback_query: types.CallbackQuery):
    """پردازش کلیک روی دکمه‌های تحلیل"""
    symbol = callback_query.data.split('_')[1]
    logging.info(f"Received analyze request for {symbol} from user_id: {callback_query.from_user.id}")
    await bot.answer_callback_query(callback_query.id)
    
    try:
        await bot.send_message(callback_query.from_user.id, f"در حال آماده‌سازی تحلیل برای {symbol}...")
        
        if not DATABASE_URL:
            await bot.send_message(callback_query.from_user.id, "خطا: سرویس دیتابیس در دسترس نیست.")
            return

        # در اینجا باید منطق تحلیل و ارسال چارت را اضافه کنیم
        # برای تست اولیه، فقط یک پیام موفقیت‌آمیز می‌فرستیم
        await bot.send_message(callback_query.from_user.id, f"تحلیل برای {symbol} با موفقیت انجام شد (این یک پیام تستی است).")

    except Exception as e:
        logging.error(f"Error during analysis for {symbol}: {e}")
        try:
            await bot.send_message(callback_query.from_user.id, "خطا در پردازش تحلیل.")
        except BotBlocked:
            logging.warning(f"Bot was blocked by user {callback_query.from_user.id}")

# --- مدیریت وب‌هوک و راه‌اندازی ---

@app.on_event("startup")
async def on_startup():
    """این تابع در هنگام راه‌اندازی FastAPI اجرا می‌شود."""
    if WEBHOOK_URL:
        logging.info(f"Setting webhook to: {WEBHOOK_URL}")
        current_webhook = await bot.get_webhook_info()
        if current_webhook.url != WEBHOOK_URL:
            await bot.set_webhook(WEBHOOK_URL)
    else:
        logging.warning("RAILWAY_STATIC_URL not found, cannot set webhook.")

@app.post(WEBHOOK_PATH)
async def bot_webhook(update: dict):
    """دریافت آپدیت‌ها از تلگرام"""
    telegram_update = types.Update(**update)
    Dispatcher.set_current(dp)
    Bot.set_current(bot)
    try:
        await dp.process_update(telegram_update)
    except Exception as e:
        logging.error(f"Error processing update: {e}")
    return {'ok': True}

@app.on_event("shutdown")
async def on_shutdown():
    """این تابع در هنگام خاموش شدن FastAPI اجرا می‌شود."""
    logging.info("Shutting down webhook...")
    await bot.delete_webhook()

@app.get("/")
def read_root():
    return {"status": "Apex Bot is running with aiogram!"}