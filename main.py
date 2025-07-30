# main.py (نسخه نهایی با اصلاح منطق پردازش کلیک‌ها)

import os
import logging
import time
import telepot
from telepot.loop import MessageLoop
from telepot.namedtuple import InlineKeyboardMarkup, InlineKeyboardButton
from fastapi import FastAPI
import uvicorn
import threading

# --- تنظیمات اولیه ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- متغیرهای محیطی ---
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

# --- برنامه FastAPI ---
app = FastAPI()
@app.get("/")
def read_root():
    return {"status": "Apex Advanced Bot is running!"}

# --- مدیریت وضعیت کاربر ---
user_states = {}

# --- توابع سازنده کیبوردها ---
def get_main_menu_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='📊 تحلیل تکنیکال', callback_data='menu_tech_analysis')],
        [InlineKeyboardButton(text='📰 اخبار و احساسات', callback_data='menu_news')],
        [InlineKeyboardButton(text='🐳 رصد نهنگ‌ها', callback_data='menu_whales')],
        [InlineKeyboardButton(text='🧠 سیگنال‌های AI', callback_data='menu_ai')]
    ])

def get_symbol_analysis_keyboard(symbol):
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='📈 نمایش چارت', callback_data=f'action_chart_{symbol}')],
        [InlineKeyboardButton(text='📉 اندیکاتورها', callback_data=f'action_indicators_{symbol}')],
        [InlineKeyboardButton(text='🗞 اخبار مرتبط', callback_data=f'action_news_{symbol}')],
        [InlineKeyboardButton(text='🔙 بازگشت (ورود نماد جدید)', callback_data='menu_tech_analysis')]
    ])

def get_back_to_main_menu_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='🔙 بازگشت به منوی اصلی', callback_data='main_menu')]
    ])

# --- منطق اصلی ربات ---

def handle_chat(msg):
    """پردازش فقط پیام‌های متنی"""
    content_type, chat_type, chat_id = telepot.glance(msg)
    
    if content_type != 'text':
        return

    text = msg['text']
    
    if user_states.get(chat_id) == 'awaiting_symbol':
        handle_symbol_input(chat_id, text)
        return
        
    if text == '/start':
        user_states[chat_id] = 'main_menu'
        bot.sendMessage(chat_id, 'به ربات هوشمند Apex خوش آمدید. چه بخشی را می‌خواهید بررسی کنید؟',
                        reply_markup=get_main_menu_keyboard())

def handle_callback_query(msg):
    """پردازش فقط کلیک روی دکمه‌ها"""
    query_id, from_id, query_data = telepot.glance(msg, flavor='callback_query')
    chat_id = from_id
    bot.answerCallbackQuery(query_id)
    logging.info(f"Callback received: {query_data} from {chat_id}")

    if query_data == 'main_menu':
        user_states[chat_id] = 'main_menu'
        # به جای ویرایش پیام، یک پیام جدید می‌فرستیم تا تمیزتر باشد
        bot.sendMessage(chat_id, 'منوی اصلی:', reply_markup=get_main_menu_keyboard())

    elif query_data == 'menu_tech_analysis':
        user_states[chat_id] = 'awaiting_symbol'
        bot.sendMessage(chat_id, 'لطفاً نماد ارز مورد نظر خود را با فرمت صحیح وارد کنید (مثلاً: BTCUSDT).',
                        reply_markup=get_back_to_main_menu_keyboard())

    elif query_data == 'menu_whales':
        user_states[chat_id] = 'whales_menu'
        message = ("🐳 **بخش رصد نهنگ‌ها (نسخه آزمایشی)**\n\n"
                   "این بخش به زودی فعال خواهد شد.")
        bot.sendMessage(chat_id, message, reply_markup=get_back_to_main_menu_keyboard())
    
    elif query_data.startswith('action_'):
        parts = query_data.split('_')
        action = parts[1]
        symbol = parts[2]
        
        if action == 'chart':
            bot.sendMessage(chat_id, f'در حال آماده‌سازی چارت برای {symbol}...')
        elif action == 'indicators':
            bot.sendMessage(chat_id, f'در حال واکشی اندیکاتورها برای {symbol}...')
        elif action == 'news':
            bot.sendMessage(chat_id, f'در حال جستجوی اخبار برای {symbol}...')

def handle_symbol_input(chat_id, text):
    """پردازش متنی که کاربر به عنوان نماد وارد کرده"""
    symbol = text.strip().upper()
    is_valid = True # TODO: Add validation logic
    
    if is_valid:
        user_states[chat_id] = f'symbol_menu_{symbol}'
        bot.sendMessage(chat_id, f'نماد {symbol} تایید شد. کدام تحلیل را نیاز دارید؟',
                        reply_markup=get_symbol_analysis_keyboard(symbol))
    else:
        bot.sendMessage(chat_id, 'خطا: نماد وارد شده معتبر نیست.')

# --- راه‌اندازی ربات و وب‌سرور ---
if not TELEGRAM_TOKEN:
    logging.fatal("TELEGRAM_TOKEN not found!")
    exit()

bot = telepot.Bot(TELEGRAM_TOKEN)
# اینجا منطق را جدا می‌کنیم
MessageLoop(bot, {'chat': handle_chat,
                  'callback_query': handle_callback_query}).run_as_thread()
logging.info('Telepot bot is listening...')

def run_web_server():
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

if os.getenv('RAILWAY_ENVIRONMENT'):
    threading.Thread(target=run_web_server, daemon=True).start()

logging.info("Bot is running. Press Ctrl+C to exit.")
while 1:
    time.sleep(10)