# main.py (نسخه نهایی با ساختار منوی پیشرفته و Telepot)

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

# --- برنامه FastAPI (برای بیدار نگه داشتن در Railway) ---
app = FastAPI()
@app.get("/")
def read_root():
    return {"status": "Apex Advanced Bot is running!"}

# --- مدیریت وضعیت کاربر ---
user_states = {} # دیکشنری برای ذخیره وضعیت هر کاربر. مثلا: {chat_id: 'awaiting_symbol'}

# --- توابع سازنده کیبوردها (برای تمیز بودن کد) ---

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

def handle(msg):
    """پردازش تمام پیام‌های ورودی (متنی و کلیک دکمه‌ها)"""
    content_type, chat_type, chat_id = telepot.glance(msg)
    
    # اگر پیام از نوع کلیک روی دکمه باشد
    if content_type == 'callback_query':
        query_id, from_id, query_data = telepot.glance(msg, flavor='callback_query')
        chat_id = from_id # برای کلیک‌ها، chat_id همان from_id است
        bot.answerCallbackQuery(query_id)
        handle_callback_query(chat_id, query_data)
        return

    # اگر پیام از نوع متنی باشد
    if content_type == 'text':
        text = msg['text']
        
        # اگر کاربر در حال وارد کردن نماد است
        if user_states.get(chat_id) == 'awaiting_symbol':
            handle_symbol_input(chat_id, text)
            return
            
        # اگر دستور /start است
        if text == '/start':
            user_states[chat_id] = 'main_menu' # کاربر را به منوی اصلی برمی‌گردانیم
            bot.sendMessage(chat_id, 'به ربات هوشمند Apex خوش آمدید. چه بخشی را می‌خواهید بررسی کنید؟',
                            reply_markup=get_main_menu_keyboard())

def handle_callback_query(chat_id, query_data):
    """پردازش منطق مربوط به کلیک روی دکمه‌ها"""
    
    # --- مدیریت منوی اصلی ---
    if query_data == 'main_menu':
        user_states[chat_id] = 'main_menu'
        bot.sendMessage(chat_id, 'منوی اصلی:', reply_markup=get_main_menu_keyboard())

    elif query_data == 'menu_tech_analysis':
        user_states[chat_id] = 'awaiting_symbol' # ربات را در حالت انتظار برای دریافت نماد قرار می‌دهیم
        bot.sendMessage(chat_id, 'لطفاً نماد ارز مورد نظر خود را با فرمت صحیح وارد کنید (مثلاً: BTCUSDT).',
                        reply_markup=get_back_to_main_menu_keyboard())

    elif query_data == 'menu_whales':
        user_states[chat_id] = 'whales_menu'
        # این بخش فعلاً نمایشی است
        message = ("🐳 **بخش رصد نهنگ‌ها (نسخه آزمایشی)**\n\n"
                   "این بخش در آینده نزدیک فعال خواهد شد و شامل موارد زیر خواهد بود:\n"
                   "- **رادار آلت‌کوین‌ها:** شناسایی ارزهای نوپایی که توسط نهنگ‌های هوشمند خریداری می‌شوند.\n"
                   "- **تراکنش‌های بزرگ:** اطلاع‌رسانی آنی از جابجایی‌های عظیم در شبکه.")
        bot.sendMessage(chat_id, message, reply_markup=get_back_to_main_menu_keyboard())
    
    # --- مدیریت اقدامات مربوط به یک نماد خاص ---
    elif query_data.startswith('action_'):
        parts = query_data.split('_')
        action = parts[1]
        symbol = parts[2]
        
        if action == 'chart':
            bot.sendMessage(chat_id, f'در حال آماده‌سازی چارت برای {symbol}...')
            # TODO: اینجا کد رسم نمودار قرار می‌گیرد
        elif action == 'indicators':
            bot.sendMessage(chat_id, f'در حال واکشی اندیکاتورها برای {symbol}...')
            # TODO: اینجا کد واکشی اندیکاتورها از دیتابیس قرار می‌گیرد
        elif action == 'news':
            bot.sendMessage(chat_id, f'در حال جستجوی اخبار برای {symbol}...')
            # TODO: اینجا کد جستجوی اخبار از NewsAPI قرار می‌گیرد


def handle_symbol_input(chat_id, text):
    """پردازش متنی که کاربر به عنوان نماد وارد کرده"""
    symbol = text.strip().upper() # حذف فاصله‌های اضافی و تبدیل به حروف بزرگ
    
    # TODO: در اینجا باید یک تابع برای اعتبارسنجی نماد از طریق API بایننس اضافه کنیم
    # فعلاً فرض می‌کنیم هر ورودی معتبر است
    is_valid = True 
    
    if is_valid:
        user_states[chat_id] = f'symbol_menu_{symbol}' # وضعیت کاربر را به منوی این نماد تغییر می‌دهیم
        bot.sendMessage(chat_id, f'نماد {symbol} تایید شد. کدام تحلیل را نیاز دارید؟',
                        reply_markup=get_symbol_analysis_keyboard(symbol))
    else:
        bot.sendMessage(chat_id, 'خطا: نماد وارد شده معتبر نیست. لطفاً دوباره تلاش کنید.')


# --- راه‌اندازی ربات و وب‌سرور ---
if not TELEGRAM_TOKEN:
    logging.fatal("TELEGRAM_TOKEN not found!")
    exit()

bot = telepot.Bot(TELEGRAM_TOKEN)
MessageLoop(bot, handle).run_as_thread()
logging.info('Telepot bot is listening...')

def run_web_server():
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# اگر در حال اجرا روی Railway هستیم، وب‌سرور را اجرا کن
if os.getenv('RAILWAY_ENVIRONMENT'):
    threading.Thread(target=run_web_server, daemon=True).start()

# نگه داشتن برنامه اصلی در حال اجرا
logging.info("Bot is running. Press Ctrl+C to exit.")
while 1:
    time.sleep(10)