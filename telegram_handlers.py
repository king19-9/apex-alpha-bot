from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import logging

logger = logging.getLogger(__name__)

def setup_handlers(application, bot):
    """تنظیم هندلرهای تلگرام"""
    # ذخیره نمونه bot در داده‌های اپلیکیشن
    application.bot_data['trading_bot'] = bot
    
    # هندلر دستور /start
    application.add_handler(CommandHandler("start", start_command))
    
    # هندلر دستور /help
    application.add_handler(CommandHandler("help", help_command))
    
    # هندلر دستور /analyze
    application.add_handler(CommandHandler("analyze", analyze_command))
    
    # هندلر دستور /price
    application.add_handler(CommandHandler("price", price_command))
    
    # هندلر دستور /news
    application.add_handler(CommandHandler("news", news_command))
    
    # هندلر دستور /signals
    application.add_handler(CommandHandler("signals", signals_command))
    
    # هندلر دستور /watchlist
    application.add_handler(CommandHandler("watchlist", watchlist_command))
    
    # هندلر دستور /settings
    application.add_handler(CommandHandler("settings", settings_command))
    
    # هندلر دستور /advanced (برای تحلیل‌های پیشرفته)
    application.add_handler(CommandHandler("advanced", advanced_command))
    
    # هندلر پیام‌های متنی
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    
    # هندلر callback query
    application.add_handler(CallbackQueryHandler(callback_query_handler))


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /start"""
    await update.message.reply_text(
        "به ربات تحلیلگر ارزهای دیجیتال خوش آمدید! 🚀\n\n"
        "با استفاده از این ربات می‌توانید:\n"
        "• تحلیل کامل ارزهای دیجیتال\n"
        "• دریافت سیگنال‌های معاملاتی\n"
        "• تحلیل اخبار و احساسات بازار\n"
        "• مدیریت واچ‌لیست شخصی\n"
        "• تحلیل‌های پیشرفته (ویچاف، پروفایل حجمی، هارمونیک و ...)\n\n"
        "برای شروع، از دستور /help استفاده کنید یا نام ارز را ارسال کنید."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /help"""
    help_text = """
    راهنمای استفاده از ربات 📚
    
    /start - شروع ربات
    /help - نمایش این راهنما
    /analyze [symbol] - تحلیل کامل ارز (مثال: /analyze BTC)
    /price [symbol] - دریافت قیمت لحظه‌ای (مثال: /price ETH)
    /news [symbol] - دریافت اخبار مرتبط (مثال: /news BTC)
    /signals - دریافت سیگنال‌های معاملاتی
    /watchlist - مدیریت واچ‌لیست شخصی
    /settings - تنظیمات کاربر
    /advanced [symbol] - تحلیل‌های پیشرفته (مثال: /advanced BTC)
    
    همچنین می‌توانید مستقیماً نام ارز را ارسال کنید تا تحلیل کامل آن را دریافت کنید.
    """
    await update.message.reply_text(help_text)


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /analyze"""
    bot = context.bot_data.get('trading_bot')
    if not bot:
        await update.message.reply_text("ربات در حال حاضر در دسترس نیست. لطفاً بعداً تلاش کنید.")
        return
    
    # استخراج نماد از پیام
    if context.args:
        symbol = context.args[0].upper()
    else:
        await update.message.reply_text("لطفاً نماد ارز را وارد کنید. مثال: /analyze BTC")
        return
    
    # ارسال پیام در حال پردازش
    processing_message = await update.message.reply_text(f"در حال تحلیل {symbol}... لطفاً صبر کنید.")
    
    try:
        # انجام تحلیل
        analysis = await bot.perform_advanced_analysis(symbol)
        
        # ایجاد پاسخ
        response = format_analysis_response(analysis)
        
        # ویرایش پیام با نتیجه تحلیل
        await processing_message.edit_text(response, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in analyze_command: {e}")
        await processing_message.edit_text("متأسفانه در تحلیل خطایی رخ داد. لطفاً دوباره تلاش کنید.")


async def advanced_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /advanced برای تحلیل‌های پیشرفته"""
    bot = context.bot_data.get('trading_bot')
    if not bot:
        await update.message.reply_text("ربات در حال حاضر در دسترس نیست. لطفاً بعداً تلاش کنید.")
        return
    
    # استخراج نماد از پیام
    if context.args:
        symbol = context.args[0].upper()
    else:
        await update.message.reply_text("لطفاً نماد ارز را وارد کنید. مثال: /advanced BTC")
        return
    
    # ارسال پیام در حال پردازش
    processing_message = await update.message.reply_text(f"در حال تحلیل پیشرفته {symbol}... لطفاً صبر کنید.")
    
    try:
        # انجام تحلیل
        analysis = await bot.perform_advanced_analysis(symbol)
        
        # ایجاد پاسخ برای تحلیل پیشرفته
        response = format_advanced_analysis_response(analysis)
        
        # ویرایش پیام با نتیجه تحلیل
        await processing_message.edit_text(response, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in advanced_command: {e}")
        await processing_message.edit_text("متأسفانه در تحلیل پیشرفته خطایی رخ داد. لطفاً دوباره تلاش کنید.")


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /price"""
    bot = context.bot_data.get('trading_bot')
    if not bot:
        await update.message.reply_text("ربات در حال حاضر در دسترس نیست. لطفاً بعداً تلاش کنید.")
        return
    
    # استخراج نماد از پیام
    if context.args:
        symbol = context.args[0].upper()
    else:
        await update.message.reply_text("لطفاً نماد ارز را وارد کنید. مثال: /price ETH")
        return
    
    # ارسال پیام در حال پردازش
    processing_message = await update.message.reply_text(f"در حال دریافت قیمت {symbol}...")
    
    try:
        # دریافت داده‌های بازار
        market_data = await bot.get_market_data(symbol)
        
        # ایجاد پاسخ
        response = f"💰 *{symbol} Price Information*\n\n"
        response += f"• قیمت: ${market_data.get('price', 0):,.2f}\n"
        response += f"• تغییر 24h: {market_data.get('price_change_24h', 0):+.2f}%\n"
        response += f"• حجم 24h: ${market_data.get('volume_24h', 0):,.0f}\n"
        response += f"• ارزش بازار: ${market_data.get('market_cap', 0):,.0f}\n"
        response += f"• منابع: {', '.join(market_data.get('sources', []))}"
        
        # ویرایش پیام با نتیجه
        await processing_message.edit_text(response, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in price_command: {e}")
        await processing_message.edit_text("متأسفانه در دریافت قیمت خطایی رخ داد. لطفاً دوباره تلاش کنید.")


async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /news"""
    bot = context.bot_data.get('trading_bot')
    if not bot:
        await update.message.reply_text("ربات در حال حاضر در دسترس نیست. لطفاً بعداً تلاش کنید.")
        return
    
    # استخراج نماد از پیام
    if context.args:
        symbol = context.args[0].upper()
    else:
        await update.message.reply_text("لطفاً نماد ارز را وارد کنید. مثال: /news BTC")
        return
    
    # ارسال پیام در حال پردازش
    processing_message = await update.message.reply_text(f"در حال دریافت اخبار {symbol}...")
    
    try:
        # دریافت اخبار
        news = await bot.fetch_news_from_multiple_sources(symbol)
        
        if not news:
            await processing_message.edit_text(f"هیچ خبری برای {symbol} یافت نشد.")
            return
        
        # ایجاد پاسخ
        response = f"📰 *اخبار {symbol}*\n\n"
        
        # نمایش حداکثر 5 خبر
        for i, item in enumerate(news[:5]):
            response += f"{i+1}. *{item['title']}*\n"
            response += f"   منبع: {item['source']}\n"
            response += f"   [لینک خبر]({item['url']})\n\n"
        
        # ویرایش پیام با نتیجه
        await processing_message.edit_text(response, parse_mode='Markdown', disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"Error in news_command: {e}")
        await processing_message.edit_text("متأسفانه در دریافت اخبار خطایی رخ داد. لطفاً دوباره تلاش کنید.")


async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /signals"""
    bot = context.bot_data.get('trading_bot')
    if not bot:
        await update.message.reply_text("ربات در حال حاضر در دسترس نیست. لطفاً بعداً تلاش کنید.")
        return
    
    # ارسال پیام در حال پردازش
    processing_message = await update.message.reply_text("در حال دریافت سیگنال‌های معاملاتی...")
    
    try:
        # در یک پیاده‌سازی واقعی، این باید از پایگاه داده یا تحلیل زنده استفاده کند
        # اینجا چند سیگنال نمونه نمایش داده می‌شود
        
        # ایجاد دکمه‌های اینلاین
        keyboard = [
            [InlineKeyboardButton("تحلیل BTC", callback_data="analyze_BTC")],
            [InlineKeyboardButton("تحلیل ETH", callback_data="analyze_ETH")],
            [InlineKeyboardButton("تحلیل BNB", callback_data="analyze_BNB")],
            [InlineKeyboardButton("刷新 سیگنال‌ها", callback_data="refresh_signals")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ایجاد پاسخ
        response = "📊 *سیگنال‌های معاملاتی امروز*\n\n"
        response += "• *BTC/USDT*: سیگنال خرید (اطمینان: 85%)\n"
        response += "• *ETH/USDT*: سیگنال نگه دار (اطمینان: 65%)\n"
        response += "• *BNB/USDT*: سیگنال فروش (اطمینان: 75%)\n\n"
        response += "برای تحلیل کامل، روی یکی از گزینه‌های زیر کلیک کنید:"
        
        # ویرایش پیام با نتیجه
        await processing_message.edit_text(response, parse_mode='Markdown', reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error in signals_command: {e}")
        await processing_message.edit_text("متأسفانه در دریافت سیگنال‌ها خطایی رخ داد. لطفاً دوباره تلاش کنید.")


async def watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /watchlist"""
    user_id = update.effective_user.id
    
    # در یک پیاده‌سازی واقعی، این باید از پایگاه داده استفاده کند
    # اینجا یک واچ‌لیست نمونه نمایش داده می‌شود
    
    # ایجاد دکمه‌های اینلاین
    keyboard = [
        [InlineKeyboardButton("افزودن ارز", callback_data="add_to_watchlist")],
        [InlineKeyboardButton("حذف ارز", callback_data="remove_from_watchlist")],
        [InlineKeyboardButton("刷新 واچ‌لیست", callback_data="refresh_watchlist")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # ایجاد پاسخ
    response = f"📋 *واچ‌لیست شما*\n\n"
    response += "• BTC/USDT\n"
    response += "• ETH/USDT\n"
    response += "• BNB/USDT\n\n"
    response += "برای مدیریت واچ‌لیست، از دکمه‌های زیر استفاده کنید:"
    
    await update.message.reply_text(response, parse_mode='Markdown', reply_markup=reply_markup)


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /settings"""
    user_id = update.effective_user.id
    
    # ایجاد دکمه‌های اینلاین
    keyboard = [
        [InlineKeyboardButton("تغییر زبان", callback_data="change_language")],
        [InlineKeyboardButton("تنظیمات هشدار", callback_data="alert_settings")],
        [InlineKeyboardButton("تنظیمات تحلیل", callback_data="analysis_settings")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # ایجاد پاسخ
    response = "⚙️ *تنظیمات*\n\n"
    response += "• زبان: فارسی\n"
    response += "• هشدارها: فعال\n"
    response += "• تحلیل پیشرفته: فعال\n\n"
    response += "برای تغییر تنظیمات، از دکمه‌های زیر استفاده کنید:"
    
    await update.message.reply_text(response, parse_mode='Markdown', reply_markup=reply_markup)


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر پیام‌های متنی"""
    bot = context.bot_data.get('trading_bot')
    text = update.message.text
    
    # اگر متن یک نماد ارز است
    if text.isalpha() and len(text) <= 10:
        symbol = text.upper()
        
        # ارسال پیام در حال پردازش
        processing_message = await update.message.reply_text(f"در حال تحلیل {symbol}... لطفاً صبر کنید.")
        
        try:
            # انجام تحلیل
            analysis = await bot.perform_advanced_analysis(symbol)
            
            # ایجاد پاسخ
            response = format_analysis_response(analysis)
            
            # ویرایش پیام با نتیجه تحلیل
            await processing_message.edit_text(response, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in message_handler: {e}")
            await processing_message.edit_text("متأسفانه در تحلیل خطایی رخ داد. لطفاً دوباره تلاش کنید.")
    else:
        await update.message.reply_text("لطفاً یک نماد ارز معتبر وارد کنید یا از دستور /help استفاده کنید.")


async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر callback query"""
    query = update.callback_query
    query.answer()
    
    data = query.data
    bot = context.bot_data.get('trading_bot')
    
    if data.startswith("analyze_"):
        symbol = data.split("_")[1]
        
        # ویرایش پیام با پیام در حال پردازش
        await query.edit_message_text(f"در حال تحلیل {symbol}... لطفاً صبر کنید.")
        
        try:
            # انجام تحلیل
            analysis = await bot.perform_advanced_analysis(symbol)
            
            # ایجاد پاسخ
            response = format_analysis_response(analysis)
            
            # ویرایش پیام با نتیجه تحلیل
            await query.edit_message_text(response, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in callback_query_handler: {e}")
            await query.edit_message_text("متأسفانه در تحلیل خطایی رخ داد. لطفاً دوباره تلاش کنید.")
    
    elif data == "refresh_signals":
        # ایجاد دکمه‌های اینلاین
        keyboard = [
            [InlineKeyboardButton("تحلیل BTC", callback_data="analyze_BTC")],
            [InlineKeyboardButton("تحلیل ETH", callback_data="analyze_ETH")],
            [InlineKeyboardButton("تحلیل BNB", callback_data="analyze_BNB")],
            [InlineKeyboardButton("刷新 سیگنال‌ها", callback_data="refresh_signals")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ایجاد پاسخ
        response = "📊 *سیگنال‌های معاملاتی امروز (به‌روز شده)*\n\n"
        response += "• *BTC/USDT*: سیگنال خرید (اطمینان: 85%)\n"
        response += "• *ETH/USDT*: سیگنال نگه دار (اطمینان: 65%)\n"
        response += "• *BNB/USDT*: سیگنال فروش (اطمینان: 75%)\n\n"
        response += "برای تحلیل کامل، روی یکی از گزینه‌های زیر کلیک کنید:"
        
        await query.edit_message_text(response, parse_mode='Markdown', reply_markup=reply_markup)
    
    elif data == "add_to_watchlist":
        await query.edit_message_text("لطفاً نماد ارزی را که می‌خواهید به واچ‌لیست اضافه کنید، ارسال کنید:")
        # در یک پیاده‌سازی واقعی، باید حالت را برای کاربر ذخیره کرد
    
    elif data == "remove_from_watchlist":
        await query.edit_message_text("لطفاً نماد ارزی را که می‌خواهید از واچ‌لیست حذف کنید، ارسال کنید:")
        # در یک پیاده‌سازی واقعی، باید حالت را برای کاربر ذخیره کرد
    
    elif data == "refresh_watchlist":
        # ایجاد دکمه‌های اینلاین
        keyboard = [
            [InlineKeyboardButton("افزودن ارز", callback_data="add_to_watchlist")],
            [InlineKeyboardButton("حذف ارز", callback_data="remove_from_watchlist")],
            [InlineKeyboardButton("刷新 واچ‌لیست", callback_data="refresh_watchlist")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ایجاد پاسخ
        response = f"📋 *واچ‌لیست شما (به‌روز شده)*\n\n"
        response += "• BTC/USDT\n"
        response += "• ETH/USDT\n"
        response += "• BNB/USDT\n\n"
        response += "برای مدیریت واچ‌لیست، از دکمه‌های زیر استفاده کنید:"
        
        await query.edit_message_text(response, parse_mode='Markdown', reply_markup=reply_markup)
    
    elif data == "change_language":
        await query.edit_message_text("تغییر زبان در حال حاضر در دسترس نیست. زبان فعلی: فارسی")
    
    elif data == "alert_settings":
        await query.edit_message_text("تنظیمات هشدار در حال حاضر در دسترس نیست")
    
    elif data == "analysis_settings":
        await query.edit_message_text("تنظیمات تحلیل در حال حاضر در دسترس نیست")


def format_analysis_response(analysis):
    """فرمت‌بندی پاسخ تحلیل"""
    symbol = analysis.get('symbol', 'UNKNOWN')
    signal = analysis.get('signal', 'UNKNOWN')
    confidence = analysis.get('confidence', 0)
    
    # ایجاد پاسخ
    response = f"📊 *تحلیل کامل {symbol}*\n\n"
    
    # سیگنال و اطمینان
    signal_emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
    response += f"{signal_emoji} *سیگنال*: {signal}\n"
    response += f"📈 *اطمینان*: {confidence:.1%}\n\n"
    
    # داده‌های بازار
    market_data = analysis.get('market_data', {})
    response += f"💰 *داده‌های بازار*\n"
    response += f"• قیمت: ${market_data.get('price', 0):,.2f}\n"
    response += f"• تغییر 24h: {market_data.get('price_change_24h', 0):+.2f}%\n"
    response += f"• حجم 24h: ${market_data.get('volume_24h', 0):,.0f}\n"
    response += f"• ارزش بازار: ${market_data.get('market_cap', 0):,.0f}\n\n"
    
    # تحلیل تکنیکال
    technical = analysis.get('technical', {})
    classical = technical.get('classical', {})
    response += f"📈 *تحلیل تکنیکال*\n"
    
    if 'rsi' in classical:
        rsi = classical['rsi'].get('14', 50)
        rsi_signal = "اشباع خرید" if rsi > 70 else "اشباع فروش" if rsi < 30 else "خنثی"
        response += f"• RSI(14): {rsi:.1f} ({rsi_signal})\n"
    
    if 'macd' in classical:
        macd = classical['macd']
        macd_signal = "صعودی" if macd.get('macd', 0) > macd.get('signal', 0) else "نزولی"
        response += f"• MACD: {macd_signal}\n"
    
    if 'trend' in classical:
        trend = classical['trend']
        response += f"• روند: {trend.get('direction', 'نامشخص')}\n"
    
    response += "\n"
    
    # تحلیل احساسات
    sentiment = analysis.get('sentiment', {})
    avg_sentiment = sentiment.get('average_sentiment', 0)
    sentiment_signal = "مثبت" if avg_sentiment > 0.2 else "منفی" if avg_sentiment < -0.2 else "خنثی"
    response += f"💭 *تحلیل احساسات*\n"
    response += f"• احساسات بازار: {sentiment_signal} ({avg_sentiment:.2f})\n"
    response += f"• تعداد اخبار: {sentiment.get('news_count', 0)}\n\n"
    
    # پیشنهاد نهایی
    if signal == "BUY":
        response += "🟢 *پیشنهاد*: خرید با ریسک متوسط\n"
    elif signal == "SELL":
        response += "🔴 *پیشنهاد*: فروش با احتیاط\n"
    else:
        response += "🟡 *پیشنهاد*: منتظر سیگنال بعدی بمانید\n"
    
    response += f"\n⏱ زمان تحلیل: {analysis.get('timestamp', 'نامشخص')}"
    
    return response


def format_advanced_analysis_response(analysis):
    """فرمت‌بندی پاسخ تحلیل پیشرفته"""
    symbol = analysis.get('symbol', 'UNKNOWN')
    signal = analysis.get('signal', 'UNKNOWN')
    confidence = analysis.get('confidence', 0)
    
    # ایجاد پاسخ
    response = f"🚀 *تحلیل پیشرفته {symbol}*\n\n"
    
    # سیگنال و اطمینان
    signal_emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
    response += f"{signal_emoji} *سیگنال*: {signal}\n"
    response += f"📈 *اطمینان*: {confidence:.1%}\n\n"
    
    # داده‌های بازار
    market_data = analysis.get('market_data', {})
    response += f"💰 *داده‌های بازار*\n"
    response += f"• قیمت: ${market_data.get('price', 0):,.2f}\n"
    response += f"• تغییر 24h: {market_data.get('price_change_24h', 0):+.2f}%\n"
    response += f"• حجم 24h: ${market_data.get('volume_24h', 0):,.0f}\n"
    response += f"• ارزش بازار: ${market_data.get('market_cap', 0):,.0f}\n\n"
    
    # تحلیل‌های پیشرفته
    advanced = analysis.get('advanced_analysis', {})
    
    # تحلیل ویچاف
    wyckoff = advanced.get('wyckoff', {})
    if wyckoff:
        response += f"🔍 *تحلیل ویچاف*\n"
        response += f"• فاز: {wyckoff.get('phase', 'نامشخص')}\n"
        response += f"• انباشت: {'بله' if wyckoff.get('accumulation_phase', False) else 'خیر'}\n"
        response += f"• توزیع: {'بله' if wyckoff.get('distribution_phase', False) else 'خیر'}\n\n"
    
    # تحلیل پروفایل حجمی
    volume_profile = advanced.get('volume_profile', {})
    if volume_profile:
        response += f"📊 *پروفایل حجمی*\n"
        poc = volume_profile.get('poc', {})
        response += f"• POC: ${poc.get('price_level', 0):.2f}\n"
        response += f"• محدوده ارزش: ${volume_profile.get('value_area_low', 0):.2f} - ${volume_profile.get('value_area_high', 0):.2f}\n\n"
    
    # تحلیل هارمونیک
    harmonic = advanced.get('harmonic_patterns', {})
    if harmonic:
        response += f"🎵 *الگوهای هارمونیک*\n"
        response += f"• تعداد الگوها: {harmonic.get('pattern_count', 0)}\n"
        patterns = harmonic.get('patterns_found', [])
        for pattern in patterns[:2]:  # نمایش 2 الگو
            response += f"• {pattern.get('pattern', '')}: {pattern.get('type', '')}\n"
        response += "\n"
    
    # تحلیل ابر ایچیموکو
    ichimoku = advanced.get('ichimoku', {})
    if ichimoku:
        response += f"☁️ *ابر ایچیموکو*\n"
        response += f"• Tenkan-sen: ${ichimoku.get('tenkan_sen', 0):.2f}\n"
        response += f"• Kijun-sen: ${ichimoku.get('kijun_sen', 0):.2f}\n"
        response += f"• قیمت بالای ابر: {'بله' if ichimoku.get('price_above_kumo', False) else 'خیر'}\n\n"
    
    # تحلیل ساختار بازار
    market_structure = advanced.get('market_structure', {})
    if market_structure:
        response += f"🏗️ *ساختار بازار*\n"
        response += f"• روند: {market_structure.get('market_trend', 'نامشخص')}\n"
        order_blocks = market_structure.get('order_blocks', [])
        response += f"• Order Block‌ها: {len(order_blocks)}\n\n"
    
    # تحلیل جریان سفارش
    order_flow = advanced.get('order_flow', {})
    if order_flow:
        response += f"🔄 *جریان سفارش*\n"
        response += f"• نسبت خرید/فروش: {order_flow.get('buy_sell_ratio', 1):.2f}\n"
        response += f"• حجم خرید: ${order_flow.get('buy_volume', 0):,.0f}\n"
        response += f"• حجم فروش: ${order_flow.get('sell_volume', 0):,.0f}\n\n"
    
    # تحلیل هوش مصنوعی
    ai_analysis = analysis.get('ai_analysis', {})
    if ai_analysis:
        response += f"🤖 *تحلیل هوش مصنوعی*\n"
        response += f"• پیش‌بینی نهایی: ${ai_analysis.get('final_prediction', 0):.2f}\n"
        response += f"• تعداد مدل‌ها: {len(ai_analysis.get('predictions', {}))}\n\n"
    
    # پیشنهاد نهایی
    if signal == "BUY":
        response += "🟢 *پیشنهاد*: خرید با ریسک متوسط\n"
    elif signal == "SELL":
        response += "🔴 *پیشنهاد*: فروش با احتیاط\n"
    else:
        response += "🟡 *پیشنهاد*: منتظر سیگنال بعدی بمانید\n"
    
    response += f"\n⏱ زمان تحلیل: {analysis.get('timestamp', 'نامشخص')}"
    
    return response