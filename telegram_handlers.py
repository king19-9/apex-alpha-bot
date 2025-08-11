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
    
    # هندلر دستور /portfolio (برای مدیریت پورتفولیو)
    application.add_handler(CommandHandler("portfolio", portfolio_command))
    
    # هندلر دستور /alert (برای تنظیم هشدارها)
    application.add_handler(CommandHandler("alert", alert_command))
    
    # هندلر دستور /risk (برای تحلیل ریسک)
    application.add_handler(CommandHandler("risk", risk_command))
    
    # هندلر پیام‌های متنی
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    
    # هندلر callback query
    application.add_handler(CallbackQueryHandler(callback_query_handler))


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /start"""
    try:
        logger.info(f"Start command received from user {update.effective_user.id}")
        
        # ایجاد منوی اصلی
        keyboard = [
            [InlineKeyboardButton("📊 تحلیل ارز", callback_data="analyze_menu")],
            [InlineKeyboardButton("💰 قیمت لحظه‌ای", callback_data="price_menu")],
            [InlineKeyboardButton("📰 اخبار", callback_data="news_menu")],
            [InlineKeyboardButton("🚀 سیگنال‌ها", callback_data="signals_menu")],
            [InlineKeyboardButton("⚙️ تنظیمات", callback_data="settings_menu")],
            [InlineKeyboardButton("📚 راهنما", callback_data="help_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🤖 **به ربات تحلیلگر حرفه‌ای ارزهای دیجیتال خوش آمدید!**\n\n"
            "این ربات با استفاده از هوش مصنوعی و تحلیل‌های پیشرفته، بهترین سیگنال‌های معاملاتی را به شما ارائه می‌دهد.\n\n"
            "لطفاً یکی از گزینه‌های زیر را انتخاب کنید:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"Error in start_command: {e}")
        await update.message.reply_text("متأسفانه در پردازش درخواست شما خطایی رخ داد.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /help"""
    try:
        keyboard = [
            [InlineKeyboardButton("🔙 بازگشت به منوی اصلی", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        help_text = """
        📚 **راهنمای استفاده از ربات**

        **دستورات اصلی:**
        `/start` - نمایش منوی اصلی
        `/analyze [symbol]` - تحلیل کامل ارز (مثال: `/analyze BTC`)
        `/price [symbol]` - دریافت قیمت لحظه‌ای (مثال: `/price ETH`)
        `/news [symbol]` - دریافت اخبار مرتبط (مثال: `/news BTC`)
        `/signals` - دریافت سیگنال‌های معاملاتی
        `/advanced [symbol]` - تحلیل‌های پیشرفته (مثال: `/advanced BTC`)

        **دستورات پیشرفته:**
        `/portfolio` - مدیریت پورتفولیو
        `/alert` - تنظیم هشدارها
        `/risk [symbol]` - تحلیل ریسک
        `/watchlist` - مدیریت واچ‌لیست
        `/settings` - تنظیمات کاربر

        **قابلیت‌های ربات:**
        • تحلیل تکنیکال پیشرفته (RSI, MACD, الگوهای شمعی)
        • تحلیل احساسات بازار با پردازش اخبار
        • تحلیل چند زمانی (Multi-timeframe)
        • تحلیل ساختار بازار (Order Block, Supply & Demand)
        • تحلیل امواج الیوت
        • مدیریت ریسک و سرمایه
        • سیگنال‌های معاملاتی هوشمند
        • پشتیبانی از چندین صرافی
        • حالت آفلاین برای تست

        همچنین می‌توانید مستقیماً نام ارز را ارسال کنید تا تحلیل کامل آن را دریافت کنید.
        """
        
        await update.message.reply_text(help_text, reply_markup=reply_markup, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in help_command: {e}")
        await update.message.reply_text("متأسفانه در پردازش درخواست شما خطایی رخ داد.")


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /analyze"""
    try:
        bot = context.bot_data.get('trading_bot')
        if not bot:
            await update.message.reply_text("ربات در حال حاضر در دسترس نیست. لطفاً بعداً تلاش کنید.")
            return
        
        # استخراج نماد از پیام
        if context.args:
            symbol = context.args[0].upper()
        else:
            keyboard = [
                [InlineKeyboardButton("BTC", callback_data="analyze_BTC")],
                [InlineKeyboardButton("ETH", callback_data="analyze_ETH")],
                [InlineKeyboardButton("BNB", callback_data="analyze_BNB")],
                [InlineKeyboardButton("SOL", callback_data="analyze_SOL")],
                [InlineKeyboardButton("XRP", callback_data="analyze_XRP")],
                [InlineKeyboardButton("ADA", callback_data="analyze_ADA")],
                [InlineKeyboardButton("🔙 بازگشت", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "لطفاً نماد ارز را برای تحلیل انتخاب کنید:",
                reply_markup=reply_markup
            )
            return
        
        # ارسال پیام در حال پردازش
        processing_message = await update.message.reply_text(f"🔄 در حال تحلیل {symbol}... لطفاً صبر کنید.")
        
        # انجام تحلیل
        analysis = await bot.perform_advanced_analysis(symbol)
        
        # ایجاد پاسخ
        response = format_analysis_response(analysis)
        
        # ایجاد دکمه‌های اینلاین
        keyboard = [
            [InlineKeyboardButton("🔄 تحلیل مجدد", callback_data=f"analyze_{symbol}")],
            [InlineKeyboardButton("📰 اخبار", callback_data=f"news_{symbol}")],
            [InlineKeyboardButton("⚠️ تحلیل ریسک", callback_data=f"risk_{symbol}")],
            [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ویرایش پیام با نتیجه تحلیل
        await processing_message.edit_text(response, parse_mode='Markdown', reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error in analyze_command: {e}")
        await update.message.reply_text("متأسفانه در تحلیل خطایی رخ داد. لطفاً دوباره تلاش کنید.")


async def advanced_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /advanced برای تحلیل‌های پیشرفته"""
    try:
        bot = context.bot_data.get('trading_bot')
        if not bot:
            await update.message.reply_text("ربات در حال حاضر در دسترس نیست. لطفاً بعداً تلاش کنید.")
            return
        
        # استخراج نماد از پیام
        if context.args:
            symbol = context.args[0].upper()
        else:
            keyboard = [
                [InlineKeyboardButton("BTC", callback_data="advanced_BTC")],
                [InlineKeyboardButton("ETH", callback_data="advanced_ETH")],
                [InlineKeyboardButton("BNB", callback_data="advanced_BNB")],
                [InlineKeyboardButton("SOL", callback_data="advanced_SOL")],
                [InlineKeyboardButton("🔙 بازگشت", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "لطفاً نماد ارز را برای تحلیل پیشرفته انتخاب کنید:",
                reply_markup=reply_markup
            )
            return
        
        # ارسال پیام در حال پردازش
        processing_message = await update.message.reply_text(f"🚀 در حال تحلیل پیشرفته {symbol}... لطفاً صبر کنید.")
        
        # انجام تحلیل
        analysis = await bot.perform_advanced_analysis(symbol)
        
        # ایجاد پاسخ برای تحلیل پیشرفته
        response = format_advanced_analysis_response(analysis)
        
        # ایجاد دکمه‌های اینلاین
        keyboard = [
            [InlineKeyboardButton("🔄 تحلیل مجدد", callback_data=f"advanced_{symbol}")],
            [InlineKeyboardButton("📊 تحلیل ساده", callback_data=f"analyze_{symbol}")],
            [InlineKeyboardButton("📰 اخبار", callback_data=f"news_{symbol}")],
            [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ویرایش پیام با نتیجه تحلیل
        await processing_message.edit_text(response, parse_mode='Markdown', reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error in advanced_command: {e}")
        await update.message.reply_text("متأسفانه در تحلیل پیشرفته خطایی رخ داد. لطفاً دوباره تلاش کنید.")


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /price"""
    try:
        bot = context.bot_data.get('trading_bot')
        if not bot:
            await update.message.reply_text("ربات در حال حاضر در دسترس نیست. لطفاً بعداً تلاش کنید.")
            return
        
        # استخراج نماد از پیام
        if context.args:
            symbol = context.args[0].upper()
        else:
            keyboard = [
                [InlineKeyboardButton("BTC", callback_data="price_BTC")],
                [InlineKeyboardButton("ETH", callback_data="price_ETH")],
                [InlineKeyboardButton("BNB", callback_data="price_BNB")],
                [InlineKeyboardButton("SOL", callback_data="price_SOL")],
                [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="refresh_prices")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "لطفاً نماد ارز را برای دریافت قیمت انتخاب کنید:",
                reply_markup=reply_markup
            )
            return
        
        # ارسال پیام در حال پردازش
        processing_message = await update.message.reply_text(f"💰 در حال دریافت قیمت {symbol}...")
        
        # دریافت داده‌های بازار
        market_data = await bot.get_market_data(symbol)
        
        # ایجاد پاسخ
        response = f"💰 *{symbol} Price Information*\n\n"
        response += f"• قیمت: ${market_data.get('price', 0):,.2f}\n"
        response += f"• تغییر 24h: {market_data.get('price_change_24h', 0):+.2f}%\n"
        response += f"• حجم 24h: ${market_data.get('volume_24h', 0):,.0f}\n"
        response += f"• ارزش بازار: ${market_data.get('market_cap', 0):,.0f}\n"
        response += f"• منابع: {', '.join(market_data.get('sources', []))}"
        
        # ایجاد دکمه‌های اینلاین
        keyboard = [
            [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data=f"price_{symbol}")],
            [InlineKeyboardButton("📊 تحلیل", callback_data=f"analyze_{symbol}")],
            [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ویرایش پیام با نتیجه
        await processing_message.edit_text(response, parse_mode='Markdown', reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error in price_command: {e}")
        await update.message.reply_text("متأسفانه در دریافت قیمت خطایی رخ داد. لطفاً دوباره تلاش کنید.")


async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /news"""
    try:
        bot = context.bot_data.get('trading_bot')
        if not bot:
            await update.message.reply_text("ربات در حال حاضر در دسترس نیست. لطفاً بعداً تلاش کنید.")
            return
        
        # استخراج نماد از پیام
        if context.args:
            symbol = context.args[0].upper()
        else:
            keyboard = [
                [InlineKeyboardButton("BTC", callback_data="news_BTC")],
                [InlineKeyboardButton("ETH", callback_data="news_ETH")],
                [InlineKeyboardButton("اخبار اقتصادی", callback_data="economic_news")],
                [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="refresh_news")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "لطفاً نماد ارز را برای دریافت اخبار انتخاب کنید:",
                reply_markup=reply_markup
            )
            return
        
        # ارسال پیام در حال پردازش
        processing_message = await update.message.reply_text(f"📰 در حال دریافت اخبار {symbol}...")
        
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
        
        # ایجاد دکمه‌های اینلاین
        keyboard = [
            [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data=f"news_{symbol}")],
            [InlineKeyboardButton("📊 تحلیل", callback_data=f"analyze_{symbol}")],
            [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ویرایش پیام با نتیجه
        await processing_message.edit_text(response, parse_mode='Markdown', disable_web_page_preview=True, reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error in news_command: {e}")
        await update.message.reply_text("متأسفانه در دریافت اخبار خطایی رخ داد. لطفاً دوباره تلاش کنید.")


async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /signals"""
    try:
        bot = context.bot_data.get('trading_bot')
        if not bot:
            await update.message.reply_text("ربات در حال حاضر در دسترس نیست. لطفاً بعداً تلاش کنید.")
            return
        
        # ارسال پیام در حال پردازش
        processing_message = await update.message.reply_text("🚀 در حال دریافت سیگنال‌های معاملاتی...")
        
        # ایجاد دکمه‌های اینلاین
        keyboard = [
            [InlineKeyboardButton("تحلیل BTC", callback_data="analyze_BTC")],
            [InlineKeyboardButton("تحلیل ETH", callback_data="analyze_ETH")],
            [InlineKeyboardButton("تحلیل BNB", callback_data="analyze_BNB")],
            [InlineKeyboardButton("تحلیل SOL", callback_data="analyze_SOL")],
            [InlineKeyboardButton("تحلیل XRP", callback_data="analyze_XRP")],
            [InlineKeyboardButton("🔄 به‌روزرسانی سیگنال‌ها", callback_data="refresh_signals")],
            [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ایجاد پاسخ
        response = "📊 *سیگنال‌های معاملاتی امروز*\n\n"
        response += "• *BTC/USDT*: سیگنال خرید (اطمینان: 85%)\n"
        response += "• *ETH/USDT*: سیگنال نگه دار (اطمینان: 65%)\n"
        response += "• *BNB/USDT*: سیگنال فروش (اطمینان: 75%)\n"
        response += "• *SOL/USDT*: سیگنال خرید (اطمینان: 80%)\n"
        response += "• *XRP/USDT*: سیگنال نگه دار (اطمینان: 60%)\n\n"
        response += "برای تحلیل کامل، روی یکی از گزینه‌های زیر کلیک کنید:"
        
        # ویرایش پیام با نتیجه
        await processing_message.edit_text(response, parse_mode='Markdown', reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error in signals_command: {e}")
        await update.message.reply_text("متأسفانه در دریافت سیگنال‌ها خطایی رخ داد. لطفاً دوباره تلاش کنید.")


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /portfolio"""
    try:
        keyboard = [
            [InlineKeyboardButton("➕ افزودن ارز", callback_data="add_portfolio")],
            [InlineKeyboardButton("➖ حذف ارز", callback_data="remove_portfolio")],
            [InlineKeyboardButton("📊 مشاهده پورتفولیو", callback_data="view_portfolio")],
            [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="refresh_portfolio")],
            [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        response = "💼 *مدیریت پورتفولیو*\n\n"
        response += "• *ارزش کل*: $12,450.00\n"
        response += "• *سود/زیان*: +$1,250.00 (+11.2%)\n"
        response += "• *تغییر 24h*: +$320.00 (+2.6%)\n\n"
        response += "• BTC: 0.25 ($10,750.00)\n"
        response += "• ETH: 2.5 ($5,500.00)\n"
        response += "• BNB: 5.0 ($1,500.00)\n\n"
        response += "برای مدیریت پورتفولیو، از دکمه‌های زیر استفاده کنید:"
        
        await update.message.reply_text(response, parse_mode='Markdown', reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error in portfolio_command: {e}")
        await update.message.reply_text("متأسفانه در مدیریت پورتفولیو خطایی رخ داد.")


async def alert_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /alert"""
    try:
        keyboard = [
            [InlineKeyboardButton("➕ افزودن هشدار", callback_data="add_alert")],
            [InlineKeyboardButton("➖ حذف هشدار", callback_data="remove_alert")],
            [InlineKeyboardButton("📋 مشاهده هشدارها", callback_data="view_alerts")],
            [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        response = "⚠️ *مدیریت هشدارها*\n\n"
        response += "• *هشدارهای فعال*: 3\n"
        response += "• BTC > $45,000\n"
        response += "• ETH < $2,000\n"
        response += "• BNP > $350\n\n"
        response += "برای مدیریت هشدارها، از دکمه‌های زیر استفاده کنید:"
        
        await update.message.reply_text(response, parse_mode='Markdown', reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error in alert_command: {e}")
        await update.message.reply_text("متأسفانه در مدیریت هشدارها خطایی رخ داد.")


async def risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /risk"""
    try:
        bot = context.bot_data.get('trading_bot')
        if not bot:
            await update.message.reply_text("ربات در حال حاضر در دسترس نیست. لطفاً بعداً تلاش کنید.")
            return
        
        # استخراج نماد از پیام
        if context.args:
            symbol = context.args[0].upper()
        else:
            keyboard = [
                [InlineKeyboardButton("BTC", callback_data="risk_BTC")],
                [InlineKeyboardButton("ETH", callback_data="risk_ETH")],
                [InlineKeyboardButton("BNB", callback_data="risk_BNB")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "لطفاً نماد ارز را برای تحلیل ریسک انتخاب کنید:",
                reply_markup=reply_markup
            )
            return
        
        # ارسال پیام در حال پردازش
        processing_message = await update.message.reply_text(f"⚠️ در حال تحلیل ریسک {symbol}...")
        
        # دریافت داده‌های تاریخی
        historical_data = bot.get_historical_data(symbol)
        market_data = await bot.get_market_data(symbol)
        
        # تحلیل ریسک
        risk_analysis = bot.analyze_risk_management(historical_data, market_data)
        
        # ایجاد پاسخ
        response = f"⚠️ *تحلیل ریسک {symbol}*\n\n"
        response += f"• *ATR*: ${risk_analysis.get('atr', 0):.2f}\n"
        response += f"• *نوسانات*: {risk_analysis.get('volatility', 0):.2f}%\n"
        response += f"• *حد ضرر*: ${risk_analysis.get('stop_loss', 0):.2f}\n"
        response += f"• *حد سود*: ${risk_analysis.get('take_profit', 0):.2f}\n"
        response += f"• *نسبت ریسک به پاداش*: {risk_analysis.get('risk_reward_ratio', 0):.2f}\n"
        response += f"• *حجم پیشنهادی*: {risk_analysis.get('position_size', 0):.2%}\n\n"
        
        if risk_analysis.get('risk_reward_ratio', 0) > 2:
            response += "🟢 *وضعیت*: ریسک به پاداش مناسب"
        elif risk_analysis.get('risk_reward_ratio', 0) > 1:
            response += "🟡 *وضعیت*: ریسک به پاداش متوسط"
        else:
            response += "🔴 *وضعیت*: ریسک به پاداش نامناسب"
        
        # ایجاد دکمه‌های اینلاین
        keyboard = [
            [InlineKeyboardButton("🔄 تحلیل مجدد", callback_data=f"risk_{symbol}")],
            [InlineKeyboardButton("📊 تحلیل کامل", callback_data=f"analyze_{symbol}")],
            [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ویرایش پیام با نتیجه
        await processing_message.edit_text(response, parse_mode='Markdown', reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error in risk_command: {e}")
        await update.message.reply_text("متأسفانه در تحلیل ریسک خطایی رخ داد. لطفاً دوباره تلاش کنید.")


async def watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /watchlist"""
    try:
        user_id = update.effective_user.id
        
        # ایجاد دکمه‌های اینلاین
        keyboard = [
            [InlineKeyboardButton("➕ افزودن ارز", callback_data="add_to_watchlist")],
            [InlineKeyboardButton("➖ حذف ارز", callback_data="remove_from_watchlist")],
            [InlineKeyboardButton("🔄 به‌روزرسانی واچ‌لیست", callback_data="refresh_watchlist")],
            [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ایجاد پاسخ
        response = f"📋 *واچ‌لیست شما*\n\n"
        response += "• BTC/USDT\n"
        response += "• ETH/USDT\n"
        response += "• BNB/USDT\n"
        response += "• SOL/USDT\n"
        response += "• XRP/USDT\n\n"
        response += "برای مدیریت واچ‌لیست، از دکمه‌های زیر استفاده کنید:"
        
        await update.message.reply_text(response, parse_mode='Markdown', reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error in watchlist_command: {e}")
        await update.message.reply_text("متأسفانه در مدیریت واچ‌لیست خطایی رخ داد.")


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /settings"""
    try:
        user_id = update.effective_user.id
        
        # ایجاد دکمه‌های اینلاین
        keyboard = [
            [InlineKeyboardButton("🌐 تغییر زبان", callback_data="change_language")],
            [InlineKeyboardButton("⚙️ تنظیمات هشدار", callback_data="alert_settings")],
            [InlineKeyboardButton("📊 تنظیمات تحلیل", callback_data="analysis_settings")],
            [InlineKeyboardButton("🔔 تنظیمات اعلان", callback_data="notification_settings")],
            [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ایجاد پاسخ
        response = "⚙️ *تنظیمات*\n\n"
        response += "• زبان: فارسی 🇮🇷\n"
        response += "• هشدارها: فعال ✅\n"
        response += "• تحلیل پیشرفته: فعال ✅\n"
        response += "• اعلان‌ها: فعال ✅\n\n"
        response += "برای تغییر تنظیمات، از دکمه‌های زیر استفاده کنید:"
        
        await update.message.reply_text(response, parse_mode='Markdown', reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error in settings_command: {e}")
        await update.message.reply_text("متأسفانه در تنظیمات خطایی رخ داد.")


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر پیام‌های متنی"""
    try:
        bot = context.bot_data.get('trading_bot')
        text = update.message.text
        
        # اگر متن یک نماد ارز است
        if text.isalpha() and len(text) <= 10:
            symbol = text.upper()
            
            # ارسال پیام در حال پردازش
            processing_message = await update.message.reply_text(f"🔄 در حال تحلیل {symbol}... لطفاً صبر کنید.")
            
            # انجام تحلیل
            analysis = await bot.perform_advanced_analysis(symbol)
            
            # ایجاد پاسخ
            response = format_analysis_response(analysis)
            
            # ایجاد دکمه‌های اینلاین
            keyboard = [
                [InlineKeyboardButton("🔄 تحلیل مجدد", callback_data=f"analyze_{symbol}")],
                [InlineKeyboardButton("🚀 تحلیل پیشرفته", callback_data=f"advanced_{symbol}")],
                [InlineKeyboardButton("📰 اخبار", callback_data=f"news_{symbol}")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # ویرایش پیام با نتیجه تحلیل
            await processing_message.edit_text(response, parse_mode='Markdown', reply_markup=reply_markup)
        else:
            keyboard = [
                [InlineKeyboardButton("📚 راهنما", callback_data="help_menu")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "لطفاً یک نماد ارز معتبر وارد کنید یا از دستور /help استفاده کنید.",
                reply_markup=reply_markup
            )
    except Exception as e:
        logger.error(f"Error in message_handler: {e}")
        await update.message.reply_text("متأسفانه در تحلیل خطایی رخ داد. لطفاً دوباره تلاش کنید.")


async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر callback query"""
    try:
        query = update.callback_query
        query.answer()
        
        data = query.data
        bot = context.bot_data.get('trading_bot')
        
        if data == "main_menu":
            keyboard = [
                [InlineKeyboardButton("📊 تحلیل ارز", callback_data="analyze_menu")],
                [InlineKeyboardButton("💰 قیمت لحظه‌ای", callback_data="price_menu")],
                [InlineKeyboardButton("📰 اخبار", callback_data="news_menu")],
                [InlineKeyboardButton("🚀 سیگنال‌ها", callback_data="signals_menu")],
                [InlineKeyboardButton("⚙️ تنظیمات", callback_data="settings_menu")],
                [InlineKeyboardButton("📚 راهنما", callback_data="help_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "🤖 **منوی اصلی ربات تحلیلگر حرفه‌ای ارزهای دیجیتال**\n\n"
                "لطفاً یکی از گزینه‌های زیر را انتخاب کنید:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        elif data == "analyze_menu":
            keyboard = [
                [InlineKeyboardButton("BTC", callback_data="analyze_BTC")],
                [InlineKeyboardButton("ETH", callback_data="analyze_ETH")],
                [InlineKeyboardButton("BNB", callback_data="analyze_BNB")],
                [InlineKeyboardButton("SOL", callback_data="analyze_SOL")],
                [InlineKeyboardButton("XRP", callback_data="analyze_XRP")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "📊 لطفاً نماد ارز را برای تحلیل انتخاب کنید:",
                reply_markup=reply_markup
            )
        
        elif data == "price_menu":
            keyboard = [
                [InlineKeyboardButton("BTC", callback_data="price_BTC")],
                [InlineKeyboardButton("ETH", callback_data="price_ETH")],
                [InlineKeyboardButton("BNB", callback_data="price_BNB")],
                [InlineKeyboardButton("SOL", callback_data="price_SOL")],
                [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="refresh_prices")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "💰 لطفاً نماد ارز را برای دریافت قیمت انتخاب کنید:",
                reply_markup=reply_markup
            )
        
        elif data == "news_menu":
            keyboard = [
                [InlineKeyboardButton("BTC", callback_data="news_BTC")],
                [InlineKeyboardButton("ETH", callback_data="news_ETH")],
                [InlineKeyboardButton("اخبار اقتصادی", callback_data="economic_news")],
                [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="refresh_news")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "📰 لطفاً نماد ارز را برای دریافت اخبار انتخاب کنید:",
                reply_markup=reply_markup
            )
        
        elif data == "signals_menu":
            keyboard = [
                [InlineKeyboardButton("تحلیل BTC", callback_data="analyze_BTC")],
                [InlineKeyboardButton("تحلیل ETH", callback_data="analyze_ETH")],
                [InlineKeyboardButton("تحلیل BNB", callback_data="analyze_BNB")],
                [InlineKeyboardButton("تحلیل SOL", callback_data="analyze_SOL")],
                [InlineKeyboardButton("تحلیل XRP", callback_data="analyze_XRP")],
                [InlineKeyboardButton("🔄 به‌روزرسانی سیگنال‌ها", callback_data="refresh_signals")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "📊 *سیگنال‌های معاملاتی امروز*\n\n"
                "• *BTC/USDT*: سیگنال خرید (اطمینان: 85%)\n"
                "• *ETH/USDT*: سیگنال نگه دار (اطمینان: 65%)\n"
                "• *BNB/USDT*: سیگنال فروش (اطمینان: 75%)\n"
                "• *SOL/USDT*: سیگنال خرید (اطمینان: 80%)\n"
                "• *XRP/USDT*: سیگنال نگه دار (اطمینان: 60%)\n\n"
                "برای تحلیل کامل، روی یکی از گزینه‌های زیر کلیک کنید:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        elif data == "settings_menu":
            keyboard = [
                [InlineKeyboardButton("🌐 تغییر زبان", callback_data="change_language")],
                [InlineKeyboardButton("⚙️ تنظیمات هشدار", callback_data="alert_settings")],
                [InlineKeyboardButton("📊 تنظیمات تحلیل", callback_data="analysis_settings")],
                [InlineKeyboardButton("🔔 تنظیمات اعلان", callback_data="notification_settings")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "⚙️ *تنظیمات*\n\n"
                "• زبان: فارسی 🇮🇷\n"
                "• هشدارها: فعال ✅\n"
                "• تحلیل پیشرفته: فعال ✅\n"
                "• اعلان‌ها: فعال ✅\n\n"
                "برای تغییر تنظیمات، از دکمه‌های زیر استفاده کنید:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        elif data == "help_menu":
            keyboard = [
                [InlineKeyboardButton("🔙 بازگشت به منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            help_text = """
            📚 **راهنمای استفاده از ربات**

            **دستورات اصلی:**
            `/start` - نمایش منوی اصلی
            `/analyze [symbol]` - تحلیل کامل ارز (مثال: `/analyze BTC`)
            `/price [symbol]` - دریافت قیمت لحظه‌ای (مثال: `/price ETH`)
            `/news [symbol]` - دریافت اخبار مرتبط (مثال: `/news BTC`)
            `/signals` - دریافت سیگنال‌های معاملاتی
            `/advanced [symbol]` - تحلیل‌های پیشرفته (مثال: `/advanced BTC`)

            **دستورات پیشرفته:**
            `/portfolio` - مدیریت پورتفولیو
            `/alert` - تنظیم هشدارها
            `/risk [symbol]` - تحلیل ریسک
            `/watchlist` - مدیریت واچ‌لیست
            `/settings` - تنظیمات کاربر

            **قابلیت‌های ربات:**
            • تحلیل تکنیکال پیشرفته (RSI, MACD, الگوهای شمعی)
            • تحلیل احساسات بازار با پردازش اخبار
            • تحلیل چند زمانی (Multi-timeframe)
            • تحلیل ساختار بازار (Order Block, Supply & Demand)
            • تحلیل امواج الیوت
            • مدیریت ریسک و سرمایه
            • سیگنال‌های معاملاتی هوشمند
            • پشتیبانی از چندین صرافی
            • حالت آفلاین برای تست

            همچنین می‌توانید مستقیماً نام ارز را ارسال کنید تا تحلیل کامل آن را دریافت کنید.
            """
            
            await query.edit_message_text(help_text, reply_markup=reply_markup, parse_mode='Markdown')
        
        elif data.startswith("analyze_"):
            symbol = data.split("_")[1]
            
            # ویرایش پیام با پیام در حال پردازش
            await query.edit_message_text(f"🔄 در حال تحلیل {symbol}... لطفاً صبر کنید.")
            
            # انجام تحلیل
            analysis = await bot.perform_advanced_analysis(symbol)
            
            # ایجاد پاسخ
            response = format_analysis_response(analysis)
            
            # ایجاد دکمه‌های اینلاین
            keyboard = [
                [InlineKeyboardButton("🔄 تحلیل مجدد", callback_data=f"analyze_{symbol}")],
                [InlineKeyboardButton("🚀 تحلیل پیشرفته", callback_data=f"advanced_{symbol}")],
                [InlineKeyboardButton("📰 اخبار", callback_data=f"news_{symbol}")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # ویرایش پیام با نتیجه تحلیل
            await query.edit_message_text(response, parse_mode='Markdown', reply_markup=reply_markup)
        
        elif data.startswith("advanced_"):
            symbol = data.split("_")[1]
            
            # ویرایش پیام با پیام در حال پردازش
            await query.edit_message_text(f"🚀 در حال تحلیل پیشرفته {symbol}... لطفاً صبر کنید.")
            
            # انجام تحلیل
            analysis = await bot.perform_advanced_analysis(symbol)
            
            # ایجاد پاسخ برای تحلیل پیشرفته
            response = format_advanced_analysis_response(analysis)
            
            # ایجاد دکمه‌های اینلاین
            keyboard = [
                [InlineKeyboardButton("🔄 تحلیل مجدد", callback_data=f"advanced_{symbol}")],
                [InlineKeyboardButton("📊 تحلیل ساده", callback_data=f"analyze_{symbol}")],
                [InlineKeyboardButton("📰 اخبار", callback_data=f"news_{symbol}")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # ویرایش پیام با نتیجه تحلیل
            await query.edit_message_text(response, parse_mode='Markdown', reply_markup=reply_markup)
        
        elif data.startswith("price_"):
            symbol = data.split("_")[1]
            
            # ویرایش پیام با پیام در حال پردازش
            await query.edit_message_text(f"💰 در حال دریافت قیمت {symbol}...")
            
            # دریافت داده‌های بازار
            market_data = await bot.get_market_data(symbol)
            
            # ایجاد پاسخ
            response = f"💰 *{symbol} Price Information*\n\n"
            response += f"• قیمت: ${market_data.get('price', 0):,.2f}\n"
            response += f"• تغییر 24h: {market_data.get('price_change_24h', 0):+.2f}%\n"
            response += f"• حجم 24h: ${market_data.get('volume_24h', 0):,.0f}\n"
            response += f"• ارزش بازار: ${market_data.get('market_cap', 0):,.0f}\n"
            response += f"• منابع: {', '.join(market_data.get('sources', []))}"
            
            # ایجاد دکمه‌های اینلاین
            keyboard = [
                [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data=f"price_{symbol}")],
                [InlineKeyboardButton("📊 تحلیل", callback_data=f"analyze_{symbol}")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # ویرایش پیام با نتیجه
            await query.edit_message_text(response, parse_mode='Markdown', reply_markup=reply_markup)
        
        elif data.startswith("news_"):
            symbol = data.split("_")[1]
            
            # ویرایش پیام با پیام در حال پردازش
            await query.edit_message_text(f"📰 در حال دریافت اخبار {symbol}...")
            
            # دریافت اخبار
            news = await bot.fetch_news_from_multiple_sources(symbol)
            
            if not news:
                await query.edit_message_text(f"هیچ خبری برای {symbol} یافت نشد.")
                return
            
            # ایجاد پاسخ
            response = f"📰 *اخبار {symbol}*\n\n"
            
            # نمایش حداکثر 5 خبر
            for i, item in enumerate(news[:5]):
                response += f"{i+1}. *{item['title']}*\n"
                response += f"   منبع: {item['source']}\n"
                response += f"   [لینک خبر]({item['url']})\n\n"
            
            # ایجاد دکمه‌های اینلاین
            keyboard = [
                [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data=f"news_{symbol}")],
                [InlineKeyboardButton("📊 تحلیل", callback_data=f"analyze_{symbol}")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # ویرایش پیام با نتیجه
            await query.edit_message_text(response, parse_mode='Markdown', disable_web_page_preview=True, reply_markup=reply_markup)
        
        elif data.startswith("risk_"):
            symbol = data.split("_")[1]
            
            # ویرایش پیام با پیام در حال پردازش
            await query.edit_message_text(f"⚠️ در حال تحلیل ریسک {symbol}...")
            
            # دریافت داده‌های تاریخی
            historical_data = bot.get_historical_data(symbol)
            market_data = await bot.get_market_data(symbol)
            
            # تحلیل ریسک
            risk_analysis = bot.analyze_risk_management(historical_data, market_data)
            
            # ایجاد پاسخ
            response = f"⚠️ *تحلیل ریسک {symbol}*\n\n"
            response += f"• *ATR*: ${risk_analysis.get('atr', 0):.2f}\n"
            response += f"• *نوسانات*: {risk_analysis.get('volatility', 0):.2f}%\n"
            response += f"• *حد ضرر*: ${risk_analysis.get('stop_loss', 0):.2f}\n"
            response += f"• *حد سود*: ${risk_analysis.get('take_profit', 0):.2f}\n"
            response += f"• *نسبت ریسک به پاداش*: {risk_analysis.get('risk_reward_ratio', 0):.2f}\n"
            response += f"• *حجم پیشنهادی*: {risk_analysis.get('position_size', 0):.2%}\n\n"
            
            if risk_analysis.get('risk_reward_ratio', 0) > 2:
                response += "🟢 *وضعیت*: ریسک به پاداش مناسب"
            elif risk_analysis.get('risk_reward_ratio', 0) > 1:
                response += "🟡 *وضعیت*: ریسک به پاداش متوسط"
            else:
                response += "🔴 *وضعیت*: ریسک به پاداش نامناسب"
            
            # ایجاد دکمه‌های اینلاین
            keyboard = [
                [InlineKeyboardButton("🔄 تحلیل مجدد", callback_data=f"risk_{symbol}")],
                [InlineKeyboardButton("📊 تحلیل کامل", callback_data=f"analyze_{symbol}")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # ویرایش پیام با نتیجه
            await query.edit_message_text(response, parse_mode='Markdown', reply_markup=reply_markup)
        
        elif data == "economic_news":
            # ویرایش پیام با پیام در حال پردازش
            await query.edit_message_text("📰 در حال دریافت اخبار اقتصادی...")
            
            # دریافت اخبار اقتصادی
            economic_news = await bot.fetch_economic_news()
            
            if not economic_news:
                await query.edit_message_text("هیچ خبر اقتصادی یافت نشد.")
                return
            
            # ایجاد پاسخ
            response = "📰 *اخبار اقتصادی*\n\n"
            
            # نمایش حداکثر 5 خبر
            for i, item in enumerate(economic_news[:5]):
                response += f"{i+1}. *{item['title']}*\n"
                response += f"   منبع: {item['source']}\n"
                response += f"   [لینک خبر]({item['url']})\n\n"
            
            # ایجاد دکمه‌های اینلاین
            keyboard = [
                [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="economic_news")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # ویرایش پیام با نتیجه
            await query.edit_message_text(response, parse_mode='Markdown', disable_web_page_preview=True, reply_markup=reply_markup)
        
        elif data == "refresh_prices":
            keyboard = [
                [InlineKeyboardButton("BTC", callback_data="price_BTC")],
                [InlineKeyboardButton("ETH", callback_data="price_ETH")],
                [InlineKeyboardButton("BNB", callback_data="price_BNB")],
                [InlineKeyboardButton("SOL", callback_data="price_SOL")],
                [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="refresh_prices")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "💰 *قیمت‌های به‌روز شده*\n\n"
                "• *BTC/USDT*: $43,250.00 (+2.3%)\n"
                "• *ETH/USDT*: $2,180.00 (+1.8%)\n"
                "• *BNB/USDT*: $310.00 (+0.9%)\n"
                "• *SOL/USDT*: $98.50 (+3.2%)\n\n"
                "برای دریافت قیمت دقیق، روی یکی از گزینه‌های زیر کلیک کنید:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        elif data == "refresh_news":
            keyboard = [
                [InlineKeyboardButton("BTC", callback_data="news_BTC")],
                [InlineKeyboardButton("ETH", callback_data="news_ETH")],
                [InlineKeyboardButton("اخبار اقتصادی", callback_data="economic_news")],
                [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="refresh_news")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "📰 *اخبار به‌روز شده*\n\n"
                "• *BTC*: بیت‌کوین به سطح مقاومت مهم رسید\n"
                "• *ETH*: اتریوم آپدیت شبکه با موفقیت انجام شد\n"
                "• *BNB*: بایننس کوین جدیدترین پروژه‌ها را معرفی کرد\n"
                "• *اقتصادی*: فدرال رزرو نرخ بهره را ثابت نگه داشت\n\n"
                "برای دریافت اخبار کامل، روی یکی از گزینه‌های زیر کلیک کنید:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        elif data == "refresh_signals":
            keyboard = [
                [InlineKeyboardButton("تحلیل BTC", callback_data="analyze_BTC")],
                [InlineKeyboardButton("تحلیل ETH", callback_data="analyze_ETH")],
                [InlineKeyboardButton("تحلیل BNB", callback_data="analyze_BNB")],
                [InlineKeyboardButton("تحلیل SOL", callback_data="analyze_SOL")],
                [InlineKeyboardButton("تحلیل XRP", callback_data="analyze_XRP")],
                [InlineKeyboardButton("🔄 به‌روزرسانی سیگنال‌ها", callback_data="refresh_signals")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "📊 *سیگنال‌های معاملاتی به‌روز شده*\n\n"
                "• *BTC/USDT*: سیگنال خرید (اطمینان: 87%)\n"
                "• *ETH/USDT*: سیگنال نگه دار (اطمینان: 68%)\n"
                "• *BNB/USDT*: سیگنال فروش (اطمینان: 72%)\n"
                "• *SOL/USDT*: سیگنال خرید (اطمینان: 82%)\n"
                "• *XRP/USDT*: سیگنال نگه دار (اطمینان: 63%)\n\n"
                "برای تحلیل کامل، روی یکی از گزینه‌های زیر کلیک کنید:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        elif data == "add_to_watchlist":
            await query.edit_message_text(
                "لطفاً نماد ارزی را که می‌خواهید به واچ‌لیست اضافه کنید، ارسال کنید:\n\n"
                "مثال: BTC",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 انصراف", callback_data="main_menu")]])
            )
            # در یک پیاده‌سازی واقعی، باید حالت را برای کاربر ذخیره کرد
        
        elif data == "remove_from_watchlist":
            await query.edit_message_text(
                "لطفاً نماد ارزی را که می‌خواهید از واچ‌لیست حذف کنید، ارسال کنید:\n\n"
                "مثال: BTC",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 انصراف", callback_data="main_menu")]])
            )
            # در یک پیاده‌سازی واقعی، باید حالت را برای کاربر ذخیره کرد
        
        elif data == "refresh_watchlist":
            keyboard = [
                [InlineKeyboardButton("➕ افزودن ارز", callback_data="add_to_watchlist")],
                [InlineKeyboardButton("➖ حذف ارز", callback_data="remove_from_watchlist")],
                [InlineKeyboardButton("🔄 به‌روزرسانی واچ‌لیست", callback_data="refresh_watchlist")],
                [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "📋 *واچ‌لیست به‌روز شده*\n\n"
                "• BTC/USDT\n"
                "• ETH/USDT\n"
                "• BNB/USDT\n"
                "• SOL/USDT\n"
                "• XRP/USDT\n\n"
                "برای مدیریت واچ‌لیست، از دکمه‌های زیر استفاده کنید:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        elif data in ["change_language", "alert_settings", "analysis_settings", "notification_settings"]:
            if data == "change_language":
                await query.edit_message_text(
                    "🌐 تغییر زبان در حال حاضر در دسترس نیست. زبان فعلی: فارسی 🇮🇷",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 بازگشت", callback_data="settings_menu")]])
                )
            elif data == "alert_settings":
                await query.edit_message_text(
                    "⚙️ تنظیمات هشدار در حال حاضر در دسترس نیست",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 بازگشت", callback_data="settings_menu")]])
                )
            elif data == "analysis_settings":
                await query.edit_message_text(
                    "📊 تنظیمات تحلیل در حال حاضر در دسترس نیست",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 بازگشت", callback_data="settings_menu")]])
                )
            elif data == "notification_settings":
                await query.edit_message_text(
                    "🔔 تنظیمات اعلان در حال حاضر در دسترس نیست",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 بازگشت", callback_data="settings_menu")]])
                )
        
        elif data in ["add_portfolio", "remove_portfolio", "view_portfolio", "refresh_portfolio"]:
            if data == "add_portfolio":
                await query.edit_message_text(
                    "➕ لطفاً نماد ارز و مقدار را برای افزودن به پورتفولیو ارسال کنید:\n\n"
                    "مثال: BTC 0.1",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 انصراف", callback_data="main_menu")]])
                )
            elif data == "remove_portfolio":
                await query.edit_message_text(
                    "➖ لطفاً نماد ارزی را که می‌خواهید از پورتفولیو حذف کنید، ارسال کنید:\n\n"
                    "مثال: BTC",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 انصراف", callback_data="main_menu")]])
                )
            elif data == "view_portfolio":
                keyboard = [
                    [InlineKeyboardButton("➕ افزودن ارز", callback_data="add_portfolio")],
                    [InlineKeyboardButton("➖ حذف ارز", callback_data="remove_portfolio")],
                    [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="refresh_portfolio")],
                    [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(
                    "💼 *پورتفولیو شما*\n\n"
                    "• *ارزش کل*: $12,450.00\n"
                    "• *سود/زیان*: +$1,250.00 (+11.2%)\n"
                    "• *تغییر 24h*: +$320.00 (+2.6%)\n\n"
                    "• BTC: 0.25 ($10,750.00)\n"
                    "• ETH: 2.5 ($5,500.00)\n"
                    "• BNB: 5.0 ($1,500.00)\n\n"
                    "برای مدیریت پورتفولیو، از دکمه‌های زیر استفاده کنید:",
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
            elif data == "refresh_portfolio":
                keyboard = [
                    [InlineKeyboardButton("➕ افزودن ارز", callback_data="add_portfolio")],
                    [InlineKeyboardButton("➖ حذف ارز", callback_data="remove_portfolio")],
                    [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="refresh_portfolio")],
                    [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(
                    "💼 *پورتفولیو به‌روز شده*\n\n"
                    "• *ارزش کل*: $12,770.00 (+12.8%)\n"
                    "• *سود/زیان*: +$1,570.00 (+14.0%)\n"
                    "• *تغییر 24h*: +$320.00 (+2.6%)\n\n"
                    "• BTC: 0.25 ($10,750.00)\n"
                    "• ETH: 2.5 ($5,500.00)\n"
                    "• BNB: 5.0 ($1,500.00)\n\n"
                    "برای مدیریت پورتفولیو، از دکمه‌های زیر استفاده کنید:",
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
        
        elif data in ["add_alert", "remove_alert", "view_alerts"]:
            if data == "add_alert":
                await query.edit_message_text(
                    "➕ لطفاً نماد ارز و شرایط هشدار را ارسال کنید:\n\n"
                    "مثال: BTC > 45000",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 انصراف", callback_data="main_menu")]])
                )
            elif data == "remove_alert":
                await query.edit_message_text(
                    "➖ لطفاً نماد ارزی را که می‌خواهید هشدار آن را حذف کنید، ارسال کنید:\n\n"
                    "مثال: BTC",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 انصراف", callback_data="main_menu")]])
                )
            elif data == "view_alerts":
                keyboard = [
                    [InlineKeyboardButton("➕ افزودن هشدار", callback_data="add_alert")],
                    [InlineKeyboardButton("➖ حذف هشدار", callback_data="remove_alert")],
                    [InlineKeyboardButton("🔙 منوی اصلی", callback_data="main_menu")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(
                    "⚠️ *هشدارهای فعال*\n\n"
                    "• *تعداد هشدارها*: 3\n"
                    "• BTC > $45,000\n"
                    "• ETH < $2,000\n"
                    "• BNP > $350\n\n"
                    "برای مدیریت هشدارها، از دکمه‌های زیر استفاده کنید:",
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
        
    except Exception as e:
        logger.error(f"Error in callback_query_handler: {e}")
        await query.edit_message_text("متأسفانه در پردازش درخواست شما خطایی رخ داد.")


def format_analysis_response(analysis):
    """فرمت‌بندی پاسخ تحلیل"""
    try:
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
        technical = analysis.get('technical', {}).get('classical', {})
        response += f"📈 *تحلیل تکنیکال*\n"
        
        if 'rsi' in technical:
            rsi = technical['rsi'].get('14', 50)
            rsi_signal = "اشباع خرید" if rsi > 70 else "اشباع فروش" if rsi < 30 else "خنثی"
            response += f"• RSI(14): {rsi:.1f} ({rsi_signal})\n"
        
        if 'macd' in technical:
            macd = technical['macd']
            macd_signal = "صعودی" if macd.get('macd', 0) > macd.get('signal', 0) else "نزولی"
            response += f"• MACD: {macd_signal}\n"
        
        if 'trend' in technical:
            trend = technical['trend']
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
    except Exception as e:
        logger.error(f"Error formatting analysis response: {e}")
        return "متأسفانه در فرمت‌بندی پاسخ تحلیل خطایی رخ داد."


def format_advanced_analysis_response(analysis):
    """فرمت‌بندی پاسخ تحلیل پیشرفته"""
    try:
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
    except Exception as e:
        logger.error(f"Error formatting advanced analysis response: {e}")
        return "متأسفانه در فرمت‌بندی پاسخ تحلیل پیشرفته خطایی رخ داد."