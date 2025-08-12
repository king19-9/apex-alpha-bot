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
            "🤖 **به ربات تحلیلگر حرفه‌ای ارزهای دیجیتال خوش آمدید!**\\n\\n"
            "این ربات با استفاده از هوش مصنوعی و تحلیل‌های پیشرفته، بهترین سیگنال‌های معاملاتی را به شما ارائه می‌دهد.\\n\\n"
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
        response = f"💰 *{symbol} Price Information*\\n\\n"
        response += f"• قیمت: ${market_data.get('price', 0):,.2f}\\n"
        response += f"• تغییر 24h: {market_data.get('price_change_24h', 0):+.2f}%\\n"
        response += f"• حجم 24h: ${market_data.get('volume_24h', 0):,.0f}\\n"
        response += f"• ارزش بازار: ${market_data.get('market_cap', 0):,.0f}\\n"
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
        response = f"📰 *اخبار {symbol}*\\n\\n"
        
        # نمایش حداکثر 5 خبر
        for i, item in enumerate(news[:5]):
            response += f"{i+1}. *{item['title']}*\\n"
            response += f"   منبع: {item['source']}\\n"
            response += f"   [لینک خبر]({item['url']})\\n\\n"
        
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
        
        # دریافت سیگنال‌ها
        signals = await bot.get_trading_signals()
        
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
        response = "📊 *سیگنال‌های معاملاتی امروز*\\n\\n"
        
        # نمایش سیگنال‌ها
        for signal in signals:
            symbol = signal.get('symbol', 'UNKNOWN')
            signal_type = signal.get('signal', 'HOLD')
            confidence = signal.get('confidence', 0.5)
            
            signal_emoji = "🟢" if signal_type == "BUY" else "🔴" if signal_type == "SELL" else "🟡"
            response += f"• *{symbol}/USDT*: سیگنال {signal_type} (اطمینان: {confidence:.1%})\\n"
        
        response += "\\nبرای تحلیل کامل، روی یکی از گزینه‌های زیر کلیک کنید:"
        
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
        
        response = "💼 *مدیریت پورتفولیو*\\n\\n"
        response += "• *ارزش کل*: $12,450.00\\n"
        response += "• *سود/زیان*: +$1,250.00 (+11.2%)\\n"
        response += "• *تغییر 24h*: +$320.00 (+2.6%)\\n\\n"
        response += "• BTC: 0.25 ($10,750.00)\\n"
        response += "• ETH: 2.5 ($5,500.00)\\n"
        response += "• BNB: 5.0 ($1,500.00)\\n\\n"
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
        
        response = "⚠️ *مدیریت هشدارها*\\n\\n"
        response += "• *هشدارهای فعال*: 3\\n"
        response += "• BTC > $45,000\\n"
        response += "• ETH < $2,000\\n"
        response += "• BNP > $350\\n\\n"
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
        response = f"⚠️ *تحلیل ریسک {symbol}*\\n\\n"
        response += f"• *ATR*: ${risk_analysis.get('atr', 0):.2f}\\n"
        response += f"• *نوسانات*: {risk_analysis.get('volatility', 0):.2f}%\\n"
        response += f"• *حد ضرر*: ${risk_analysis.get('stop_loss', 0):.2f}\\n"
        response += f"• *حد سود*: ${risk_analysis.get('take_profit', 0):.2f}\\n"
        response += f"• *نسبت ریسک به پاداش*: {risk_analysis.get('risk_reward_ratio', 0):.2f}\\n"
        response += f"• *حجم پیشنهادی*: {risk_analysis.get('position_size', 0):.2%}\\n\\n"
        
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
        response = f"📋 *واچ‌لیست شما*\\n\\n"
        response += "• BTC/USDT\\n"
        response += "• ETH/USDT\\n"
        response += "• BNB/USDT\\n"
        response += "• SOL/USDT\\n"
        response += "• XRP/USDT\\n\\n"
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
        response = "⚙️ *تنظیمات*\\n\\n"
        response += "• زبان: فارسی 🇮🇷\\n"
        response += "• هشدارها: فعال ✅\\n"
        response += "• تحلیل پیشرفته: فعال ✅\\n"
        response += "• اعلان‌ها: فعال ✅\\n\\n"
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
                "🤖 **منوی اصلی ربات تحلیلگر حرفه‌ای ارزهای دیجیتال**\\n\\n"
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
                "📊 *سیگنال‌های معاملاتی امروز*\\n\\n"
                "• *BTC/USDT*: سیگنال خرید (اطمینان: 85%)\\n"
                "• *ETH/USDT*: سیگنال نگه دار (اطمینان: 65%)\\n"
                "• *BNB/USDT*: سیگنال فروش (اطمینان: 75%)\\n"
                "• *SOL/USDT*: سیگنال خرید (اطمینان: 80%)\\n"
                "• *XRP/USDT*: سیگنال نگه دار (اطمینان: 60%)\\n\\n"
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
                "⚙️ *تنظیمات*\\n\\n"
                "• زبان: فارسی 🇮🇷\\n"
                "• هشدارها: فعال ✅\\n"
                "• تحلیل پیشرفته: فعال ✅\\n"
                "• اعلان‌ها: فعال ✅\\n\\n"
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
            response = f"💰 *{symbol} Price Information*\\n\\n"
            response += f"• قیمت: ${market_data.get('price', 0):,.2f}\\n"
            response += f"• تغییر 24h: {market_data.get('price_change_24h', 0):+.2f}%\\n"
            response += f"• حجم 24h: ${market_data.get('volume_24h', 0):,.0f}\\n"
            response += f"• ارزش بازار: ${market_data.get('market_cap', 0):,.0f}\\n"
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
            response = f"📰 *اخبار {symbol}*\\n\\n"
            
            # نمایش حداکثر 5 خبر
            for i, item in enumerate(news[:5]):
                response += f"{i+1}. *{item['title']}*\\n"
                response += f"   منبع: {item['source']}\\n"
                response += f"   [لینک خبر]({item['url']})\\n\\n"
            
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
            response = f"⚠️ *تحلیل ریسک {symbol}*\\n\\n"
            response += f"• *ATR*: ${risk_analysis.get('atr', 0):.2f}\\n"
            response += f"• *نوسانات*: {risk_analysis.get('volatility', 0):.2f}%\\n"
            response += f"• *حد ضرر*: ${risk_analysis.get('stop_loss', 0):.2f}\\n"
            response += f"• *حد سود*: ${risk_analysis.get('take_profit', 0):.2f}\\n"
            response += f"• *نسبت ریسک به پاداش*: {risk_analysis.get('risk_reward_ratio', 0):.2f}\\n"
            response += f"• *حجم پیشنهادی*: {risk_analysis.get('position_size', 0):.2%}\\n\\n"
            
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
            response = "📰 *اخبار اقتصادی*\\n\\n"
            
            # نمایش حداکثر 5 خبر
            for i, item in enumerate(economic_news[:5]):
                response += f"{i+1}. *{item['title']}*\\n"
                response += f"   منبع: {item['source']}\\n"
                response += f"   [لینک خبر]({item['url']})\\n\\n"
            
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
                "💰 *قیمت‌های به‌روز شده*\\n\\n"
                "• *BTC/USDT*: $43,250.00 (+2.3%)\\n"
                "• *ETH/USDT*: $2,180.00 (+1.8%)\\n"
                "• *BNB/USDT*: $310.00 (+0.9%)\\n"
                "• *SOL/USDT*: $98.50 (+3.2%)\\n\\n"
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
                "📰 *اخبار به‌روز شده*\\n\\n"
                "• *BTC*: بیت‌کوین به سطح مقاومت مهم رسید\\n"
                "• *ETH*: اتریوم آپدیت شبکه با موفقیت انجام شد\\n"
                "• *BNB*: بایننس کوین جدیدترین پروژه‌ها را معرفی کرد\\n"
                "• *اقتصادی*: فدرال رزرو نرخ بهره را ثابت نگه داشت\\n\\n"
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
                "📊 *سیگنال‌های معاملاتی به‌روز شده*\\n\\n"
                "• *BTC/USDT*: سیگنال خرید (اطمینان: 87%)\\n"
                "• *ETH/USDT*: سیگنال نگه دار (اطمینان: 68%)\\n"
                "• *BNB/USDT*: سیگنال فروش (اطمینان: 72%)\\n"
                "• *SOL/USDT*: سیگنال خرید (اطمینان: 82%)\\n"
                "• *XRP/USDT*: سیگنال نگه دار (اطمینان: 63%)\\n\\n"
                "برای تحلیل کامل، روی یکی از گزینه‌های زیر کلیک کنید:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        elif data == "add_to_watchlist":
            await query.edit_message_text(
                "لطفاً نماد ارزی را که می‌خواهید به واچ‌لیست اضافه کنید، ارسال کنید:\\n\\n"
                "مثال: BTC",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 انصراف", callback_data="main_menu")]])
            )
            # در یک پیاده‌سازی واقعی، باید حالت را برای کاربر ذخیره کرد
        
        elif data == "remove_from_watchlist":
            await query.edit_message_text(
                "لطفاً نماد ارزی را که می‌خواهید از واچ‌لیست حذف کنید، ارسال کنید:\\n\\n"
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
                "📋 *واچ‌لیست به‌روز شده*\\n\\n"
                "• BTC/USDT\\n"
                "• ETH/USDT\\n"
                "• BNB/USDT\\n"
                "• SOL/USDT\\n"
                "• XRP/USDT\\n\\n"
                "برای مدیریت واچ‌لیست، از دکمه‌های زیر استفاده کنید:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        # سایر callbackها را به صورت مشابه مدیریت کنید
        # ...
        
    except Exception as e:
        logger.error(f"Error in callback_query_handler: {e}")
        await query.edit_message_text("متأسفانه در پردازش درخواست شما خطایی رخ داد.")


def format_analysis_response(analysis):
    """قالب‌بندی پاسخ تحلیل ساده"""
    try:
        symbol = analysis.get('symbol', 'UNKNOWN')
        market_data = analysis.get('market_data', {})
        signal = analysis.get('signal', 'HOLD')
        confidence = analysis.get('confidence', 0.5)
        sentiment = analysis.get('sentiment', {})
        technical = analysis.get('technical', {})
        
        response = f"📊 *تحلیل {symbol}*\\n\\n"
        
        # اطلاعات بازار
        if market_data:
            response += f"💰 *قیمت*: ${market_data.get('price', 0):,.2f}\\n"
            response += f"📈 *تغییر 24h*: {market_data.get('price_change_24h', 0):+.2f}%\\n"
            response += f"🔄 *حجم 24h*: ${market_data.get('volume_24h', 0):,.0f}\\n"
            response += f"💎 *ارزش بازار*: ${market_data.get('market_cap', 0):,.0f}\\n\\n"
        
        # سیگنال
        signal_emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
        response += f"{signal_emoji} *سیگنال*: {signal}\\n"
        response += f"🎯 *اطمینان*: {confidence:.1%}\\n\\n"
        
        # تحلیل احساسات
        if sentiment:
            avg_sentiment = sentiment.get('average_sentiment', 0)
            sentiment_emoji = "😊" if avg_sentiment > 0.2 else "😔" if avg_sentiment < -0.2 else "😐"
            response += f"{sentiment_emoji} *احساسات بازار*: {avg_sentiment:.2f}\\n"
            
            topics = sentiment.get('topics', [])
            if topics:
                response += f"🏷️ *موضوعات*: {', '.join(topics)}\\n\\n"
        
        # تحلیل تکنیکال
        if technical:
            rsi = technical.get('rsi', 50)
            rsi_signal = "اشباع خرید" if rsi > 70 else "اشباع فروش" if rsi < 30 else "خنثی"
            response += f"📉 *RSI*: {rsi:.2f} ({rsi_signal})\\n"
            
            macd = technical.get('macd', {})
            if macd:
                response += f"📊 *MACD*: {macd.get('macd', 0):.4f}\\n"
                response += f"📈 *سیگنال MACD*: {macd.get('signal', 0):.4f}\\n"
            
            sma = technical.get('sma', {})
            if sma:
                response += f"📉 *SMA 20*: {sma.get('sma20', 0):.2f}\\n"
                response += f"📈 *SMA 50*: {sma.get('sma50', 0):.2f}\\n\\n"
        
        # منابع داده
        sources = market_data.get('sources', [])
        if sources:
            response += f"🔗 *منابع*: {', '.join(sources)}\\n"
        
        return response
    except Exception as e:
        logger.error(f"Error formatting analysis response: {e}")
        return f"📊 تحلیل {symbol}\\n\\nخطا در قالب‌بندی تحلیل: {str(e)}"


def format_advanced_analysis_response(analysis):
    """قالب‌بندی پاسخ تحلیل پیشرفته"""
    try:
        symbol = analysis.get('symbol', 'UNKNOWN')
        market_data = analysis.get('market_data', {})
        signal = analysis.get('signal', 'HOLD')
        confidence = analysis.get('confidence', 0.5)
        sentiment = analysis.get('sentiment', {})
        economic_sentiment = analysis.get('economic_sentiment', {})
        technical = analysis.get('technical', {})
        elliott = analysis.get('elliott', {})
        supply_demand = analysis.get('supply_demand', {})
        market_structure = analysis.get('market_structure', {})
        risk_management = analysis.get('risk_management', {})
        ai_analysis = analysis.get('ai_analysis', {})
        advanced_analysis = analysis.get('advanced_analysis', {})
        
        response = f"🚀 *تحلیل پیشرفته {symbol}*\\n\\n"
        
        # اطلاعات بازار
        if market_data:
            response += f"💰 *قیمت*: ${market_data.get('price', 0):,.2f}\\n"
            response += f"📈 *تغییر 24h*: {market_data.get('price_change_24h', 0):+.2f}%\\n"
            response += f"🔄 *حجم 24h*: ${market_data.get('volume_24h', 0):,.0f}\\n"
            response += f"💎 *ارزش بازار*: ${market_data.get('market_cap', 0):,.0f}\\n\\n"
        
        # سیگنال
        signal_emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
        response += f"{signal_emoji} *سیگنال*: {signal}\\n"
        response += f"🎯 *اطمینان*: {confidence:.1%}\\n\\n"
        
        # تحلیل احساسات
        if sentiment:
            avg_sentiment = sentiment.get('average_sentiment', 0)
            sentiment_emoji = "😊" if avg_sentiment > 0.2 else "😔" if avg_sentiment < -0.2 else "😐"
            response += f"{sentiment_emoji} *احساسات بازار*: {avg_sentiment:.2f}\\n"
            
            topics = sentiment.get('topics', [])
            if topics:
                response += f"🏷️ *موضوعات*: {', '.join(topics)}\\n"
        
        # تحلیل احساسات اقتصادی
        if economic_sentiment:
            avg_economic_sentiment = economic_sentiment.get('average_sentiment', 0)
            economic_sentiment_emoji = "😊" if avg_economic_sentiment > 0.2 else "😔" if avg_economic_sentiment < -0.2 else "😐"
            response += f"📰 *احساسات اقتصادی*: {avg_economic_sentiment:.2f}\\n"
            
            economic_topics = economic_sentiment.get('topics', [])
            if economic_topics:
                response += f"🏷️ *موضوعات اقتصادی*: {', '.join(economic_topics)}\\n\\n"
        
        # تحلیل تکنیکال
        if technical:
            response += "📊 *تحلیل تکنیکال*:\\n"
            
            rsi = technical.get('rsi', 50)
            rsi_signal = "اشباع خرید" if rsi > 70 else "اشباع فروش" if rsi < 30 else "خنثی"
            response += f"  📉 *RSI*: {rsi:.2f} ({rsi_signal})\\n"
            
            macd = technical.get('macd', {})
            if macd:
                response += f"  📊 *MACD*: {macd.get('macd', 0):.4f}\\n"
                response += f"  📈 *سیگنال MACD*: {macd.get('signal', 0):.4f}\\n"
            
            sma = technical.get('sma', {})
            if sma:
                response += f"  📉 *SMA 20*: {sma.get('sma20', 0):.2f}\\n"
                response += f"  📈 *SMA 50*: {sma.get('sma50', 0):.2f}\\n"
            
            bollinger = technical.get('bollinger', {})
            if bollinger:
                response += f"  📊 *بولینگر بالایی*: {bollinger.get('upper', 0):.2f}\\n"
                response += f"  📊 *بولینگر میانی*: {bollinger.get('middle', 0):.2f}\\n"
                response += f"  📊 *بولینگر پایینی*: {bollinger.get('lower', 0):.2f}\\n"
            
            response += "\\n"
        
        # تحلیل امواج الیوت
        if elliott:
            response += "🌊 *تحلیل امواج الیوت*:\\n"
            response += f"  🔄 *الگوی فعلی*: {elliott.get('current_pattern', 'ناشناخته')}\\n"
            response += f"  📈 *موج فعلی*: {elliott.get('current_wave', 'ناشناخته')}\\n"
            response += f"  🎯 *هدف بعدی*: {elliott.get('next_target', 'ناشناخته')}\\n\\n"
        
        # تحلیل عرضه و تقاضا
        if supply_demand:
            response += "⚖️ *تحلیل عرضه و تقاضا*:\\n"
            response += f"  📊 *عدم تعادل*: {supply_demand.get('imbalance', 0):.2f}\\n"
            response += f"  📈 *مناطق تقاضا*: {supply_demand.get('demand_zones', 'ناشناخته')}\\n"
            response += f"  📉 *مناطق عرضه*: {supply_demand.get('supply_zones', 'ناشناخته')}\\n\\n"
        
        # تحلیل ساختار بازار
        if market_structure:
            response += "🏗️ *ساختار بازار*:\\n"
            response += f"  📊 *روند*: {market_structure.get('trend', 'ناشناخته')}\\n"
            response += f"  🔄 *فاز بازار*: {market_structure.get('phase', 'ناشناخته')}\\n"
            response += f"  🎯 *سطح حمایتی*: {market_structure.get('support_level', 0):.2f}\\n"
            response += f"  🎯 *سطح مقاومتی*: {market_structure.get('resistance_level', 0):.2f}\\n\\n"
        
        # تحلیل مدیریت ریسک
        if risk_management:
            response += "⚠️ *مدیریت ریسک*:\\n"
            response += f"  📊 *ATR*: {risk_management.get('atr', 0):.2f}\\n"
            response += f"  📈 *نوسانات*: {risk_management.get('volatility', 0):.2f}%\\n"
            response += f"  🛑 *حد ضرر*: {risk_management.get('stop_loss', 0):.2f}\\n"
            response += f"  🎯 *حد سود*: {risk_management.get('take_profit', 0):.2f}\\n"
            response += f"  ⚖️ *نسبت ریسک به پاداش*: {risk_management.get('risk_reward_ratio', 0):.2f}\\n"
            response += f"  📊 *حجم پیشنهادی*: {risk_management.get('position_size', 0):.2%}\\n\\n"
        
        # تحلیل هوش مصنوعی
        if ai_analysis:
            response += "🤖 *تحلیل هوش مصنوعی*:\\n"
            response += f"  📊 *پیش‌بینی قیمت*: {ai_analysis.get('price_prediction', 0):.2f}\\n"
            response += f"  📈 *اطمینان پیش‌بینی*: {ai_analysis.get('prediction_confidence', 0):.2f}\\n"
            response += f"  🔄 *روند پیش‌بینی*: {ai_analysis.get('predicted_trend', 'ناشناخته')}\\n\\n"
        
        # تحلیل‌های پیشرفته
        if advanced_analysis:
            response += "🔬 *تحلیل‌های پیشرفته*:\\n"
            
            # تحلیل ویکاف
            wyckoff = advanced_analysis.get('wyckoff', {})
            if wyckoff:
                response += f"  📊 *ویکاف*: {wyckoff.get('phase', 'ناشناخته')}\\n"
            
            # تحلیل پروفایل حجم
            volume_profile = advanced_analysis.get('volume_profile', {})
            if volume_profile:
                response += f"  📊 *ناحیه ارزش*: {volume_profile.get('value_area', 'ناشناخته')}\\n"
            
            # تحلیل فیبوناچی
            fibonacci = advanced_analysis.get('fibonacci', {})
            if fibonacci:
                response += f"  📊 *سطوح فیبوناچی*: {fibonacci.get('levels', 'ناشناخته')}\\n"
            
            # تحلیل الگوهای هارمونیک
            harmonic_patterns = advanced_analysis.get('harmonic_patterns', {})
            if harmonic_patterns:
                response += f"  📊 *الگوی هارمونیک*: {harmonic_patterns.get('pattern', 'ناشناخته')}\\n"
            
            # تحلیل ایچیموکو
            ichimoku = advanced_analysis.get('ichimoku', {})
            if ichimoku:
                response += f"  📊 *ایچیموکو*: {ichimoku.get('signal', 'ناشناخته')}\\n"
            
            response += "\\n"
        
        # منابع داده
        sources = market_data.get('sources', [])
        if sources:
            response += f"🔗 *منابع*: {', '.join(sources)}\\n"
        
        return response
    except Exception as e:
        logger.error(f"Error formatting advanced analysis response: {e}")
        return f"🚀 تحلیل پیشرفته {symbol}\\n\\nخطا در قالب‌بندی تحلیل: {str(e)}"