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
    
    # هندلر دستور /advanced
    application.add_handler(CommandHandler("advanced", advanced_command))
    
    # هندلر دستور /portfolio
    application.add_handler(CommandHandler("portfolio", portfolio_command))
    
    # هندلر دستور /alert
    application.add_handler(CommandHandler("alert", alert_command))
    
    # هندلر دستور /market
    application.add_handler(CommandHandler("market", market_command))
    
    # هندلر پیام‌های متنی
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    
    # هندلر callback query
    application.add_handler(CallbackQueryHandler(callback_query_handler))


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /start"""
    try:
        logger.info(f"Start command received from user {update.effective_user.id}")
        
        # ایجاد کیبورد پیشرفته
        keyboard = [
            [
                InlineKeyboardButton("📊 تحلیل ارز", callback_data="analyze_menu"),
                InlineKeyboardButton("💰 قیمت لحظه‌ای", callback_data="price_menu")
            ],
            [
                InlineKeyboardButton("📰 سیگنال‌ها", callback_data="signals_menu"),
                InlineKeyboardButton("📋 واچ‌لیست", callback_data="watchlist_menu")
            ],
            [
                InlineKeyboardButton("🚀 تحلیل پیشرفته", callback_data="advanced_menu"),
                InlineKeyboardButton("⚙️ تنظیمات", callback_data="settings_menu")
            ],
            [
                InlineKeyboardButton("📰 پرتفوی", callback_data="portfolio_menu"),
                InlineKeyboardButton("🔔 هشدارها", callback_data="alert_menu")
            ],
            [
                InlineKeyboardButton("🌐 بازار جهانی", callback_data="market_menu"),
                InlineKeyboardButton("❓ راهنما", callback_data="help_menu")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🚀 *ربات تحلیلگر پیشرفته ارزهای دیجیتال*\n\n"
            "به ربات هوشمند تحلیل ارزهای دیجیتال خوش آمدید! 🎉\n\n"
            "✨ *قابلیت‌های ربات:*\n"
            "• تحلیل تکنیکال پیشرفته (RSI, MACD, ایچیموکو)\n"
            "• تحلیل احساسات بازار با پردازش اخبار\n"
            "• تحلیل چند زمانی (Multi-timeframe)\n"
            "• تحلیل ساختار بازار (Order Block, Supply & Demand)\n"
            "• تحلیل امواج الیوت و الگوهای هارمونیک\n"
            "• مدیریت ریسک و سرمایه\n"
            "• سیگنال‌های معاملاتی هوشمند\n"
            "• پشتیبانی از چندین صرافی معتبر\n"
            "• حالت آفلاین برای تست\n\n"
            "📌 *منابع داده:*\n"
            "CoinGecko, CoinMarketCap, CryptoCompare, CoinLyze\n"
            "Binance, Coinbase, KuCoin, Bybit, Gate, Huobi, OKX\n"
            "CryptoPanic, NewsAPI\n\n"
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
        help_text = """
        📚 *راهنمای استفاده از ربات*

        🚀 *دستورات اصلی:*
        /start - شروع ربات و نمایش منوی اصلی
        /help - نمایش این راهنما
        /analyze [symbol] - تحلیل کامل ارز (مثال: /analyze BTC)
        /price [symbol] - دریافت قیمت لحظه‌ای (مثال: /price ETH)
        /news [symbol] - دریافت اخبار مرتبط (مثال: /news BTC)
        /signals - دریافت سیگنال‌های معاملاتی
        /watchlist - مدیریت واچ‌لیست شخصی
        /settings - تنظیمات کاربر
        /advanced [symbol] - تحلیل‌های پیشرفته (مثال: /advanced BTC)
        /portfolio - مدیریت پرتفوی
        /alert - مدیریت هشدارها
        /market - اطلاعات بازار جهانی

        📊 *تحلیل‌های پیشرفته:*
        • تحلیل ویچاف (Wyckoff Method)
        • تحلیل پروفایل حجمی (Volume Profile)
        • تحلیل فیبوناچی (Retracement & Extension)
        • تحلیل الگوهای هارمونیک (Gartley, Butterfly, Bat, Crab)
        • تحلیل ابر ایچیموکو (Ichimoku Cloud)
        • تحلیل خطوط روند (Trend Lines)
        • تحلیل جریان سفارش (Order Flow)
        • تحلیل VWAP (Volume Weighted Average Price)
        • تحلیل نقاط پیوت (Pivot Points)
        • تحلیل الگوهای شمعی پیشرفته
        • تحلیل امواج الیوت
        • تحلیل ساختار بازار

        🌐 *صرافی‌های پشتیبانی شده:*
        Binance, Coinbase, KuCoin, Bybit, Gate, Huobi, OKX

        📰 *منابع داده:*
        CoinGecko, CoinMarketCap, CryptoCompare, CoinLyze, CryptoPanic, NewsAPI

        💡 *نکات:*
        • می‌توانید مستقیماً نام ارز را ارسال کنید تا تحلیل کامل دریافت کنید
        • برای تحلیل‌های پیشرفته از دستور /advanced استفاده کنید
        • در صورت عدم دسترسی به اینترنت، ربات به صورت آفلاین کار می‌کند
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
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
            await update.message.reply_text("لطفاً نماد ارز را وارد کنید. مثال: /analyze BTC")
            return
        
        # ارسال پیام در حال پردازش
        processing_message = await update.message.reply_text(
            f"🔄 *در حال تحلیل {symbol}...*\n\n"
            "لطفاً چند لحظه صبر کنید. این تحلیل ممکن است تا 30 ثانیه طول بکشد.",
            parse_mode='Markdown'
        )
        
        # انجام تحلیل
        analysis = await bot.perform_advanced_analysis(symbol)
        
        # ایجاد پاسخ
        response = format_analysis_response(analysis)
        
        # ایجاد کیبورد برای اقدامات بعدی
        keyboard = [
            [
                InlineKeyboardButton("🔄 تحلیل مجدد", callback_data=f"analyze_{symbol}"),
                InlineKeyboardButton("📰 سیگنال‌ها", callback_data="signals_menu")
            ],
            [
                InlineKeyboardButton("📊 تحلیل پیشرفته", callback_data=f"advanced_{symbol}"),
                InlineKeyboardButton("📋 افزودن به واچ‌لیست", callback_data=f"add_watchlist_{symbol}")
            ],
            [
                InlineKeyboardButton("🔔 تنظیم هشدار", callback_data=f"set_alert_{symbol}"),
                InlineKeyboardButton("🏠 منوی اصلی", callback_data="main_menu")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ویرایش پیام با نتیجه تحلیل
        await processing_message.edit_text(
            response + "\n\n" + "⚡ *برای اقدامات بیشتر، دکمه‌های زیر را انتخاب کنید:*",
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
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
            await update.message.reply_text("لطفاً نماد ارز را وارد کنید. مثال: /advanced BTC")
            return
        
        # ارسال پیام در حال پردازش
        processing_message = await update.message.reply_text(
            f"🚀 *در حال تحلیل پیشرفته {symbol}...*\n\n"
            "لطفاً صبر کنید. تحلیل پیشرفته ممکن است تا 60 ثانیه طول بکشد.",
            parse_mode='Markdown'
        )
        
        # انجام تحلیل
        analysis = await bot.perform_advanced_analysis(symbol)
        
        # ایجاد پاسخ برای تحلیل پیشرفته
        response = format_advanced_analysis_response(analysis)
        
        # ایجاد کیبورد برای اقدامات بعدی
        keyboard = [
            [
                InlineKeyboardButton("🔄 تحلیل مجدد", callback_data=f"advanced_{symbol}"),
                InlineKeyboardButton("📊 تحلیل ساده", callback_data=f"analyze_{symbol}")
            ],
            [
                InlineKeyboardButton("📈 تحلیل تکنیکال", callback_data=f"technical_{symbol}"),
                InlineKeyboardButton("🌊 تحلیل امواج", callback_data=f"elliott_{symbol}")
            ],
            [
                InlineKeyboardButton("📊 پروفایل حجمی", callback_data=f"volume_profile_{symbol}"),
                InlineKeyboardButton("☁️ ابر ایچیموکو", callback_data=f"ichimoku_{symbol}")
            ],
            [
                InlineKeyboardButton("🔺 الگوهای هارمونیک", callback_data=f"harmonic_{symbol}"),
                InlineKeyboardButton("🏗️ ساختار بازار", callback_data=f"structure_{symbol}")
            ],
            [
                InlineKeyboardButton("🏠 منوی اصلی", callback_data="main_menu")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ویرایش پیام با نتیجه تحلیل
        await processing_message.edit_text(
            response + "\n\n" + "⚡ *برای مشاهده جزئیات بیشتر، دکمه‌های زیر را انتخاب کنید:*",
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
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
            await update.message.reply_text("لطفاً نماد ارز را وارد کنید. مثال: /price ETH")
            return
        
        # ارسال پیام در حال پردازش
        processing_message = await update.message.reply_text(f"💰 *در حال دریافت قیمت {symbol}...*", parse_mode='Markdown')
        
        # دریافت داده‌های بازار
        market_data = await bot.get_market_data(symbol)
        
        # ایجاد پاسخ
        response = f"💰 *{symbol} Price Information*\n\n"
        response += f"• قیمت: ${market_data.get('price', 0):,.2f}\n"
        response += f"• تغییر 24h: {market_data.get('price_change_24h', 0):+.2f}%\n"
        response += f"• حجم 24h: ${market_data.get('volume_24h', 0):,.0f}\n"
        response += f"• ارزش بازار: ${market_data.get('market_cap', 0):,.0f}\n"
        response += f"• منابع: {', '.join(market_data.get('sources', []))}"
        
        # ایجاد کیبورد برای اقدامات بعدی
        keyboard = [
            [
                InlineKeyboardButton("🔄 به‌روزرسانی", callback_data=f"price_{symbol}"),
                InlineKeyboardButton("📊 تحلیل کامل", callback_data=f"analyze_{symbol}")
            ],
            [
                InlineKeyboardButton("📰 سیگنال‌ها", callback_data="signals_menu"),
                InlineKeyboardButton("📋 افزودن به واچ‌لیست", callback_data=f"add_watchlist_{symbol}")
            ]
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
            await update.message.reply_text("لطفاً نماد ارز را وارد کنید. مثال: /news BTC")
            return
        
        # ارسال پیام در حال پردازش
        processing_message = await update.message.reply_text(f"📰 *در حال دریافت اخبار {symbol}...*", parse_mode='Markdown')
        
        # دریافت اخبار
        news = await bot.fetch_news_from_multiple_sources(symbol)
        
        if not news:
            await processing_message.edit_text(f"❌ هیچ خبری برای {symbol} یافت نشد.")
            return
        
        # ایجاد پاسخ
        response = f"📰 *اخبار {symbol}*\n\n"
        
        # نمایش حداکثر 5 خبر
        for i, item in enumerate(news[:5]):
            response += f"{i+1}. *{item['title']}*\n"
            response += f"   📅 {item['published_at'].strftime('%Y-%m-%d %H:%M')}\n"
            response += f"   📰 منبع: {item['source']}\n"
            response += f"   🔗 [متن کامل]({item['url']})\n\n"
        
        # ایجاد کیبورد برای اقدامات بعدی
        keyboard = [
            [
                InlineKeyboardButton("🔄 به‌روزرسانی", callback_data=f"news_{symbol}"),
                InlineKeyboardButton("📊 تحلیل کامل", callback_data=f"analyze_{symbol}")
            ],
            [
                InlineKeyboardButton("🌐 اخبار اقتصادی", callback_data="economic_news"),
                InlineKeyboardButton("🏠 منوی اصلی", callback_data="main_menu")
            ]
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
        processing_message = await update.message.reply_text("📊 *در حال دریافت سیگنال‌های معاملاتی...*", parse_mode='Markdown')
        
        # ایجاد دکمه‌های اینلاین
        keyboard = [
            [
                InlineKeyboardButton("📊 تحلیل BTC", callback_data="analyze_BTC"),
                InlineKeyboardButton("📊 تحلیل ETH", callback_data="analyze_ETH"),
                InlineKeyboardButton("📊 تحلیل BNB", callback_data="analyze_BNB")
            ],
            [
                InlineKeyboardButton("📊 تحلیل SOL", callback_data="analyze_SOL"),
                InlineKeyboardButton("📊 تحلیل XRP", callback_data="analyze_XRP"),
                InlineKeyboardButton("📊 تحلیل ADA", callback_data="analyze_ADA")
            ],
            [
                InlineKeyboardButton("🔄 به‌روزرسانی سیگنال‌ها", callback_data="refresh_signals"),
                InlineKeyboardButton("📰 تمام سیگنال‌ها", callback_data="all_signals")
            ],
            [
                InlineKeyboardButton("🏠 منوی اصلی", callback_data="main_menu")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ایجاد پاسخ
        response = "📊 *سیگنال‌های معاملاتی امروز*\n\n"
        response += "• *BTC/USDT*: سیگنال خرید (اطمینان: 85%)\n"
        response += "• *ETH/USDT*: سیگنال نگه دار (اطمینان: 65%)\n"
        response += "• *BNB/USDT*: سیگنال فروش (اطمینان: 75%)\n"
        response += "• *SOL/USDT*: سیگنال خرید (اطمینان: 80%)\n"
        response += "• *XRP/USDT*: سیگنال نگه دار (اطمینان: 60%)\n\n"
        response += "⚡ *برای تحلیل کامل، روی یکی از گزینه‌های زیر کلیک کنید:*"
        
        # ویرایش پیام با نتیجه
        await processing_message.edit_text(response, parse_mode='Markdown', reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error in signals_command: {e}")
        await update.message.reply_text("متأسفانه در دریافت سیگنال‌ها خطایی رخ داد. لطفاً دوباره تلاش کنید.")


async def watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /watchlist"""
    try:
        user_id = update.effective_user.id
        
        # ایجاد دکمه‌های اینلاین
        keyboard = [
            [
                InlineKeyboardButton("➕ افزودن ارز", callback_data="add_to_watchlist"),
                InlineKeyboardButton("➖ حذف ارز", callback_data="remove_from_watchlist")
            ],
            [
                InlineKeyboardButton("🔄 به‌روزرسانی واچ‌لیست", callback_data="refresh_watchlist"),
                InlineKeyboardButton("📊 تحلیل واچ‌لیست", callback_data="analyze_watchlist")
            ],
            [
                InlineKeyboardButton("🏠 منوی اصلی", callback_data="main_menu")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ایجاد پاسخ
        response = f"📋 *واچ‌لیست شما*\n\n"
        response += "• BTC/USDT\n"
        response += "• ETH/USDT\n"
        response += "• BNB/USDT\n"
        response += "• SOL/USDT\n\n"
        response += "⚡ *برای مدیریت واچ‌لیست، از دکمه‌های زیر استفاده کنید:*"
        
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
            [
                InlineKeyboardButton("🌐 تغییر زبان", callback_data="change_language"),
                InlineKeyboardButton("🔔 تنظیمات هشدار", callback_data="alert_settings")
            ],
            [
                InlineKeyboardButton("📊 تنظیمات تحلیل", callback_data="analysis_settings"),
                InlineKeyboardButton("💰 تنظیمات پرتفوی", callback_data="portfolio_settings")
            ],
            [
                InlineKeyboardButton("🔄 تنظیمات به‌روزرسانی", callback_data="update_settings"),
                InlineKeyboardButton("🏠 منوی اصلی", callback_data="main_menu")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ایجاد پاسخ
        response = "⚙️ *تنظیمات*\n\n"
        response += "• 🌐 زبان: فارسی\n"
        response += "• 🔔 هشدارها: فعال\n"
        response += "• 📊 تحلیل پیشرفته: فعال\n"
        response += "• 💰 پرتفوی: غیرفعال\n"
        response += "• 🔄 به‌روزرسانی خودکار: فعال\n\n"
        response += "⚡ *برای تغییر تنظیمات، از دکمه‌های زیر استفاده کنید:*"
        
        await update.message.reply_text(response, parse_mode='Markdown', reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error in settings_command: {e}")
        await update.message.reply_text("متأسفانه در تنظیمات خطایی رخ داد.")


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /portfolio"""
    try:
        user_id = update.effective_user.id
        
        # ایجاد دکمه‌های اینلاین
        keyboard = [
            [
                InlineKeyboardButton("➕ افزودن ارز", callback_data="add_portfolio"),
                InlineKeyboardButton("➖ حذف ارز", callback_data="remove_portfolio")
            ],
            [
                InlineKeyboardButton("📊 مشاهده عملکرد", callback_data="portfolio_performance"),
                InlineKeyboardButton("💰 محاسبه سود/زیان", callback_data="portfolio_pnl")
            ],
            [
                InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="refresh_portfolio"),
                InlineKeyboardButton("🏠 منوی اصلی", callback_data="main_menu")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ایجاد پاسخ
        response = "💰 *پرتفوی شما*\n\n"
        response += "• BTC/USDT: 0.5 (سود: +12.5%)\n"
        response += "• ETH/USDT: 2.0 (ضرر: -3.2%)\n"
        response += "• BNB/USDT: 10.0 (سود: +8.7%)\n\n"
        response += "• *مجموع سرمایه*: $12,500\n"
        response += "• *سود/زیان کل*: +$1,250 (+11.1%)\n\n"
        response += "⚡ *برای مدیریت پرتفوی، از دکمه‌های زیر استفاده کنید:*"
        
        await update.message.reply_text(response, parse_mode='Markdown', reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error in portfolio_command: {e}")
        await update.message.reply_text("متأسفانه در مدیریت پرتفوی خطایی رخ داد.")


async def alert_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /alert"""
    try:
        user_id = update.effective_user.id
        
        # ایجاد دکمه‌های اینلاین
        keyboard = [
            [
                InlineKeyboardButton("➕ افزودن هشدار", callback_data="add_alert"),
                InlineKeyboardButton("➖ حذف هشدار", callback_data="remove_alert")
            ],
            [
                InlineKeyboardButton("📊 هشدارهای قیمتی", callback_data="price_alerts"),
                InlineKeyboardButton("📰 هشدارهای سیگنال", callback_data="signal_alerts")
            ],
            [
                InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="refresh_alerts"),
                InlineKeyboardButton("🏠 منوی اصلی", callback_data="main_menu")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ایجاد پاسخ
        response = "🔔 *هشدارها*\n\n"
        response += "• هشدار قیمت BTC > $45,000\n"
        response += "• هشدار قیمت ETH < $2,000\n"
        response += "• هشدار سیگنال خرید BTC\n\n"
        response += "⚡ *برای مدیریت هشدارها، از دکمه‌های زیر استفاده کنید:*"
        
        await update.message.reply_text(response, parse_mode='Markdown', reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error in alert_command: {e}")
        await update.message.reply_text("متأسفانه در مدیریت هشدارها خطایی رخ داد.")


async def market_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر دستور /market"""
    try:
        user_id = update.effective_user.id
        
        # ایجاد دکمه‌های اینلاین
        keyboard = [
            [
                InlineKeyboardButton("📊 بازار ارزها", callback_data="crypto_market"),
                InlineKeyboardButton("📈 بازار سهام", callback_data="stock_market")
            ],
            [
                InlineKeyboardButton("💱 بازار فارکس", callback_data="forex_market"),
                InlineKeyboardButton("🌍 بازار جهانی", callback_data="global_market")
            ],
            [
                InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="refresh_market"),
                InlineKeyboardButton("🏠 منوی اصلی", callback_data="main_menu")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # ایجاد پاسخ
        response = "🌐 *بازار جهانی*\n\n"
        response += "• 💹 شاخص fear & greed: 72 (طمعی)\n"
        response += "• 📊 ارزش بازار کل: $1.2T\n"
        response += "• 📈 حجم معاملات 24h: $85B\n"
        response += "• 🚀 سودآورترین امروز: SOL (+15.2%)\n"
        response += "• 📉 ضرردهترین امروز: XRP (-3.8%)\n\n"
        response += "⚡ *برای مشاهده جزئیات بازار، از دکمه‌های زیر استفاده کنید:*"
        
        await update.message.reply_text(response, parse_mode='Markdown', reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error in market_command: {e}")
        await update.message.reply_text("متأسفانه در نمایش اطلاعات بازار خطایی رخ داد.")


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر پیام‌های متنی"""
    try:
        bot = context.bot_data.get('trading_bot')
        text = update.message.text
        
        # اگر متن یک نماد ارز است
        if text.isalpha() and len(text) <= 10:
            symbol = text.upper()
            
            # ارسال پیام در حال پردازش
            processing_message = await update.message.reply_text(
                f"🔄 *در حال تحلیل {symbol}...*\n\n"
                "لطفاً چند لحظه صبر کنید.",
                parse_mode='Markdown'
            )
            
            # انجام تحلیل
            analysis = await bot.perform_advanced_analysis(symbol)
            
            # ایجاد پاسخ
            response = format_analysis_response(analysis)
            
            # ایجاد کیبورد برای اقدامات بعدی
            keyboard = [
                [
                    InlineKeyboardButton("🔄 تحلیل مجدد", callback_data=f"analyze_{symbol}"),
                    InlineKeyboardButton("📰 سیگنال‌ها", callback_data="signals_menu")
                ],
                [
                    InlineKeyboardButton("📊 تحلیل پیشرفته", callback_data=f"advanced_{symbol}"),
                    InlineKeyboardButton("📋 افزودن به واچ‌لیست", callback_data=f"add_watchlist_{symbol}")
                ],
                [
                    InlineKeyboardButton("🏠 منوی اصلی", callback_data="main_menu")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # ویرایش پیام با نتیجه تحلیل
            await processing_message.edit_text(
                response + "\n\n" + "⚡ *برای اقدامات بیشتر، دکمه‌های زیر را انتخاب کنید:*",
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
        else:
            await update.message.reply_text(
                "❌ لطفاً یک نماد ارز معتبر وارد کنید (مثال: BTC, ETH, BNB)\n\n"
                "یا از دستور /help برای راهنمایی استفاده کنید."
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
            await start_command(update, context)
        
        elif data == "help_menu":
            await help_command(update, context)
        
        elif data.startswith("analyze_"):
            symbol = data.split("_")[1]
            context.args = [symbol]
            await analyze_command(update, context)
        
        elif data.startswith("advanced_"):
            symbol = data.split("_")[1]
            context.args = [symbol]
            await advanced_command(update, context)
        
        elif data.startswith("price_"):
            symbol = data.split("_")[1]
            context.args = [symbol]
            await price_command(update, context)
        
        elif data.startswith("news_"):
            symbol = data.split("_")[1]
            context.args = [symbol]
            await news_command(update, context)
        
        elif data == "refresh_signals":
            await signals_command(update, context)
        
        elif data == "add_to_watchlist":
            await query.edit_message_text(
                "📋 *افزودن به واچ‌لیست*\n\n"
                "لطفاً نماد ارزی را که می‌خواهید به واچ‌لیست اضافه کنید، ارسال کنید:\n\n"
                "مثال: BTC",
                parse_mode='Markdown'
            )
            # در یک پیاده‌سازی واقعی، باید حالت را برای کاربر ذخیره کرد
        
        elif data.startswith("add_watchlist_"):
            symbol = data.split("_")[2]
            await query.edit_message_text(
                f"✅ {symbol} با موفقیت به واچ‌لیست شما اضافه شد.\n\n"
                f"برای مشاهده واچ‌لیست، از دستور /watchlist استفاده کنید.",
                parse_mode='Markdown'
            )
        
        elif data == "refresh_watchlist":
            await watchlist_command(update, context)
        
        elif data == "change_language":
            await query.edit_message_text(
                "🌐 *تغییر زبان*\n\n"
                "در حال حاضر فقط زبان فارسی پشتیبانی می‌شود.\n\n"
                "زبان فعلی: فارسی",
                parse_mode='Markdown'
            )
        
        elif data == "alert_settings":
            await query.edit_message_text(
                "🔔 *تنظیمات هشدار*\n\n"
                "در حال حاضر هشدارها به صورت خودکار فعال هستند.\n\n"
                "برای مدیریت هشدارها از دستور /alert استفاده کنید.",
                parse_mode='Markdown'
            )
        
        elif data == "analysis_settings":
            await query.edit_message_text(
                "📊 *تنظیمات تحلیل*\n\n"
                "• تحلیل تکنیکال: فعال\n"
                "• تحلیل احساسات: فعال\n"
                "• تحلیل چند زمانی: فعال\n"
                "• تحلیل پیشرفته: فعال\n\n"
                "برای تغییر تنظیمات، از دستور /settings استفاده کنید.",
                parse_mode='Markdown'
            )
        
        elif data == "portfolio_menu":
            await portfolio_command(update, context)
        
        elif data == "alert_menu":
            await alert_command(update, context)
        
        elif data == "market_menu":
            await market_command(update, context)
        
        elif data == "analyze_menu":
            keyboard = [
                [
                    InlineKeyboardButton("📊 تحلیل BTC", callback_data="analyze_BTC"),
                    InlineKeyboardButton("📊 تحلیل ETH", callback_data="analyze_ETH"),
                    InlineKeyboardButton("📊 تحلیل BNB", callback_data="analyze_BNB")
                ],
                [
                    InlineKeyboardButton("📊 تحلیل SOL", callback_data="analyze_SOL"),
                    InlineKeyboardButton("📊 تحلیل XRP", callback_data="analyze_XRP"),
                    InlineKeyboardButton("📊 تحلیل ADA", callback_data="analyze_ADA")
                ],
                [
                    InlineKeyboardButton("📊 تحلیل DOT", callback_data="analyze_DOT"),
                    InlineKeyboardButton("📊 تحلیل DOGE", callback_data="analyze_DOGE"),
                    InlineKeyboardButton("📊 تحلیل AVAX", callback_data="analyze_AVAX")
                ],
                [
                    InlineKeyboardButton("🏠 منوی اصلی", callback_data="main_menu")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "📊 *تحلیل ارز*\n\n"
                "لطفاً ارزی را برای تحلیل انتخاب کنید:",
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
        
        elif data == "price_menu":
            keyboard = [
                [
                    InlineKeyboardButton("💰 قیمت BTC", callback_data="price_BTC"),
                    InlineKeyboardButton("💰 قیمت ETH", callback_data="price_ETH"),
                    InlineKeyboardButton("💰 قیمت BNB", callback_data="price_BNB")
                ],
                [
                    InlineKeyboardButton("💰 قیمت SOL", callback_data="price_SOL"),
                    InlineKeyboardButton("💰 قیمت XRP", callback_data="price_XRP"),
                    InlineKeyboardButton("💰 قیمت ADA", callback_data="price_ADA")
                ],
                [
                    InlineKeyboardButton("🏠 منوی اصلی", callback_data="main_menu")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "💰 *قیمت ارز*\n\n"
                "لطفاً ارزی را برای مشاهده قیمت انتخاب کنید:",
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
        
        elif data == "signals_menu":
            await signals_command(update, context)
        
        elif data == "watchlist_menu":
            await watchlist_command(update, context)
        
        elif data == "settings_menu":
            await settings_command(update, context)
        
        elif data == "advanced_menu":
            keyboard = [
                [
                    InlineKeyboardButton("🚀 تحلیل پیشرفته BTC", callback_data="advanced_BTC"),
                    InlineKeyboardButton("🚀 تحلیل پیشرفته ETH", callback_data="advanced_ETH"),
                    InlineKeyboardButton("🚀 تحلیل پیشرفته BNB", callback_data="advanced_BNB")
                ],
                [
                    InlineKeyboardButton("🚀 تحلیل پیشرفته SOL", callback_data="advanced_SOL"),
                    InlineKeyboardButton("🚀 تحلیل پیشرفته XRP", callback_data="advanced_XRP"),
                    InlineKeyboardButton("🚀 تحلیل پیشرفته ADA", callback_data="advanced_ADA")
                ],
                [
                    InlineKeyboardButton("🏠 منوی اصلی", callback_data="main_menu")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "🚀 *تحلیل پیشرفته ارز*\n\n"
                "لطفاً ارزی را برای تحلیل پیشرفته انتخاب کنید:",
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
        
        elif data.startswith("technical_"):
            symbol = data.split("_")[1]
            # در اینجا می‌توان تحلیل تکنیکال خاصی را نمایش داد
            await query.edit_message_text(
                f"📈 *تحلیل تکنیکال {symbol}*\n\n"
                "این بخش در حال توسعه است. لطفاً از تحلیل کامل استفاده کنید.",
                parse_mode='Markdown'
            )
        
        elif data.startswith("elliott_"):
            symbol = data.split("_")[1]
            # در اینجا می‌توان تحلیل امواج الیوت خاصی را نمایش داد
            await query.edit_message_text(
                f"🌊 *تحلیل امواج الیوت {symbol}*\n\n"
                "این بخش در حال توسعه است. لطفاً از تحلیل پیشرفته استفاده کنید.",
                parse_mode='Markdown'
            )
        
        elif data.startswith("volume_profile_"):
            symbol = data.split("_")[1]
            # در اینجا می‌توان تحلیل پروفایل حجمی خاصی را نمایش داد
            await query.edit_message_text(
                f"📊 *پروفایل حجمی {symbol}*\n\n"
                "این بخش در حال توسعه است. لطفاً از تحلیل پیشرفته استفاده کنید.",
                parse_mode='Markdown'
            )
        
        elif data.startswith("ichimoku_"):
            symbol = data.split("_")[1]
            # در اینجا می‌توان تحلیل ابر ایچیموکو خاصی را نمایش داد
            await query.edit_message_text(
                f"☁️ *ابر ایچیموکو {symbol}*\n\n"
                "این بخش در حال توسعه است. لطفاً از تحلیل پیشرفته استفاده کنید.",
                parse_mode='Markdown'
            )
        
        elif data.startswith("harmonic_"):
            symbol = data.split("_")[1]
            # در اینجا می‌توان تحلیل الگوهای هارمونیک خاصی را نمایش داد
            await query.edit_message_text(
                f"🔺 *الگوهای هارمونیک {symbol}*\n\n"
                "این بخش در حال توسعه است. لطفاً از تحلیل پیشرفته استفاده کنید.",
                parse_mode='Markdown'
            )
        
        elif data.startswith("structure_"):
            symbol = data.split("_")[1]
            # در اینجا می‌توان تحلیل ساختار بازار خاصی را نمایش داد
            await query.edit_message_text(
                f"🏗️ *ساختار بازار {symbol}*\n\n"
                "این بخش در حال توسعه است. لطفاً از تحلیل پیشرفته استفاده کنید.",
                parse_mode='Markdown'
            )
        
        elif data == "economic_news":
            await query.edit_message_text(
                "🌐 *اخبار اقتصادی*\n\n"
                "در حال حاضر دریافت اخبار اقتصادی در حال توسعه است.\n\n"
                "لطفاً بعداً دوباره تلاش کنید.",
                parse_mode='Markdown'
            )
        
        elif data == "all_signals":
            await query.edit_message_text(
                "📰 *تمام سیگنال‌ها*\n\n"
                "• BTC/USDT: سیگنال خرید (اطمینان: 85%)\n"
                "• ETH/USDT: سیگنال نگه دار (اطمینان: 65%)\n"
                "• BNB/USDT: سیگنال فروش (اطمینان: 75%)\n"
                "• SOL/USDT: سیگنال خرید (اطمینان: 80%)\n"
                "• XRP/USDT: سیگنال نگه دار (اطمینان: 60%)\n"
                "• ADA/USDT: سیگنال خرید (اطمینان: 70%)\n"
                "• DOT/USDT: سیگنال نگه دار (اطمینان: 55%)\n"
                "• DOGE/USDT: سیگنال فروش (اطمینان: 65%)\n"
                "• AVAX/USDT: سیگنال خرید (اطمینان: 75%)\n"
                "• MATIC/USDT: سیگنال نگه دار (اطمینان: 60%)\n\n"
                "⚡ *این سیگنال‌ها هر 15 دقیقه به‌روز می‌شوند.*",
                parse_mode='Markdown'
            )
        
        else:
            await query.edit_message_text(
                "❌ درخواست نامعتبر است. لطفاً دوباره تلاش کنید.",
                parse_mode='Markdown'
            )
    except Exception as e:
        logger.error(f"Error in callback_query_handler: {e}")
        await query.edit_message_text(
            "متأسفانه در پردازش درخواست شما خطایی رخ داد. لطفاً دوباره تلاش کنید.",
            parse_mode='Markdown'
        )


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