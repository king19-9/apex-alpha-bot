import logging
from datetime import datetime
from typing import Dict, List, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext

from main import AdvancedCryptoBot
from database import DatabaseManager
from config import Config
from utils import ValidationUtils

logger = logging.getLogger(__name__)

class TelegramHandlers:
    """هندلرهای تلگرام"""
    
    def __init__(self):
        self.bot = AdvancedCryptoBot()
        self.db = DatabaseManager()
        self.config = Config()
    
    async def start_command(self, update: Update, context: CallbackContext):
        """دستور شروع"""
        user = update.effective_user
        user_data = {
            'user_id': user.id,
            'username': user.username or user.first_name,
            'first_seen': datetime.now(),
            'last_seen': datetime.now(),
            'preferences': {}
        }
        
        self.db.save_user(user_data)
        
        welcome_text = f"""
🤖 خوش آمدید به ربات تحلیل ارز دیجیتال پیشرفته!

من یک ربات هوشمند برای تحلیل بازار ارزهای دیجیتال هستم. با استفاده از هوش مصنوعی و یادگیری ماشین، می‌توانم بهترین سیگنال‌های معاملاتی را برای شما ارائه دهم.

🔹 *قابلیت‌های اصلی:*
• تحلیل تکنیکال پیشرفته
• تحلیل احساسات بازار
• تحلیل امواج الیوت
• تحلیل ویکاف
• تحلیل فیبوناچی
• تحلیل الگوهای هارمونیک
• تحلیل رفتار نهنگ‌ها
• تحلیل زنجیره‌ای
• مدیریت ریسک هوشمند

📋 *دستورات موجود:*
• /analyze <symbol> - تحلیل یک ارز خاص
• /signals - دریافت سیگنال‌های تمام ارزها
• /watchlist <symbol> - افزودن به واچ‌لیست
• /alerts - مدیریت هشدارها
• /help - راهنما

برای شروع، یکی از دستورات بالا را استفاده کنید.
        """
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 تحلیل BTC", callback_data="analyze_BTC")],
            [InlineKeyboardButton("📈 سیگنال‌ها", callback_data="signals")],
            [InlineKeyboardButton("❓ راهنما", callback_data="help")]
        ])
        
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=welcome_text,
            reply_markup=keyboard,
            parse_mode='Markdown'
        )
    
    async def help_command(self, update: Update, context: CallbackContext):
        """دستور راهنما"""
        help_text = """
📚 *راهنمای ربات تحلیل ارز دیجیتال*

🔹 *دستورات اصلی:*

• `/start` - شروع ربات
• `/help` - نمایش این راهنما
• `/analyze <symbol>` - تحلیل یک ارز خاص
• `/signals` - دریافت سیگنال‌های معاملاتی
• `/watchlist <symbol>` - افزودن به واچ‌لیست
• `/alerts` - مدیریت هشدارها
• `/performance <symbol>` - مشاهده عملکرد تحلیل‌ها

🔹 *تحلیل ارز:*
برای تحلیل یک ارز خاص، از دستور `/analyze` به همراه نماد ارز استفاده کنید:
مثال: `/analyze BTC`

🔹 *سیگنال‌های معاملاتی:*
با دستور `/signals` می‌توانید لیست سیگنال‌های معاملاتی برای تمام ارزها را دریافت کنید.

🔹 *واچ‌لیست:*
• `/watchlist <symbol>` - افزودن ارز به واچ‌لیست
• `/watchlist list` - نمایش واچ‌لیست شما
• `/watchlist remove <symbol>` - حذف از واچ‌لیست

🔹 *هشدارها:*
• `/alerts` - نمایش هشدارهای فعال
• `/alerts add <symbol> <price> <type>` - افزودن هشدار
• `/alerts remove <id>` - حذف هشدار

نوع‌های هشدار:
• `above` - بالاتر از قیمت
• `below` - پایین‌تر از قیمت
• `change` - تغییر قیمت

مثال: `/alerts add BTC 50000 above`

🔹 *توضیحات تحلیل:*
• 🟢 سیگنال خرید: پیشنهاد ورود به موقعیت
• 🔴 سیگنال فروش: پیشنهاد خروج از موقعیت
• 🟡 سیگنال خنثی: نظاره‌گر بازار باشید
• 🎯 اطمینان: دقت سیگنال (0-100%)
• 🛑 حد ضرر: قیمت پیشنهادی برای حد ضرر
• 🎯 حد سود: قیمت پیشنهادی برای حد سود
• ⚖️ نسبت ریسک به پاداش: نسبت ریسک به پاداش

🔹 *منابع داده:*
ربات از چندین منبع داده استفاده می‌کند:
• CoinGecko
• CoinMarketCap
• CryptoCompare
• CryptoPanic
• Whale Alert
• GlassNode

برای شروع به تحلیل، یکی از دستورات بالا را استفاده کنید.
        """
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 بازگشت", callback_data="back_to_main")],
            [InlineKeyboardButton("📊 تحلیل BTC", callback_data="analyze_BTC")],
            [InlineKeyboardButton("📈 سیگنال‌ها", callback_data="signals")]
        ])
        
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=help_text,
            reply_markup=keyboard,
            parse_mode='Markdown'
        )
    
    async def analyze_command(self, update: Update, context: CallbackContext):
        """دستور تحلیل ارز"""
        try:
            # استخراج نماد از پیام
            message_text = update.message.text
            parts = message_text.split()
            
            if len(parts) < 2:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="❌ لطفاً نماد ارز را وارد کنید.\nمثال: `/analyze BTC`"
                )
                return
            
            symbol = parts[1].upper()
            
            if not ValidationUtils.validate_symbol(symbol):
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="❌ نماد ارز نامعتبر است.\nلطفاً یک نماد معتبر وارد کنید."
                )
                return
            
            # ارسال پیام در حال پردازش
            processing_msg = await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"⏳ در حال تحلیل {symbol}..."
            )
            
            # انجام تحلیل
            analysis = await self.bot.perform_intelligent_analysis(symbol)
            
            # حذف پیام پردازش
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=processing_msg.message_id
            )
            
            # ارسال نتیجه تحلیل
            response = self.bot.format_analysis_response(analysis)
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 تحلیل مجدد", callback_data=f"analyze_{symbol}")],
                [InlineKeyboardButton("📊 سیگنال‌ها", callback_data="signals")],
                [InlineKeyboardButton("📋 افزودن به واچ‌لیست", callback_data=f"watchlist_add_{symbol}")],
                [InlineKeyboardButton("⚠️ افزودن هشدار", callback_data=f"alert_add_{symbol}")]
            ])
            
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=response,
                reply_markup=keyboard,
                parse_mode='Markdown'
            )
            
            # ذخیره تحلیل در پایگاه داده
            self.db.save_analysis_performance({
                'symbol': symbol,
                'method': 'user_request',
                'timestamp': analysis['timestamp'],
                'signal': analysis['signal'],
                'confidence': analysis['confidence'],
                'market_conditions': {
                    'price': analysis.get('market_data', {}).get('price', 0),
                    'volume_24h': analysis.get('market_data', {}).get('volume_24h', 0),
                    'market_cap': analysis.get('market_data', {}).get('market_cap', 0)
                }
            })
            
        except Exception as e:
            logger.error(f"Error in analyze command: {e}")
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ خطا در تحلیل. لطفاً دوباره تلاش کنید."
            )
    
    async def signals_command(self, update: Update, context: CallbackContext):
        """دستور دریافت سیگنال‌ها"""
        try:
            # ارسال پیام در حال پردازش
            processing_msg = await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⏳ در حال دریافت سیگنال‌های معاملاتی..."
            )
            
            # دریافت سیگنال‌ها
            signals = await self.bot.get_trading_signals()
            
            # حذف پیام پردازش
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=processing_msg.message_id
            )
            
            if not signals:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="❌ سیگنالی برای نمایش وجود ندارد."
                )
                return
            
            # ایجاد متن پاسخ
            response = "📈 *سیگنال‌های معاملاتی*\n\n"
            
            # نمایش 10 سیگنال برتر
            for i, signal in enumerate(signings[:10]):
                signal_emoji = "🟢" if signal['signal'] == "BUY" else "🔴" if signal['signal'] == "SELL" else "🟡"
                
                response += f"{signal_emoji} *{signal['symbol']}*\n"
                response += f"💰 قیمت: ${signal['price']:,.2f}\n"
                response += f"📊 تغییر 24h: {signal['price_change_24h']:+.2f}%\n"
                response += f"🎯 سیگنال: {signal['signal']}\n"
                response += f"📈 اطمینان: {signal['confidence']:.1%}\n"
                
                if signal.get('stop_loss') and signal.get('take_profit'):
                    response += f"🛑 حد ضرر: ${signal['stop_loss']:,.2f}\n"
                    response += f"🎯 حد سود: ${signal['take_profit']:,.2f}\n"
                    response += f"⚖️ نسبت ریسک به پاداش: {signal.get('risk_reward_ratio', 0):.2f}\n"
                
                response += "\\n"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="refresh_signals")],
                [InlineKeyboardButton("📊 تحلیل BTC", callback_data="analyze_BTC")],
                [InlineKeyboardButton("❓ راهنما", callback_data="help")]
            ])
            
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=response,
                reply_markup=keyboard,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Error in signals command: {e}")
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ خطا در دریافت سیگنال‌ها. لطفاً دوباره تلاش کنید."
            )
    
    async def watchlist_command(self, update: Update, context: CallbackContext):
        """دستور واچ‌لیست"""
        try:
            message_text = update.message.text
            parts = message_text.split()
            
            user_id = update.effective_user.id
            
            if len(parts) == 1:
                # نمایش واچ‌لیست
                watchlist = self.db.get_watchlist(user_id)
                
                if not watchlist:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text="📋 واچ‌لیست شما خالی است.\nبرای افزودن ارز از دستور `/watchlist <symbol>` استفاده کنید."
                    )
                    return
                
                response = "📋 *واچ‌لیست شما*\n\n"
                for symbol in watchlist:
                    response += f"• {symbol}\n"
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="refresh_watchlist")],
                    [InlineKeyboardButton("📊 سیگنال‌ها", callback_data="signals")]
                ])
                
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=response,
                    reply_markup=keyboard
                )
                
            elif len(parts) >= 2 and parts[1].lower() == 'remove':
                # حذف از واچ‌لیست
                if len(parts) < 3:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text="❌ لطفاً نماد ارز را برای حذف وارد کنید.\nمثال: `/watchlist remove BTC`"
                    )
                    return
                
                symbol = parts[2].upper()
                
                if not ValidationUtils.validate_symbol(symbol):
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text="❌ نماد ارز نامعتبر است."
                    )
                    return
                
                self.db.remove_from_watchlist(user_id, symbol)
                
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"✅ {symbol} از واچ‌لیست حذف شد."
                )
                
            else:
                # افزودن به واچ‌لیست
                symbol = parts[1].upper()
                
                if not ValidationUtils.validate_symbol(symbol):
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text="❌ نماد ارز نامعتبر است."
                    )
                    return
                
                self.db.add_to_watchlist(user_id, symbol)
                
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"✅ {symbol} به واچ‌لیست اضافه شد."
                )
                
                # پیشنهاد تحلیل
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton(f"📊 تحلیل {symbol}", callback_data=f"analyze_{symbol}")],
                    [InlineKeyboardButton("❌ خیر", callback_data="cancel")]
                ])
                
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"آیا مایلید تحلیل {symbol} را ببینید؟",
                    reply_markup=keyboard
                )
                
        except Exception as e:
            logger.error(f"Error in watchlist command: {e}")
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ خطا در مدیریت واچ‌لیست. لطفاً دوباره تلاش کنید."
            )
    
    async def alerts_command(self, update: Update, context: CallbackContext):
        """دستور مدیریت هشدارها"""
        try:
            message_text = update.message.text
            parts = message_text.split()
            
            user_id = update.effective_user.id
            
            if len(parts) == 1:
                # نمایش هشدارها
                alerts = self.db.get_alerts(user_id)
                
                if not alerts:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text="⚠️ هشدار فعالی وجود ندارد.\nبرای افزودن هشدار از دستور `/alerts add <symbol> <price> <type>` استفاده کنید."
                    )
                    return
                
                response = "⚠️ *هشدارهای فعال*\n\n"
                for alert in alerts:
                    alert_type = "بالاتر از" if alert['alert_type'] == 'above' else "پایین‌تر از" if alert['alert_type'] == 'below' else "تغییر"
                    response += f"• {alert['symbol']}: {alert_type} ${alert['target_price']:,.2f}\n"
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="refresh_alerts")],
                    [InlineKeyboardButton("📊 سیگنال‌ها", callback_data="signals")]
                ])
                
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=response,
                    reply_markup=keyboard
                )
                
            elif len(parts) >= 2 and parts[1].lower() == 'add':
                # افزودن هشدار
                if len(parts) < 4:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text="❌ لطفاً پارامترهای هشدار را کامل وارد کنید.\nمثال: `/alerts add BTC 50000 above`"
                    )
                    return
                
                symbol = parts[2].upper()
                
                if not ValidationUtils.validate_symbol(symbol):
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text="❌ نماد ارز نامعتبر است."
                    )
                    return
                
                try:
                    target_price = float(parts[3])
                    alert_type = parts[4].lower() if len(parts) > 4 else 'above'
                    
                    if alert_type not in ['above', 'below', 'change']:
                        await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text="❌ نوع هشدار نامعتبر است.\nنوع‌های مجاز: above, below, change"
                        )
                        return
                    
                    if not ValidationUtils.validate_price(target_price):
                        await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text="❌ قیمت نامعتبر است."
                        )
                        return
                    
                    # افزودن هشدار
                    alert_data = {
                        'user_id': user_id,
                        'symbol': symbol,
                        'alert_type': alert_type,
                        'target_price': target_price
                    }
                    
                    self.db.add_alert(alert_data)
                    
                    type_text = "بالاتر از" if alert_type == 'above' else "پایین‌تر از" if alert_type == 'below' else "تغییر"
                    
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=f"✅ هشدار {type_text} ${target_price:,.2f} برای {symbol} اضافه شد."
                    )
                    
                except ValueError:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text="❌ قیمت نامعتبر است."
                    )
                    
            elif len(parts) >= 2 and parts[1].lower() == 'remove':
                # حذف هشدار
                if len(parts) < 3:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text="❌ لطفاً شناسه هشدار را وارد کنید.\nمثال: `/alerts remove 1`"
                    )
                    return
                
                try:
                    alert_id = int(parts[2])
                    self.db.trigger_alert(alert_id)
                    
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=f"✅ هشدار با شناسه {alert_id} حذف شد."
                    )
                    
                except ValueError:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text="❌ شناسه هشدار نامعتبر است."
                    )
                    
            else:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="❌ دستور نامعتبر است.\nبرای راهنما از `/help` استفاده کنید."
                )
                
        except Exception as e:
            logger.error(f"Error in alerts command: {e}")
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ خطا در مدیریت هشدارها. لطفاً دوباره تلاش کنید."
            )
    
    async def performance_command(self, update: Update, context: CallbackContext):
        """دستور مشاهده عملکرد"""
        try:
            message_text = update.message.text
            parts = message_text.split()
            
            if len(parts) < 2:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="❌ لطفاً نماد ارز را وارد کنید.\nمثال: `/performance BTC`"
                )
                return
            
            symbol = parts[1].upper()
            
            if not ValidationUtils.validate_symbol(symbol):
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="❌ نماد ارز نامعتبر است."
                )
                return
            
            # دریافت عملکرد روش‌های تحلیلی
            methods = ['technical', 'sentiment', 'elliott_wave', 'quantum', 'wyckoff', 'fibonacci']
            performance_data = {}
            
            for method in methods:
                perf = self.db.get_method_performance(symbol, method)
                performance_data[method] = perf
            
            # ایجاد متن پاسخ
            response = f"📊 *عملکرد تحلیل {symbol}*\n\n"
            
            for method, perf in performance_data.items():
                response += f"🔬 *{method.replace('_', ' ').title()}*\n"
                response += f"  • نرخ موفقیت: {perf['success_rate']:.1%}\n"
                response += f"  • میانگین اطمینان: {perf['avg_confidence']:.1%}\n"
                response += f"  • میانگین سود/زیان: {perf['avg_profit_loss']:+.2%}\n"
                response += f"  • تعداد تحلیل‌ها: {perf['total_trades']}\n"
                response += f"  • تحلیل‌های موفق: {perf['winning_trades']}\n\n"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton(f"📊 تحلیل {symbol}", callback_data=f"analyze_{symbol}")],
                [InlineKeyboardButton("📈 سیگنال‌ها", callback_data="signals")]
            ])
            
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=response,
                reply_markup=keyboard,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Error in performance command: {e}")
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ خطا در دریافت عملکرد. لطفاً دوباره تلاش کنید."
            )
    
    async def button_callback(self, update: Update, context: CallbackContext):
        """پرداخ به دکمه‌های اینلاین"""
        query = update.callback_query
        data = query.data
        
        if data.startswith("analyze_"):
            symbol = data.split("_")[1]
            await self._handle_analyze_button(update, context, symbol)
        elif data == "signals":
            await self._handle_signals_button(update, context)
        elif data == "refresh_signals":
            await self._handle_refresh_signals_button(update, context)
        elif data.startswith("watchlist_add_"):
            symbol = data.split("_")[2]
            await self._handle_watchlist_add_button(update, context, symbol)
        elif data == "refresh_watchlist":
            await self._handle_refresh_watchlist_button(update, context)
        elif data.startswith("alert_add_"):
            symbol = data.split("_")[2]
            await self._handle_alert_add_button(update, context, symbol)
        elif data == "refresh_alerts":
            await self._handle_refresh_alerts_button(update, context)
        elif data == "help":
            await self.help_command(update, context)
        elif data == "back_to_main":
            await self.start_command(update, context)
        elif data == "cancel":
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=query.message.message_id
            )
    
    async def _handle_analyze_button(self, update: Update, context: CallbackContext, symbol: str):
        """پرداخ به دکمه تحلیل"""
        try:
            # ویرایش پیام اصلی
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=update.callback_query.message.message_id,
                text=f"⏳ در حال تحلیل {symbol}..."
            )
            
            # انجام تحلیل
            analysis = await self.bot.perform_intelligent_analysis(symbol)
            
            # ارسال نتیجه تحلیل
            response =