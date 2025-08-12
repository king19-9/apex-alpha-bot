import os
import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from dotenv import load_dotenv

from telegram_handlers import setup_handlers
from utils.database import DatabaseManager
from utils.market_data import MarketDataManager
from utils.analysis import AnalysisEngine
from utils.risk_management import RiskManager
from ai_models.technical_analyzer import TechnicalAnalyzer
from ai_models.sentiment_analyzer import SentimentAnalyzer
from ai_models.signal_generator import SignalGenerator

# بارگذاری متغیرهای محیطی
load_dotenv()

# تنظیمات لاگینگ
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class AdvancedTradingBot:
    def __init__(self):
        """مقداردهی اولیه ربات"""
        logger.info("Initializing Advanced Trading Bot...")
        
        # مقداردهی اجزای اصلی
        self.db = DatabaseManager()
        self.market_data = MarketDataManager()
        self.analysis_engine = AnalysisEngine()
        self.risk_manager = RiskManager()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.signal_generator = SignalGenerator()
        
        logger.info("Advanced Trading Bot initialized successfully")
    
    async def perform_analysis(self, symbol):
        """انجام تحلیل کامل برای یک ارز"""
        try:
            # دریافت داده‌های بازار
            market_data = await self.market_data.get_data(symbol)
            
            # تحلیل تکنیکال
            technical_analysis = await self.technical_analyzer.analyze(market_data)
            
            # تحلیل احساسات
            sentiment_analysis = await self.sentiment_analyzer.analyze(symbol)
            
            # تولید سیگنال
            signal = await self.signal_generator.generate_signal(
                symbol, technical_analysis, sentiment_analysis
            )
            
            return {
                'symbol': symbol,
                'market_data': market_data,
                'technical': technical_analysis,
                'sentiment': sentiment_analysis,
                'signal': signal
            }
        except Exception as e:
            logger.error(f"Error in analysis for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}

async def main():
    """تابع اصلی اجرای ربات"""
    # ایجاد نمونه ربات
    bot = AdvancedTradingBot()
    
    # تنظیمات ربات تلگرام
    application = Application.builder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()
    
    # تنظیم هندلرها
    setup_handlers(application, bot)
    
    # اجرای ربات
    logger.info("Starting bot...")
    await application.run_polling()

if __name__ == '__main__':
    # اجرای برنامه
    asyncio.run(main())