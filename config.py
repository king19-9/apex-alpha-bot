
import os
from typing import Dict, Any

class Config:
    """تنظیمات اصلی برنامه"""
    
    # کلیدهای API
    API_KEYS = {
        'coingecko': os.getenv('COINGECKO_API_KEY'),
        'coinmarketcap': os.getenv('COINMARKETCAP_API_KEY'),
        'cryptocompare': os.getenv('CRYPTOCOMPARE_API_KEY'),
        'cryptopanic': os.getenv('CRYPTOPANIC_API_KEY'),
        'news': os.getenv('NEWS_API_KEY'),
        'whale_alert': os.getenv('WHALE_ALERT_API_KEY'),
        'glassnode': os.getenv('GLASSNODE_API_KEY'),
        'economic_calendar': os.getenv('ECONOMIC_CALENDAR_API_KEY'),
    }
    
    # توکن تلگرام
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    
    # تنظیمات تحلیل
    CONFIDENCE_THRESHOLD = 0.85
    RISK_REWARD_RATIO = 3  # نسبت ریسک به پاداش 1:3
    MAX_POSITION_SIZE = 0.1
    
    # تنظیمات پایگاه داده
    DATABASE_PATH = 'data/crypto_bot.db'
    
    # تنظیمات کش
    CACHE_EXPIRY_MINUTES = 10
    
    # تنظیمات لاگینگ
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # تنظیمات API
    API_TIMEOUT = 30
    API_RETRY_COUNT = 3
    
    # تنظیمات مدل‌ها
    MODEL_CONFIGS = {
        'signal_classifier': {
            'n_estimators': 300,
            'max_depth': 15,
            'min_samples_split': 3,
            'learning_rate': 0.05,
        },
        'elliott_wave': {
            'hidden_layer_sizes': (300, 200, 100, 50),
            'max_iter': 2000,
            'activation': 'relu',
        },
        'quantum_pattern': {
            'n_estimators': 500,
            'max_depth': 20,
            'min_samples_split': 2,
        },
        'whale_behavior': {
            'n_estimators': 200,
            'learning_rate': 0.03,
            'max_depth': 10,
        }
    }
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """دریافت مقدار تنظیمات"""
        return getattr(cls, key, default)