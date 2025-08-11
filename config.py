# config.py
import os
from dotenv import load_dotenv
import aiohttp

load_dotenv()

class Config:
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

    PROXY_URL = os.getenv("PROXY_URL")
    PROXY_USERNAME = os.getenv("PROXY_USERNAME")
    PROXY_PASSWORD = os.getenv("PROXY_PASSWORD")
    
    REQUESTS_PROXY_DICT = {'http': PROXY_URL, 'https': PROXY_URL} if PROXY_URL else {}
    AIOHTTP_PROXY_URL = PROXY_URL
    AIOHTTP_PROXY_AUTH = aiohttp.BasicAuth(PROXY_USERNAME, PROXY_PASSWORD) if PROXY_USERNAME else None

    DATABASE_URL = os.getenv("DATABASE_URL")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

    API_KEYS = {
        'coingecko': os.getenv('COINGECKO_API_KEY'),
        'news': os.getenv('NEWS_API_KEY'),
        'cryptopanic': os.getenv('CRYPTOPANIC_API_KEY'),
        'cryptocompare': os.getenv('CRYPTOCOMPARE_API_KEY'),
        'coinalyze': os.getenv('COINANALYZE_API_KEY'),
        'glassnode': os.getenv('GLASSNODE_API_KEY'), # برای تحلیل On-chain
        'binance': {'apiKey': os.getenv('BINANCE_API_KEY'), 'secret': os.getenv('BINANCE_SECRET_KEY')}
    }

    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "True").lower() == "true"
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 300))
    ENABLED_EXCHANGES = ['binance', 'kucoin', 'bybit', 'gateio', 'okx']
    SUPPORTED_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w']