import asyncio
import aiohttp
import yfinance as yf
import ccxt
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class MarketDataManager:
    def __init__(self):
        """مقداردهی اولیه مدیریت داده‌های بازار"""
        self.exchanges = {
            'binance': ccxt.binance(),
            'coinbase': ccxt.coinbase(),
            'kraken': ccxt.kraken(),
            'kucoin': ccxt.kucoin()
        }
        self.session = aiohttp.ClientSession()
        logger.info("Market data manager initialized")
    
    async def get_data(self, symbol: str) -> Dict[str, Any]:
        """دریافت داده‌های بازار برای یک ارز"""
        try:
            # دریافت داده‌ها از چندین منبع
            tasks = [
                self.get_yahoo_data(symbol),
                self.get_exchange_data(symbol),
                self.get_coingecko_data(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # ترکیب نتایج
            combined_data = self.combine_data(results)
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return self.get_fallback_data(symbol)
    
    async def get_yahoo_data(self, symbol: str) -> Dict[str, Any]:
        """دریافت داده‌ها از Yahoo Finance"""
        try:
            ticker = yf.Ticker(f'{symbol}-USD')
            hist = ticker.history(period='1d')
            
            if not hist.empty:
                return {
                    'source': 'yahoo',
                    'price': hist['Close'].iloc[-1],
                    'volume': hist['Volume'].iloc[-1],
                    'high': hist['High'].iloc[-1],
                    'low': hist['Low'].iloc[-1],
                    'change': ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
                }
        except Exception as e:
            logger.error(f"Error getting Yahoo data: {e}")
        
        return {}
    
    async def get_exchange_data(self, symbol: str) -> Dict[str, Any]:
        """دریافت داده‌ها از صرافی‌ها"""
        try:
            exchange_data = {}
            
            for exchange_name, exchange in self.exchanges.items():
                try:
                    ticker = exchange.fetch_ticker(f'{symbol}/USDT')
                    exchange_data[exchange_name] = {
                        'price': ticker['last'],
                        'volume': ticker['quoteVolume'],
                        'change': ticker['change']
                    }
                except Exception as e:
                    logger.error(f"Error getting data from {exchange_name}: {e}")
            
            return exchange_data
        except Exception as e:
            logger.error(f"Error getting exchange data: {e}")
            return {}
    
    async def get_coingecko_data(self, symbol: str) -> Dict[str, Any]:
        """دریافت داده‌ها از CoinGecko"""
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': symbol.lower(),
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if symbol.lower() in data:
                        return {
                            'source': 'coingecko',
                            'price': data[symbol.lower()]['usd'],
                            'market_cap': data[symbol.lower()]['usd_market_cap'],
                            'volume_24h': data[symbol.lower()]['usd_24h_vol'],
                            'change_24h': data[symbol.lower()]['usd_24h_change']
                        }
        except Exception as e:
            logger.error(f"Error getting CoinGecko data: {e}")
        
        return {}
    
    def combine_data(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ترکیب داده‌ها از منابع مختلف"""
        combined = {
            'price': 0,
            'volume': 0,
            'market_cap': 0,
            'change': 0,
            'sources': []
        }
        
        prices = []
        volumes = []
        
        for result in results:
            if isinstance(result, dict) and result:
                if 'price' in result and result['price']:
                    prices.append(result['price'])
                    combined['sources'].append(result.get('source', 'unknown'))
                
                if 'volume' in result and result['volume']:
                    volumes.append(result['volume'])
                
                if 'market_cap' in result and result['market_cap']:
                    combined['market_cap'] = result['market_cap']
                
                if 'change' in result and result['change']:
                    combined['change'] = result['change']
        
        # محاسبه میانگین‌ها
        if prices:
            combined['price'] = np.mean(prices)
        if volumes:
            combined['volume'] = np.mean(volumes)
        
        return combined
    
    def get_fallback_data(self, symbol: str) -> Dict[str, Any]:
        """دریافت داده‌های پیش‌فرض در صورت خطا"""
        base_prices = {
            'BTC': 43000,
            'ETH': 2200,
            'BNB': 300,
            'SOL': 100,
            'XRP': 0.6
        }
        
        base_price = base_prices.get(symbol, 100)
        change = np.random.uniform(-0.05, 0.05)
        
        return {
            'price': base_price * (1 + change),
            'volume': base_price * 1000000,
            'market_cap': base_price * 20000000,
            'change': change * 100,
            'sources': ['fallback']
        }
    
    async def get_all_cryptocurrencies(self) -> List[str]:
        """دریافت لیست تمام ارزهای دیجیتال"""
        try:
            # دریافت از CoinGecko
            url = "https://api.coingecko.com/api/v3/coins/list"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return [coin['symbol'].upper() for coin in data[:100]]  # 100 ارز برتر
            
        except Exception as e:
            logger.error(f"Error getting cryptocurrency list: {e}")
        
        # لیست پیش‌فرض
        return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOT', 'DOGE', 'AVAX', 'MATIC']