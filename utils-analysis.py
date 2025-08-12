import asyncio
import logging
import numpy as np
import pandas as pd
import talib
from typing import Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AnalysisEngine:
    def __init__(self):
        """مقداردهی اولیه موتور تحلیل"""
        logger.info("Analysis engine initialized")
    
    async def perform_comprehensive_analysis(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """انجام تحلیل جامع"""
        try:
            # دریافت داده‌های تاریخی
            historical_data = await self.get_historical_data(symbol)
            
            if historical_data.empty:
                return self.get_fallback_analysis(symbol, market_data)
            
            # تحلیل تکنیکال
            technical_analysis = self.perform_technical_analysis(historical_data)
            
            # تحلیل روند
            trend_analysis = self.analyze_trend(historical_data)
            
            # تحلیل حجم
            volume_analysis = self.analyze_volume(historical_data)
            
            # تحلیل نوسانات
            volatility_analysis = self.analyze_volatility(historical_data)
            
            return {
                'symbol': symbol,
                'market_data': market_data,
                'technical': technical_analysis,
                'trend': trend_analysis,
                'volume': volume_analysis,
                'volatility': volatility_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
            return self.get_fallback_analysis(symbol, market_data)
    
    async def get_historical_data(self, symbol: str, period: str = '60d') -> pd.DataFrame:
        """دریافت داده‌های تاریخی"""
        try:
            ticker = yf.Ticker(f'{symbol}-USD')
            hist = ticker.history(period=period)
            return hist
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def perform_technical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """انجام تحلیل تکنیکال"""
        try:
            if data.empty:
                return {}
            
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            volume = data['Volume'].values
            
            # محاسبه شاخص‌ها
            rsi = talib.RSI(close, timeperiod=14)
            macd, macdsignal, macdhist = talib.MACD(close)
            upper, middle, lower = talib.BBANDS(close)
            
            return {
                'rsi': rsi[-1] if not np.isnan(rsi[-1]) else 50,
                'macd': {
                    'macd': macd[-1] if not np.isnan(macd[-1]) else 0,
                    'signal': macdsignal[-1] if not np.isnan(macdsignal[-1]) else 0,
                    'histogram': macdhist[-1] if not np.isnan(macdhist[-1]) else 0
                },
                'bollinger': {
                    'upper': upper[-1] if not np.isnan(upper[-1]) else 0,
                    'middle': middle[-1] if not np.isnan(middle[-1]) else 0,
                    'lower': lower[-1] if not np.isnan(lower[-1]) else 0
                },
                'sma_20': talib.SMA(close, timeperiod=20)[-1] if not np.isnan(talib.SMA(close, timeperiod=20)[-1]) else 0,
                'sma_50': talib.SMA(close, timeperiod=50)[-1] if not np.isnan(talib.SMA(close, timeperiod=50)[-1]) else 0
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {}
    
    def analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """تحلیل روند"""
        try:
            if data.empty:
                return {'trend': 'unknown', 'strength': 0}
            
            close = data['Close'].values
            sma20 = talib.SMA(close, timeperiod=20)
            sma50 = talib.SMA(close, timeperiod=50)
            
            if sma20[-1] > sma50[-1]:
                trend = 'uptrend'
            elif sma20[-1] < sma50[-1]:
                trend = 'downtrend'
            else:
                trend = 'sideways'
            
            strength = abs(sma20[-1] - sma50[-1]) / sma50[-1] if sma50[-1] != 0 else 0
            
            return {
                'trend': trend,
                'strength': strength
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {'trend': 'unknown', 'strength': 0}
    
    def analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """تحلیل حجم"""
        try:
            if data.empty:
                return {'volume_trend': 'unknown', 'volume_spike': False}
            
            volume = data['Volume'].values
            volume_ma = talib.SMA(volume, timeperiod=20)
            
            if volume_ma[-1] > volume_ma[-2]:
                volume_trend = 'increasing'
            elif volume_ma[-1] < volume_ma[-2]:
                volume_trend = 'decreasing'
            else:
                volume_trend = 'stable'
            
            volume_spike = volume[-1] > 2 * volume_ma[-1]
            
            return {
                'volume_trend': volume_trend,
                'volume_spike': volume_spike
            }
            
        except Exception as e:
            logger.error(f"Error in volume analysis: {e}")
            return {'volume_trend': 'unknown', 'volume_spike': False}
    
    def analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """تحلیل نوسانات"""
        try:
            if data.empty:
                return {'volatility': 0, 'atr': 0}
            
            high = data['High'].values
            low = data['Low'].values
            close = data['Close'].values
            
            atr = talib.ATR(high, low, close, timeperiod=14)
            returns = np.diff(close) / close[:-1]
            volatility = np.std(returns) * np.sqrt(252)
            
            return {
                'volatility': volatility,
                'atr': atr[-1] if not np.isnan(atr[-1]) else 0
            }
            
        except Exception as e:
            logger.error(f"Error in volatility analysis: {e}")
            return {'volatility': 0, 'atr': 0}
    
    def get_fallback_analysis(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """تحلیل پیش‌فرض در صورت خطا"""
        return {
            'symbol': symbol,
            'market_data': market_data,
            'technical': {
                'rsi': 50,
                'macd': {'macd': 0, 'signal': 0, 'histogram': 0},
                'bollinger': {'upper': 0, 'middle': 0, 'lower': 0},
                'sma_20': 0,
                'sma_50': 0
            },
            'trend': {'trend': 'unknown', 'strength': 0},
            'volume': {'volume_trend': 'unknown', 'volume_spike': False},
            'volatility': {'volatility': 0, 'atr': 0},
            'timestamp': datetime.now().isoformat()
        }