import numpy as np
import pandas as pd
import talib
import logging
from typing import Dict, Any
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    def __init__(self):
        """مقداردهی اولیه تحلیلگر تکنیکال"""
        logger.info("Technical analyzer initialized")
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """انجام تحلیل تکنیکال"""
        try:
            if data.empty:
                return {}
            
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            
            # محاسبه شاخص‌ها
            rsi = talib.RSI(close, timeperiod=14)
            macd, macdsignal, macdhist = talib.MACD(close)
            
            # شناسایی الگوهای شمعی
            patterns = self.detect_candlestick_patterns(data)
            
            # شناسایی سطوح حمایت و مقاومت
            support_resistance = self.find_support_resistance(data)
            
            return {
                'rsi': rsi[-1] if not np.isnan(rsi[-1]) else 50,
                'macd': {
                    'macd': macd[-1] if not np.isnan(macd[-1]) else 0,
                    'signal': macdsignal[-1] if not np.isnan(macdsignal[-1]) else 0,
                    'histogram': macdhist[-1] if not np.isnan(macdhist[-1]) else 0
                },
                'patterns': patterns,
                'support_resistance': support_resistance
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {}
    
    def detect_candlestick_patterns(self, data: pd.DataFrame) -> Dict[str, bool]:
        """شناسایی الگوهای شمعی"""
        try:
            if len(data) < 2:
                return {}
            
            open_prices = data['Open'].values
            close_prices = data['Close'].values
            high_prices = data['High'].values
            low_prices = data['Low'].values
            
            patterns = {}
            
            # الگوی پوشاننده
            if len(data) >= 2:
                body1 = abs(close_prices[-2] - open_prices[-2])
                body2 = abs(close_prices[-1] - open_prices[-1])
                
                if (close_prices[-2] > open_prices[-2] and  # شمع اول صعودی
                    close_prices[-1] < open_prices[-1] and  # شمع دوم نزولی
                    close_prices[-1] < open_prices[-2] and  # بسته شدن زیر شمع اول
                    close_prices[-2] < open_prices[-1] and  # باز شدن بالای شمع اول
                    body2 > body1):  # بدنه بزرگتر
                    patterns['bearish_engulfing'] = True
                
                if (close_prices[-2] < open_prices[-2] and  # شمع اول نزولی
                    close_prices[-1] > open_prices[-1] and  # شمع دوم صعودی
                    close_prices[-1] > open_prices[-2] and  # بسته شدن بالای شمع اول
                    close_prices[-2] > open_prices[-1] and  # باز شدن زیر شمع اول
                    body2 > body1):  # بدنه بزرگتر
                    patterns['bullish_engulfing'] = True
            
            # الگوی چکش
            if len(data) >= 1:
                body = abs(close_prices[-1] - open_prices[-1])
                lower_shadow = min(open_prices[-1], close_prices[-1]) - low_prices[-1]
                upper_shadow = high_prices[-1] - max(open_prices[-1], close_prices[-1])
                
                if (lower_shadow > 2 * body and  # سایه پایینی بلند
                    upper_shadow < 0.1 * body):  # سایه بالایی کوتاه
                    patterns['hammer'] = True
                
                if (upper_shadow > 2 * body and  # سایه بالایی بلند
                    lower_shadow < 0.1 * body):  # سایه پایینی کوتاه
                    patterns['shooting_star'] = True
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}")
            return {}
    
    def find_support_resistance(self, data: pd.DataFrame) -> Dict[str, list]:
        """پیدا کردن سطوح حمایت و مقاومت"""
        try:
            if len(data) < 20:
                return {'support': [], 'resistance': []}
            
            highs = data['High'].values
            lows = data['Low'].values
            
            # پیدا کردن قله‌ها و دره‌ها
            peaks, _ = find_peaks(highs, distance=5)
            troughs, _ = find_peaks(-lows, distance=5)
            
            support_levels = lows[troughs].tolist()
            resistance_levels = highs[peaks].tolist()
            
            return {
                'support': support_levels[-5:] if support_levels else [],  # 5 سطح آخر
                'resistance': resistance_levels[-5:] if resistance_levels else []  # 5 سطح آخر
            }
            
        except Exception as e:
            logger.error(f"Error finding support/resistance: {e}")
            return {'support': [], 'resistance': []}