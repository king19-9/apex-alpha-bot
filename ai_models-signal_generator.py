import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SignalGenerator:
    def __init__(self):
        """مقداردهی اولیه تولیدکننده سیگنال"""
        logger.info("Signal generator initialized")
    
    async def generate_signal(self, symbol: str, technical_analysis: Dict[str, Any], 
                            sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """تولید سیگنال معاملاتی"""
        try:
            # محاسبه امتیاز سیگنال
            signal_score = self.calculate_signal_score(technical_analysis, sentiment_analysis)
            
            # تعیین نوع سیگنال
            if signal_score > 0.7:
                signal_type = 'BUY'
            elif signal_score < 0.3:
                signal_type = 'SELL'
            else:
                signal_type = 'HOLD'
            
            # محاسبه اطمینان
            confidence = abs(signal_score - 0.5) * 2  # تبدیل به 0-1
            
            return {
                'symbol': symbol,
                'signal': signal_type,
                'confidence': confidence,
                'score': signal_score,
                'timestamp': str(np.datetime64('now'))
            }
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {
                'symbol': symbol,
                'signal': 'HOLD',
                'confidence': 0.5,
                'score': 0.5,
                'timestamp': str(np.datetime64('now'))
            }
    
    def calculate_signal_score(self, technical_analysis: Dict[str, Any], 
                              sentiment_analysis: Dict[str, Any]) -> float:
        """محاسبه امتیاز سیگنال"""
        try:
            score = 0.5  # امتیاز پایه
            
            # تحلیل تکنیکال (وزن 70%)
            if technical_analysis:
                score += self.analyze_technical_signals(technical_analysis) * 0.7
            
            # تحلیل احساسات (وزن 30%)
            if sentiment_analysis:
                score += self.analyze_sentiment_signals(sentiment_analysis) * 0.3
            
            # نرمال‌سازی امتیاز
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating signal score: {e}")
            return 0.5
    
    def analyze_technical_signals(self, technical_analysis: Dict[str, Any]) -> float:
        """تحلیل سیگنال‌های تکنیکال"""
        try:
            score = 0.5
            
            # RSI
            rsi = technical_analysis.get('rsi', 50)
            if rsi < 30:  # اشباع فروش
                score += 0.3
            elif rsi > 70:  # اشباع خرید
                score -= 0.3
            
            # MACD
            macd = technical_analysis.get('macd', {})
            if macd:
                macd_value = macd.get('macd', 0)
                macd_signal = macd.get('signal', 0)
                if macd_value > macd_signal:  # سیگنال خرید
                    score += 0.2
                elif macd_value < macd_signal:  # سیگنال فروش
                    score -= 0.2
            
            # الگوهای شمعی
            patterns = technical_analysis.get('patterns', {})
            if patterns.get('bullish_engulfing'):
                score += 0.2
            elif patterns.get('bearish_engulfing'):
                score -= 0.2
            
            if patterns.get('hammer'):
                score += 0.1
            elif patterns.get('shooting_star'):
                score -= 0.1
            
            return score
            
        except Exception as e:
            logger.error(f"Error analyzing technical signals: {e}")
            return 0.5
    
    def analyze_sentiment_signals(self, sentiment_analysis: Dict[str, Any]) -> float:
        """تحلیل سیگنال‌های احساسات"""
        try:
            score = 0.5
            
            sentiment_score = sentiment_analysis.get('sentiment_score', 0)
            
            # تبدیل امتیاز احساسات به تأثیر روی سیگنال
            score += sentiment_score * 0.5
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment signals: {e}")
            return 0.5