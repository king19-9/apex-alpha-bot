import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self):
        """مقداردهی اولیه مدیریت ریسک"""
        logger.info("Risk manager initialized")
    
    def calculate_position_size(self, account_balance: float, risk_per_trade: float, stop_loss: float) -> float:
        """محاسبه حجم معامله"""
        try:
            risk_amount = account_balance * risk_per_trade
            position_size = risk_amount / stop_loss
            return min(position_size, account_balance * 0.1)  # حداکثر 10% از حساب
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def calculate_stop_loss(self, current_price: float, atr: float, multiplier: float = 2.0) -> float:
        """محاسبه حد ضرر"""
        try:
            return current_price - (atr * multiplier)
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return current_price * 0.95  # 5% حد ضرر پیش‌فرض
    
    def calculate_take_profit(self, current_price: float, stop_loss: float, risk_reward_ratio: float = 2.0) -> float:
        """محاسبه حد سود"""
        try:
            risk_amount = current_price - stop_loss
            return current_price + (risk_amount * risk_reward_ratio)
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            return current_price * 1.1  # 10% حد سود پیش‌فرض
    
    def analyze_risk(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """تحلیل ریسک"""
        try:
            current_price = analysis.get('market_data', {}).get('price', 0)
            atr = analysis.get('volatility', {}).get('atr', 0)
            
            if current_price == 0:
                return {}
            
            stop_loss = self.calculate_stop_loss(current_price, atr)
            take_profit = self.calculate_take_profit(current_price, stop_loss)
            
            risk_reward_ratio = (take_profit - current_price) / (current_price - stop_loss)
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': risk_reward_ratio,
                'position_size': 0.02  # 2% از حساب
            }
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            return {}