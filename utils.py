import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import talib

logger = logging.getLogger(__name__)

class AnalysisUtils:
    """توابع کمکی برای تحلیل‌ها"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """محاسبه شاخص RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def interpret_rsi(rsi: float) -> str:
        """تفسیر شاخص RSI"""
        if rsi > 70:
            return "اشباع خرید"
        elif rsi < 30:
            return "اشباع فروش"
        else:
            return "خنثی"
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """محاسبه شاخص MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def interpret_macd(macd: float, signal: float) -> str:
        """تفسیر شاخص MACD"""
        if macd > signal:
            return "صعودی"
        elif macd < signal:
            return "نزولی"
        else:
            return "خنثی"
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict:
        """محاسبه بولینگر باند"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    @staticmethod
    def interpret_bollinger_bands(price: float, upper: float, lower: float) -> str:
        """تفسیر بولینگر باند"""
        if price > upper:
            return "بالاتر از باند بالایی"
        elif price < lower:
            return "پایین‌تر از باند پایینی"
        else:
            return "درون باندها"
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """محاسبه میانگین دامنه واقعی (ATR)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1]
    
    @staticmethod
    def identify_pivot_points(df: pd.DataFrame) -> List[Dict]:
        """شناسایی نقاط چرخش"""
        pivot_points = []
        
        for i in range(1, len(df) - 1):
            # نقطه چرخش بالایی
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i+1]):
                pivot_points.append({
                    'price': df['high'].iloc[i],
                    'type': 'resistance',
                    'index': i
                })
            
            # نقطه چرخش پایینی
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i+1]):
                pivot_points.append({
                    'price': df['low'].iloc[i],
                    'type': 'support',
                    'index': i
                })
        
        return pivot_points
    
    @staticmethod
    def calculate_zone_strength(df: pd.DataFrame, index: int, zone_type: str) -> float:
        """محاسبه قدرت ناحیه عرضه یا تقاضا"""
        price = df['low'].iloc[index] if zone_type == 'demand' else df['high'].iloc[index]
        tolerance = price * 0.01  # 1% tolerance
        
        reactions = 0
        for i in range(len(df)):
            if i == index:
                continue
            
            if zone_type == 'demand':
                if abs(df['low'].iloc[i] - price) < tolerance:
                    reactions += 1
            else:
                if abs(df['high'].iloc[i] - price) < tolerance:
                    reactions += 1
        
        volume_factor = df['volume'].iloc[index] / df['volume'].mean()
        strength = reactions * volume_factor
        
        return min(strength, 10)
    
    @staticmethod
    def identify_order_blocks(df: pd.DataFrame) -> List[Dict]:
        """شناسایی Order Blocks"""
        order_blocks = []
        
        for i in range(2, len(df) - 2):
            # Order Block صعودی
            if (df['close'].iloc[i] > df['open'].iloc[i] and
                df['close'].iloc[i-1] < df['open'].iloc[i-1] and
                df['close'].iloc[i-2] < df['open'].iloc[i-2] and
                df['volume'].iloc[i] > df['volume'].iloc[i-1] * 1.5):
                
                order_blocks.append({
                    'price': df['low'].iloc[i],
                    'type': 'bullish',
                    'strength': df['volume'].iloc[i] / df['volume'].mean(),
                    'timeframe': '1d'
                })
            
            # Order Block نزولی
            if (df['close'].iloc[i] < df['open'].iloc[i] and
                df['close'].iloc[i-1] > df['open'].iloc[i-1] and
                df['close'].iloc[i-2] > df['open'].iloc[i-2] and
                df['volume'].iloc[i] > df['volume'].iloc[i-1] * 1.5):
                
                order_blocks.append({
                    'price': df['high'].iloc[i],
                    'type': 'bearish',
                    'strength': df['volume'].iloc[i] / df['volume'].mean(),
                    'timeframe': '1d'
                })
        
        return order_blocks
    
    @staticmethod
    def find_support_resistance(df: pd.DataFrame) -> Tuple[float, float]:
        """پیدا کردن حمایت و مقاومت"""
        if len(df) < 20:
            return 0, 0
        
        highs = df['high'].rolling(window=5).max()
        lows = df['low'].rolling(window=5).min()
        
        pivot_highs = []
        pivot_lows = []
        
        for i in range(2, len(df)-2):
            if df['high'].iloc[i] == highs.iloc[i] and df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1]:
                pivot_highs.append(df['high'].iloc[i])
            
            if df['low'].iloc[i] == lows.iloc[i] and df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1]:
                pivot_lows.append(df['low'].iloc[i])
        
        current_price = df['close'].iloc[-1]
        
        if pivot_lows:
            support = max([low for low in pivot_lows if low < current_price], default=current_price * 0.95)
        else:
            support = current_price * 0.95
        
        if pivot_highs:
            resistance = min([high for high in pivot_highs if high > current_price], default=current_price * 1.05)
        else:
            resistance = current_price * 1.05
        
        return support, resistance
    
    @staticmethod
    def determine_trend(df: pd.DataFrame) -> str:
        """تعیین روند قیمت"""
        if len(df) < 50:
            return 'neutral'
        
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        current_price = df['close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            return 'strong_bullish'
        elif current_price > sma_20 and sma_20 > sma_50:
            return 'bullish'
        elif current_price < sma_20 < sma_50:
            return 'strong_bearish'
        elif current_price < sma_20 and sma_20 < sma_50:
            return 'bearish'
        else:
            return 'neutral'
    
    @staticmethod
    def interpret_volume(current_volume: float, avg_volume: float) -> str:
        """تفسیر حجم معاملات"""
        if current_volume > avg_volume * 1.5:
            return "حجم بالا"
        elif current_volume < avg_volume * 0.5:
            return "حجم پایین"
        else:
            return "حجم عادی"
    
    @staticmethod
    def interpret_moving_averages(sma_20: float, sma_50: float) -> str:
        """تفسیر میانگین‌های متحرک"""
        if sma_20 > sma_50:
            return "صعودی"
        elif sma_20 < sma_50:
            return "نزولی"
        else:
            return "خنثی"
    
    @staticmethod
    def interpret_market_position(current_price: float, nearest_supply: float, nearest_demand: float) -> str:
        """تفسیر موقعیت فعلی قیمت در ساختار بازار"""
        if nearest_supply and nearest_demand:
            distance_to_supply = (nearest_supply - current_price) / current_price
            distance_to_demand = (current_price - nearest_demand) / current_price
            
            if distance_to_supply < distance_to_demand:
                return "نزدیک به مقاومت"
            else:
                return "نزدیک به حمایت"
        elif nearest_supply:
            return "نزدیک به مقاومت"
        elif nearest_demand:
            return "نزدیک به حمایت"
        else:
            return "در محدوده خنثی"
    
    @staticmethod
    def calculate_fractal_dimension(df: pd.DataFrame) -> float:
        """محاسبه بعد فرکتال"""
        prices = df['close'].values
        n = len(prices)
        
        if n < 10:
            return 1.0
        
        scales = np.logspace(0.1, 1, num=10)
        counts = []
        
        for scale in scales:
            boxes = np.floor(np.arange(n) / scale).astype(int)
            box_counts = np.bincount(boxes)
            counts.append(len(box_counts))
        
        coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
        return -coeffs[0]
    
    @staticmethod
    def calculate_entropy(df: pd.DataFrame) -> float:
        """محاسبه آنتروپی"""
        prices = df['close'].values
        n = len(prices)
        
        if n < 10:
            return 0.0
        
        diffs = np.diff(prices)
        hist, _ = np.histogram(diffs, bins=20)
        hist = hist / np.sum(hist)
        
        entropy = 0
        for p in hist:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    @staticmethod
    def calculate_lyapunov_exponent(df: pd.DataFrame) -> float:
        """محاسبه نمای لیاپانوف"""
        prices = df['close'].values
        n = len(prices)
        
        if n < 20:
            return 0.0
        
        m = 3  # بعد جاسازی
        tau = 1  # تأخیر زمانی
        
        embedded = np.zeros((n - (m-1)*tau, m))
        for i in range(m):
            embedded[:, i] = prices[i*tau : i*tau + len(embedded)]
        
        max_iter = min(100, len(embedded) - 10)
        lyapunov_sum = 0
        
        for i in range(max_iter):
            distances = np.sqrt(np.sum((embedded - embedded[i])**2, axis=1))
            distances[i] = np.inf
            
            nearest_idx = np.argmin(distances)
            initial_distance = distances[nearest_idx]
            
            if initial_distance == 0:
                continue
            
            j = min(i + 10, len(embedded) - 1)
            final_distance = np.sqrt(np.sum((embedded[j] - embedded[nearest_idx + (j-i)])**2))
            
            if final_distance > 0:
                lyapunov_sum += np.log(final_distance / initial_distance)
        
        if max_iter > 0:
            return lyapunov_sum / (max_iter * 10)
        return 0.0
    
    @staticmethod
    def identify_elliott_wave_patterns(df: pd.DataFrame) -> List[str]:
        """شناسایی الگوهای امواج الیوت"""
        patterns = []
        
        # الگوی ایمپالس
        if AnalysisUtils._is_impulse_pattern(df):
            patterns.append("ایمپالس")
        
        # الگوی اصلاحی
        if AnalysisUtils._is_corrective_pattern(df):
            patterns.append("اصلاحی")
        
        # الگوی مثلث
        if AnalysisUtils._is_triangle_pattern(df):
            patterns.append("مثلث")
        
        # الگوی مسطح
        if AnalysisUtils._is_flat_pattern(df):
            patterns.append("مسطح")
        
        return patterns
    
    @staticmethod
    def _is_impulse_pattern(df: pd.DataFrame) -> bool:
        """بررسی الگوی ایمپالس"""
        if len(df) < 20:
            return False
        
        waves = AnalysisUtils._identify_waves(df)
        if len(waves) >= 5:
            if (waves[2]['height'] > waves[0]['height'] and 
                waves[2]['height'] > waves[4]['height']):
                return True
        return False
    
    @staticmethod
    def _is_corrective_pattern(df: pd.DataFrame) -> bool:
        """بررسی الگوی اصلاحی"""
        if len(df) < 15:
            return False
        
        waves = AnalysisUtils._identify_waves(df)
        if len(waves) >= 3:
            if waves[1]['height'] < waves[0]['height']:
                return True
        return False
    
    @staticmethod
    def _is_triangle_pattern(df: pd.DataFrame) -> bool:
        """بررسی الگوی مثلث"""
        if len(df) < 20:
            return False
        
        highs = df['high'].rolling(window=5).max().dropna()
        lows = df['low'].rolling(window=5).min().dropna()
        
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        
        if high_slope < 0 and low_slope > 0:
            return True
        return False
    
    @staticmethod
    def _is_flat_pattern(df: pd.DataFrame) -> bool:
        """بررسی الگوی مسطح"""
        if len(df) < 15:
            return False
        
        price_range = (df['high'].max() - df['low'].min()) / df['close'].mean()
        if price_range < 0.05:
            return True
        return False
    
    @staticmethod
    def _identify_waves(df: pd.DataFrame) -> List[Dict]:
        """شناسایی امواج قیمت"""
        pivot_points = AnalysisUtils.identify_pivot_points(df)
        
        if len(pivot_points) < 2:
            return []
        
        pivot_points.sort(key=lambda x: x['index'])
        
        waves = []
        for i in range(len(pivot_points) - 1):
            start_point = pivot_points[i]
            end_point = pivot_points[i + 1]
            
            wave = {
                'start_price': start_point['price'],
                'end_price': end_point['price'],
                'start_index': start_point['index'],
                'end_index': end_point['index'],
                'type': 'bullish' if end_point['price'] > start_point['price'] else 'bearish',
                'height': abs(end_point['price'] - start_point['price'])
            }
            
            waves.append(wave)
        
        return waves

class DataUtils:
    """توابع کمکی برای مدیریت داده‌ها"""
    
    @staticmethod
    def extract_price_data(data: Dict) -> List[Dict]:
        """استخراج داده‌های قیمت"""
        price_data = []
        
        # استخراج داده‌ها از CoinGecko
        if 'coingecko' in data and 'market_data' in data['coingecko']:
            market_data = data['coingecko']['market_data']
            if 'sparkline_7d' in market_data and 'price' in market_data['sparkline_7d']:
                prices = market_data['sparkline_7d']['price']
                for i, price in enumerate(prices):
                    price_data.append({
                        'timestamp': i,
                        'close': price,
                        'high': price * 1.01,
                        'low': price * 0.99,
                        'volume': market_data.get('total_volume', {}).get('usd', 0) / len(prices)
                    })
        
        # استخراج داده‌ها از CryptoCompare
        if 'cryptocompare' in data and 'Data' in data['cryptocompare'] and 'Data' in data['cryptocompare']['Data']:
            for item in data['cryptocompare']['Data']['Data']:
                price_data.append({
                    'timestamp': item['time'],
                    'close': item['close'],
                    'high': item['high'],
                    'low': item['low'],
                    'volume': item['volumeto']
                })
        
        return price_data
    
    @staticmethod
    def extract_news(data: Dict) -> List[Dict]:
        """استخراج اخبار"""
        news_data = []
        
        if 'cryptopanic' in data and 'results' in data['cryptopanic']:
            for item in data['cryptopanic']['results']:
                news_data.append({
                    'title': item.get('title', ''),
                    'description': item.get('metadata', {}).get('description', ''),
                    'url': item.get('url', ''),
                    'published_at': item.get('published_at', ''),
                    'source': item.get('source', {}).get('title', ''),
                    'keywords': item.get('metadata', {}).get('keywords', []),
                    'impact': item.get('metadata', {}).get('impact', 0.5)
                })
        
        return news_data
    
    @staticmethod
    def extract_market_data(data: Dict) -> Dict:
        """استخراج داده‌های بازار"""
        market_data = {}
        
        # استخراج داده‌ها از CoinGecko
        if 'coingecko' in data and 'market_data' in data['coingecko']:
            cg_data = data['coingecko']['market_data']
            market_data = {
                'price': cg_data.get('current_price', {}).get('usd', 0),
                'price_change_24h': cg_data.get('price_change_percentage_24h', 0),
                'volume_24h': cg_data.get('total_volume', {}).get('usd', 0),
                'market_cap': cg_data.get('market_cap', {}).get('usd', 0),
                'circulating_supply': cg_data.get('circulating_supply', 0),
                'total_supply': cg_data.get('total_supply', 0),
                'all_time_high': cg_data.get('ath', {}).get('usd', 0),
                'all_time_low': cg_data.get('atl', {}).get('usd', 0),
                'price_change_percentage_7d': cg_data.get('price_change_percentage_7d', 0),
                'price_change_percentage_14d': cg_data.get('price_change_percentage_14d', 0),
                'price_change_percentage_30d': cg_data.get('price_change_percentage_30d', 0),
                'price_change_percentage_60d': cg_data.get('price_change_percentage_60d', 0),
                'price_change_percentage_200d': cg_data.get('price_change_percentage_200d', 0),
                'price_change_percentage_1y': cg_data.get('price_change_percentage_1y', 0),
            }
        
        # استخراج داده‌ها از CoinMarketCap
        if 'coinmarketcap' in data and 'quote' in data['coinmarketcap']:
            cmc_data = data['coinmarketcap']['quote']['USD']
            market_data.update({
                'price': cmc_data.get('price', market_data.get('price', 0)),
                'volume_24h': cmc_data.get('volume_24h', market_data.get('volume_24h', 0)),
                'market_cap': cmc_data.get('market_cap', market_data.get('market_cap', 0)),
                'percent_change_1h': cmc_data.get('percent_change_1h', 0),
                'percent_change_24h': cmc_data.get('percent_change_24h', market_data.get('price_change_24h', 0)),
                'percent_change_7d': cmc_data.get('percent_change_7d', market_data.get('price_change_percentage_7d', 0)),
                'percent_change_30d': cmc_data.get('percent_change_30d', market_data.get('price_change_percentage_30d', 0)),
                'market_cap_dominance': cmc_data.get('market_cap_dominance', 0),
            })
        
        return market_data

class ValidationUtils:
    """توابع کمکی برای اعتبارسنجی"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """اعتبارسنجی نماد ارز"""
        if not symbol or len(symbol) < 2 or len(symbol) > 10:
            return False
        
        # بررسی اینکه نماد فقط حروف باشد
        return symbol.isalpha()
    
    @staticmethod
    def validate_price(price: float) -> bool:
        """اعتبارسنجی قیمت"""
        return price > 0 and price < 1000000  # حداکثر 1 میلیون دلار
    
    @staticmethod
    def validate_confidence(confidence: float) -> bool:
        """اعتبارسنجی اطمینان"""
        return 0 <= confidence <= 1
    
    @staticmethod
    def validate_timestamp(timestamp: str) -> bool:
        """اعتبارسنجی زمان"""
        try:
            datetime.fromisoformat(timestamp)
            return True
        except:
            return False
    
    @staticmethod
    def validate_analysis_data(analysis: Dict) -> bool:
        """اعتبارسنجی داده‌های تحلیل"""
        required_fields = ['symbol', 'timestamp', 'signal', 'confidence']
        
        for field in required_fields:
            if field not in analysis:
                return False
        
        return (
            ValidationUtils.validate_symbol(analysis['symbol']) and
            ValidationUtils.validate_timestamp(analysis['timestamp']) and
            ValidationUtils.validate_confidence(analysis['confidence'])
        )