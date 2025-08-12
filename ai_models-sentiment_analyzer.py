import asyncio
import aiohttp
import logging
from typing import Dict, Any, List
from collections import Counter
import re

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        """مقداردهی اولیه تحلیلگر احساسات"""
        self.positive_keywords = [
            'bullish', 'buy', 'long', 'profit', 'gain', 'growth', 'increase', 'rise',
            'surge', 'rally', 'breakout', 'support', 'resistance', 'momentum', 'uptrend'
        ]
        
        self.negative_keywords = [
            'bearish', 'sell', 'short', 'loss', 'decline', 'decrease', 'fall', 'drop',
            'crash', 'dump', 'breakdown', 'pressure', 'downtrend', 'resistance'
        ]
        
        logger.info("Sentiment analyzer initialized")
    
    async def analyze(self, symbol: str) -> Dict[str, Any]:
        """تحلیل احساسات بازار"""
        try:
            # دریافت اخبار
            news = await self.get_news(symbol)
            
            if not news:
                return {
                    'sentiment_score': 0.0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0,
                    'topics': []
                }
            
            # تحلیل احساسات اخبار
            sentiment_scores = []
            topics = []
            
            for news_item in news:
                text = f"{news_item.get('title', '')} {news_item.get('content', '')}"
                
                # محاسبه امتیاز احساسات
                score = self.calculate_sentiment_score(text)
                sentiment_scores.append(score)
                
                # استخراج موضوعات
                item_topics = self.extract_topics(text)
                topics.extend(item_topics)
            
            # محاسبه آمار احساسات
            positive_count = len([s for s in sentiment_scores if s > 0.2])
            negative_count = len([s for s in sentiment_scores if s < -0.2])
            neutral_count = len(sentiment_scores) - positive_count - negative_count
            
            # محاسبه امتیاز میانگین
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            # شمارش موضوعات
            topic_counts = Counter(topics)
            top_topics = [topic for topic, count in topic_counts.most_common(5)]
            
            return {
                'sentiment_score': avg_sentiment,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'topics': top_topics
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis for {symbol}: {e}")
            return {
                'sentiment_score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'topics': []
            }
    
    async def get_news(self, symbol: str) -> List[Dict[str, Any]]:
        """دریافت اخبار"""
        try:
            news = []
            
            # دریافت اخبار از چند منبع
            sources = [
                self.get_cryptopanic_news,
                self.get_cryptocompare_news
            ]
            
            for source_func in sources:
                try:
                    source_news = await source_func(symbol)
                    news.extend(source_news)
                except Exception as e:
                    logger.error(f"Error getting news from source: {e}")
            
            return news[:10]  # حداکثر 10 خبر
            
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {e}")
            return []
    
    async def get_cryptopanic_news(self, symbol: str) -> List[Dict[str, Any]]:
        """دریافت اخبار از CryptoPanic"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://cryptopanic.com/api/v1/posts/"
                params = {
                    'auth_token': 'YOUR_CRYPTOPANIC_API_KEY',  # باید در متغیرهای محیطی تنظیم شود
                    'kind': 'news',
                    'filter': 'hot',
                    'currencies': symbol.lower()
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            {
                                'title': item['title'],
                                'content': item.get('metadata', {}).get('description', ''),
                                'url': item['url']
                            }
                            for item in data['results']
                        ]
            
        except Exception as e:
            logger.error(f"Error getting CryptoPanic news: {e}")
        
        return []
    
    async def get_cryptocompare_news(self, symbol: str) -> List[Dict[str, Any]]:
        """دریافت اخبار از CryptoCompare"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://min-api.cryptocompare.com/data/v2/news/"
                params = {
                    'categories': symbol.lower(),
                    'excludeCategories': 'Sponsored'
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            {
                                'title': item['title'],
                                'content': item.get('body', ''),
                                'url': item['url']
                            }
                            for item in data['Data']
                        ]
            
        except Exception as e:
            logger.error(f"Error getting CryptoCompare news: {e}")
        
        return []
    
    def calculate_sentiment_score(self, text: str) -> float:
        """محاسبه امتیاز احساسات"""
        try:
            text_lower = text.lower()
            
            positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
            negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
            
            total_keywords = positive_count + negative_count
            if total_keywords == 0:
                return 0.0
            
            return (positive_count - negative_count) / total_keywords
            
        except Exception as e:
            logger.error(f"Error calculating sentiment score: {e}")
            return 0.0
    
    def extract_topics(self, text: str) -> List[str]:
        """استخراج موضوعات از متن"""
        try:
            text_lower = text.lower()
            
            topics = []
            
            # موضوعات کلیدی
            topic_keywords = {
                'regulation': ['regulation', 'law', 'government', 'policy', 'legal'],
                'technology': ['technology', 'blockchain', 'innovation', 'upgrade', 'development'],
                'market': ['market', 'trading', 'price', 'volume', 'liquidity'],
                'security': ['security', 'hack', 'breach', 'safety', 'protection'],
                'adoption': ['adoption', 'integration', 'partnership', 'implementation']
            }
            
            for topic, keywords in topic_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    topics.append(topic)
            
            return topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []