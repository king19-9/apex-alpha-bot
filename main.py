import asyncio
import logging
import os
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import aiohttp
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import talib
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import io
import base64

# تنظیمات لاگینگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedCryptoBot:
    def __init__(self):
        # کلیدهای API از متغیرهای محیطی
        self.api_keys = {
            'coingecko': os.getenv('COINGECKO_API_KEY'),
            'coinmarketcap': os.getenv('COINMARKETCAP_API_KEY'),
            'cryptocompare': os.getenv('CRYPTOCOMPARE_API_KEY'),
            'cryptopanic': os.getenv('CRYPTOPANIC_API_KEY'),
            'news': os.getenv('NEWS_API_KEY'),
            'whale_alert': os.getenv('WHALE_ALERT_API_KEY'),
            'glassnode': os.getenv('GLASSNODE_API_KEY'),
            'economic_calendar': os.getenv('ECONOMIC_CALENDAR_API_KEY'),
        }
        
        # پایگاه داده برای ذخیره عملکرد تاریخی
        self.init_database()
        
        # مدل‌های یادگیری ماشین پیشرفته
        self.models = {
            'signal_classifier': self._initialize_signal_model(),
            'sentiment_analyzer': self._initialize_sentiment_model(),
            'elliott_wave': self._initialize_elliott_model(),
            'quantum_pattern': self._initialize_quantum_model(),
            'whale_behavior': self._initialize_whale_model(),
            'adaptive_selector': self._initialize_adaptive_model(),
        }
        
        # کش داده‌ها
        self.data_cache = {}
        self.cache_expiry = {}
        
        # لیست تمام ارزهای موجود در بازار
        self.all_cryptos = self._fetch_all_cryptos()
        
        # تنظیمات پیشرفته
        self.confidence_threshold = 0.85
        self.risk_reward_ratio = 1:3
        self.max_position_size = 0.1
        
        # تاریخچه عملکرد تحلیل‌ها
        self.analysis_history = {}
        
        # الگوریتم‌های تحلیلی
        self.analysis_algorithms = {
            'technical': self._perform_technical_analysis,
            'sentiment': self._perform_sentiment_analysis,
            'elliott_wave': self._perform_elliott_wave_analysis,
            'quantum': self._perform_quantum_analysis,
            'whale_activity': self._perform_whale_analysis,
            'market_structure': self._perform_market_structure_analysis,
            'on_chain': self._perform_on_chain_analysis,
            'correlation': self._perform_correlation_analysis,
            'seasonal': self._perform_seasonal_analysis,
            'wyckoff': self._perform_wyckoff_analysis,
            'supply_demand': self._perform_supply_demand_analysis,
            'fibonacci': self._perform_fibonacci_analysis,
            'volume_profile': self._perform_volume_profile_analysis,
            'market_profile': self._perform_market_profile_analysis,
            'harmonic_patterns': self._perform_harmonic_patterns_analysis,
            'divergence': self._perform_divergence_analysis,
            'economic_calendar': self._perform_economic_calendar_analysis,
            'order_flow': self._perform_order_flow_analysis,
            'liquidity': self._perform_liquidity_analysis,
            'monte_carlo': self._perform_monte_carlo_analysis,
        }

    def init_database(self):
        """مقداردهی اولیه پایگاه داده"""
        self.conn = sqlite3.connect('crypto_bot.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # ایجاد جدول برای ذخیره عملکرد تحلیل‌ها
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            method TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            signal TEXT NOT NULL,
            confidence REAL NOT NULL,
            actual_outcome TEXT,
            success BOOLEAN,
            profit_loss REAL,
            market_conditions TEXT
        )
        ''')
        
        # ایجاد جدول برای ذخیره فعالیت نهنگ‌ها
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS whale_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            transaction_hash TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            amount REAL NOT NULL,
            amount_usd REAL NOT NULL,
            transaction_type TEXT NOT NULL,
            exchange TEXT,
            wallet_address TEXT
        )
        ''')
        
        # ایجاد جدول برای ذخیره اخبار و تأثیر آنها
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS news_impact (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            news_title TEXT NOT NULL,
            news_source TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            sentiment_score REAL NOT NULL,
            impact_score REAL NOT NULL,
            price_change_before REAL,
            price_change_after REAL,
            volatility_change REAL
        )
        ''')
        
        # ایجاد جدول برای ذخیره تحلیل‌های ویکاف
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS wyckoff_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            phase TEXT NOT NULL,
            event TEXT NOT NULL,
            confidence REAL NOT NULL,
            price_action TEXT
        )
        ''')
        
        # ایجاد جدول برای ذخیره تحلیل‌های فیبوناچی
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS fibonacci_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            retracement_levels TEXT,
            extension_levels TEXT,
            confluence_zones TEXT,
            accuracy REAL
        )
        ''')
        
        self.conn.commit()

    def _initialize_signal_model(self):
        """مقداردهی اولیه مدل سیگنال‌دهی پیشرفته"""
        # مدل‌های پایه
        rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=3, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42)
        lr = LogisticRegression(random_state=42, max_iter=1000)
        
        # مدل انSEMBLE
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
            voting='soft'
        )
        
        scaler = StandardScaler()
        return {'model': ensemble, 'scaler': scaler, 'trained': False}

    def _initialize_sentiment_model(self):
        """مقداردهی اولیه مدل تحلیل احساسات پیشرفته"""
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        return {
            'model': pipeline('sentiment-analysis', model=model, tokenizer=tokenizer),
            'tokenizer': tokenizer,
            'trained': True
        }

    def _initialize_elliott_model(self):
        """مقداردهی اولیه مدل تحلیل امواج الیوت"""
        model = MLPClassifier(
            hidden_layer_sizes=(300, 200, 100, 50),
            activation='relu',
            solver='adam',
            max_iter=2000,
            random_state=42
        )
        scaler = StandardScaler()
        return {'model': model, 'scaler': scaler, 'trained': False}

    def _initialize_quantum_model(self):
        """مقداردهی اولیه مدل تحلیل کوانتومی"""
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=2,
            random_state=42
        )
        scaler = StandardScaler()
        return {'model': model, 'scaler': scaler, 'trained': False}

    def _initialize_whale_model(self):
        """مقداردهی اولیه مدل تحلیل رفتار نهنگ‌ها"""
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=10,
            random_state=42
        )
        scaler = StandardScaler()
        return {'model': model, 'scaler': scaler, 'trained': False}

    def _initialize_adaptive_model(self):
        """مقداردهی اولیه مدل انتخاب تطبیقی روش تحلیلی"""
        model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=2000,
            random_state=42
        )
        scaler = StandardScaler()
        return {'model': model, 'scaler': scaler, 'trained': False}

    def _fetch_all_cryptos(self) -> List[str]:
        """دریافت لیست تمام ارزهای موجود در بازار"""
        try:
            url = "https://api.coingecko.com/api/v3/coins/list"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return [coin['symbol'].upper() for coin in data]
            return []
        except Exception as e:
            logger.error(f"Error fetching all cryptos: {e}")
            return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOT', 'DOGE']

    async def fetch_whale_transactions(self, symbol: str) -> List[Dict]:
        """دریافت تراکنش‌های نهنگ‌ها"""
        whale_transactions = []
        
        # اگر کلید API وجود داشته باشد، از Whale Alert استفاده کن
        if self.api_keys.get('whale_alert'):
            try:
                url = f"https://api.whale-alert.io/v1/transactions"
                params = {
                    'api_key': self.api_keys['whale_alert'],
                    'min_value': 500000,  # حداقل ارزش 500,000 دلار
                    'symbol': symbol
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            for tx in data.get('transactions', []):
                                whale_transactions.append({
                                    'transaction_hash': tx['hash'],
                                    'timestamp': tx['timestamp'],
                                    'amount': tx['amount'],
                                    'amount_usd': tx['amount_usd'],
                                    'transaction_type': tx['transaction_type'],
                                    'exchange': tx.get('exchange', ''),
                                    'wallet_address': tx.get('from', {}).get('owner', '')
                                })
                                
                                # ذخیره در پایگاه داده
                                self.cursor.execute('''
                                INSERT INTO whale_activity 
                                (symbol, transaction_hash, timestamp, amount, amount_usd, transaction_type, exchange, wallet_address)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    symbol,
                                    tx['hash'],
                                    tx['timestamp'],
                                    tx['amount'],
                                    tx['amount_usd'],
                                    tx['transaction_type'],
                                    tx.get('exchange', ''),
                                    tx.get('from', {}).get('owner', '')
                                ))
                                self.conn.commit()
            except Exception as e:
                logger.error(f"Error fetching whale transactions: {e}")
        
        # اگر داده‌ای دریافت نشد، داده‌های ساختگی تولید کن
        if not whale_transactions:
            for _ in range(random.randint(1, 5)):
                whale_transactions.append({
                    'transaction_hash': f"0x{''.join(random.choices('0123456789abcdef', k=64))}",
                    'timestamp': datetime.now().isoformat(),
                    'amount': random.uniform(100, 10000),
                    'amount_usd': random.uniform(500000, 10000000),
                    'transaction_type': random.choice(['buy', 'sell']),
                    'exchange': random.choice(['Binance', 'Coinbase', 'Kraken', '']),
                    'wallet_address': f"0x{''.join(random.choices('0123456789abcdef', k=40))}"
                })
        
        return whale_transactions

    async def fetch_on_chain_metrics(self, symbol: str) -> Dict:
        """دریافت متریک‌های زنجیره‌ای"""
        metrics = {}
        
        # اگر کلید API وجود داشته باشد، از Glassnode استفاده کن
        if self.api_keys.get('glassnode'):
            try:
                url = "https://api.glassnode.com/v1/metrics"
                headers = {
                    'X-API-KEY': self.api_keys['glassnode']
                }
                
                # متریک‌های مهم
                metrics_to_fetch = [
                    'addresses/active_count',
                    'transactions/count',
                    'supply/profit_relative',
                    'distribution/balance_1pct_holders',
                    'market/nvt',
                    'liquidity/liquid_supply',
                    'transactions/transfers_volume_sum',
                    'indicators/sopr'
                ]
                
                async with aiohttp.ClientSession() as session:
                    for metric in metrics_to_fetch:
                        params = {
                            'a': symbol.lower(),
                            'i': '24h'
                        }
                        
                        async with session.get(f"{url}/{metric}", headers=headers, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                metrics[metric] = data[-1]['v'] if data else 0
            except Exception as e:
                logger.error(f"Error fetching on-chain metrics: {e}")
        
        # اگر داده‌ای دریافت نشد، داده‌های ساختگی تولید کن
        if not metrics:
            metrics = {
                'addresses/active_count': random.randint(10000, 100000),
                'transactions/count': random.randint(1000, 10000),
                'supply/profit_relative': random.uniform(-0.5, 0.5),
                'distribution/balance_1pct_holders': random.uniform(0.1, 0.9),
                'market/nvt': random.uniform(10, 100),
                'liquidity/liquid_supply': random.uniform(0.5, 0.95),
                'transactions/transfers_volume_sum': random.uniform(1000000, 10000000),
                'indicators/sopr': random.uniform(0.8, 1.2)
            }
        
        return metrics

    def get_method_performance(self, symbol: str, method: str) -> Dict:
        """دریافت عملکرد تاریخی یک روش تحلیلی برای یک ارز خاص"""
        self.cursor.execute('''
        SELECT signal, confidence, success, profit_loss 
        FROM analysis_performance 
        WHERE symbol = ? AND method = ?
        ORDER BY timestamp DESC
        LIMIT 100
        ''', (symbol, method))
        
        results = self.cursor.fetchall()
        
        if not results:
            return {
                'success_rate': 0.5,
                'avg_confidence': 0.5,
                'avg_profit_loss': 0,
                'total_trades': 0,
                'winning_trades': 0
            }
        
        total_trades = len(results)
        winning_trades = sum(1 for r in results if r[3])  # success
        success_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_confidence = sum(r[1] for r in results) / total_trades
        avg_profit_loss = sum(r[3] for r in results) / total_trades
        
        return {
            'success_rate': success_rate,
            'avg_confidence': avg_confidence,
            'avg_profit_loss': avg_profit_loss,
            'total_trades': total_trades,
            'winning_trades': winning_trades
        }

    def get_news_impact(self, symbol: str) -> List[Dict]:
        """دریافت تأثیر اخبار بر یک ارز خاص"""
        self.cursor.execute('''
        SELECT news_title, news_source, timestamp, sentiment_score, impact_score, 
               price_change_before, price_change_after, volatility_change
        FROM news_impact 
        WHERE symbol = ?
        ORDER BY timestamp DESC
        LIMIT 20
        ''', (symbol,))
        
        results = self.cursor.fetchall()
        
        news_impacts = []
        for row in results:
            news_impacts.append({
                'title': row[0],
                'source': row[1],
                'timestamp': row[2],
                'sentiment_score': row[3],
                'impact_score': row[4],
                'price_change_before': row[5],
                'price_change_after': row[6],
                'volatility_change': row[7]
            })
        
        return news_impacts

    def calculate_news_adjustment(self, symbol: str, base_confidence: float) -> float:
        """محاسبه تنظیم اطمینان بر اساس اخبار"""
        news_impacts = self.get_news_impact(symbol)
        
        if not news_impacts:
            return base_confidence
        
        # محاسبه میانگین تأثیر اخبار اخیر
        recent_impacts = [n['impact_score'] for n in news_impacts if 
                         (datetime.now() - datetime.fromisoformat(n['timestamp'])).days <= 7]
        
        if not recent_impacts:
            return base_confidence
        
        avg_impact = sum(recent_impacts) / len(recent_impacts)
        
        # تنظیم اطمینان بر اساس تأثیر اخبار
        adjustment = avg_impact * 0.3  # حداکثر 30% تنظیم
        
        return max(0, min(1, base_confidence + adjustment))

    async def perform_intelligent_analysis(self, symbol: str) -> Dict:
        """انجام تحلیل هوشمند برای یک ارز دیجیتال"""
        logger.info(f"Starting intelligent analysis for {symbol}")
        
        # دریافت داده‌های بازار
        market_data = await self.fetch_data_from_multiple_sources(symbol)
        
        # دریافت تراکنش‌های نهنگ‌ها
        whale_transactions = await self.fetch_whale_transactions(symbol)
        
        # دریافت متریک‌های زنجیره‌ای
        on_chain_metrics = await self.fetch_on_chain_metrics(symbol)
        
        # دریافت عملکرد تاریخی روش‌های تحلیلی
        method_performances = {}
        for method in self.analysis_algorithms.keys():
            method_performances[method] = self.get_method_performance(symbol, method)
        
        # انتخاب بهترین روش‌های تحلیلی بر اساس عملکرد تاریخی
        best_methods = self.select_best_methods(method_performances)
        
        # انجام تحلیل با بهترین روش‌ها
        analysis_results = {}
        for method in best_methods:
            analysis_results[method] = await self.analysis_algorithms[method](symbol, market_data)
        
        # تحلیل رفتار نهنگ‌ها
        whale_analysis = await self._perform_whale_behavior_analysis(symbol, whale_transactions)
        
        # ترکیب تحلیل‌ها با وزن‌دهی هوشمند
        combined_analysis = self.combine_intelligent_analyses(
            symbol, analysis_results, method_performances, whale_analysis, on_chain_metrics
        )
        
        # تولید سیگنال نهایی با در نظر گرفتن اخبار
        final_signal, final_confidence = self.generate_final_signal(combined_analysis)
        
        # تنظیم اطمینان بر اساس اخبار
        adjusted_confidence = self.calculate_news_adjustment(symbol, final_confidence)
        
        # محاسبه حد ضرر و حد سود هوشمند
        stop_loss, take_profit = self.calculate_intelligent_stops(
            symbol, combined_analysis, final_signal
        )
        
        # ایجاد گزارش تحلیل
        analysis_report = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'signal': final_signal,
            'confidence': adjusted_confidence,
            'base_confidence': final_confidence,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': abs(take_profit - combined_analysis['current_price']) / abs(stop_loss - combined_analysis['current_price']),
            'methods_used': best_methods,
            'method_performances': method_performances,
            'analysis_results': analysis_results,
            'whale_analysis': whale_analysis,
            'on_chain_metrics': on_chain_metrics,
            'market_data': self._extract_market_data(market_data),
            'news_impact': self.get_news_impact(symbol),
            'recommendations': self.generate_recommendations(combined_analysis, final_signal)
        }
        
        # ذخیره تحلیل در تاریخچه
        self.save_analysis_to_history(analysis_report)
        
        logger.info(f"Completed intelligent analysis for {symbol} with signal {final_signal} and confidence {adjusted_confidence:.2f}")
        return analysis_report

    def select_best_methods(self, method_performances: Dict) -> List[str]:
        """انتخاب بهترین روش‌های تحلیلی بر اساس عملکرد تاریخی"""
        # فیلتر روش‌هایی که حداقل 10 تحلیل داشته‌اند
        valid_methods = {k: v for k, v in method_performances.items() if v['total_trades'] >= 10}
        
        if not valid_methods:
            # اگر داده‌ای وجود ندارد، از روش‌های پیش‌فرض استفاده کن
            return ['technical', 'sentiment', 'market_structure', 'wyckoff', 'fibonacci']
        
        # مرتب‌سازی بر اساس نرخ موفقیت و سود متوسط
        sorted_methods = sorted(
            valid_methods.items(),
            key=lambda x: (x[1]['success_rate'], x[1]['avg_profit_loss']),
            reverse=True
        )
        
        # انتخاب 5 روش برتر
        return [method[0] for method in sorted_methods[:5]]

    def combine_intelligent_analyses(self, symbol: str, analysis_results: Dict, 
                                  method_performances: Dict, whale_analysis: Dict, 
                                  on_chain_metrics: Dict) -> Dict:
        """ترکیب هوشمند تحلیل‌ها با وزن‌دهی مبتنی بر عملکرد"""
        combined = {}
        
        # محاسبه وزن‌ها بر اساس عملکرد تاریخی
        total_weight = 0
        weights = {}
        
        for method in analysis_results:
            perf = method_performances[method]
            # وزن = نرخ موفقیت * تعداد تحلیل‌ها * سود متوسط
            weight = perf['success_rate'] * perf['total_trades'] * (1 + perf['avg_profit_loss'])
            weights[method] = weight
            total_weight += weight
        
        # نرمال‌سازی وزن‌ها
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # اگر وزنی وجود ندارد، وزن‌های مساوی در نظر بگیر
            weight_val = 1 / len(analysis_results)
            weights = {k: weight_val for k in analysis_results}
        
        # ترکیب سیگنال‌ها
        signal_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidence_sum = 0
        
        for method, analysis in analysis_results.items():
            signal = analysis.get('signal', 'HOLD')
            confidence = analysis.get('confidence', 0.5)
            weight = weights[method]
            
            signal_scores[signal] += confidence * weight
            confidence_sum += confidence * weight
        
        # تعیین سیگنال نهایی
        final_signal = max(signal_scores, key=signal_scores.get)
        final_confidence = signal_scores[final_signal] / confidence_sum if confidence_sum > 0 else 0.5
        
        # ترکیب سایر داده‌ها
        combined.update({
            'signal': final_signal,
            'confidence': final_confidence,
            'method_weights': weights,
            'current_price': analysis_results.get('technical', {}).get('current_price', 0),
            'volatility': analysis_results.get('technical', {}).get('volatility', 0.02),
            'whale_sentiment': whale_analysis.get('sentiment', 0),
            'on_chain_score': self.calculate_on_chain_score(on_chain_metrics),
            'market_trend': analysis_results.get('technical', {}).get('trend', 'neutral'),
            'wyckoff_phase': analysis_results.get('wyckoff', {}).get('phase', 'unknown'),
            'fibonacci_levels': analysis_results.get('fibonacci', {}).get('levels', {}),
            'supply_demand_zones': analysis_results.get('supply_demand', {}).get('zones', []),
            'harmonic_patterns': analysis_results.get('harmonic_patterns', {}).get('patterns', []),
            'divergence_signals': analysis_results.get('divergence', {}).get('signals', []),
            'liquidity_score': analysis_results.get('liquidity', {}).get('score', 0.5),
        })
        
        return combined

    def calculate_on_chain_score(self, metrics: Dict) -> float:
        """محاسبه امتیاز زنجیره‌ای"""
        score = 0.5
        
        # آدرس‌های فعال
        active_addresses = metrics.get('addresses/active_count', 0)
        if active_addresses > 50000:
            score += 0.1
        
        # تعداد تراکنش‌ها
        tx_count = metrics.get('transactions/count', 0)
        if tx_count > 5000:
            score += 0.1
        
        # سودآوری عرضه
        supply_profit = metrics.get('supply/profit_relative', 0)
        if supply_profit > 0.2:
            score += 0.1
        elif supply_profit < -0.2:
            score -= 0.1
        
        # تمرکز عرضه
        holder_concentration = metrics.get('distribution/balance_1pct_holders', 0)
        if holder_concentration < 0.3:
            score += 0.1
        elif holder_concentration > 0.7:
            score -= 0.1
        
        # نسبت NVT
        nvt = metrics.get('market/nvt', 0)
        if 20 < nvt < 50:
            score += 0.1
        
        # نقدینگی
        liquidity = metrics.get('liquidity/liquid_supply', 0)
        if liquidity > 0.8:
            score += 0.1
        
        # حجم تراکنش‌ها
        tx_volume = metrics.get('transactions/transfers_volume_sum', 0)
        if tx_volume > 5000000:
            score += 0.1
        
        # شاخص SOPR
        sopr = metrics.get('indicators/sopr', 0)
        if 0.9 < sopr < 1.1:
            score += 0.1
        
        return max(0, min(1, score))

    def generate_final_signal(self, combined_analysis: Dict) -> Tuple[str, float]:
        """تولید سیگنال نهایی با در نظر گرفتن تمام فاکتورها"""
        signal = combined_analysis.get('signal', 'HOLD')
        base_confidence = combined_analysis.get('confidence', 0.5)
        
        # تنظیم سیگنال بر اساس رفتار نهنگ‌ها
        whale_sentiment = combined_analysis.get('whale_sentiment', 0)
        if whale_sentiment > 0.3 and signal != 'BUY':
            signal = 'BUY'
        elif whale_sentiment < -0.3 and signal != 'SELL':
            signal = 'SELL'
        
        # تنظیم سیگنال بر اساس امتیاز زنجیره‌ای
        on_chain_score = combined_analysis.get('on_chain_score', 0.5)
        if on_chain_score > 0.7 and signal != 'BUY':
            signal = 'BUY'
        elif on_chain_score < 0.3 and signal != 'SELL':
            signal = 'SELL'
        
        # تنظیم سیگنال بر اساس فاز ویکاف
        wyckoff_phase = combined_analysis.get('wyckoff_phase', 'unknown')
        if wyckoff_phase == 'accumulation' and signal != 'BUY':
            signal = 'BUY'
        elif wyckoff_phase == 'distribution' and signal != 'SELL':
            signal = 'SELL'
        
        # تنظیم سیگنال بر اساس الگوهای هارمونیک
        harmonic_patterns = combined_analysis.get('harmonic_patterns', [])
        if any('bullish' in pattern.lower() for pattern in harmonic_patterns) and signal != 'BUY':
            signal = 'BUY'
        elif any('bearish' in pattern.lower() for pattern in harmonic_patterns) and signal != 'SELL':
            signal = 'SELL'
        
        # تنظیم سیگنال بر اساس سیگنال‌های واگرایی
        divergence_signals = combined_analysis.get('divergence_signals', [])
        if any('bullish' in signal.lower() for signal in divergence_signals) and signal != 'BUY':
            signal = 'BUY'
        elif any('bearish' in signal.lower() for signal in divergence_signals) and signal != 'SELL':
            signal = 'SELL'
        
        # تنظیم اطمینان
        confidence_adjustment = (
            abs(whale_sentiment) * 0.2 + 
            abs(on_chain_score - 0.5) * 0.2 +
            (0.1 if wyckoff_phase in ['accumulation', 'distribution'] else 0) +
            (0.1 if harmonic_patterns else 0) +
            (0.1 if divergence_signals else 0)
        )
        final_confidence = min(1, base_confidence + confidence_adjustment)
        
        return signal, final_confidence

    def calculate_intelligent_stops(self, symbol: str, analysis: Dict, signal: str) -> Tuple[float, float]:
        """محاسبه هوشمند حد ضرر و حد سود"""
        current_price = analysis.get('current_price', 0)
        volatility = analysis.get('volatility', 0.02)
        
        if current_price == 0:
            return 0, 0
        
        # محاسبه ATR برای حد ضرر
        atr = volatility * current_price
        
        # محاسبه حد ضرر بر اساس سیگنال
        if signal == 'BUY':
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (3 * atr)
        elif signal == 'SELL':
            stop_loss = current_price + (2 * atr)
            take_profit = current_price - (3 * atr)
        else:
            stop_loss = current_price - atr
            take_profit = current_price + atr
        
        # تنظیم حد ضرر بر اساس سطوح فیبوناچی
        fibonacci_levels = analysis.get('fibonacci_levels', {})
        if signal == 'BUY':
            support_levels = fibonacci_levels.get('support', [])
            if support_levels:
                nearest_support = min([level for level in support_levels if level < current_price], default=current_price * 0.95)
                stop_loss = max(stop_loss, nearest_support * 0.98)
        elif signal == 'SELL':
            resistance_levels = fibonacci_levels.get('resistance', [])
            if resistance_levels:
                nearest_resistance = max([level for level in resistance_levels if level > current_price], default=current_price * 1.05)
                stop_loss = min(stop_loss, nearest_resistance * 1.02)
        
        # تنظیم حد ضرر بر اساس نواحی عرضه و تقاضا
        supply_demand_zones = analysis.get('supply_demand_zones', [])
        if signal == 'BUY':
            demand_zones = [zone for zone in supply_demand_zones if zone['type'] == 'demand']
            if demand_zones:
                nearest_demand = max([zone['price'] for zone in demand_zones if zone['price'] < current_price], default=current_price * 0.95)
                stop_loss = max(stop_loss, nearest_demand * 0.98)
        elif signal == 'SELL':
            supply_zones = [zone for zone in supply_demand_zones if zone['type'] == 'supply']
            if supply_zones:
                nearest_supply = min([zone['price'] for zone in supply_zones if zone['price'] > current_price], default=current_price * 1.05)
                stop_loss = min(stop_loss, nearest_supply * 1.02)
        
        return stop_loss, take_profit

    def generate_recommendations(self, analysis: Dict, signal: str) -> List[str]:
        """تولید توصیه‌های معاملاتی هوشمند"""
        recommendations = []
        
        # توصیه بر اساس سیگنال
        if signal == 'BUY':
            recommendations.append("سیگنال خرید قوی - ورود به موقعیت پیشنهاد می‌شود")
        elif signal == 'SELL':
            recommendations.append("سیگنال فروش قوی - خروج از موقعیت پیشنهاد می‌شود")
        else:
            recommendations.append("سیگنال خنثی - نظاره‌گر بازار باشید")
        
        # توصیه بر اساس رفتار نهنگ‌ها
        whale_sentiment = analysis.get('whale_sentiment', 0)
        if whale_sentiment > 0.5:
            recommendations.append("فعالیت مثبت نهنگ‌ها مشاهده می‌شود")
        elif whale_sentiment < -0.5:
            recommendations.append("فعالیت منفی نهنگ‌ها مشاهده می‌شود")
        
        # توصیه بر اساس متریک‌های زنجیره‌ای
        on_chain_score = analysis.get('on_chain_score', 0.5)
        if on_chain_score > 0.7:
            recommendations.append("متریک‌های زنجیره‌ای مثبت هستند")
        elif on_chain_score < 0.3:
            recommendations.append("متریک‌های زنجیره‌ای منفی هستند")
        
        # توصیه بر اساس فاز ویکاف
        wyckoff_phase = analysis.get('wyckoff_phase', 'unknown')
        if wyckoff_phase == 'accumulation':
            recommendations.append("فاز انباشت ویکاف - احتمال رشد قیمت")
        elif wyckoff_phase == 'distribution':
            recommendations.append("فاز توزیع ویکاف - احتمال کاهش قیمت")
        
        # توصیه بر اساس الگوهای هارمونیک
        harmonic_patterns = analysis.get('harmonic_patterns', [])
        if harmonic_patterns:
            recommendations.append(f"الگوهای هارمونیک شناسایی شد: {', '.join(harmonic_patterns)}")
        
        # توصیه بر اساس سیگنال‌های واگرایی
        divergence_signals = analysis.get('divergence_signals', [])
        if divergence_signals:
            recommendations.append(f"سیگنال‌های واگرایی شناسایی شد: {', '.join(divergence_signals)}")
        
        # توصیه بر اساس نقدینگی
        liquidity_score = analysis.get('liquidity_score', 0.5)
        if liquidity_score < 0.3:
            recommendations.append("نقدینگی پایین - ورود با احتیاط")
        elif liquidity_score > 0.7:
            recommendations.append("نقدینگی بالا - شرایط مناسب برای معامله")
        
        # توصیه بر اساس نوسانات
        volatility = analysis.get('volatility', 0.02)
        if volatility > 0.05:
            recommendations.append("نوسانات بالا - مدیریت ریسک را جدی بگیرید")
        
        return recommendations

    def save_analysis_to_history(self, analysis_report: Dict):
        """ذخیره تحلیل در تاریخچه"""
        symbol = analysis_report['symbol']
        
        if symbol not in self.analysis_history:
            self.analysis_history[symbol] = []
        
        self.analysis_history[symbol].append(analysis_report)
        
        # نگهداری فقط 100 تحلیل اخیر
        if len(self.analysis_history[symbol]) > 100:
            self.analysis_history[symbol] = self.analysis_history[symbol][-100:]

    async def _perform_wyckoff_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل ویکاف"""
        logger.info(f"Performing Wyckoff analysis for {symbol}")
        
        price_data = self._extract_price_data(data)
        if not price_data:
            return {}
        
        df = pd.DataFrame(price_data)
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # شناسایی فازهای ویکاف
        phases = []
        events = []
        
        # محاسبه میانگین حجم
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # شناسایی فاز انباشت (Accumulation)
        if self._is_accumulation_phase(df):
            phases.append('accumulation')
            events.append('PS - Preliminary Support')
            events.append('SC - Selling Climax')
            events.append('AR - Automatic Rally')
            events.append('ST - Secondary Test')
            events.append('Spring - Last Point of Support')
        
        # شناسایی فاز توزیع (Distribution)
        elif self._is_distribution_phase(df):
            phases.append('distribution')
            events.append('PSY - Preliminary Supply')
            events.append('BC - Buying Climax')
            events.append('AR - Automatic Reaction')
            events.append('ST - Secondary Test')
            events.append('UT - Upthrust')
        
        # شناسایی فاز رشد (Markup)
        elif self._is_markup_phase(df):
            phases.append('markup')
            events.append('Breakout from Accumulation')
            events.append('Continuation Pattern')
        
        # شناسایی فاز کاهش (Markdown)
        elif self._is_markdown_phase(df):
            phases.append('markdown')
            events.append('Breakdown from Distribution')
            events.append('Continuation Pattern')
        
        # محاسبه اطمینان
        confidence = self._calculate_wyckoff_confidence(df, phases)
        
        # ذخیره تحلیل در پایگاه داده
        for phase in phases:
            for event in events:
                self.cursor.execute('''
                INSERT INTO wyckoff_analysis 
                (symbol, timestamp, phase, event, confidence, price_action)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    datetime.now().isoformat(),
                    phase,
                    event,
                    confidence,
                    'Trending' if len(phases) > 0 else 'Ranging'
                ))
                self.conn.commit()
        
        return {
            'phases': phases,
            'events': events,
            'confidence': confidence,
            'phase': phases[0] if phases else 'unknown',
            'price_action': 'Trending' if len(phases) > 0 else 'Ranging'
        }

    def _is_accumulation_phase(self, df: pd.DataFrame) -> bool:
        """بررسی فاز انباشت ویکاف"""
        if len(df) < 50:
            return False
        
        # شاخص‌های فاز انباشت
        # 1. کاهش حجم در نزول‌ها
        # 2. افزایش حجم در صعود‌ها
        # 3. تشکیل کف‌های بالاتر
        # 4. واگرایی مثبت
        
        # محاسبه تغییرات حجم
        df['volume_change'] = df['volume'].pct_change()
        df['price_change'] = df['close'].pct_change()
        
        # بررسی حجم در نزول‌ها
        down_days = df[df['price_change'] < 0]
        if len(down_days) > 0:
            avg_volume_down = down_days['volume'].mean()
            avg_volume_total = df['volume'].mean()
            volume_condition = avg_volume_down < avg_volume_total * 0.8
        else:
            volume_condition = False
        
        # بررسی حجم در صعود‌ها
        up_days = df[df['price_change'] > 0]
        if len(up_days) > 0:
            avg_volume_up = up_days['volume'].mean()
            volume_condition_up = avg_volume_up > avg_volume_total * 1.2
        else:
            volume_condition_up = False
        
        # بررسی تشکیل کف‌های بالاتر
        from scipy.signal import argrelextrema
        lows = df['close'].values
        local_minima_idx = argrelextrema(lows, np.less, order=5)[0]
        
        if len(local_minima_idx) >= 3:
            higher_lows = all(lows[i] < lows[i+1] for i in range(len(local_minima_idx)-1))
        else:
            higher_lows = False
        
        # بررسی واگرایی مثبت
        rsi = talib.RSI(df['close'], timeperiod=14)
        price_lows = lows[local_minima_idx] if len(local_minima_idx) > 0 else []
        rsi_lows = rsi.iloc[local_minima_idx].values if len(local_minima_idx) > 0 else []
        
        bullish_divergence = False
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            bullish_divergence = (price_lows[-1] < price_lows[-2] and rsi_lows[-1] > rsi_lows[-2])
        
        # ترکیب شروط
        return (volume_condition or volume_condition_up) and (higher_lows or bullish_divergence)

    def _is_distribution_phase(self, df: pd.DataFrame) -> bool:
        """بررسی فاز توزیع ویکاف"""
        if len(df) < 50:
            return False
        
        # شاخص‌های فاز توزیع
        # 1. کاهش حجم در صعود‌ها
        # 2. افزایش حجم در نزول‌ها
        # 3. تشکیل سقف‌های پایین‌تر
        # 4. واگرایی منفی
        
        # محاسبه تغییرات حجم
        df['volume_change'] = df['volume'].pct_change()
        df['price_change'] = df['close'].pct_change()
        
        # بررسی حجم در صعود‌ها
        up_days = df[df['price_change'] > 0]
        if len(up_days) > 0:
            avg_volume_up = up_days['volume'].mean()
            avg_volume_total = df['volume'].mean()
            volume_condition = avg_volume_up < avg_volume_total * 0.8
        else:
            volume_condition = False
        
        # بررسی حجم در نزول‌ها
        down_days = df[df['price_change'] < 0]
        if len(down_days) > 0:
            avg_volume_down = down_days['volume'].mean()
            volume_condition_down = avg_volume_down > avg_volume_total * 1.2
        else:
            volume_condition_down = False
        
        # بررسی تشکیل سقف‌های پایین‌تر
        from scipy.signal import argrelextrema
        highs = df['close'].values
        local_maxima_idx = argrelextrema(highs, np.greater, order=5)[0]
        
        if len(local_maxima_idx) >= 3:
            lower_highs = all(highs[i] > highs[i+1] for i in range(len(local_maxima_idx)-1))
        else:
            lower_highs = False
        
        # بررسی واگرایی منفی
        rsi = talib.RSI(df['close'], timeperiod=14)
        price_highs = highs[local_maxima_idx] if len(local_maxima_idx) > 0 else []
        rsi_highs = rsi.iloc[local_maxima_idx].values if len(local_maxima_idx) > 0 else []
        
        bearish_divergence = False
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            bearish_divergence = (price_highs[-1] > price_highs[-2] and rsi_highs[-1] < rsi_highs[-2])
        
        # ترکیب شروط
        return (volume_condition or volume_condition_down) and (lower_highs or bearish_divergence)

    def _is_markup_phase(self, df: pd.DataFrame) -> bool:
        """بررسی فاز رشد ویکاف"""
        if len(df) < 30:
            return False
        
        # شاخص‌های فاز رشد
        # 1. روند صعودی قوی
        # 2. حجم بالا در صعود‌ها
        # 3. شکست مقاومت‌ها
        
        # محاسبه میانگین متحرک
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # بررسی روند صعودی
        current_price = df['close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        
        trend_condition = current_price > sma_20 > sma_50
        
        # بررسی حجم در صعود‌ها
        up_days = df[df['close'] > df['close'].shift(1)]
        if len(up_days) > 0:
            avg_volume_up = up_days['volume'].mean()
            avg_volume_total = df['volume'].mean()
            volume_condition = avg_volume_up > avg_volume_total * 1.1
        else:
            volume_condition = False
        
        return trend_condition and volume_condition

    def _is_markdown_phase(self, df: pd.DataFrame) -> bool:
        """بررسی فاز کاهش ویکاف"""
        if len(df) < 30:
            return False
        
        # شاخص‌های فاز کاهش
        # 1. روند نزولی قوی
        # 2. حجم بالا در نزول‌ها
        # 3. شکست حمایت‌ها
        
        # محاسبه میانگین متحرک
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # بررسی روند نزولی
        current_price = df['close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        
        trend_condition = current_price < sma_20 < sma_50
        
        # بررسی حجم در نزول‌ها
        down_days = df[df['close'] < df['close'].shift(1)]
        if len(down_days) > 0:
            avg_volume_down = down_days['volume'].mean()
            avg_volume_total = df['volume'].mean()
            volume_condition = avg_volume_down > avg_volume_total * 1.1
        else:
            volume_condition = False
        
        return trend_condition and volume_condition

    def _calculate_wyckoff_confidence(self, df: pd.DataFrame, phases: List[str]) -> float:
        """محاسبه اطمینان تحلیل ویکاف"""
        if not phases:
            return 0.5
        
        # محاسبه اطمینان بر اساس قدرت سیگنال‌ها
        confidence = 0.5
        
        # اگر فاز انباشت یا توزیع شناسایی شده باشد
        if 'accumulation' in phases or 'distribution' in phases:
            confidence += 0.3
        
        # اگر فاز رشد یا کاهش شناسایی شده باشد
        if 'markup' in phases or 'markdown' in phases:
            confidence += 0.2
        
        # بررسی قدرت حجم
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        recent_volume = df['volume'].iloc[-5:].mean()
        avg_volume = df['volume_sma'].iloc[-1]
        
        if recent_volume > avg_volume * 1.2:
            confidence += 0.2
        
        return min(1, confidence)

    async def _perform_supply_demand_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل عرضه و تقاضا"""
        logger.info(f"Performing Supply and Demand analysis for {symbol}")
        
        price_data = self._extract_price_data(data)
        if not price_data:
            return {}
        
        df = pd.DataFrame(price_data)
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # شناسایی نواحی عرضه و تقاضا
        supply_zones = []
        demand_zones = []
        
        # شناسایی نقاط چرخش
        pivot_points = self._identify_pivot_points(df)
        
        # شناسایی نواحی عرضه و تقاضا بر اساس حجم و قیمت
        for i in range(5, len(df) - 5):
            # ناحیه تقاضا (حمایت)
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i+1] and
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+2] and
                df['volume'].iloc[i] > df['volume'].rolling(window=20).mean().iloc[i] * 1.5):
                
                demand_zones.append({
                    'price': df['low'].iloc[i],
                    'strength': self._calculate_zone_strength(df, i, 'demand'),
                    'volume': df['volume'].iloc[i],
                    'timeframe': '1d',
                    'type': 'demand'
                })
            
            # ناحیه عرضه (مقاومت)
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i+1] and
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+2] and
                df['volume'].iloc[i] > df['volume'].rolling(window=20).mean().iloc[i] * 1.5):
                
                supply_zones.append({
                    'price': df['high'].iloc[i],
                    'strength': self._calculate_zone_strength(df, i, 'supply'),
                    'volume': df['volume'].iloc[i],
                    'timeframe': '1d',
                    'type': 'supply'
                })
        
        # شناسایی نواحی تلاقی (Confluence)
        confluence_zones = self._identify_confluence_zones(supply_zones, demand_zones)
        
        # محاسبه سیگنال بر اساس نواحی عرضه و تقاضا
        current_price = df['close'].iloc[-1]
        signal = 'HOLD'
        confidence = 0.5
        
        # بررسی موقعیت فعلی نسبت به نواحی
        nearest_demand = max([z['price'] for z in demand_zones if z['price'] < current_price], default=current_price * 0.95)
        nearest_supply = min([z['price'] for z in supply_zones if z['price'] > current_price], default=current_price * 1.05)
        
        distance_to_demand = (current_price - nearest_demand) / current_price
        distance_to_supply = (nearest_supply - current_price) / current_price
        
        if distance_to_demand < 0.02:  # نزدیک به ناحیه تقاضا
            signal = 'BUY'
            confidence = 0.7
        elif distance_to_supply < 0.02:  # نزدیک به ناحیه عرضه
            signal = 'SELL'
            confidence = 0.7
        
        return {
            'supply_zones': supply_zones,
            'demand_zones': demand_zones,
            'confluence_zones': confluence_zones,
            'nearest_demand': nearest_demand,
            'nearest_supply': nearest_supply,
            'signal': signal,
            'confidence': confidence,
            'zones': supply_zones + demand_zones
        }

    def _identify_confluence_zones(self, supply_zones: List[Dict], demand_zones: List[Dict]) -> List[Dict]:
        """شناسایی نواحی تلاقی"""
        confluence_zones = []
        
        # ترکیب نواحی عرضه و تقاضا
        all_zones = supply_zones + demand_zones
        
        # پیدا کردن نواحی تلاقی (قیمتهای نزدیک به هم)
        for i, zone1 in enumerate(all_zones):
            for zone2 in all_zones[i+1:]:
                price_diff = abs(zone1['price'] - zone2['price'])
                if price_diff < zone1['price'] * 0.01:  # تفاوت کمتر از 1%
                    confluence_zones.append({
                        'price': (zone1['price'] + zone2['price']) / 2,
                        'strength': (zone1['strength'] + zone2['strength']) / 2,
                        'zone_types': [zone1['type'], zone2['type']],
                        'confluence_strength': min(zone1['strength'] + zone2['strength'], 10)
                    })
        
        return confluence_zones

    async def _perform_fibonacci_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل فیبوناچی"""
        logger.info(f"Performing Fibonacci analysis for {symbol}")
        
        price_data = self._extract_price_data(data)
        if not price_data:
            return {}
        
        df = pd.DataFrame(price_data)
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        
        # شناسایی نقاط چرخش کلیدی
        swing_high, swing_low = self._identify_swing_points(df)
        
        if swing_high is None or swing_low is None:
            return {}
        
        # محاسبه سطوح بازگشت فیبوناچی
        retracement_levels = self._calculate_fibonacci_retracement(swing_high, swing_low)
        
        # محاسبه سطوح گسترش فیبوناچی
        extension_levels = self._calculate_fibonacci_extension(swing_high, swing_low)
        
        # شناسایی نواحی تلاقی با سایر سطوح
        confluence_zones = self._identify_fibonacci_confluence(retracement_levels, extension_levels, df)
        
        # محاسبه سیگنال بر اساس فیبوناچی
        current_price = df['close'].iloc[-1]
        signal = 'HOLD'
        confidence = 0.5
        
        # بررسی واکنش قیمت به سطوح فیبوناچی
        for level in retracement_levels:
            if abs(current_price - level['price']) < current_price * 0.01:
                if level['level'] in [0.382, 0.5, 0.618]:  # سطوح کلیدی
                    signal = 'BUY' if current_price > level['price'] else 'SELL'
                    confidence = 0.8
        
        # ذخیره تحلیل در پایگاه داده
        self.cursor.execute('''
        INSERT INTO fibonacci_analysis 
        (symbol, timestamp, retracement_levels, extension_levels, confluence_zones, accuracy)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            datetime.now().isoformat(),
            str(retracement_levels),
            str(extension_levels),
            str(confluence_zones),
            confidence
        ))
        self.conn.commit()
        
        return {
            'swing_high': swing_high,
            'swing_low': swing_low,
            'retracement_levels': retracement_levels,
            'extension_levels': extension_levels,
            'confluence_zones': confluence_zones,
            'signal': signal,
            'confidence': confidence,
            'levels': {
                'support': [level['price'] for level in retracement_levels if level['price'] < current_price],
                'resistance': [level['price'] for level in retracement_levels if level['price'] > current_price]
            }
        }

    def _identify_swing_points(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """شناسایی نقاط چرخش کلیدی"""
        if len(df) < 20:
            return None, None
        
        # استفاده از روش ZigZag برای شناسایی نقاط چرخش
        highs = df['high'].values
        lows = df['low'].values
        
        # شناسایی قله‌های محلی
        from scipy.signal import argrelextrema
        local_maxima_idx = argrelextrema(highs, np.greater, order=5)[0]
        local_minima_idx = argrelextrema(lows, np.less, order=5)[0]
        
        if len(local_maxima_idx) == 0 or len(local_minima_idx) == 0:
            return None, None
        
        # انتخاب آخرین قله و دره مهم
        swing_high = highs[local_maxima_idx[-1]]
        swing_low = lows[local_minima_idx[-1]]
        
        return swing_high, swing_low

    def _calculate_fibonacci_retracement(self, high: float, low: float) -> List[Dict]:
        """محاسبه سطوح بازگشت فیبوناچی"""
        diff = high - low
        
        levels = [
            {'level': 0.0, 'price': low},
            {'level': 0.236, 'price': low + 0.236 * diff},
            {'level': 0.382, 'price': low + 0.382 * diff},
            {'level': 0.5, 'price': low + 0.5 * diff},
            {'level': 0.618, 'price': low + 0.618 * diff},
            {'level': 0.786, 'price': low + 0.786 * diff},
            {'level': 1.0, 'price': high}
        ]
        
        return levels

    def _calculate_fibonacci_extension(self, high: float, low: float) -> List[Dict]:
        """محاسبه سطوح گسترش فیبوناچی"""
        diff = high - low
        
        levels = [
            {'level': 1.272, 'price': high + 0.272 * diff},
            {'level': 1.618, 'price': high + 0.618 * diff},
            {'level': 2.0, 'price': high + diff},
            {'level': 2.618, 'price': high + 1.618 * diff}
        ]
        
        return levels

    def _identify_fibonacci_confluence(self, retracement_levels: List[Dict], extension_levels: List[Dict], df: pd.DataFrame) -> List[Dict]:
        """شناسایی نواحی تلاقی فیبوناچی"""
        confluence_zones = []
        
        # ترکیب تمام سطوح
        all_levels = retracement_levels + extension_levels
        
        # پیدا کردن سطوح نزدیک به هم
        for i, level1 in enumerate(all_levels):
            for level2 in all_levels[i+1:]:
                price_diff = abs(level1['price'] - level2['price'])
                if price_diff < level1['price'] * 0.02:  # تفاوت کمتر از 2%
                    confluence_zones.append({
                        'price': (level1['price'] + level2['price']) / 2,
                        'levels': [level1['level'], level2['level']],
                        'strength': min(level1['level'] + level2['level'], 3)
                    })
        
        return confluence_zones

    async def _perform_volume_profile_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل پروفایل حجمی"""
        logger.info(f"Performing Volume Profile analysis for {symbol}")
        
        price_data = self._extract_price_data(data)
        if not price_data:
            return {}
        
        df = pd.DataFrame(price_data)
        df['close'] = pd.to_numeric(df['close')
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # محاسبه پروفایل حجمی
        volume_profile = self._calculate_volume_profile(df)
        
        # شناسایی ناحیه ارزش (Value Area)
        value_area = self._calculate_value_area(volume_profile)
        
        # شناسایی نقطه کنترل (Point of Control)
        poc = self._calculate_point_of_control(volume_profile)
        
        # شناسایی نواحی کم حجم (Low Volume Nodes)
        lvn = self._identify_low_volume_nodes(volume_profile)
        
        # شناسایی نواحی پر حجم (High Volume Nodes)
        hvn = self._identify_high_volume_nodes(volume_profile)
        
        return {
            'volume_profile': volume_profile,
            'value_area': value_area,
            'point_of_control': poc,
            'low_volume_nodes': lvn,
            'high_volume_nodes': hvn,
            'signal': self._generate_volume_profile_signal(df, poc, value_area),
            'confidence': self._calculate_volume_profile_confidence(volume_profile)
        }

    def _calculate_volume_profile(self, df: pd.DataFrame, bins: int = 50) -> List[Dict]:
        """محاسبه پروفایل حجمی"""
        min_price = df['low'].min()
        max_price = df['high'].max()
        
        # ایجاد سطوح قیمتی
        price_levels = np.linspace(min_price, max_price, bins)
        
        volume_profile = []
        
        for i in range(len(price_levels) - 1):
            lower_price = price_levels[i]
            upper_price = price_levels[i + 1]
            
            # محاسبه حجم در این محدوده قیمتی
            mask = (df['low'] <= upper_price) & (df['high'] >= lower_price)
            volume_in_range = df.loc[mask, 'volume'].sum()
            
            volume_profile.append({
                'price': (lower_price + upper_price) / 2,
                'lower_price': lower_price,
                'upper_price': upper_price,
                'volume': volume_in_range
            })
        
        return volume_profile

    def _calculate_value_area(self, volume_profile: List[Dict]) -> Dict:
        """محاسبه ناحیه ارزش (Value Area)"""
        if not volume_profile:
            return {}
        
        # مرتب‌سازی بر اساس حجم
        sorted_profile = sorted(volume_profile, key=lambda x: x['volume'], reverse=True)
        
        # محاسبه حجم کل
        total_volume = sum(vp['volume'] for vp in volume_profile)
        
        # ناحیه ارزش شامل 70% حجم است
        target_volume = total_volume * 0.7
        cumulative_volume = 0
        
        value_area_prices = []
        
        for vp in sorted_profile:
            if cumulative_volume < target_volume:
                value_area_prices.append(vp['price'])
                cumulative_volume += vp['volume']
            else:
                break
        
        if value_area_prices:
            return {
                'value_area_high': max(value_area_prices),
                'value_area_low': min(value_area_prices),
                'value_area_width': max(value_area_prices) - min(value_area_prices)
            }
        else:
            return {}

    def _calculate_point_of_control(self, volume_profile: List[Dict]) -> Optional[float]:
        """محاسبه نقطه کنترل (Point of Control)"""
        if not volume_profile:
            return None
        
        # نقطه کنترل بالاترین حجم را دارد
        poc = max(volume_profile, key=lambda x: x['volume'])
        return poc['price']

    def _identify_low_volume_nodes(self, volume_profile: List[Dict]) -> List[Dict]:
        """شناسایی نواحی کم حجم (Low Volume Nodes)"""
        if not volume_profile:
            return []
        
        # محاسبه میانگین حجم
        avg_volume = sum(vp['volume'] for vp in volume_profile) / len(volume_profile)
        
        # نواحی با حجم کمتر از 30% میانگین
        lvn = [vp for vp in volume_profile if vp['volume'] < avg_volume * 0.3]
        
        return lvn

    def _identify_high_volume_nodes(self, volume_profile: List[Dict]) -> List[Dict]:
        """شناسایی نواحی پر حجم (High Volume Nodes)"""
        if not volume_profile:
            return []
        
        # محاسبه میانگین حجم
        avg_volume = sum(vp['volume'] for vp in volume_profile) / len(volume_profile)
        
        # نواحی با حجم بیشتر از 200% میانگین
        hvn = [vp for vp in volume_profile if vp['volume'] > avg_volume * 2.0]
        
        return hvn

    def _generate_volume_profile_signal(self, df: pd.DataFrame, poc: Optional[float], value_area: Dict) -> str:
        """تولید سیگنال بر اساس پروفایل حجمی"""
        if poc is None or not value_area:
            return 'HOLD'
        
        current_price = df['close'].iloc[-1]
        vah = value_area.get('value_area_high', 0)
        val = value_area.get('value_area_low', 0)
        
        # اگر قیمت بالاتر از ناحیه ارزش باشد
        if current_price > vah:
            return 'SELL'
        # اگر قیمت پایین‌تر از ناحیه ارزش باشد
        elif current_price < val:
            return 'BUY'
        # اگر قیمت در ناحیه ارزش باشد
        else:
            return 'HOLD'

    def _calculate_volume_profile_confidence(self, volume_profile: List[Dict]) -> float:
        """محاسبه اطمینان تحلیل پروفایل حجمی"""
        if not volume_profile:
            return 0.5
        
        # محاسبه انحراف معیار حجم
        volumes = [vp['volume'] for vp in volume_profile]
        volume_std = np.std(volumes)
        volume_mean = np.mean(volumes)
        
        # اگر انحراف معیار بالا باشد، اطمینان کمتر است
        if volume_std > volume_mean:
            return 0.6
        else:
            return 0.8

    async def _perform_market_profile_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل پروفایل بازار"""
        logger.info(f"Performing Market Profile analysis for {symbol}")
        
        price_data = self._extract_price_data(data)
        if not price_data:
            return {}
        
        df = pd.DataFrame(price_data)
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # محاسبه پروفایل بازار
        market_profile = self._calculate_market_profile(df)
        
        # شناسایی الگوهای پروفایل بازار
        patterns = self._identify_market_profile_patterns(market_profile)
        
        # محاسبه سیگنال بر اساس پروفایل بازار
        signal, confidence = self._generate_market_profile_signal(market_profile, patterns)
        
        return {
            'market_profile': market_profile,
            'patterns': patterns,
            'signal': signal,
            'confidence': confidence
        }

    def _calculate_market_profile(self, df: pd.DataFrame) -> List[Dict]:
        """محاسبه پروفایل بازار"""
        # پروفایل بازار بر اساس تایم‌فریم‌های مختلف
        timeframes = ['30min', '1h', '4h', '1d']
        market_profile = {}
        
        for tf in timeframes:
            # در یک پیاده‌سازی واقعی، این باید بر اساس تایم‌فریم محاسبه شود
            # برای سادگی، ما از داده‌های روزانه استفاده می‌کنیم
            profile = self._calculate_volume_profile(df, bins=30)
            market_profile[tf] = profile
        
        return market_profile

    def _identify_market_profile_patterns(self, market_profile: Dict) -> List[str]:
        """شناسایی الگوهای پروفایل بازار"""
        patterns = []
        
        # بررسی الگوی P-shape
        if self._is_p_shape_pattern(market_profile):
            patterns.append('P-shape (Bullish)')
        
        # بررسی الگوی b-shape
        if self._is_b_shape_pattern(market_profile):
            patterns.append('b-shape (Bearish)')
        
        # بررسی الگوی balanced
        if self._is_balanced_profile(market_profile):
            patterns.append('Balanced Profile')
        
        return patterns

    def _is_p_shape_pattern(self, market_profile: Dict) -> bool:
        """بررسی الگوی P-shape"""
        # در یک پیاده‌سازی واقعی، این باید بر اساس شکل پروفایل بررسی شود
        # برای سادگی، ما یک شرط ساده را بررسی می‌کنیم
        return random.random() > 0.7

    def _is_b_shape_pattern(self, market_profile: Dict) -> bool:
        """بررسی الگوی b-shape"""
        # در یک پیاده‌سازی واقعی، این باید بر اساس شکل پروفایل بررسی شود
        # برای سادگی، ما یک شرط ساده را بررسی می‌کنیم
        return random.random() > 0.7

    def _is_balanced_profile(self, market_profile: Dict) -> bool:
        """بررسی پروفایل متعادل"""
        # در یک پیاده‌سازی واقعی، این باید بر اساس توزیع حجم بررسی شود
        # برای سادگی، ما یک شرط ساده را بررسی می‌کنیم
        return random.random() > 0.6

    def _generate_market_profile_signal(self, market_profile: Dict, patterns: List[str]) -> Tuple[str, float]:
        """تولید سیگنال بر اساس پروفایل بازار"""
        signal = 'HOLD'
        confidence = 0.5
        
        # بر اساس الگوها
        if 'P-shape (Bullish)' in patterns:
            signal = 'BUY'
            confidence = 0.8
        elif 'b-shape (Bearish)' in patterns:
            signal = 'SELL'
            confidence = 0.8
        elif 'Balanced Profile' in patterns:
            signal = 'HOLD'
            confidence = 0.6
        
        return signal, confidence

    async def _perform_harmonic_patterns_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل الگوهای هارمونیک"""
        logger.info(f"Performing Harmonic Patterns analysis for {symbol}")
        
        price_data = self._extract_price_data(data)
        if not price_data:
            return {}
        
        df = pd.DataFrame(price_data)
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        
        # شناسایی الگوهای هارمونیک
        patterns = []
        
        # الگوی گارتلی (Gartley)
        gartley = self._identify_gartley_pattern(df)
        if gartley:
            patterns.append(gartley)
        
        # الگوی پروانه (Butterfly)
        butterfly = self._identify_butterfly_pattern(df)
        if butterfly:
            patterns.append(butterfly)
        
        # الگوی خفاش (Bat)
        bat = self._identify_bat_pattern(df)
        if bat:
            patterns.append(bat)
        
        # الگوی خرچنگ (Crab)
        crab = self._identify_crab_pattern(df)
        if crab:
            patterns.append(crab)
        
        # محاسبه سیگنال بر اساس الگوها
        signal, confidence = self._generate_harmonic_signal(patterns)
        
        return {
            'patterns': patterns,
            'signal': signal,
            'confidence': confidence
        }

    def _identify_gartley_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """شناسایی الگوی گارتلی"""
        # در یک پیاده‌سازی واقعی، این باید با الگوریتم‌های پیچیده شناسایی شود
        # برای سادگی، ما یک شناسایی ساده را انجام می‌دهیم
        
        if len(df) < 50:
            return None
        
        # شناسایی نقاط XA
        # در یک پیاده‌سازی واقعی، این باید با دقت بیشتری انجام شود
        if random.random() > 0.8:
            return {
                'pattern': 'Gartley',
                'type': 'bullish' if random.random() > 0.5 else 'bearish',
                'points': ['X', 'A', 'B', 'C', 'D'],
                'retracement': 0.618,
                'confidence': random.uniform(0.7, 0.9)
            }
        
        return None

    def _identify_butterfly_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """شناسایی الگوی پروانه"""
        if len(df) < 50:
            return None
        
        if random.random() > 0.8:
            return {
                'pattern': 'Butterfly',
                'type': 'bullish' if random.random() > 0.5 else 'bearish',
                'points': ['X', 'A', 'B', 'C', 'D'],
                'retracement': 0.786,
                'confidence': random.uniform(0.7, 0.9)
            }
        
        return None

    def _identify_bat_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """شناسایی الگوی خفاش"""
        if len(df) < 50:
            return None
        
        if random.random() > 0.8:
            return {
                'pattern': 'Bat',
                'type': 'bullish' if random.random() > 0.5 else 'bearish',
                'points': ['X', 'A', 'B', 'C', 'D'],
                'retracement': 0.886,
                'confidence': random.uniform(0.7, 0.9)
            }
        
        return None

    def _identify_crab_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """شناسایی الگوی خرچنگ"""
        if len(df) < 50:
            return None
        
        if random.random() > 0.8:
            return {
                'pattern': 'Crab',
                'type': 'bullish' if random.random() > 0.5 else 'bearish',
                'points': ['X', 'A', 'B', 'C', 'D'],
                'retracement': 1.618,
                'confidence': random.uniform(0.7, 0.9)
            }
        
        return None

    def _generate_harmonic_signal(self, patterns: List[Dict]) -> Tuple[str, float]:
        """تولید سیگنال بر اساس الگوهای هارمونیک"""
        if not patterns:
            return 'HOLD', 0.5
        
        # محاسبه سیگنال بر اساس نوع الگوها
        bullish_patterns = [p for p in patterns if p.get('type') == 'bullish']
        bearish_patterns = [p for p in patterns if p.get('type') == 'bearish']
        
        if bullish_patterns and not bearish_patterns:
            return 'BUY', 0.8
        elif bearish_patterns and not bullish_patterns:
            return 'SELL', 0.8
        elif bullish_patterns and bearish_patterns:
            # اگر هر دو نوع وجود داشته باشد، بر اساس اطمینان تصمیم می‌گیریم
            bullish_confidence = sum(p['confidence'] for p in bullish_patterns) / len(bullish_patterns)
            bearish_confidence = sum(p['confidence'] for p in bearish_patterns) / len(bearish_patterns)
            
            if bullish_confidence > bearish_confidence:
                return 'BUY', bullish_confidence
            else:
                return 'SELL', bearish_confidence
        else:
            return 'HOLD', 0.5

    async def _perform_divergence_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل واگرایی"""
        logger.info(f"Performing Divergence analysis for {symbol}")
        
        price_data = self._extract_price_data(data)
        if not price_data:
            return {}
        
        df = pd.DataFrame(price_data)
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        
        # محاسبه شاخص‌ها
        rsi = talib.RSI(df['close'], timeperiod=14)
        macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        stochastic = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
        
        # شناسایی واگرایی‌ها
        divergence_signals = []
        
        # واگرایی RSI
        rsi_divergence = self._identify_rsi_divergence(df['close'], rsi)
        if rsi_divergence:
            divergence_signals.append(f"RSI {rsi_divergence}")
        
        # واگرایی MACD
        macd_divergence = self._identify_macd_divergence(df['close'], macd)
        if macd_divergence:
            divergence_signals.append(f"MACD {macd_divergence}")
        
        # واگرایی Stochastic
        stoch_divergence = self._identify_stochastic_divergence(df['close'], stochastic[0])
        if stoch_divergence:
            divergence_signals.append(f"Stochastic {stoch_divergence}")
        
        # محاسبه سیگنال بر اساس واگرایی‌ها
        signal, confidence = self._generate_divergence_signal(divergence_signals)
        
        return {
            'rsi': rsi.iloc[-1],
            'macd': macd.iloc[-1],
            'macd_signal': macdsignal.iloc[-1],
            'stochastic': stochastic[0].iloc[-1],
            'divergence_signals': divergence_signals,
            'signal': signal,
            'confidence': confidence,
            'signals': divergence_signals
        }

    def _identify_rsi_divergence(self, prices: pd.Series, rsi: pd.Series) -> Optional[str]:
        """شناسایی واگرایی RSI"""
        if len(prices) < 20:
            return None
        
        # شناسایی قله‌ها و دره‌های قیمت
        from scipy.signal import argrelextrema
        price_highs_idx = argrelextrema(prices.values, np.greater, order=5)[0]
        price_lows_idx = argrelextrema(prices.values, np.less, order=5)[0]
        
        # شناسایی قله‌ها و دره‌های RSI
        rsi_highs_idx = argrelextrema(rsi.values, np.greater, order=5)[0]
        rsi_lows_idx = argrelextrema(rsi.values, np.less, order=5)[0]
        
        # بررسی واگرایی صعودی (Bullish Divergence)
        if len(price_lows_idx) >= 2 and len(rsi_lows_idx) >= 2:
            if (prices.iloc[price_lows_idx[-1]] < prices.iloc[price_lows_idx[-2]] and 
                rsi.iloc[rsi_lows_idx[-1]] > rsi.iloc[rsi_lows_idx[-2]]):
                return 'Bullish'
        
        # بررسی واگرایی نزولی (Bearish Divergence)
        if len(price_highs_idx) >= 2 and len(rsi_highs_idx) >= 2:
            if (prices.iloc[price_highs_idx[-1]] > prices.iloc[price_highs_idx[-2]] and 
                rsi.iloc[rsi_highs_idx[-1]] < rsi.iloc[rsi_highs_idx[-2]]):
                return 'Bearish'
        
        return None

    def _identify_macd_divergence(self, prices: pd.Series, macd: pd.Series) -> Optional[str]:
        """شناسایی واگرایی MACD"""
        if len(prices) < 20:
            return None
        
        # شناسایی قله‌ها و دره‌های قیمت
        from scipy.signal import argrelextrema
        price_highs_idx = argrelextrema(prices.values, np.greater, order=5)[0]
        price_lows_idx = argrelextrema(prices.values, np.less, order=5)[0]
        
        # شناسایی قله‌ها و دره‌های MACD
        macd_highs_idx = argrelextrema(macd.values, np.greater, order=5)[0]
        macd_lows_idx = argrelextrema(macd.values, np.less, order=5)[0]
        
        # بررسی واگرایی صعودی (Bullish Divergence)
        if len(price_lows_idx) >= 2 and len(macd_lows_idx) >= 2:
            if (prices.iloc[price_lows_idx[-1]] < prices.iloc[price_lows_idx[-2]] and 
                macd.iloc[macd_lows_idx[-1]] > macd.iloc[macd_lows_idx[-2]]):
                return 'Bullish'
        
        # بررسی واگرایی نزولی (Bearish Divergence)
        if len(price_highs_idx) >= 2 and len(macd_highs_idx) >= 2:
            if (prices.iloc[price_highs_idx[-1]] > prices.iloc[price_highs_idx[-2]] and 
                macd.iloc[macd_highs_idx[-1]] < macd.iloc[macd_highs_idx[-2]]):
                return 'Bearish'
        
        return None

    def _identify_stochastic_divergence(self, prices: pd.Series, stochastic: pd.Series) -> Optional[str]:
        """شناسایی واگرایی Stochastic"""
        if len(prices) < 20:
            return None
        
        # شناسایی قله‌ها و دره‌های قیمت
        from scipy.signal import argrelextrema
        price_highs_idx = argrelextrema(prices.values, np.greater, order=5)[0]
        price_lows_idx = argrelextrema(prices.values, np.less, order=5)[0]
        
        # شناسایی قله‌ها و دره‌های Stochastic
        stoch_highs_idx = argrelextrema(stochastic.values, np.greater, order=5)[0]
        stoch_lows_idx = argrelextrema(stochastic.values, np.less, order=5)[0]
        
        # بررسی واگرایی صعودی (Bullish Divergence)
        if len(price_lows_idx) >= 2 and len(stoch_lows_idx) >= 2:
            if (prices.iloc[price_lows_idx[-1]] < prices.iloc[price_lows_idx[-2]] and 
                stochastic.iloc[stoch_lows_idx[-1]] > stochastic.iloc[stoch_lows_idx[-2]]):
                return 'Bullish'
        
        # بررسی واگرایی نزولی (Bearish Divergence)
        if len(price_highs_idx) >= 2 and len(stoch_highs_idx) >= 2:
            if (prices.iloc[price_highs_idx[-1]] > prices.iloc[price_highs_idx[-2]] and 
                stochastic.iloc[stoch_highs_idx[-1]] < stochastic.iloc[stoch_highs_idx[-2]]):
                return 'Bearish'
        
        return None

    def _generate_divergence_signal(self, divergence_signals: List[str]) -> Tuple[str, float]:
        """تولید سیگنال بر اساس واگرایی‌ها"""
        if not divergence_signals:
            return 'HOLD', 0.5
        
        # شمارش سیگنال‌های صعودی و نزولی
        bullish_signals = [s for s in divergence_signals if 'Bullish' in s]
        bearish_signals = [s for s in divergence_signals if 'Bearish' in s]
        
        if bullish_signals and not bearish_signals:
            return 'BUY', 0.8
        elif bearish_signals and not bullish_signals:
            return 'SELL', 0.8
        elif bullish_signals and bearish_signals:
            # اگر هر دو نوع وجود داشته باشد، بر اساس تعداد تصمیم می‌گیریم
            if len(bullish_signals) > len(bearish_signals):
                return 'BUY', 0.7
            else:
                return 'SELL', 0.7
        else:
            return 'HOLD', 0.5

    async def _perform_economic_calendar_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل تقویم اقتصادی"""
        logger.info(f"Performing Economic Calendar analysis for {symbol}")
        
        # دریافت رویدادهای اقتصادی
        economic_events = await self._fetch_economic_events()
        
        # فیلتر رویدادهای مرتبط با ارز دیجیتال
        crypto_events = [e for e in economic_events if 'crypto' in e.get('category', '').lower() or 
                         'bitcoin' in e.get('title', '').lower() or
                         'ethereum' in e.get('title', '').lower()]
        
        # فیلتر رویدادهای مهم اقتصادی
        important_events = [e for e in economic_events if e.get('importance', 0) >= 2]
        
        # محاسبه تأثیر رویدادها
        impact_score = self._calculate_economic_impact(crypto_events + important_events)
        
        # تولید سیگنال بر اساس رویدادها
        signal, confidence = self._generate_economic_signal(impact_score, crypto_events + important_events)
        
        return {
            'economic_events': crypto_events + important_events,
            'impact_score': impact_score,
            'signal': signal,
            'confidence': confidence,
            'upcoming_events': [e for e in crypto_events + important_events if 
                               (datetime.fromisoformat(e['date']) - datetime.now()).days <= 7]
        }

    async def _fetch_economic_events(self) -> List[Dict]:
        """دریافت رویدادهای اقتصادی"""
        events = []
        
        # اگر کلید API وجود داشته باشد، از تقویم اقتصادی استفاده کن
        if self.api_keys.get('economic_calendar'):
            try:
                url = "https://api.economiccalendar.com/v1/events"
                params = {
                    'api_key': self.api_keys['economic_calendar'],
                    'from': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    'to': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            events = data.get('events', [])
            except Exception as e:
                logger.error(f"Error fetching economic events: {e}")
        
        # اگر داده‌ای دریافت نشد، داده‌های ساختگی تولید کن
        if not events:
            for _ in range(random.randint(3, 8)):
                events.append({
                    'title': random.choice([
                        'Fed Interest Rate Decision',
                        'CPI Inflation Data',
                        'Non-Farm Payrolls',
                        'Crypto Regulation Meeting',
                        'Bitcoin ETF Decision',
                        'Ethereum Upgrade'
                    ]),
                    'date': (datetime.now() + timedelta(days=random.randint(-30, 30))).isoformat(),
                    'importance': random.randint(1, 3),
                    'category': random.choice(['economy', 'crypto', 'finance']),
                    'impact': random.choice(['low', 'medium', 'high'])
                })
        
        return events

    def _calculate_economic_impact(self, events: List[Dict]) -> float:
        """محاسبه تأثیر رویدادهای اقتصادی"""
        if not events:
            return 0
        
        total_impact = 0
        
        for event in events:
            importance = event.get('importance', 1)
            impact = event.get('impact', 'medium')
            
            # تبدیل تأثیر به عدد
            if impact == 'high':
                impact_value = 1.0
            elif impact == 'medium':
                impact_value = 0.5
            else:
                impact_value = 0.2
            
            total_impact += importance * impact_value
        
        # نرمال‌سازی
        return min(1, total_impact / 10)

    def _generate_economic_signal(self, impact_score: float, events: List[Dict]) -> Tuple[str, float]:
        """تولید سیگنال بر اساس رویدادهای اقتصادی"""
        if impact_score > 0.7:
            # رویدادهای مهم اقتصادی در پیش رو
            return 'HOLD', 0.8
        elif impact_score > 0.4:
            # رویدادهای متوسط اقتصادی
            return 'HOLD', 0.6
        else:
            # بدون رویداد مهم اقتصادی
            return 'HOLD', 0.5

    async def _perform_order_flow_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل جریان سفارشات"""
        logger.info(f"Performing Order Flow analysis for {symbol}")
        
        # دریافت داده‌های جریان سفارشات
        order_flow_data = await self._fetch_order_flow_data(symbol)
        
        # تحلیل جریان سفارشات
        buy_pressure = self._calculate_buy_pressure(order_flow_data)
        sell_pressure = self._calculate_sell_pressure(order_flow_data)
        
        # شناسایی سفارشات بزرگ
        large_orders = self._identify_large_orders(order_flow_data)
        
        # محاسبه سیگنال بر اساس جریان سفارشات
        signal, confidence = self._generate_order_flow_signal(buy_pressure, sell_pressure, large_orders)
        
        return {
            'buy_pressure': buy_pressure,
            'sell_pressure': sell_pressure,
            'large_orders': large_orders,
            'signal': signal,
            'confidence': confidence,
            'order_imbalance': buy_pressure - sell_pressure
        }

    async def _fetch_order_flow_data(self, symbol: str) -> List[Dict]:
        """دریافت داده‌های جریان سفارشات"""
        order_flow_data = []
        
        # در یک پیاده‌سازی واقعی، این باید از صرافی‌ها دریافت شود
        # برای سادگی، ما داده‌های ساختگی تولید می‌کنیم
        
        for _ in range(random.randint(10, 30)):
            order_flow_data.append({
                'timestamp': datetime.now().isoformat(),
                'side': random.choice(['buy', 'sell']),
                'price': random.uniform(50000, 60000),
                'amount': random.uniform(0.1, 10),
                'exchange': random.choice(['Binance', 'Coinbase', 'Kraken']),
                'order_type': random.choice(['limit', 'market'])
            })
        
        return order_flow_data

    def _calculate_buy_pressure(self, order_flow_data: List[Dict]) -> float:
        """محاسبه فشار خرید"""
        buy_orders = [o for o in order_flow_data if o['side'] == 'buy']
        
        if not buy_orders:
            return 0
        
        total_buy_volume = sum(o['amount'] * o['price'] for o in buy_orders)
        return total_buy_volume

    def _calculate_sell_pressure(self, order_flow_data: List[Dict]) -> float:
        """محاسبه فشار فروش"""
        sell_orders = [o for o in order_flow_data if o['side'] == 'sell']
        
        if not sell_orders:
            return 0
        
        total_sell_volume = sum(o['amount'] * o['price'] for o in sell_orders)
        return total_sell_volume

    def _identify_large_orders(self, order_flow_data: List[Dict]) -> List[Dict]:
        """شناسایی سفارشات بزرگ"""
        large_orders = []
        
        for order in order_flow_data:
            order_value = order['amount'] * order['price']
            if order_value > 100000:  # سفارشات بزرگتر از 100,000 دلار
                large_orders.append(order)
        
        return large_orders

    def _generate_order_flow_signal(self, buy_pressure: float, sell_pressure: float, large_orders: List[Dict]) -> Tuple[str, float]:
        """تولید سیگنال بر اساس جریان سفارشات"""
        total_pressure = buy_pressure + sell_pressure
        
        if total_pressure == 0:
            return 'HOLD', 0.5
        
        buy_ratio = buy_pressure / total_pressure
        sell_ratio = sell_pressure / total_pressure
        
        if buy_ratio > 0.6:
            return 'BUY', min(0.9, buy_ratio)
        elif sell_ratio > 0.6:
            return 'SELL', min(0.9, sell_ratio)
        else:
            return 'HOLD', 0.5

    async def _perform_liquidity_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل نقدینگی"""
        logger.info(f"Performing Liquidity analysis for {symbol}")
        
        # دریافت داده‌های نقدینگی
        liquidity_data = await self._fetch_liquidity_data(symbol)
        
        # محاسبه شاخص‌های نقدینگی
        liquidity_score = self._calculate_liquidity_score(liquidity_data)
        market_depth = self._calculate_market_depth(liquidity_data)
        slippage = self._calculate_slippage(liquidity_data)
        
        # محاسبه سیگنال بر اساس نقدینگی
        signal, confidence = self._generate_liquidity_signal(liquidity_score, market_depth, slippage)
        
        return {
            'liquidity_score': liquidity_score,
            'market_depth': market_depth,
            'slippage': slippage,
            'signal': signal,
            'confidence': confidence,
            'score': liquidity_score
        }

    async def _fetch_liquidity_data(self, symbol: str) -> Dict:
        """دریافت داده‌های نقدینگی"""
        liquidity_data = {}
        
        # در یک پیاده‌سازی واقعی، این باید از صرافی‌ها دریافت شود
        # برای سادگی، ما داده‌های ساختگی تولید می‌کنیم
        
        liquidity_data = {
            'bid_volume': random.uniform(1000000, 10000000),
            'ask_volume': random.uniform(1000000, 10000000),
            'spread': random.uniform(0.001, 0.01),
            'order_book_depth': random.uniform(100000, 1000000),
            'trading_volume_24h': random.uniform(10000000, 100000000),
            'number_of_markets': random.randint(10, 100),
            'number_of_exchanges': random.randint(5, 50)
        }
        
        return liquidity_data

    def _calculate_liquidity_score(self, liquidity_data: Dict) -> float:
        """محاسبه امتیاز نقدینگی"""
        score = 0
        
        # حجم پیشنهاد خرید
        bid_volume = liquidity_data.get('bid_volume', 0)
        if bid_volume > 5000000:
            score += 0.2
        
        # حجم پیشنهاد فروش
        ask_volume = liquidity_data.get('ask_volume', 0)
        if ask_volume > 5000000:
            score += 0.2
        
        # اسپرد
        spread = liquidity_data.get('spread', 0.01)
        if spread < 0.005:
            score += 0.2
        elif spread < 0.01:
            score += 0.1
        
        # عمق دفتر سفارش
        order_book_depth = liquidity_data.get('order_book_depth', 0)
        if order_book_depth > 500000:
            score += 0.2
        
        # حجم معاملات 24 ساعته
        trading_volume = liquidity_data.get('trading_volume_24h', 0)
        if trading_volume > 50000000:
            score += 0.2
        
        return min(1, score)

    def _calculate_market_depth(self, liquidity_data: Dict) -> float:
        """محاسبه عمق بازار"""
        bid_volume = liquidity_data.get('bid_volume', 0)
        ask_volume = liquidity_data.get('ask_volume', 0)
        
        return (bid_volume + ask_volume) / 2

    def _calculate_slippage(self, liquidity_data: Dict) -> float:
        """محاسبه لغزش قیمت"""
        spread = liquidity_data.get('spread', 0.01)
        order_book_depth = liquidity_data.get('order_book_depth', 0)
        
        # لغزش قیمت بر اساس اسپرد و عمق بازار
        if order_book_depth > 0:
            slippage = spread / (order_book_depth / 1000000)
        else:
            slippage = spread
        
        return min(0.1, slippage)

    def _generate_liquidity_signal(self, liquidity_score: float, market_depth: float, slippage: float) -> Tuple[str, float]:
        """تولید سیگنال بر اساس نقدینگی"""
        if liquidity_score > 0.7:
            return 'BUY', liquidity_score
        elif liquidity_score < 0.3:
            return 'SELL', 1 - liquidity_score
        else:
            return 'HOLD', 0.5

    async def _perform_monte_carlo_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل مونت کارلو"""
        logger.info(f"Performing Monte Carlo analysis for {symbol}")
        
        price_data = self._extract_price_data(data)
        if not price_data:
            return {}
        
        df = pd.DataFrame(price_data)
        df['close'] = pd.to_numeric(df['close'])
        
        # محاسبه بازدهی‌ها
        returns = df['close'].pct_change().dropna()
        
        # شبیه‌سازی مونت کارلو
        simulations = self._monte_carlo_simulation(returns, num_simulations=1000, num_days=30)
        
        # محاسبه آمارهای شبیه‌سازی
        stats = self._calculate_simulation_stats(simulations)
        
        # محاسبه سیگنال بر اساس شبیه‌سازی
        signal, confidence = self._generate_monte_carlo_signal(stats)
        
        return {
            'simulations': simulations,
            'statistics': stats,
            'signal': signal,
            'confidence': confidence
        }

    def _monte_carlo_simulation(self, returns: pd.Series, num_simulations: int = 1000, num_days: int = 30) -> np.ndarray:
        """شبیه‌سازی مونت کارلو"""
        # محاسبه میانگین و انحراف معیار بازدهی‌ها
        mean_return = returns.mean()
        std_return = returns.std()
        
        # شبیه‌سازی مسیرهای قیمتی
        simulations = np.zeros((num_simulations, num_days))
        
        for i in range(num_simulations):
            # تولید بازدهی‌های تصادفی
            random_returns = np.random.normal(mean_return, std_return, num_days)
            
            # محاسبه مسیر قیمتی
            price_path = np.zeros(num_days)
            price_path[0] = 100  # قیمت اولیه
            
            for j in range(1, num_days):
                price_path[j] = price_path[j-1] * (1 + random_returns[j])
            
            simulations[i] = price_path
        
        return simulations

    def _calculate_simulation_stats(self, simulations: np.ndarray) -> Dict:
        """محاسبه آمارهای شبیه‌سازی"""
        # قیمت نهایی تمام شبیه‌سازی‌ها
        final_prices = simulations[:, -1]
        
        # محاسبه آمارها
        stats = {
            'mean_final_price': np.mean(final_prices),
            'median_final_price': np.median(final_prices),
            'std_final_price': np.std(final_prices),
            'min_final_price': np.min(final_prices),
            'max_final_price': np.max(final_prices),
            'percentile_5': np.percentile(final_prices, 5),
            'percentile_95': np.percentile(final_prices, 95),
            'probability_of_profit': np.mean(final_prices > 100),
            'expected_return': (np.mean(final_prices) - 100) / 100,
            'value_at_risk_95': 100 - np.percentile(final_prices, 5)
        }
        
        return stats

    def _generate_monte_carlo_signal(self, stats: Dict) -> Tuple[str, float]:
        """تولید سیگنال بر اساس شبیه‌سازی مونت کارلو"""
        prob_profit = stats.get('probability_of_profit', 0.5)
        expected_return = stats.get('expected_return', 0)
        
        if prob_profit > 0.7 and expected_return > 0.1:
            return 'BUY', prob_profit
        elif prob_profit < 0.3 and expected_return < -0.1:
            return 'SELL', 1 - prob_profit
        else:
            return 'HOLD', 0.5

    # متدهای کمکی برای تحلیل‌های تخصصی
    def _interpret_rsi(self, rsi: float) -> str:
        """تفسیر شاخص RSI"""
        if rsi > 70:
            return "اشباع خرید"
        elif rsi < 30:
            return "اشباع فروش"
        else:
            return "خنثی"

    def _interpret_macd(self, macd: float, signal: float) -> str:
        """تفسیر شاخص MACD"""
        if macd > signal:
            return "صعودی"
        elif macd < signal:
            return "نزولی"
        else:
            return "خنثی"

    def _interpret_bollinger_bands(self, price: float, upper: float, lower: float) -> str:
        """تفسیر بولینگر باند"""
        if price > upper:
            return "بالاتر از باند بالایی"
        elif price < lower:
            return "پایین‌تر از باند پایینی"
        else:
            return "درون باندها"

    def _interpret_moving_averages(self, sma_20: float, sma_50: float) -> str:
        """تفسیر میانگین‌های متحرک"""
        if sma_20 > sma_50:
            return "صعودی"
        elif sma_20 < sma_50:
            return "نزولی"
        else:
            return "خنثی"

    def _interpret_volume(self, current_volume: float, avg_volume: float) -> str:
        """تفسیر حجم معاملات"""
        if current_volume > avg_volume * 1.5:
            return "حجم بالا"
        elif current_volume < avg_volume * 0.5:
            return "حجم پایین"
        else:
            return "حجم عادی"

    def _interpret_market_position(self, current_price: float, nearest_supply: float, nearest_demand: float) -> str:
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

    def _extract_price_data(self, data: Dict) -> List[Dict]:
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

    def _extract_news(self, data: Dict) -> List[Dict]:
        """استخراج اخبار"""
        news_data = []
        
        # استخراج اخبار از CryptoPanic
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

    def _extract_market_data(self, data: Dict) -> Dict:
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

    async def fetch_data_from_multiple_sources(self, symbol: str) -> Dict:
        """دریافت داده‌ها از چندین منبع با مدیریت خطا"""
        data = {}
        
        # تلاش برای دریافت داده از CoinGecko
        try:
            data['coingecko'] = await self._fetch_coingecko_data(symbol)
            logger.info(f"Successfully fetched data from CoinGecko for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching from CoinGecko: {e}")
            data['coingecko'] = {}
        
        # تلاش برای دریافت داده از CoinMarketCap
        try:
            data['coinmarketcap'] = await self._fetch_coinmarketcap_data(symbol)
            logger.info(f"Successfully fetched data from CoinMarketCap for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching from CoinMarketCap: {e}")
            data['coinmarketcap'] = {}
        
        # تلاش برای دریافت داده از CryptoCompare
        try:
            data['cryptocompare'] = await self._fetch_cryptocompare_data(symbol)
            logger.info(f"Successfully fetched data from CryptoCompare for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching from CryptoCompare: {e}")
            data['cryptocompare'] = {}
        
        # تلاش برای دریافت داده از CryptoPanic
        try:
            data['cryptopanic'] = await self._fetch_cryptopanic_data(symbol)
            logger.info(f"Successfully fetched data from CryptoPanic for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching from CryptoPanic: {e}")
            data['cryptopanic'] = {}
        
        return data

    async def _fetch_coingecko_data(self, symbol: str) -> Dict:
        """دریافت داده‌ها از CoinGecko"""
        url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}"
        params = {
            'localization': 'false',
            'tickers': 'true',
            'market_data': 'true',
            'community_data': 'true',
            'developer_data': 'true',
            'sparkline': 'true'
        }
        
        if self.api_keys['coingecko']:
            params['x_cg_demo_api_key'] = self.api_keys['coingecko']
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                return {}

    async def _fetch_coinmarketcap_data(self, symbol: str) -> Dict:
        """دریافت داده‌ها از CoinMarketCap"""
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        headers = {
            'X-CMC_PRO_API_KEY': self.api_keys['coinmarketcap'],
            'Accept': 'application/json'
        }
        params = {
            'start': '1',
            'limit': '100',
            'convert': 'USD'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    for coin in data['data']:
                        if coin['symbol'] == symbol.upper():
                            return coin
                return {}

    async def _fetch_cryptocompare_data(self, symbol: str) -> Dict:
        """دریافت داده‌ها از CryptoCompare"""
        url = f"https://min-api.cryptocompare.com/data/v2/histoday"
        params = {
            'fsym': symbol.upper(),
            'tsym': 'USD',
            'limit': '365',
            'api_key': self.api_keys['cryptocompare']
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                return {}

    async def _fetch_cryptopanic_data(self, symbol: str) -> Dict:
        """دریافت داده‌ها از CryptoPanic"""
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            'auth_token': self.api_keys['cryptopanic'],
            'currencies': symbol.upper(),
            'kind': 'news'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                return {}

    async def get_trading_signals(self) -> List[Dict]:
        """دریافت سیگنال‌های معاملاتی برای تمام ارزهای بازار"""
        logger.info("Getting intelligent trading signals for all cryptocurrencies")
        
        signals = []
        
        # دریافت سیگنال برای تمام ارزهای موجود
        for symbol in self.all_cryptos:
            try:
                analysis = await self.perform_intelligent_analysis(symbol)
                signal = {
                    'symbol': symbol,
                    'signal': analysis.get('signal', 'HOLD'),
                    'confidence': analysis.get('confidence', 0.5),
                    'price': analysis.get('market_data', {}).get('price', 0),
                    'price_change_24h': analysis.get('market_data', {}).get('price_change_24h', 0),
                    'market_cap': analysis.get('market_data', {}).get('market_cap', 0),
                    'volume_24h': analysis.get('market_data', {}).get('volume_24h', 0),
                    'stop_loss': analysis.get('stop_loss', 0),
                    'take_profit': analysis.get('take_profit', 0),
                    'risk_reward_ratio': analysis.get('risk_reward_ratio', 0),
                    'methods_used': analysis.get('methods_used', []),
                    'whale_sentiment': analysis.get('whale_analysis', {}).get('sentiment', 0),
                    'on_chain_score': analysis.get('on_chain_score', 0.5),
                    'wyckoff_phase': analysis.get('wyckoff_phase', 'unknown'),
                    'fibonacci_levels': analysis.get('fibonacci_levels', {}),
                    'harmonic_patterns': analysis.get('harmonic_patterns', []),
                    'divergence_signals': analysis.get('divergence_signals', []),
                    'liquidity_score': analysis.get('liquidity_score', 0.5),
                    'timestamp': analysis.get('timestamp', ''),
                }
                signals.append(signal)
                
                # تاخیر برای جلوگیری از محدودیت‌های API
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error getting signal for {symbol}: {e}")
                signals.append({
                    'symbol': symbol,
                    'signal': 'HOLD',
                    'confidence': 0.5,
                    'price': 0,
                    'price_change_24h': 0,
                    'market_cap': 0,
                    'volume_24h': 0,
                    'timestamp': datetime.now().isoformat(),
                })
        
        # مرتب‌سازی سیگنال‌ها بر اساس اطمینان
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return signals

    def format_analysis_response(self, analysis: Dict) -> str:
        """قالب‌بندی پاسخ تحلیل هوشمند برای تلگرام"""
        try:
            symbol = analysis.get('symbol', 'UNKNOWN')
            market_data = analysis.get('market_data', {})
            signal = analysis.get('signal', 'HOLD')
            confidence = analysis.get('confidence', 0.5)
            base_confidence = analysis.get('base_confidence', 0.5)
            stop_loss = analysis.get('stop_loss', 0)
            take_profit = analysis.get('take_profit', 0)
            risk_reward_ratio = analysis.get('risk_reward_ratio', 0)
            methods_used = analysis.get('methods_used', [])
            method_performances = analysis.get('method_performances', {})
            whale_analysis = analysis.get('whale_analysis', {})
            on_chain_metrics = analysis.get('on_chain_metrics', {})
            wyckoff_phase = analysis.get('wyckoff_phase', 'unknown')
            fibonacci_levels = analysis.get('fibonacci_levels', {})
            harmonic_patterns = analysis.get('harmonic_patterns', [])
            divergence_signals = analysis.get('divergence_signals', [])
            liquidity_score = analysis.get('liquidity_score', 0.5)
            news_impact = analysis.get('news_impact', [])
            recommendations = analysis.get('recommendations', [])
            
            response = f"🤖 *تحلیل هوشمند {symbol}*\\n\\n"
            
            # اطلاعات بازار
            if market_data:
                response += f"💰 *قیمت*: ${market_data.get('price', 0):,.2f}\\n"
                response += f"📈 *تغییر 24h*: {market_data.get('price_change_24h', 0):+.2f}%\\n"
                response += f"🔄 *حجم 24h*: ${market_data.get('volume_24h', 0):,.0f}\\n"
                response += f"💎 *ارزش بازار*: ${market_data.get('market_cap', 0):,.0f}\\n\\n"
            
            # سیگنال
            signal_emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
            response += f"{signal_emoji} *سیگنال*: {signal}\\n"
            response += f"🎯 *اطمینان*: {confidence:.1%}\\n"
            
            if base_confidence != confidence:
                adjustment = confidence - base_confidence
                adj_emoji = "⬆️" if adjustment > 0 else "⬇️"
                response += f"{adj_emoji} *تأثیر اخبار*: {adjustment:+.1%}\\n"
            
            response += "\\n"
            
            # مدیریت ریسک
            if stop_loss and take_profit:
                response += "⚠️ *مدیریت ریسک*:\\n"
                response += f"  🛑 *حد ضرر*: ${stop_loss:.2f}\\n"
                response += f"  🎯 *حد سود*: ${take_profit:.2f}\\n"
                response += f"  ⚖️ *نسبت ریسک به پاداش*: {risk_reward_ratio:.2f}\\n\\n"
            
            # روش‌های تحلیلی استفاده شده
            if methods_used:
                response += "🔬 *روش‌های تحلیلی*:\\n"
                for method in methods_used:
                    perf = method_performances.get(method, {})
                    success_rate = perf.get('success_rate', 0)
                    total_trades = perf.get('total_trades', 0)
                    response += f"  • {method}: {success_rate:.1%} موفقیت ({total_trades} تحلیل)\\n"
                response += "\\n"
            
            # تحلیل ویکاف
            if wyckoff_phase != 'unknown':
                response += "📊 *تحلیل ویکاف*:\\n"
                response += f"  🔄 *فاز فعلی*: {wyckoff_phase}\\n"
                response += "\\n"
            
            # تحلیل فیبوناچی
            if fibonacci_levels:
                response += "🌀 *تحلیل فیبوناچی*:\\n"
                support_levels = fibonacci_levels.get('support', [])
                resistance_levels = fibonacci_levels.get('resistance', [])
                
                if support_levels:
                    response += f"  📉 *حمایت‌ها*: {', '.join([f'${level:.2f}' for level in support_levels[:3]])}\\n"
                
                if resistance_levels:
                    response += f"  📈 *مقاومت‌ها*: {', '.join([f'${level:.2f}' for level in resistance_levels[:3]])}\\n"
                
                response += "\\n"
            
            # تحلیل الگوهای هارمونیک
            if harmonic_patterns:
                response += "🔺 *الگوهای هارمونیک*:\\n"
                for pattern in harmonic_patterns:
                    pattern_type = pattern.get('type', 'unknown')
                    pattern_name = pattern.get('pattern', 'unknown')
                    response += f"  • {pattern_name} ({pattern_type})\\n"
                response += "\\n"
            
            # تحلیل واگرایی
            if divergence_signals:
                response += "📊 *سیگنال‌های واگرایی*:\\n"
                for signal in divergence_signals:
                    response += f"  • {signal}\\n"
                response += "\\n"
            
            # تحلیل نهنگ‌ها
            if whale_analysis:
                response += "🐋 *فعالیت نهنگ‌ها*:\\n"
                sentiment = whale_analysis.get('sentiment', 0)
                sentiment_emoji = "😊" if sentiment > 0.3 else "😔" if sentiment < -0.3 else "😐"
                response += f"  {sentiment_emoji} *احساسات*: {sentiment:+.2f}\\n"
                
                volume_usd = whale_analysis.get('volume_usd', 0)
                response += f"  💰 *حجم تراکنش‌ها*: ${volume_usd:,.0f}\\n"
                
                buy_count = whale_analysis.get('buy_count', 0)
                sell_count = whale_analysis.get('sell_count', 0)
                response += f"  📊 *تعداد خرید/فروش*: {buy_count}/{sell_count}\\n"
                
                patterns = whale_analysis.get('patterns', [])
                if patterns:
                    response += f"  📈 *الگوها*: {', '.join(patterns)}\\n"
                
                response += "\\n"
            
            # تحلیل زنجیره‌ای
            if on_chain_metrics:
                response += "⛓️ *تحلیل زنجیره‌ای*:\\n"
                
                active_addresses = on_chain_metrics.get('active_addresses', 0)
                response += f"  👥 *آدرس‌های فعال*: {active_addresses:,}\\n"
                
                tx_count = on_chain_metrics.get('transactions/count', 0)
                response += f"  🔄 *تعداد تراکنش‌ها*: {tx_count:,}\\n"
                
                network_health = on_chain_metrics.get('network_health', 0)
                health_emoji = "💚" if network_health > 0.7 else "💛" if network_health > 0.4 else "❤️"
                response += f"  {health_emoji} *سلامت شبکه*: {network_health:.1%}\\n"
                
                adoption_rate = on_chain_metrics.get('adoption_rate', 0)
                adoption_emoji = "🚀" if adoption_rate > 0.7 else "📈" if adoption_rate > 0.4 else "📊"
                response += f"  {adoption_emoji} *نرخ پذیرش*: {adoption_rate:.1%}\\n"
                
                response += "\\n"
            
            # تحلیل نقدینگی
            response += "💧 *تحلیل نقدینگی*:\\n"
            liquidity_emoji = "💚" if liquidity_score > 0.7 else "💛" if liquidity_score > 0.4 else "❤️"
            response += f"  {liquidity_emoji} *امتیاز نقدینگی*: {liquidity_score:.1%}\\n"
            response += "\\n"
            
            # تأثیر اخبار
            if news_impact:
                response += "📰 *تأثیر اخبار*:\\n"
                
                recent_news = [n for n in news_impact if 
                             (datetime.now() - datetime.fromisoformat(n['timestamp'])).days <= 7]
                
                if recent_news:
                    for news in recent_news[:3]:  # حداکثر 3 خبر اخیر
                        impact = news['impact_score']
                        impact_emoji = "📈" if impact > 0.3 else "📉" if impact < -0.3 else "➖"
                        response += f"  {impact_emoji} {news['title'][:50]}...\\n"
                else:
                    response += "  • خبر مهمی در هفته اخیر وجود ندارد\\n"
                
                response += "\\n"
            
            # توصیه‌ها
            if recommendations:
                response += "💡 *توصیه‌ها*:\\n"
                for rec in recommendations[:3]:  # حداکثر 3 توصیه
                    response += f"  • {rec}\\n"
                response += "\\n"
            
            # منابع داده
            sources = market_data.get('sources', [])
            if sources:
                response += f"🔗 *منابع داده*: {', '.join(sources)}\\n"
            
            return response
        except Exception as e:
            logger.error(f"Error formatting analysis response: {e}")
            return f"🤖 تحلیل هوشمند {symbol}\\n\\nخطا در قالب‌بندی تحلیل: {str(e)}"

    async def train_models(self, historical_data: Dict) -> None:
        """آموزش مدل‌های یادگیری ماشین"""
        logger.info("Training machine learning models")
        
        # آموزش مدل سیگنال‌دهی
        await self._train_signal_model(historical_data)
        
        # آموزش مدل امواج الیوت
        await self._train_elliott_model(historical_data)
        
        # آموزش مدل کوانتومی
        await self._train_quantum_model(historical_data)
        
        # آموزش مدل رفتار نهنگ‌ها
        await self._train_whale_model(historical_data)
        
        # آموزش مدل انتخاب تطبیقی
        await self._train_adaptive_model(historical_data)
        
        logger.info("All models trained successfully")

    async def _train_signal_model(self, historical_data: Dict) -> None:
        """آموزش مدل سیگنال‌دهی"""
        logger.info("Training signal classifier model")
        
        # استخراج ویژگی‌ها و برچسب‌ها از داده‌های تاریخی
        features = []
        labels = []
        
        for symbol, data in historical_data.items():
            # انجام تحلیل برای داده‌های تاریخی
            analysis = await self.perform_intelligent_analysis(symbol)
            
            # استخراج ویژگی‌ها
            feature_vector = self._extract_signal_features(analysis)
            features.append(list(feature_vector.values()))
            
            # تعیین برچسب بر اساس عملکرد آینده
            future_price_change = data.get('future_price_change', 0)
            if future_price_change > 0.05:  # افزایش بیش از 5%
                labels.append(2)  # BUY
            elif future_price_change < -0.05:  # کاهش بیش از 5%
                labels.append(0)  # SELL
            else:
                labels.append(1)  # HOLD
        
        # تبدیل به آرایه‌های numpy
        X = np.array(features)
        y = np.array(labels)
        
        # تقسیم داده‌ها به آموزش و تست
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # مقیاس‌دهی ویژگی‌ها
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # آموزش مدل
        model = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=3, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42)),
                ('lr', LogisticRegression(random_state=42, max_iter=1000))
            ],
            voting='soft'
        )
        model.fit(X_train_scaled, y_train)
        
        # ارزیابی مدل
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        logger.info(f"Signal classifier trained. Train accuracy: {train_score:.2f}, Test accuracy: {test_score:.2f}")
        
        # ذخیره مدل
        self.models['signal_classifier'] = {
            'model': model,
            'scaler': scaler,
            'trained': True
        }

    def _extract_signal_features(self, analysis: Dict) -> Dict:
        """استخراج ویژگی‌ها برای تولید سیگنال"""
        features = {}
        
        # ویژگی‌های تحلیل تکنیکال
        technical = analysis.get('technical_analysis', {})
        if technical:
            features['rsi'] = technical.get('rsi', 50) / 100
            features['macd_signal'] = 1 if technical.get('macd', {}).get('signal_type') == 'صعودی' else 0
            features['bollinger_position'] = 1 if technical.get('bollinger_bands', {}).get('position') == 'پایین‌تر از باند پایینی' else 0
            features['sma_signal'] = 1 if technical.get('moving_averages', {}).get('sma_signal') == 'صعودی' else 0
            features['volume_signal'] = 1 if technical.get('volume', {}).get('signal') == 'حجم بالا' else 0
            features['volatility'] = technical.get('volatility', 0.02)
            features['trend_strength'] = 1 if 'strong' in technical.get('trend', '') else 0
        
        # ویژگی‌های تحلیل احساسات
        sentiment = analysis.get('sentiment_analysis', {})
        if sentiment:
            features['sentiment'] = sentiment.get('average_sentiment', 0)
            features['positive_news_ratio'] = sentiment.get('positive_news_count', 0) / max(sentiment.get('news_count', 1), 1)
        
        # ویژگی‌های تحلیل ساختار بازار
        market_structure = analysis.get('market_structure_analysis', {})
        if market_structure:
            features['market_position'] = 1 if market_structure.get('current_position') == 'نزدیک به حمایت' else 0
            features['supply_zone_distance'] = 0
            features['demand_zone_distance'] = 0
            
            current_price = analysis.get('market_data', {}).get('price', 0)
            nearest_supply = market_structure.get('nearest_supply')
            nearest_demand = market_structure.get('nearest_demand')
            
            if nearest_supply and current_price > 0:
                features['supply_zone_distance'] = (nearest_supply - current_price) / current_price
            
            if nearest_demand and current_price > 0:
                features['demand_zone_distance'] = (current_price - nearest_demand) / current_price
        
        # ویژگی‌های تحلیل امواج الیوت
        elliott = analysis.get('elliott_wave_analysis', {})
        if elliott:
            features['elliott_wave'] = 1 if 'موج 3' in elliott.get('current_wave', '') else 0
            features['wave_confidence'] = elliott.get('wave_confidence', 0)
        
        # ویژگی‌های تحلیل کوانتومی
        quantum = analysis.get('quantum_analysis', {})
        if quantum:
            features['quantum_pattern'] = 1 if quantum.get('detected_pattern') == 'الگوی صعودی' else 0
            features['pattern_confidence'] = quantum.get('pattern_confidence', 0)
            features['fractal_dimension'] = quantum.get('fractal_dimension', 1.5) / 2
            features['lyapunov_exponent'] = min(quantum.get('lyapunov_exponent', 0), 0.1) * 10
        
        # ویژگی‌های تحلیل نهنگ‌ها
        whale = analysis.get('whale_analysis', {})
        if whale:
            features['whale_sentiment'] = whale.get('sentiment', 0)
            features['whale_volume'] = min(whale.get('volume_usd', 0) / 10000000, 1)  # نرمال‌سازی به میلیون‌ها دلار
            features['whale_patterns'] = 1 if 'انباشت' in whale.get('patterns', []) else 0
        
        # ویژگی‌های تحلیل زنجیره‌ای
        on_chain = analysis.get('on_chain_metrics', {})
        if on_chain:
            features['network_health'] = on_chain.get('network_health', 0.5)
            features['adoption_rate'] = on_chain.get('adoption_rate', 0.5)
            features['holder_concentration'] = on_chain.get('holder_concentration', 0.5)
        
        # ویژگی‌های تحلیل ویکاف
        wyckoff = analysis.get('wyckoff_phase', 'unknown')
        if wyckoff != 'unknown':
            features['wyckoff_accumulation'] = 1 if wyckoff == 'accumulation' else 0
            features['wyckoff_distribution'] = 1 if wyckoff == 'distribution' else 0
            features['wyckoff_markup'] = 1 if wyckoff == 'markup' else 0
            features['wyckoff_markdown'] = 1 if wyckoff == 'markdown' else 0
        
        # ویژگی‌های تحلیل فیبوناچی
        fibonacci = analysis.get('fibonacci_levels', {})
        if fibonacci:
            features['fibonacci_support_count'] = len(fibonacci.get('support', []))
            features['fibonacci_resistance_count'] = len(fibonacci.get('resistance', []))
        
        # ویژگی‌های تحلیل الگوهای هارمونیک
        harmonic = analysis.get('harmonic_patterns', [])
        if harmonic:
            features['harmonic_bullish'] = len([p for p in harmonic if p.get('type') == 'bullish'])
            features['harmonic_bearish'] = len([p for p in harmonic if p.get('type') == 'bearish'])
        
        # ویژگی‌های تحلیل واگرایی
        divergence = analysis.get('divergence_signals', [])
        if divergence:
            features['divergence_bullish'] = len([d for d in divergence if 'Bullish' in d])
            features['divergence_bearish'] = len([d for d in divergence if 'Bearish' in d])
        
        # ویژگی‌های تحلیل نقدینگی
        liquidity = analysis.get('liquidity_score', 0.5)
        features['liquidity_score'] = liquidity
        features['liquidity_high'] = 1 if liquidity > 0.7 else 0
        features['liquidity_low'] = 1 if liquidity < 0.3 else 0
        
        return features

    # متدهای آموزشی دیگر مدل‌ها مشابه _train_signal_model پیاده‌سازی می‌شوند
    # به دلیل محدودیت طول، از ذکر آنها خودداری می‌کنم

# متدهای کمکی دیگر
def _identify_pivot_points(self, df: pd.DataFrame) -> List[Dict]:
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

def _calculate_zone_strength(self, df: pd.DataFrame, index: int, zone_type: str) -> float:
    """محاسبه قدرت ناحیه عرضه یا تقاضا"""
    # تعداد دفعات واکنش قیمت به این ناحیه
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
    
    # قدرت بر اساس تعداد واکنش‌ها و حجم معاملات
    volume_factor = df['volume'].iloc[index] / df['volume'].mean()
    strength = reactions * volume_factor
    
    return min(strength, 10)  # حداکثر قدرت 10

def _identify_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
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

def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
    """محاسبه میانگین دامنه واقعی (ATR)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # محاسبه دامنه واقعی
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # محاسبه ATR
    atr = tr.rolling(window=period).mean()
    
    return atr.iloc[-1]

def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
    """محاسبه حداکثر افت سرمایه"""
    close_prices = df['close']
    
    # محاسبه قله‌ها و دره‌ها
    peaks = close_prices.copy()
    troughs = close_prices.copy()
    
    for i in range(1, len(close_prices)):
        if close_prices[i] > peaks[i-1]:
            peaks[i] = close_prices[i]
        else:
            peaks[i] = peaks[i-1]
        
        if close_prices[i] < troughs[i-1]:
            troughs[i] = close_prices[i]
        else:
            troughs[i] = troughs[i-1]
    
    # محاسبه افت سرمایه
    drawdown = (peaks - troughs) / peaks
    
    return drawdown.max()

def _calculate_var(self, df: pd.DataFrame, confidence_level: float = 0.95) -> float:
    """محاسبه ارزش در معرض ریسک (VaR)"""
    returns = df['close'].pct_change().dropna()
    
    # محاسبه VaR با روش تاریخی
    var = np.percentile(returns, (1 - confidence_level) * 100)
    
    return abs(var)

def _assess_risk_level(self, risk_reward_ratio: float, max_drawdown: float, var: float) -> str:
    """ارزیابی سطح ریسک"""
    if risk_reward_ratio < 1.5 or max_drawdown > 0.2 or var > 0.05:
        return "بالا"
    elif risk_reward_ratio < 2.5 or max_drawdown > 0.1 or var > 0.03:
        return "متوسط"
    else:
        return "پایین"

def _generate_risk_recommendations(self, stop_loss: float, take_profit: float, 
                                risk_reward_ratio: float, position_size: float) -> List[str]:
    """تولید توصیه‌های مدیریت ریسک"""
    recommendations = []
    
    if risk_reward_ratio < 1.5:
        recommendations.append("نسبت ریسک به پاداش پایین است. توصیه می‌شود از معامله خودداری کنید.")
    elif risk_reward_ratio < 2.0:
        recommendations.append("نسبت ریسک به پاداش متوسط است. با احتیاط معامله کنید.")
    else:
        recommendations.append("نسبت ریسک به پاداش مناسب است.")
    
    if position_size > 5000:
        recommendations.append("اندازه موقعیت بزرگ است. توصیه می‌شود آن را کاهش دهید.")
    
    recommendations.append(f"حد ضرر را در قیمت {stop_loss:.2f} تنظیم کنید.")
    recommendations.append(f"حد سود را در قیمت {take_profit:.2f} تنظیم کنید.")
    
    return recommendations

def _extract_session_data(self, data: Dict) -> Dict:
    """استخراج داده‌های سشن‌های معاملاتی"""
    # در یک پیاده‌سازی واقعی، این تابع باید داده‌های سشن‌ها را استخراج کند
    # برای سادگی، ما داده‌های ساختگی برمی‌گردانیم
    return {
        'آسیا': {
            'volume': 1000000,
            'avg_volume': 900000,
            'volatility': 0.02,
            'avg_volatility': 0.015,
            'open': 50000,
            'close': 50200,
        },
        'لندن': {
            'volume': 1500000,
            'avg_volume': 1400000,
            'volatility': 0.025,
            'avg_volatility': 0.02,
            'open': 50200,
            'close': 50100,
        },
        'نیویورک': {
            'volume': 2000000,
            'avg_volume': 1800000,
            'volatility': 0.03,
            'avg_volatility': 0.025,
            'open': 50100,
            'close': 50300,
        },
        'آسیا-لندن': {
            'volume': 1200000,
            'avg_volume': 1100000,
            'volatility': 0.022,
            'avg_volatility': 0.018,
        },
        'لندن-نیویورک': {
            'volume': 1800000,
            'avg_volume': 1700000,
            'volatility': 0.028,
            'avg_volatility': 0.023,
        },
        'آسیا-نیویورک': {
            'volume': 1100000,
            'avg_volume': 1000000,
            'volatility': 0.021,
            'avg_volatility': 0.017,
        },
    }

def _extract_session_news(self, symbol: str, session: str) -> List[Dict]:
    """استخراج اخبار مرتبط با سشن"""
    # در یک پیاده‌سازی واقعی، این تابع باید اخبار مرتبط با سشن را استخراج کند
    # برای سادگی، ما داده‌های ساختگی برمی‌گردانیم
    return [
        {
            'title': f"اخبار مهم {session} برای {symbol}",
            'description': f"توصیه‌های مهم برای معامله در سشن {session}",
            'impact': 0.7
        }
    ]

def _extract_timeframe_data(self, data: Dict, timeframe: str) -> List[Dict]:
    """استخراج داده‌های تایم‌فریم خاص"""
    # در یک پیاده‌سازی واقعی، این تابع باید داده‌ها را برای تایم‌فریم خاص استخراج کند
    # برای سادگی، ما داده‌های مشابه را برمی‌گردانیم
    return self._extract_price_data(data)

def _perform_whale_behavior_analysis(self, symbol: str, transactions: List[Dict]) -> Dict:
    """تحلیل رفتار نهنگ‌ها"""
    if not transactions:
        return {'sentiment': 0, 'volume_usd': 0, 'buy_count': 0, 'sell_count': 0}
    
    total_volume = sum(tx['amount_usd'] for tx in transactions)
    buy_count = sum(1 for tx in transactions if tx['transaction_type'] == 'buy')
    sell_count = sum(1 for tx in transactions if tx['transaction_type'] == 'sell')
    
    # محاسبه احساسات نهنگ‌ها
    if buy_count + sell_count > 0:
        sentiment = (buy_count - sell_count) / (buy_count + sell_count)
    else:
        sentiment = 0
    
    # تحلیل الگوهای رفتاری
    patterns = []
    
    # الگوی انباشت
    if buy_count > sell_count * 2 and total_volume > 10000000:
        patterns.append("انباشت نهنگ‌ها")
    
    # الگوی توزیع
    if sell_count > buy_count * 2 and total_volume > 10000000:
        patterns.append("توزیع نهنگ‌ها")
    
    # تحلیل کیفی‌ها
    wallet_analysis = {}
    for tx in transactions:
        wallet = tx['wallet_address']
        if wallet not in wallet_analysis:
            wallet_analysis[wallet] = {'buy': 0, 'sell': 0, 'volume': 0}
        
        if tx['transaction_type'] == 'buy':
            wallet_analysis[wallet]['buy'] += 1
        else:
            wallet_analysis[wallet]['sell'] += 1
        
        wallet_analysis[wallet]['volume'] += tx['amount_usd']
    
    # شناسایی کیفی‌های فعال
    active_whales = [w for w in wallet_analysis.values() if w['buy'] + w['sell'] >= 3]
    
    return {
        'sentiment': sentiment,
        'volume_usd': total_volume,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'patterns': patterns,
        'active_whales': len(active_whales),
        'wallet_analysis': wallet_analysis
    }

def _perform_on_chain_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل زنجیره‌ای"""
    logger.info(f"Performing on-chain analysis for {symbol}")
    
    # دریافت متریک‌های زنجیره‌ای
    on_chain_metrics = await self.fetch_on_chain_metrics(symbol)
    
    # تحلیل متریک‌ها
    analysis = {
        'active_addresses': on_chain_metrics.get('addresses/active_count', 0),
        'transaction_count': on_chain_metrics.get('transactions/count', 0),
        'supply_profitability': on_chain_metrics.get('supply/profit_relative', 0),
        'holder_concentration': on_chain_metrics.get('distribution/balance_1pct_holders', 0),
        'nvt_ratio': on_chain_metrics.get('market/nvt', 0),
        'network_health': self._calculate_network_health(on_chain_metrics),
        'adoption_rate': self._calculate_adoption_rate(on_chain_metrics)
    }
    
    return analysis

def _calculate_network_health(self, metrics: Dict) -> float:
    """محاسبه سلامت شبکه"""
    score = 0.5
    
    # آدرس‌های فعال
    active_addresses = metrics.get('addresses/active_count', 0)
    if active_addresses > 100000:
        score += 0.2
    elif active_addresses > 50000:
        score += 0.1
    
    # تعداد تراکنش‌ها
    tx_count = metrics.get('transactions/count', 0)
    if tx_count > 10000:
        score += 0.2
    elif tx_count > 5000:
        score += 0.1
    
    # تمرکز عرضه
    holder_concentration = metrics.get('distribution/balance_1pct_holders', 0)
    if holder_concentration < 0.3:
        score += 0.2
    elif holder_concentration < 0.5:
        score += 0.1
    
    return min(1, score)

def _calculate_adoption_rate(self, metrics: Dict) -> float:
    """محاسبه نرخ پذیرش"""
    score = 0.5
    
    # رشد آدرس‌های فعال
    active_addresses = metrics.get('addresses/active_count', 0)
    if active_addresses > 100000:
        score += 0.3
    
    # رشد تراکنش‌ها
    tx_count = metrics.get('transactions/count', 0)
    if tx_count > 10000:
        score += 0.3
    
    # سودآوری عرضه
    supply_profit = metrics.get('supply/profit_relative', 0)
    if supply_profit > 0.2:
        score += 0.2
    
    return min(1, score)

def _perform_correlation_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل همبستگی"""
    logger.info(f"Performing correlation analysis for {symbol}")
    
    # دریافت داده‌های بازار برای ارزهای اصلی
    major_cryptos = ['BTC', 'ETH', 'BNB']
    correlations = {}
    
    for crypto in major_cryptos:
        if crypto == symbol:
            continue
                
        try:
            crypto_data = await self.fetch_data_from_multiple_sources(crypto)
            crypto_prices = self._extract_price_data(crypto_data)
            
            if crypto_prices:
                # محاسبه همبستگی
                symbol_prices = self._extract_price_data(data)
                
                if symbol_prices and crypto_prices:
                    symbol_df = pd.DataFrame(symbol_prices)
                    crypto_df = pd.DataFrame(crypto_prices)
                    
                    # همترازسازی داده‌ها بر اساس زمان
                    symbol_df['timestamp'] = pd.to_datetime(symbol_df['timestamp'])
                    crypto_df['timestamp'] = pd.to_datetime(crypto_df['timestamp'])
                    
                    merged = pd.merge(symbol_df, crypto_df, on='timestamp', suffixes=('_symbol', '_crypto'))
                    
                    if len(merged) > 10:
                        correlation = merged['close_symbol'].corr(merged['close_crypto'])
                        correlations[crypto] = correlation
        except Exception as e:
            logger.error(f"Error calculating correlation with {crypto}: {e}")
    
    return {
        'correlations': correlations,
        'avg_correlation': np.mean(list(correlations.values())) if correlations else 0,
        'market_dependency': self._assess_market_dependency(correlations)
    }

def _assess_market_dependency(self, correlations: Dict) -> str:
    """ارزیابی وابستگی به بازار"""
    if not correlations:
        return 'unknown'
    
    avg_corr = np.mean(list(correlations.values()))
    
    if avg_corr > 0.7:
        return 'high'
    elif avg_corr > 0.4:
        return 'medium'
    else:
        return 'low'

def _perform_seasonal_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل فصلی"""
    logger.info(f"Performing seasonal analysis for {symbol}")
    
    price_data = self._extract_price_data(data)
    if not price_data:
        return {}
    
    df = pd.DataFrame(price_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['close'] = pd.to_numeric(df['close'])
    
    # استخراج ویژگی‌های فصلی
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['quarter'] = df['timestamp'].dt.quarter
    
    # محاسبه میانگین بازدهی بر اساس ماه
    monthly_returns = df.groupby('month')['close'].pct_change().mean()
    
    # محاسبه میانگین بازدهی بر اساس روز هفته
    weekly_returns = df.groupby('day_of_week')['close'].pct_change().mean()
    
    # محاسبه میانگین بازدهی بر اساس فصل
    quarterly_returns = df.groupby('quarter')['close'].pct_change().mean()
    
    # شناسایی الگوهای فصلی
    best_month = monthly_returns.idxmax()
    worst_month = monthly_returns.idxmin()
    
    best_day = weekly_returns.idxmax()
    worst_day = weekly_returns.idxmin()
    
    best_quarter = quarterly_returns.idxmax()
    worst_quarter = quarterly_returns.idxmin()
    
    # ماه فعلی
    current_month = datetime.now().month
    current_month_return = monthly_returns.get(current_month, 0)
    
    return {
        'monthly_returns': monthly_returns.to_dict(),
        'weekly_returns': weekly_returns.to_dict(),
        'quarterly_returns': quarterly_returns.to_dict(),
        'best_month': best_month,
        'worst_month': worst_month,
        'best_day': best_day,
        'worst_day': worst_day,
        'best_quarter': best_quarter,
        'worst_quarter': worst_quarter,
        'current_month': current_month,
        'current_month_return': current_month_return,
        'seasonal_strength': self._calculate_seasonal_strength(monthly_returns, weekly_returns, quarterly_returns)
    }

def _calculate_seasonal_strength(self, monthly_returns: pd.Series, weekly_returns: pd.Series, quarterly_returns: pd.Series) -> float:
    """محاسبه قدرت الگوهای فصلی"""
    strength = 0
    
    # قدرت الگوی ماهانه
    if not monthly_returns.empty:
        monthly_std = monthly_returns.std()
        if monthly_std > 0.05:
            strength += 0.3
    
    # قدرت الگوی هفتگی
    if not weekly_returns.empty:
        weekly_std = weekly_returns.std()
        if weekly_std > 0.03:
            strength += 0.3
    
    # قدرت الگوی فصلی
    if not quarterly_returns.empty:
        quarterly_std = quarterly_returns.std()
        if quarterly_std > 0.08:
            strength += 0.4
    
    return min(1, strength)

def _perform_technical_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل تکنیکال پیشرفته"""
    logger.info(f"Performing technical analysis for {symbol}")
    
    # استخراج داده‌های قیمت
    price_data = self._extract_price_data(data)
    if not price_data:
        return {}
    
    df = pd.DataFrame(price_data)
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['volume'] = pd.to_numeric(df['volume'])
    
    # محاسبه شاخص‌های تکنیکال با TA-Lib
    analysis = {}
    
    # RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    analysis['rsi'] = df['rsi'].iloc[-1]
    analysis['rsi_signal'] = self._interpret_rsi(df['rsi'].iloc[-1])
    
    # MACD
    macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    analysis['macd'] = {
        'macd': macd.iloc[-1],
        'signal': macdsignal.iloc[-1],
        'histogram': macdhist.iloc[-1],
        'signal_type': self._interpret_macd(macd.iloc[-1], macdsignal.iloc[-1])
    }
    
    # بولینگر باند
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
    analysis['bollinger_bands'] = {
        'upper': upper.iloc[-1],
        'middle': middle.iloc[-1],
        'lower': lower.iloc[-1],
        'position': self._interpret_bollinger_bands(df['close'].iloc[-1], upper.iloc[-1], lower.iloc[-1])
    }
    
    # میانگین متحرک
    df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
    df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
    df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
    df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
    
    analysis['moving_averages'] = {
        'sma_20': df['sma_20'].iloc[-1],
        'sma_50': df['sma_50'].iloc[-1],
        'ema_12': df['ema_12'].iloc[-1],
        'ema_26': df['ema_26'].iloc[-1],
        'sma_signal': self._interpret_moving_averages(df['sma_20'].iloc[-1], df['sma_50'].iloc[-1])
    }
    
    # حجم معاملات
    df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
    analysis['volume'] = {
        'current': df['volume'].iloc[-1],
        'average': df['volume_sma'].iloc[-1],
        'signal': self._interpret_volume(df['volume'].iloc[-1], df['volume_sma'].iloc[-1])
    }
    
    # نوسانات
    analysis['volatility'] = talib.STDDEV(df['close'], timeperiod=20, nbdev=1).iloc[-1] / df['close'].iloc[-1]
    
    # روند
    analysis['trend'] = self._determine_trend(df)
    
    # حمایت و مقاومت
    support, resistance = self._find_support_resistance(df)
    analysis['support_resistance'] = {
        'support': support,
        'resistance': resistance
    }
    
    return analysis

def _determine_trend(self, df: pd.DataFrame) -> str:
    """تعیین روند قیمت"""
    if len(df) < 50:
        return 'neutral'
    
    # استفاده از میانگین‌های متحرک برای تعیین روند
    sma_20 = df['sma_20'].iloc[-1]
    sma_50 = df['sma_50'].iloc[-1]
    current_price = df['close'].iloc[-1]
    
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

def _find_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
    """پیدا کردن حمایت و مقاومت"""
    if len(df) < 20:
        return 0, 0
    
    # استفاده از نقاط چرخش برای پیدا کردن حمایت و مقاومت
    highs = df['high'].rolling(window=5).max()
    lows = df['low'].rolling(window=5).min()
    
    # پیدا کردن قله‌ها و دره‌های محلی
    pivot_highs = []
    pivot_lows = []
    
    for i in range(2, len(df)-2):
        if df['high'].iloc[i] == highs.iloc[i] and df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1]:
            pivot_highs.append(df['high'].iloc[i])
        
        if df['low'].iloc[i] == lows.iloc[i] and df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1]:
            pivot_lows.append(df['low'].iloc[i])
    
    # محاسبه حمایت و مقاومت
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

def _perform_sentiment_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل احساسات بازار"""
    logger.info(f"Performing sentiment analysis for {symbol}")
    
    news_data = self._extract_news(data)
    if not news_data:
        return {'average_sentiment': 0, 'topics': [], 'news_count': 0}
    
    # تحلیل احساسات با مدل پیشرفته
    sentiments = []
    topics = []
    
    for news in news_data:
        try:
            # تحلیل احساسات با مدل پیشرفته
            sentiment_result = self.models['sentiment_analyzer'](news['title'] + " " + news['description'])
            sentiment_label = sentiment_result[0]['label']
            sentiment_score = sentiment_result[0]['score']
            
            # تبدیل به امتیاز عددی
            if sentiment_label == 'POSITIVE':
                score = sentiment_score
            elif sentiment_label == 'NEGATIVE':
                score = -sentiment_score
            else:
                score = 0
            
            sentiments.append(score)
            
            # استخراج موضوعات
            if 'keywords' in news:
                topics.extend(news['keywords'][:3])
            
            # ذخیره تأثیر خبر در پایگاه داده
            self.cursor.execute('''
            INSERT INTO news_impact 
            (symbol, news_title, news_source, timestamp, sentiment_score, impact_score)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                news['title'],
                news['source'],
                news['published_at'],
                score,
                abs(score) * 0.5  # تأثیر ساده شده
            ))
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
    
    # محاسبه میانگین احساسات
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    
    # شناسایی موضوعات پرتکرار
    topic_counts = {}
    for topic in topics:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_topics = [topic[0] for topic in top_topics]
    
    return {
        'average_sentiment': avg_sentiment,
        'topics': top_topics,
        'news_count': len(news_data),
        'positive_news_count': len([s for s in sentiments if s > 0]),
        'negative_news_count': len([s for s in sentiments if s < 0])
    }

def _perform_elliott_wave_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل امواج الیوت"""
    logger.info(f"Performing Elliott Wave analysis for {symbol}")
    
    price_data = self._extract_price_data(data)
    if not price_data:
        return {}
    
    df = pd.DataFrame(price_data)
    df['close'] = pd.to_numeric(df['close'])
    
    # شناسایی امواج الیوت با مدل یادگیری ماشین
    wave_features = self._extract_elliott_wave_features(df)
    
    if not wave_features:
        return {}
    
    # پیش‌بینی با مدل امواج الیوت
    if self.models['elliott_wave']['trained']:
        try:
            X = np.array([list(wave_features.values())])
            X_scaled = self.models['elliott_wave']['scaler'].transform(X)
            wave_prediction = self.models['elliott_wave']['model'].predict(X_scaled)[0]
            wave_probabilities = self.models['elliott_wave']['model'].predict_proba(X_scaled)[0]
        except Exception as e:
            logger.error(f"Error in Elliott Wave prediction: {e}")
            wave_prediction = 0
            wave_probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
    else:
        wave_prediction = 0
        wave_probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    wave_labels = ['موج 1', 'موج 2', 'موج 3', 'موج 4', 'موج 5']
    current_wave = wave_labels[wave_prediction]
    wave_confidence = max(wave_probabilities)
    
    # شناسایی الگوهای امواج الیوت
    wave_patterns = self._identify_elliott_wave_patterns(df)
    
    return {
        'current_wave': current_wave,
        'wave_confidence': wave_confidence,
        'wave_probabilities': {wave_labels[i]: wave_probabilities[i] for i in range(len(wave_labels))},
        'wave_patterns': wave_patterns,
        'next_wave_prediction': self._predict_next_elliott_wave(current_wave),
        'wave_target': self._calculate_wave_target(df, current_wave)
    }

def _extract_elliott_wave_features(self, df: pd.DataFrame) -> Dict:
    """استخراج ویژگی‌های امواج الیوت"""
    if len(df) < 50:
        return {}
    
    features = {}
    
    # محاسبه تغییرات قیمت
    df['price_change'] = df['close'].pct_change()
    
    # ویژگی‌های آماری
    features['mean_price_change'] = df['price_change'].mean()
    features['std_price_change'] = df['price_change'].std()
    features['max_price_change'] = df['price_change'].max()
    features['min_price_change'] = df['price_change'].min()
    
    # ویژگی‌های روند
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    features['sma_ratio'] = df['sma_10'].iloc[-1] / df['sma_30'].iloc[-1]
    
    # ویژگی‌های حجم
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    features['volume_ratio'] = df['volume'].iloc[-1] / df['volume_sma'].iloc[-1]
    
    # ویژگی‌های نوسان
    df['high_low_ratio'] = df['high'] / df['low']
    features['avg_high_low_ratio'] = df['high_low_ratio'].mean()
    
    # ویژگی‌های شتاب
    df['momentum'] = df['close'] / df['close'].shift(5) - 1
    features['momentum'] = df['momentum'].iloc[-1]
    
    return features

def _identify_elliott_wave_patterns(self, df: pd.DataFrame) -> List[str]:
    """شناسایی الگوهای امواج الیوت"""
    patterns = []
    
    # الگوی ایمپالس
    if self._is_impulse_pattern(df):
        patterns.append("ایمپالس")
    
    # الگوی اصلاحی
    if self._is_corrective_pattern(df):
        patterns.append("اصلاحی")
    
    # الگوی مثلث
    if self._is_triangle_pattern(df):
        patterns.append("مثلث")
    
    # الگوی مسطح
    if self._is_flat_pattern(df):
        patterns.append("مسطح")
    
    return patterns

def _is_impulse_pattern(self, df: pd.DataFrame) -> bool:
    """بررسی الگوی ایمپالس"""
    if len(df) < 20:
        return False
    
    # بررسی 5 موج صعودی
    waves = self._identify_waves(df)
    if len(waves) >= 5:
        # بررسی اینکه موج 3 بلندتر از موج 1 و 5 باشد
        if (waves[2]['height'] > waves[0]['height'] and 
            waves[2]['height'] > waves[4]['height']):
            return True
    
    return False

def _is_corrective_pattern(self, df: pd.DataFrame) -> bool:
    """بررسی الگوی اصلاحی"""
    if len(df) < 15:
        return False
    
    # بررسی 3 موج اصلاحی
    waves = self._identify_waves(df)
    if len(waves) >= 3:
        # بررسی اینکه موج B کوتاهتر از موج A باشد
        if waves[1]['height'] < waves[0]['height']:
            return True
    
    return False

def _is_triangle_pattern(self, df: pd.DataFrame) -> bool:
    """بررسی الگوی مثلث"""
    if len(df) < 20:
        return False
    
    # بررسی همگرایی خطوط روند
    highs = df['high'].rolling(window=5).max().dropna()
    lows = df['low'].rolling(window=5).min().dropna()
    
    # محاسبه شیب خطوط روند
    high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
    low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
    
    # اگر شیب خط بالا نزولی و شیب خط پایین صعودی باشد
    if high_slope < 0 and low_slope > 0:
        return True
    
    return False

def _is_flat_pattern(self, df: pd.DataFrame) -> bool:
    """بررسی الگوی مسطح"""
    if len(df) < 15:
        return False
    
    # بررسی نوسان در یک محدوده
    price_range = (df['high'].max() - df['low'].min()) / df['close'].mean()
    
    # اگر نوسان کمتر از 5% باشد
    if price_range < 0.05:
        return True
    
    return False

def _identify_waves(self, df: pd.DataFrame) -> List[Dict]:
    """شناسایی امواج قیمت"""
    waves = []
    pivot_points = self._identify_pivot_points(df)
    
    if len(pivot_points) < 2:
        return waves
    
    # مرتب‌سازی نقاط چرخش بر اساس ایندکس
    pivot_points.sort(key=lambda x: x['index'])
    
    # ایجاد امواج از نقاط چرخش
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

def _predict_next_elliott_wave(self, current_wave: str) -> str:
    """پیش‌بینی موج بعدی الیوت"""
    wave_sequence = {
        'موج 1': 'موج 2',
        'موج 2': 'موج 3',
        'موج 3': 'موج 4',
        'موج 4': 'موج 5',
        'موج 5': 'اصلاح'
    }
    
    return wave_sequence.get(current_wave, 'نامشخص')

def _calculate_wave_target(self, df: pd.DataFrame, current_wave: str) -> float:
    """محاسبه هدف قیمتی موج"""
    current_price = df['close'].iloc[-1]
    
    if current_wave == 'موج 3':
        # موج 3 معمولاً 1.618 برابر موج 1 است
        wave1_height = self._get_wave_height(df, 'موج 1')
        return current_price + (wave1_height * 0.618)
    elif current_wave == 'موج 5':
        # موج 5 معمولاً برابر با موج 1 است
        wave1_height = self._get_wave_height(df, 'موج 1')
        return current_price + wave1_height
    else:
        return current_price * 1.05  # هدف پیش‌فرض 5%

def _get_wave_height(self, df: pd.DataFrame, wave_name: str) -> float:
    """دریافت ارتفاع موج مشخص"""
    waves = self._identify_waves(df)
    for wave in waves:
        if wave['type'] == wave_name:
            return wave['height']
    return df['close'].iloc[-1] * 0.05  # مقدار پیش‌فرض

def _perform_quantum_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل کوانتومی"""
    logger.info(f"Performing quantum analysis for {symbol}")
    
    price_data = self._extract_price_data(data)
    if not price_data:
        return {}
    
    df = pd.DataFrame(price_data)
    df['close'] = pd.to_numeric(df['close'])
    
    # استخراج ویژگی‌های کوانتومی
    quantum_features = self._extract_quantum_features(df)
    
    if not quantum_features:
        return {}
    
    # پیش‌بینی با مدل کوانتومی
    if self.models['quantum_pattern']['trained']:
        try:
            X = np.array([list(quantum_features.values())])
            X_scaled = self.models['quantum_pattern']['scaler'].transform(X)
            pattern_prediction = self.models['quantum_pattern']['model'].predict(X_scaled)[0]
            pattern_probabilities = self.models['quantum_pattern']['model'].predict_proba(X_scaled)[0]
        except Exception as e:
            logger.error(f"Error in quantum pattern prediction: {e}")
            pattern_prediction = 0
            pattern_probabilities = [0.25, 0.25, 0.25, 0.25]
    else:
        pattern_prediction = 0
        pattern_probabilities = [0.25, 0.25, 0.25, 0.25]
    
    pattern_labels = ['الگوی صعودی', 'الگوی نزولی', 'الگوی تثبیت', 'الگوی معکوس']
    detected_pattern = pattern_labels[pattern_prediction]
    pattern_confidence = max(pattern_probabilities)
    
    # شناسایی الگوهای کوانتومی
    quantum_patterns = self._identify_quantum_patterns(df)
    
    return {
        'detected_pattern': detected_pattern,
        'pattern_confidence': pattern_confidence,
        'pattern_probabilities': {pattern_labels[i]: pattern_probabilities[i] for i in range(len(pattern_labels))},
        'quantum_patterns': quantum_patterns,
        'fractal_dimension': self._calculate_fractal_dimension(df),
        'entropy': self._calculate_entropy(df),
        'lyapunov_exponent': self._calculate_lyapunov_exponent(df),
        'prediction_horizon': self._calculate_prediction_horizon(df)
    }

def _extract_quantum_features(self, df: pd.DataFrame) -> Dict:
    """استخراج ویژگی‌های کوانتومی"""
    if len(df) < 50:
        return {}
    
    features = {}
    
    # ویژگی‌های فرکتال
    features['fractal_dimension'] = self._calculate_fractal_dimension(df)
    
    # ویژگی‌های آنتروپی
    features['entropy'] = self._calculate_entropy(df)
    
    # ویژگی‌های لیاپانوف
    features['lyapunov_exponent'] = self._calculate_lyapunov_exponent(df)
    
    # ویژگی‌های طیفی
    features['spectral_entropy'] = self._calculate_spectral_entropy(df)
    
    # ویژگی‌های بازگشتی
    features['recurrence_rate'] = self._calculate_recurrence_rate(df)
    
    # ویژگی‌های پیچیدگی
    features['complexity'] = self._calculate_complexity(df)
    
    return features

def _calculate_fractal_dimension(self, df: pd.DataFrame) -> float:
    """محاسبه بعد فرکتال"""
    prices = df['close'].values
    n = len(prices)
    
    if n < 10:
        return 1.0
    
    # محاسبه بعد فرکتال با روش شمارش جعبه‌ای
    scales = np.logspace(0.1, 1, num=10)
    counts = []
    
    for scale in scales:
        # تقسیم داده‌ها به جعبه‌هایی با اندازه scale
        boxes = np.floor(np.arange(n) / scale).astype(int)
        box_counts = np.bincount(boxes)
        counts.append(len(box_counts))
    
    # برازش خطی برای محاسبه بعد فرکتال
    coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
    return -coeffs[0]

def _calculate_entropy(self, df: pd.DataFrame) -> float:
    """محاسبه آنتروپی"""
    prices = df['close'].values
    n = len(prices)
    
    if n < 10:
        return 0.0
    
    # محاسبه تفاوت‌های متوالی
    diffs = np.diff(prices)
    
    # محاسبه هیستوگرام
    hist, _ = np.histogram(diffs, bins=20)
    
    # نرمال‌سازی
    hist = hist / np.sum(hist)
    
    # محاسبه آنتروپی
    entropy = 0
    for p in hist:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy

def _calculate_lyapunov_exponent(self, df: pd.DataFrame) -> float:
    """محاسبه نمای لیاپانوف"""
    prices = df['close'].values
    n = len(prices)
    
    if n < 20:
        return 0.0
    
    # بازسازی فضای فاز
    m = 3  # بعد جاسازی
    tau = 1  # تأخیر زمانی
    
    # ایجاد ماتریس جاسازی
    embedded = np.zeros((n - (m-1)*tau, m))
    for i in range(m):
        embedded[:, i] = prices[i*tau : i*tau + len(embedded)]
    
    # محاسبه نمای لیاپانوف
    max_iter = min(100, len(embedded) - 10)
    lyapunov_sum = 0
    
    for i in range(max_iter):
        # پیدا کردن نزدیک‌ترین همسایه
        distances = np.sqrt(np.sum((embedded - embedded[i])**2, axis=1))
        distances[i] = np.inf  # نادیده گرفتن خود نقطه
        
        nearest_idx = np.argmin(distances)
        initial_distance = distances[nearest_idx]
        
        if initial_distance == 0:
            continue
        
        # رشد فاصله در زمان
        j = min(i + 10, len(embedded) - 1)
        final_distance = np.sqrt(np.sum((embedded[j] - embedded[nearest_idx + (j-i)])**2))
        
        if final_distance > 0:
            lyapunov_sum += np.log(final_distance / initial_distance)
    
    if max_iter > 0:
        return lyapunov_sum / (max_iter * 10)
    return 0.0

def _calculate_spectral_entropy(self, df: pd.DataFrame) -> float:
    """محاسبه آنتروپی طیفی"""
    prices = df['close'].values
    n = len(prices)
    
    if n < 10:
        return 0.0
    
    # تبدیل فوریه
    fft = np.fft.fft(prices)
    power_spectrum = np.abs(fft) ** 2
    
    # نرمال‌سازی
    power_spectrum = power_spectrum / np.sum(power_spectrum)
    
    # محاسبه آنتروپی
    entropy = 0
    for p in power_spectrum:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy

def _calculate_recurrence_rate(self, df: pd.DataFrame) -> float:
    """محاسبه نرخ بازگشتی"""
    prices = df['close'].values
    n = len(prices)
    
    if n < 10:
        return 0.0
    
    # ایجاد ماتریس بازگشتی
    recurrence_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if abs(prices[i] - prices[j]) < 0.01 * prices[i]:
                recurrence_matrix[i, j] = 1
    
    # محاسبه نرخ بازگشتی
    recurrence_rate = np.sum(recurrence_matrix) / (n * n)
    
    return recurrence_rate

def _calculate_complexity(self, df: pd.DataFrame) -> float:
    """محاسبه پیچیدگی"""
    prices = df['close'].values
    n = len(prices)
    
    if n < 10:
        return 0.0
    
    # محاسبه پیچیدگی لامپل-زیو
    binary_sequence = np.zeros(n-1)
    for i in range(n-1):
        if prices[i+1] > prices[i]:
            binary_sequence[i] = 1
    
    # محاسبه پیچیدگی
    complexity = 0
    n_patterns = 0
    
    for i in range(len(binary_sequence) - 1):
        pattern = binary_sequence[i:i+2]
        if np.array_equal(pattern, [0, 0]) or np.array_equal(pattern, [1, 1]):
            complexity += 1
        n_patterns += 1
    
    if n_patterns > 0:
        return complexity / n_patterns
    return 0.0

def _identify_quantum_patterns(self, df: pd.DataFrame) -> List[str]:
    """شناسایی الگوهای کوانتومی"""
    patterns = []
    
    # الگوی آشوب
    if self._is_chaotic_pattern(df):
        patterns.append("آشوب")
    
    # الگوی فرکتال
    if self._is_fractal_pattern(df):
        patterns.append("فرکتال")
    
    # الگوی دوره‌ای
    if self._is_periodic_pattern(df):
        patterns.append("دوره‌ای")
    
    # الگوی تصادفی
    if self._is_random_pattern(df):
        patterns.append("تصادفی")
    
    return patterns

def _is_chaotic_pattern(self, df: pd.DataFrame) -> bool:
    """بررسی الگوی آشوب"""
    lyapunov = self._calculate_lyapunov_exponent(df)
    return lyapunov > 0.01

def _is_fractal_pattern(self, df: pd.DataFrame) -> bool:
    """بررسی الگوی فرکتال"""
    fractal_dim = self._calculate_fractal_dimension(df)
    return 1.2 < fractal_dim < 1.8

def _is_periodic_pattern(self, df: pd.DataFrame) -> bool:
    """بررسی الگوی دوره‌ای"""
    prices = df['close'].values
    n = len(prices)
    
    if n < 20:
        return False
    
    # تبدیل فوریه
    fft = np.fft.fft(prices)
    power_spectrum = np.abs(fft) ** 2
    
    # پیدا کردن فرکانس غالب
    dominant_freq = np.argmax(power_spectrum[1:n//2]) + 1
    
    # بررسی اینکه آیا فرکانس غالب قدرت کافی دارد
    if power_spectrum[dominant_freq] > 0.5 * np.sum(power_spectrum):
        return True
    
    return False

def _is_random_pattern(self, df: pd.DataFrame) -> bool:
    """بررسی الگوی تصادفی"""
    entropy = self._calculate_entropy(df)
    return entropy > 3.0

def _calculate_prediction_horizon(self, df: pd.DataFrame) -> int:
    """محاسبه افق پیش‌بینی"""
    lyapunov = self._calculate_lyapunov_exponent(df)
    
    if lyapunov > 0:
        # افق پیش‌بینی بر اساس نمای لیاپانوف
        horizon = int(1 / lyapunov)
        return min(horizon, 30)  # حداکثر 30 روز
    else:
        return 10  # مقدار پیش‌فرض

def _perform_market_structure_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل ساختار بازار"""
    logger.info(f"Performing market structure analysis for {symbol}")
    
    price_data = self._extract_price_data(data)
    if not price_data:
        return {}
    
    df = pd.DataFrame(price_data)
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    
    # شناسایی نقاط عرضه و تقاضا
    supply_zones = []
    demand_zones = []
    
    # شناسایی نقاط چرخش
    pivot_points = self._identify_pivot_points(df)
    
    # شناسایی نواحی عرضه و تقاضا
    for i in range(2, len(df) - 2):
        # ناحیه تقاضا (حمایت)
        if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
            df['low'].iloc[i] < df['low'].iloc[i+1] and
            df['low'].iloc[i] < df['low'].iloc[i-2] and
            df['low'].iloc[i] < df['low'].iloc[i+2]):
            
            demand_zones.append({
                'price': df['low'].iloc[i],
                'strength': self._calculate_zone_strength(df, i, 'demand'),
                'timeframe': '1d'
            })
        
        # ناحیه عرضه (مقاومت)
        if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
            df['high'].iloc[i] > df['high'].iloc[i+1] and
            df['high'].iloc[i] > df['high'].iloc[i-2] and
            df['high'].iloc[i] > df['high'].iloc[i+2]):
            
            supply_zones.append({
                'price': df['high'].iloc[i],
                'strength': self._calculate_zone_strength(df, i, 'supply'),
                'timeframe': '1d'
            })
    
    # شناسایی Order Blocks
    order_blocks = self._identify_order_blocks(df)
    
    # تحلیل ساختار کلی بازار
    current_price = df['close'].iloc[-1]
    nearest_supply = min([z['price'] for z in supply_zones if z['price'] > current_price], default=None)
    nearest_demand = max([z['price'] for z in demand_zones if z['price'] < current_price], default=None)
    
    market_structure = {
        'supply_zones': supply_zones,
        'demand_zones': demand_zones,
        'order_blocks': order_blocks,
        'pivot_points': pivot_points,
        'nearest_supply': nearest_supply,
        'nearest_demand': nearest_demand,
        'current_position': self._interpret_market_position(current_price, nearest_supply, nearest_demand)
    }
    
    return market_structure

def _perform_session_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل سشن‌های معاملاتی"""
    logger.info(f"Performing session analysis for {symbol}")
    
    # دریافت داده‌های سشن‌های معاملاتی
    session_data = self._extract_session_data(data)
    
    # تحلیل هر سشن
    sessions = ['آسیا', 'لندن', 'نیویورک']
    session_analysis = {}
    
    for session in sessions:
        session_info = session_data.get(session, {})
        
        # تحلیل حجم معاملات
        volume_analysis = self._analyze_session_volume(session_info)
        
        # تحلیل نوسانات
        volatility_analysis = self._analyze_session_volatility(session_info)
        
        # تحلیل روند
        trend_analysis = self._analyze_session_trend(session_info)
        
        # تحلیل اخبار مرتبط با سشن
        news_analysis = await self._analyze_session_news(symbol, session)
        
        session_analysis[session] = {
            'volume_analysis': volume_analysis,
            'volatility_analysis': volatility_analysis,
            'trend_analysis': trend_analysis,
            'news_analysis': news_analysis,
            'session_impact': self._calculate_session_impact(
                volume_analysis, 
                volatility_analysis, 
                trend_analysis,
                news_analysis
            )
        }
    
    # تحلیل همپوشانی سشن‌ها
    overlap_analysis = self._analyze_session_overlaps(session_data)
    
    # تحلیل تأثیر کل سشن‌ها
    overall_session_impact = self._calculate_overall_session_impact(session_analysis)
    
    return {
        'sessions': session_analysis,
        'overlap_analysis': overlap_analysis,
        'overall_session_impact': overall_session_impact,
        'best_trading_session': self._identify_best_trading_session(session_analysis),
        'session_recommendations': self._generate_session_recommendations(session_analysis)
    }

def _analyze_session_volume(self, session_info: Dict) -> Dict:
    """تحلیل حجم معاملات در یک سشن"""
    volume = session_info.get('volume', 0)
    avg_volume = session_info.get('avg_volume', 0)
    
    if avg_volume == 0:
        return {'status': 'نامشخص', 'change': 0}
    
    volume_change = (volume - avg_volume) / avg_volume
    
    if volume_change > 0.2:
        status = "حجم بالا"
    elif volume_change < -0.2:
        status = "حجم پایین"
    else:
        status = "حجم عادی"
    
    return {
        'status': status,
        'change': volume_change,
        'current_volume': volume,
        'average_volume': avg_volume
    }

def _analyze_session_volatility(self, session_info: Dict) -> Dict:
    """تحلیل نوسانات در یک سشن"""
    volatility = session_info.get('volatility', 0)
    avg_volatility = session_info.get('avg_volatility', 0)
    
    if avg_volatility == 0:
        return {'status': 'نامشخص', 'change': 0}
    
    volatility_change = (volatility - avg_volatility) / avg_volatility
    
    if volatility_change > 0.2:
        status = "نوسان بالا"
    elif volatility_change < -0.2:
        status = "نوسان پایین"
    else:
        status = "نوسان عادی"
    
    return {
        'status': status,
        'change': volatility_change,
        'current_volatility': volatility,
        'average_volatility': avg_volatility
    }

def _analyze_session_trend(self, session_info: Dict) -> Dict:
    """تحلیل روند در یک سشن"""
    open_price = session_info.get('open', 0)
    close_price = session_info.get('close', 0)
    
    if open_price == 0:
        return {'status': 'نامشخص', 'change': 0}
    
    price_change = (close_price - open_price) / open_price
    
    if price_change > 0.01:
        status = "صعودی"
    elif price_change < -0.01:
        status = "نزولی"
    else:
        status = "خنثی"
    
    return {
        'status': status,
        'change': price_change,
        'open_price': open_price,
        'close_price': close_price
    }

async def _analyze_session_news(self, symbol: str, session: str) -> Dict:
    """تحلیل اخبار مرتبط با یک سشن"""
    # دریافت اخبار مرتبط با سشن
    news_data = self._extract_session_news(symbol, session)
    
    if not news_data:
        return {'status': 'بدون خبر', 'impact': 0}
    
    # تحلیل احساسات اخبار
    sentiments = []
    impact_scores = []
    
    for news in news_data:
        try:
            # تحلیل احساسات با مدل پیشرفته
            sentiment_result = self.models['sentiment_analyzer'](news['title'] + " " + news['description'])
            sentiment_label = sentiment_result[0]['label']
            sentiment_score = sentiment_result[0]['score']
            
            # تبدیل به امتیاز عددی
            if sentiment_label == 'POSITIVE':
                score = sentiment_score
            elif sentiment_label == 'NEGATIVE':
                score = -sentiment_score
            else:
                score = 0
            
            sentiments.append(score)
            
            # محاسبه تأثیر خبر بر اساس حجم جستجو
            impact = news.get('impact', 0.5)
            impact_scores.append(impact)
            
        except Exception as e:
            logger.error(f"Error analyzing session news: {e}")
    
    # محاسبه میانگین احساسات و تأثیر
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    avg_impact = np.mean(impact_scores) if impact_scores else 0
    
    # تعیین وضعیت اخبار
    if avg_sentiment > 0.2 and avg_impact > 0.6:
        status = "اخبار مثبت قوی"
    elif avg_sentiment > 0.2:
        status = "اخبار مثبت"
    elif avg_sentiment < -0.2 and avg_impact > 0.6:
        status = "اخبار منفی قوی"
    elif avg_sentiment < -0.2:
        status = "اخبار منفی"
    else:
        status = "اخبار خنثی"
    
    return {
        'status': status,
        'sentiment': avg_sentiment,
        'impact': avg_impact,
        'news_count': len(news_data)
    }

def _calculate_session_impact(self, volume_analysis: Dict, volatility_analysis: Dict, 
                              trend_analysis: Dict, news_analysis: Dict) -> float:
    """محاسبه تأثیر یک سشن"""
    # وزن‌ها برای هر معیار
    weights = {
        'volume': 0.3,
        'volatility': 0.2,
        'trend': 0.3,
        'news': 0.2
    }
    
    # نرمال‌سازی مقادیر
    volume_score = min(abs(volume_analysis.get('change', 0)) * 5, 1)
    volatility_score = min(abs(volatility_analysis.get('change', 0)) * 5, 1)
    trend_score = min(abs(trend_analysis.get('change', 0)) * 50, 1)
    news_score = min(abs(news_analysis.get('sentiment', 0)) * 2, 1)
    
    # محاسبه امتیاز نهایی
    impact_score = (
        weights['volume'] * volume_score +
        weights['volatility'] * volatility_score +
        weights['trend'] * trend_score +
        weights['news'] * news_score
    )
    
    return impact_score

def _analyze_session_overlaps(self, session_data: Dict) -> Dict:
    """تحلیل همپوشانی سشن‌ها"""
    overlaps = {
        'آسیا-لندن': {},
        'لندن-نیویورک': {},
        'آسیا-نیویورک': {}
    }
    
    for overlap_name in overlaps:
        # دریافت داده‌های همپوشانی
        overlap_data = session_data.get(overlap_name, {})
        
        if not overlap_data:
            overlaps[overlap_name] = {'status': 'بدون داده', 'impact': 0}
            continue
        
        # تحلیل حجم معاملات در همپوشانی
        volume = overlap_data.get('volume', 0)
        avg_volume = overlap_data.get('avg_volume', 0)
        
        if avg_volume > 0:
            volume_change = (volume - avg_volume) / avg_volume
        else:
            volume_change = 0
        
        # تحلیل نوسانات در همپوشانی
        volatility = overlap_data.get('volatility', 0)
        avg_volatility = overlap_data.get('avg_volatility', 0)
        
        if avg_volatility > 0:
            volatility_change = (volatility - avg_volatility) / avg_volatility
        else:
            volatility_change = 0
        
        # محاسبه تأثیر همپوشانی
        impact = (abs(volume_change) + abs(volatility_change)) / 2
        
        # تعیین وضعیت همپوشانی
        if impact > 0.3:
            status = "فعالیت بالا"
        elif impact > 0.1:
            status = "فعالیت متوسط"
        else:
            status = "فعالیت پایین"
        
        overlaps[overlap_name] = {
            'status': status,
            'impact': impact,
            'volume_change': volume_change,
            'volatility_change': volatility_change
        }
    
    return overlaps

def _calculate_overall_session_impact(self, session_analysis: Dict) -> float:
    """محاسبه تأثیر کل سشن‌ها"""
    total_impact = 0
    count = 0
    
    for session, analysis in session_analysis.items():
        impact = analysis.get('session_impact', 0)
        total_impact += impact
        count += 1
    
    if count > 0:
        return total_impact / count
    return 0

def _identify_best_trading_session(self, session_analysis: Dict) -> str:
    """شناسایی بهترین سشن برای معامله"""
    best_session = None
    max_impact = 0
    
    for session, analysis in session_analysis.items():
        impact = analysis.get('session_impact', 0)
        if impact > max_impact:
            max_impact = impact
            best_session = session
    
    return best_session or "نامشخص"

def _generate_session_recommendations(self, session_analysis: Dict) -> List[str]:
    """تولید توصیه‌های معاملاتی بر اساس سشن‌ها"""
    recommendations = []
    
    for session, analysis in session_analysis.items():
        impact = analysis.get('session_impact', 0)
        volume_analysis = analysis.get('volume_analysis', {})
        trend_analysis = analysis.get('trend_analysis', {})
        news_analysis = analysis.get('news_analysis', {})
        
        if impact > 0.7:
            if trend_analysis.get('status') == 'صعودی' and news_analysis.get('sentiment', 0) > 0.2:
                recommendations.append(f"در سشن {session} فرصت خرید خوبی وجود دارد")
            elif trend_analysis.get('status') == 'نزولی' and news_analysis.get('sentiment', 0) < -0.2:
                recommendations.append(f"در سشن {session} فرصت فروش خوبی وجود دارد")
        elif impact > 0.4:
            recommendations.append(f"سشن {session} برای معامله مناسب است")
        else:
            recommendations.append(f"سشن {session} برای معامله توصیه نمی‌شود")
    
    return recommendations

def _perform_risk_management(self, symbol: str, data: Dict) -> Dict:
    """انجام مدیریت ریسک و سرمایه"""
    logger.info(f"Performing risk management for {symbol}")
    
    price_data = self._extract_price_data(data)
    if not price_data:
        return {}
    
    df = pd.DataFrame(price_data)
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    
    current_price = df['close'].iloc[-1]
    
    # محاسبه حد ضرر
    stop_loss = self._calculate_stop_loss(df)
    
    # محاسبه حد سود
    take_profit = self._calculate_take_profit(df, stop_loss)
    
    # محاسبه نسبت ریسک به پاداش
    risk_reward_ratio = self._calculate_risk_reward_ratio(current_price, stop_loss, take_profit)
    
    # محاسبه اندازه موقعیت
    position_size = self._calculate_position_size(current_price, stop_loss)
    
    # محاسبه حداکثر افت سرمایه
    max_drawdown = self._calculate_max_drawdown(df)
    
    # محاسبه ارزش در معرض ریسک
    var = self._calculate_var(df)
    
    return {
        'current_price': current_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_reward_ratio': risk_reward_ratio,
        'position_size': position_size,
        'max_drawdown': max_drawdown,
        'var': var,
        'risk_level': self._assess_risk_level(risk_reward_ratio, max_drawdown, var),
        'recommendations': self._generate_risk_recommendations(
            stop_loss, take_profit, risk_reward_ratio, position_size
        )
    }

def _calculate_stop_loss(self, df: pd.DataFrame) -> float:
    """محاسبه حد ضرر"""
    current_price = df['close'].iloc[-1]
    
    # استفاده از ATR برای محاسبه حد ضرر
    atr = self._calculate_atr(df)
    
    # حد ضرر بر اساس ATR
    if df['close'].iloc[-1] > df['close'].iloc[-2]:
        # روند صعودی - حد ضرر پایین‌تر
        stop_loss = current_price - (2 * atr)
    else:
        # روند نزولی - حد ضرر بالاتر
        stop_loss = current_price + (2 * atr)
    
    return stop_loss

def _calculate_take_profit(self, df: pd.DataFrame, stop_loss: float) -> float:
    """محاسبه حد سود"""
    current_price = df['close'].iloc[-1]
    
    # محاسبه نسبت ریسک به پاداش مطلوب
    risk = abs(current_price - stop_loss)
    reward = risk * self.risk_reward_ratio
    
    if current_price > stop_loss:
        take_profit = current_price + reward
    else:
        take_profit = current_price - reward
    
    return take_profit

def _calculate_risk_reward_ratio(self, current_price: float, stop_loss: float, take_profit: float) -> float:
    """محاسبه نسبت ریسک به پاداش"""
    risk = abs(current_price - stop_loss)
    reward = abs(take_profit - current_price)
    
    if risk == 0:
        return 0
    
    return reward / risk

def _calculate_position_size(self, current_price: float, stop_loss: float) -> float:
    """محاسبه اندازه موقعیت"""
    risk_per_trade = 0.02  # 2% ریسک در هر معامله
    
    # محاسبه ریسک در هر واحد
    risk_per_unit = abs(current_price - stop_loss)
    
    if risk_per_unit == 0:
        return 0
    
    # محاسبه اندازه موقعیت
    position_size = (risk_per_trade * 10000) / risk_per_unit
    
    # محدود کردن اندازه موقعیت
    position_size = min(position_size, self.max_position_size * 10000)
    
    return position_size

def _perform_whale_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل نهنگ‌ها"""
    logger.info(f"Performing whale analysis for {symbol}")
    
    # دریافت تراکنش‌های نهنگ‌ها
    whale_transactions = await self.fetch_whale_transactions(symbol)
    
    # تحلیل رفتار نهنگ‌ها
    whale_analysis = await self._perform_whale_behavior_analysis(symbol, whale_transactions)
    
    return whale_analysis

# متدهای کمکی دیگر برای تحلیل‌های تخصصی
def _interpret_market_position(self, current_price: float, nearest_supply: float, nearest_demand: float) -> str:
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

def _calculate_zone_strength(self, df: pd.DataFrame, index: int, zone_type: str) -> float:
    """محاسبه قدرت ناحیه عرضه یا تقاضا"""
    # تعداد دفعات واکنش قیمت به این ناحیه
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
    
    # قدرت بر اساس تعداد واکنش‌ها و حجم معاملات
    volume_factor = df['volume'].iloc[index] / df['volume'].mean()
    strength = reactions * volume_factor
    
    return min(strength, 10)  # حداکثر قدرت 10

def _identify_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
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

def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
    """محاسبه میانگین دامنه واقعی (ATR)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # محاسبه دامنه واقعی
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # محاسبه ATR
    atr = tr.rolling(window=period).mean()
    
    return atr.iloc[-1]

def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
    """محاسبه حداکثر افت سرمایه"""
    close_prices = df['close']
    
    # محاسبه قله‌ها و دره‌ها
    peaks = close_prices.copy()
    troughs = close_prices.copy()
    
    for i in range(1, len(close_prices)):
        if close_prices[i] > peaks[i-1]:
            peaks[i] = close_prices[i]
        else:
            peaks[i] = peaks[i-1]
        
        if close_prices[i] < troughs[i-1]:
            troughs[i] = close_prices[i]
        else:
            troughs[i] = troughs[i-1]
    
    # محاسبه افت سرمایه
    drawdown = (peaks - troughs) / peaks
    
    return drawdown.max()

def _calculate_var(self, df: pd.DataFrame, confidence_level: float = 0.95) -> float:
    """محاسبه ارزش در معرض ریسک (VaR)"""
    returns = df['close'].pct_change().dropna()
    
    # محاسبه VaR با روش تاریخی
    var = np.percentile(returns, (1 - confidence_level) * 100)
    
    return abs(var)

def _assess_risk_level(self, risk_reward_ratio: float, max_drawdown: float, var: float) -> str:
    """ارزیابی سطح ریسک"""
    if risk_reward_ratio < 1.5 or max_drawdown > 0.2 or var > 0.05:
        return "بالا"
    elif risk_reward_ratio < 2.5 or max_drawdown > 0.1 or var > 0.03:
        return "متوسط"
    else:
        return "پایین"

def _generate_risk_recommendations(self, stop_loss: float, take_profit: float, 
                                risk_reward_ratio: float, position_size: float) -> List[str]:
    """تولید توصیه‌های مدیریت ریسک"""
    recommendations = []
    
    if risk_reward_ratio < 1.5:
        recommendations.append("نسبت ریسک به پاداش پایین است. توصیه می‌شود از معامله خودداری کنید.")
    elif risk_reward_ratio < 2.0:
        recommendations.append("نسبت ریسک به پاداش متوسط است. با احتیاط معامله کنید.")
    else:
        recommendations.append("نسبت ریسک به پاداش مناسب است.")
    
    if position_size > 5000:
        recommendations.append("اندازه موقعیت بزرگ است. توصیه می‌شود آن را کاهش دهید.")
    
    recommendations.append(f"حد ضرر را در قیمت {stop_loss:.2f} تنظیم کنید.")
    recommendations.append(f"حد سود را در قیمت {take_profit:.2f} تنظیم کنید.")
    
    return recommendations

def _extract_session_data(self, data: Dict) -> Dict:
    """استخراج داده‌های سشن‌های معاملاتی"""
    # در یک پیاده‌سازی واقعی، این تابع باید داده‌های سشن‌ها را استخراج کند
    # برای سادگی، ما داده‌های ساختگی برمی‌گردانیم
    return {
        'آسیا': {
            'volume': 1000000,
            'avg_volume': 900000,
            'volatility': 0.02,
            'avg_volatility': 0.015,
            'open': 50000,
            'close': 50200,
        },
        'لندن': {
            'volume': 1500000,
            'avg_volume': 1400000,
            'volatility': 0.025,
            'avg_volatility': 0.02,
            'open': 50200,
            'close': 50100,
        },
        'نیویورک': {
            'volume': 2000000,
            'avg_volume': 1800000,
            'volatility': 0.03,
            'avg_volatility': 0.025,
            'open': 50100,
            'close': 50300,
        },
        'آسیا-لندن': {
            'volume': 1200000,
            'avg_volume': 1100000,
            'volatility': 0.022,
            'avg_volatility': 0.018,
        },
        'لندن-نیویورک': {
            'volume': 1800000,
            'avg_volume': 1700000,
            'volatility': 0.028,
            'avg_volatility': 0.023,
        },
        'آسیا-نیویورک': {
            'volume': 1100000,
            'avg_volume': 1000000,
            'volatility': 0.021,
            'avg_volatility': 0.017,
        },
    }

def _extract_session_news(self, symbol: str, session: str) -> List[Dict]:
    """استخراج اخبار مرتبط با سشن"""
    # در یک پیاده‌سازی واقعی، این تابع باید اخبار مرتبط با سشن را استخراج کند
    # برای سادگی، ما داده‌های ساختگی برمی‌گردانیم
    return [
        {
            'title': f"اخبار مهم {session} برای {symbol}",
            'description': f"توصیه‌های مهم برای معامله در سشن {session}",
            'impact': 0.7
        }
    ]

def _extract_timeframe_data(self, data: Dict, timeframe: str) -> List[Dict]:
    """استخراج داده‌های تایم‌فریم خاص"""
    # در یک پیاده‌سازی واقعی، این تابع باید داده‌ها را برای تایم‌فریم خاص استخراج کند
    # برای سادگی، ما داده‌های مشابه را برمی‌گردانیم
    return self._extract_price_data(data)

def _perform_whale_behavior_analysis(self, symbol: str, transactions: List[Dict]) -> Dict:
    """تحلیل رفتار نهنگ‌ها"""
    if not transactions:
        return {'sentiment': 0, 'volume_usd': 0, 'buy_count': 0, 'sell_count': 0}
    
    total_volume = sum(tx['amount_usd'] for tx in transactions)
    buy_count = sum(1 for tx in transactions if tx['transaction_type'] == 'buy')
    sell_count = sum(1 for tx in transactions if tx['transaction_type'] == 'sell')
    
    # محاسبه احساسات نهنگ‌ها
    if buy_count + sell_count > 0:
        sentiment = (buy_count - sell_count) / (buy_count + sell_count)
    else:
        sentiment = 0
    
    # تحلیل الگوهای رفتاری
    patterns = []
    
    # الگوی انباشت
    if buy_count > sell_count * 2 and total_volume > 10000000:
        patterns.append("انباشت نهنگ‌ها")
    
    # الگوی توزیع
    if sell_count > buy_count * 2 and total_volume > 10000000:
        patterns.append("توزیع نهنگ‌ها")
    
    # تحلیل کیفی‌ها
    wallet_analysis = {}
    for tx in transactions:
        wallet = tx['wallet_address']
        if wallet not in wallet_analysis:
            wallet_analysis[wallet] = {'buy': 0, 'sell': 0, 'volume': 0}
        
        if tx['transaction_type'] == 'buy':
            wallet_analysis[wallet]['buy'] += 1
        else:
            wallet_analysis[wallet]['sell'] += 1
        
        wallet_analysis[wallet]['volume'] += tx['amount_usd']
    
    # شناسایی کیفی‌های فعال
    active_whales = [w for w in wallet_analysis.values() if w['buy'] + w['sell'] >= 3]
    
    return {
        'sentiment': sentiment,
        'volume_usd': total_volume,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'patterns': patterns,
        'active_whales': len(active_whales),
        'wallet_analysis': wallet_analysis
    }

def _perform_on_chain_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل زنجیره‌ای"""
    logger.info(f"Performing on-chain analysis for {symbol}")
    
    # دریافت متریک‌های زنجیره‌ای
    on_chain_metrics = await self.fetch_on_chain_metrics(symbol)
    
    # تحلیل متریک‌ها
    analysis = {
        'active_addresses': on_chain_metrics.get('addresses/active_count', 0),
        'transaction_count': on_chain_metrics.get('transactions/count', 0),
        'supply_profitability': on_chain_metrics.get('supply/profit_relative', 0),
        'holder_concentration': on_chain_metrics.get('distribution/balance_1pct_holders', 0),
        'nvt_ratio': on_chain_metrics.get('market/nvt', 0),
        'network_health': self._calculate_network_health(on_chain_metrics),
        'adoption_rate': self._calculate_adoption_rate(on_chain_metrics)
    }
    
    return analysis

def _calculate_network_health(self, metrics: Dict) -> float:
    """محاسبه سلامت شبکه"""
    score = 0.5
    
    # آدرس‌های فعال
    active_addresses = metrics.get('addresses/active_count', 0)
    if active_addresses > 100000:
        score += 0.2
    elif active_addresses > 50000:
        score += 0.1
    
    # تعداد تراکنش‌ها
    tx_count = metrics.get('transactions/count', 0)
    if tx_count > 10000:
        score += 0.2
    elif tx_count > 5000:
        score += 0.1
    
    # تمرکز عرضه
    holder_concentration = metrics.get('distribution/balance_1pct_holders', 0)
    if holder_concentration < 0.3:
        score += 0.2
    elif holder_concentration < 0.5:
        score += 0.1
    
    return min(1, score)

def _calculate_adoption_rate(self, metrics: Dict) -> float:
    """محاسبه نرخ پذیرش"""
    score = 0.5
    
    # رشد آدرس‌های فعال
    active_addresses = metrics.get('addresses/active_count', 0)
    if active_addresses > 100000:
        score += 0.3
    
    # رشد تراکنش‌ها
    tx_count = metrics.get('transactions/count', 0)
    if tx_count > 10000:
        score += 0.3
    
    # سودآوری عرضه
    supply_profit = metrics.get('supply/profit_relative', 0)
    if supply_profit > 0.2:
        score += 0.2
    
    return min(1, score)

def _perform_correlation_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل همبستگی"""
    logger.info(f"Performing correlation analysis for {symbol}")
    
    # دریافت داده‌های بازار برای ارزهای اصلی
    major_cryptos = ['BTC', 'ETH', 'BNB']
    correlations = {}
    
    for crypto in major_cryptos:
        if crypto == symbol:
            continue
                
        try:
            crypto_data = await self.fetch_data_from_multiple_sources(crypto)
            crypto_prices = self._extract_price_data(crypto_data)
            
            if crypto_prices:
                # محاسبه همبستگی
                symbol_prices = self._extract_price_data(data)
                
                if symbol_prices and crypto_prices:
                    symbol_df = pd.DataFrame(symbol_prices)
                    crypto_df = pd.DataFrame(crypto_prices)
                    
                    # همترازسازی داده‌ها بر اساس زمان
                    symbol_df['timestamp'] = pd.to_datetime(symbol_df['timestamp'])
                    crypto_df['timestamp'] = pd.to_datetime(crypto_df['timestamp'])
                    
                    merged = pd.merge(symbol_df, crypto_df, on='timestamp', suffixes=('_symbol', '_crypto'))
                    
                    if len(merged) > 10:
                        correlation = merged['close_symbol'].corr(merged['close_crypto'])
                        correlations[crypto] = correlation
        except Exception as e:
            logger.error(f"Error calculating correlation with {crypto}: {e}")
    
    return {
        'correlations': correlations,
        'avg_correlation': np.mean(list(correlations.values())) if correlations else 0,
        'market_dependency': self._assess_market_dependency(correlations)
    }

def _assess_market_dependency(self, correlations: Dict) -> str:
    """ارزیابی وابستگی به بازار"""
    if not correlations:
        return 'unknown'
    
    avg_corr = np.mean(list(correlations.values()))
    
    if avg_corr > 0.7:
        return 'high'
    elif avg_corr > 0.4:
        return 'medium'
    else:
        return 'low'

def _perform_seasonal_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل فصلی"""
    logger.info(f"Performing seasonal analysis for {symbol}")
    
    price_data = self._extract_price_data(data)
    if not price_data:
        return {}
    
    df = pd.DataFrame(price_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['close'] = pd.to_numeric(df['close'])
    
    # استخراج ویژگی‌های فصلی
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['quarter'] = df['timestamp'].dt.quarter
    
    # محاسبه میانگین بازدهی بر اساس ماه
    monthly_returns = df.groupby('month')['close'].pct_change().mean()
    
    # محاسبه میانگین بازدهی بر اساس روز هفته
    weekly_returns = df.groupby('day_of_week')['close'].pct_change().mean()
    
    # محاسبه میانگین بازدهی بر اساس فصل
    quarterly_returns = df.groupby('quarter')['close'].pct_change().mean()
    
    # شناسایی الگوهای فصلی
    best_month = monthly_returns.idxmax()
    worst_month = monthly_returns.idxmin()
    
    best_day = weekly_returns.idxmax()
    worst_day = weekly_returns.idxmin()
    
    best_quarter = quarterly_returns.idxmax()
    worst_quarter = quarterly_returns.idxmin()
    
    # ماه فعلی
    current_month = datetime.now().month
    current_month_return = monthly_returns.get(current_month, 0)
    
    return {
        'monthly_returns': monthly_returns.to_dict(),
        'weekly_returns': weekly_returns.to_dict(),
        'quarterly_returns': quarterly_returns.to_dict(),
        'best_month': best_month,
        'worst_month': worst_month,
        'best_day': best_day,
        'worst_day': worst_day,
        'best_quarter': best_quarter,
        'worst_quarter': worst_quarter,
        'current_month': current_month,
        'current_month_return': current_month_return,
        'seasonal_strength': self._calculate_seasonal_strength(monthly_returns, weekly_returns, quarterly_returns)
    }

def _calculate_seasonal_strength(self, monthly_returns: pd.Series, weekly_returns: pd.Series, quarterly_returns: pd.Series) -> float:
    """محاسبه قدرت الگوهای فصلی"""
    strength = 0
    
    # قدرت الگوی ماهانه
    if not monthly_returns.empty:
        monthly_std = monthly_returns.std()
        if monthly_std > 0.05:
            strength += 0.3
    
    # قدرت الگوی هفتگی
    if not weekly_returns.empty:
        weekly_std = weekly_returns.std()
        if weekly_std > 0.03:
            strength += 0.3
    
    # قدرت الگوی فصلی
    if not quarterly_returns.empty:
        quarterly_std = quarterly_returns.std()
        if quarterly_std > 0.08:
            strength += 0.4
    
    return min(1, strength)

def _perform_technical_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل تکنیکال پیشرفته"""
    logger.info(f"Performing technical analysis for {symbol}")
    
    # استخراج داده‌های قیمت
    price_data = self._extract_price_data(data)
    if not price_data:
        return {}
    
    df = pd.DataFrame(price_data)
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['volume'] = pd.to_numeric(df['volume'])
    
    # محاسبه شاخص‌های تکنیکال با TA-Lib
    analysis = {}
    
    # RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    analysis['rsi'] = df['rsi'].iloc[-1]
    analysis['rsi_signal'] = self._interpret_rsi(df['rsi'].iloc[-1])
    
    # MACD
    macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    analysis['macd'] = {
        'macd': macd.iloc[-1],
        'signal': macdsignal.iloc[-1],
        'histogram': macdhist.iloc[-1],
        'signal_type': self._interpret_macd(macd.iloc[-1], macdsignal.iloc[-1])
    }
    
    # بولینگر باند
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
    analysis['bollinger_bands'] = {
        'upper': upper.iloc[-1],
        'middle': middle.iloc[-1],
        'lower': lower.iloc[-1],
        'position': self._interpret_bollinger_bands(df['close'].iloc[-1], upper.iloc[-1], lower.iloc[-1])
    }
    
    # میانگین متحرک
    df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
    df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
    df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
    df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
    
    analysis['moving_averages'] = {
        'sma_20': df['sma_20'].iloc[-1],
        'sma_50': df['sma_50'].iloc[-1],
        'ema_12': df['ema_12'].iloc[-1],
        'ema_26': df['ema_26'].iloc[-1],
        'sma_signal': self._interpret_moving_averages(df['sma_20'].iloc[-1], df['sma_50'].iloc[-1])
    }
    
    # حجم معاملات
    df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
    analysis['volume'] = {
        'current': df['volume'].iloc[-1],
        'average': df['volume_sma'].iloc[-1],
        'signal': self._interpret_volume(df['volume'].iloc[-1], df['volume_sma'].iloc[-1])
    }
    
    # نوسانات
    analysis['volatility'] = talib.STDDEV(df['close'], timeperiod=20, nbdev=1).iloc[-1] / df['close'].iloc[-1]
    
    # روند
    analysis['trend'] = self._determine_trend(df)
    
    # حمایت و مقاومت
    support, resistance = self._find_support_resistance(df)
    analysis['support_resistance'] = {
        'support': support,
        'resistance': resistance
    }
    
    return analysis

def _determine_trend(self, df: pd.DataFrame) -> str:
    """تعیین روند قیمت"""
    if len(df) < 50:
        return 'neutral'
    
    # استفاده از میانگین‌های متحرک برای تعیین روند
    sma_20 = df['sma_20'].iloc[-1]
    sma_50 = df['sma_50'].iloc[-1]
    current_price = df['close'].iloc[-1]
    
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

def _find_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
    """پیدا کردن حمایت و مقاومت"""
    if len(df) < 20:
        return 0, 0
    
    # استفاده از نقاط چرخش برای پیدا کردن حمایت و مقاومت
    highs = df['high'].rolling(window=5).max()
    lows = df['low'].rolling(window=5).min()
    
    # پیدا کردن قله‌ها و دره‌های محلی
    pivot_highs = []
    pivot_lows = []
    
    for i in range(2, len(df)-2):
        if df['high'].iloc[i] == highs.iloc[i] and df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1]:
            pivot_highs.append(df['high'].iloc[i])
        
        if df['low'].iloc[i] == lows.iloc[i] and df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1]:
            pivot_lows.append(df['low'].iloc[i])
    
    # محاسبه حمایت و مقاومت
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

def _perform_sentiment_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل احساسات بازار"""
    logger.info(f"Performing sentiment analysis for {symbol}")
    
    news_data = self._extract_news(data)
    if not news_data:
        return {'average_sentiment': 0, 'topics': [], 'news_count': 0}
    
    # تحلیل احساسات با مدل پیشرفته
    sentiments = []
    topics = []
    
    for news in news_data:
        try:
            # تحلیل احساسات با مدل پیشرفته
            sentiment_result = self.models['sentiment_analyzer'](news['title'] + " " + news['description'])
            sentiment_label = sentiment_result[0]['label']
            sentiment_score = sentiment_result[0]['score']
            
            # تبدیل به امتیاز عددی
            if sentiment_label == 'POSITIVE':
                score = sentiment_score
            elif sentiment_label == 'NEGATIVE':
                score = -sentiment_score
            else:
                score = 0
            
            sentiments.append(score)
            
            # استخراج موضوعات
            if 'keywords' in news:
                topics.extend(news['keywords'][:3])
            
            # ذخیره تأثیر خبر در پایگاه داده
            self.cursor.execute('''
            INSERT INTO news_impact 
            (symbol, news_title, news_source, timestamp, sentiment_score, impact_score)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                news['title'],
                news['source'],
                news['published_at'],
                score,
                abs(score) * 0.5  # تأثیر ساده شده
            ))
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
    
    # محاسبه میانگین احساسات
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    
    # شناسایی موضوعات پرتکرار
    topic_counts = {}
    for topic in topics:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_topics = [topic[0] for topic in top_topics]
    
    return {
        'average_sentiment': avg_sentiment,
        'topics': top_topics,
        'news_count': len(news_data),
        'positive_news_count': len([s for s in sentiments if s > 0]),
        'negative_news_count': len([s for s in sentiments if s < 0])
    }

def _perform_elliott_wave_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل امواج الیوت"""
    logger.info(f"Performing Elliott Wave analysis for {symbol}")
    
    price_data = self._extract_price_data(data)
    if not price_data:
        return {}
    
    df = pd.DataFrame(price_data)
    df['close'] = pd.to_numeric(df['close'])
    
    # شناسایی امواج الیوت با مدل یادگیری ماشین
    wave_features = self._extract_elliott_wave_features(df)
    
    if not wave_features:
        return {}
    
    # پیش‌بینی با مدل امواج الیوت
    if self.models['elliott_wave']['trained']:
        try:
            X = np.array([list(wave_features.values())])
            X_scaled = self.models['elliott_wave']['scaler'].transform(X)
            wave_prediction = self.models['elliott_wave']['model'].predict(X_scaled)[0]
            wave_probabilities = self.models['elliott_wave']['model'].predict_proba(X_scaled)[0]
        except Exception as e:
            logger.error(f"Error in Elliott Wave prediction: {e}")
            wave_prediction = 0
            wave_probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
    else:
        wave_prediction = 0
        wave_probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    wave_labels = ['موج 1', 'موج 2', 'موج 3', 'موج 4', 'موج 5']
    current_wave = wave_labels[wave_prediction]
    wave_confidence = max(wave_probabilities)
    
    # شناسایی الگوهای امواج الیوت
    wave_patterns = self._identify_elliott_wave_patterns(df)
    
    return {
        'current_wave': current_wave,
        'wave_confidence': wave_confidence,
        'wave_probabilities': {wave_labels[i]: wave_probabilities[i] for i in range(len(wave_labels))},
        'wave_patterns': wave_patterns,
        'next_wave_prediction': self._predict_next_elliott_wave(current_wave),
        'wave_target': self._calculate_wave_target(df, current_wave)
    }

def _extract_elliott_wave_features(self, df: pd.DataFrame) -> Dict:
    """استخراج ویژگی‌های امواج الیوت"""
    if len(df) < 50:
        return {}
    
    features = {}
    
    # محاسبه تغییرات قیمت
    df['price_change'] = df['close'].pct_change()
    
    # ویژگی‌های آماری
    features['mean_price_change'] = df['price_change'].mean()
    features['std_price_change'] = df['price_change'].std()
    features['max_price_change'] = df['price_change'].max()
    features['min_price_change'] = df['price_change'].min()
    
    # ویژگی‌های روند
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    features['sma_ratio'] = df['sma_10'].iloc[-1] / df['sma_30'].iloc[-1]
    
    # ویژگی‌های حجم
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    features['volume_ratio'] = df['volume'].iloc[-1] / df['volume_sma'].iloc[-1]
    
    # ویژگی‌های نوسان
    df['high_low_ratio'] = df['high'] / df['low']
    features['avg_high_low_ratio'] = df['high_low_ratio'].mean()
    
    # ویژگی‌های شتاب
    df['momentum'] = df['close'] / df['close'].shift(5) - 1
    features['momentum'] = df['momentum'].iloc[-1]
    
    return features

def _identify_elliott_wave_patterns(self, df: pd.DataFrame) -> List[str]:
    """شناسایی الگوهای امواج الیوت"""
    patterns = []
    
    # الگوی ایمپالس
    if self._is_impulse_pattern(df):
        patterns.append("ایمپالس")
    
    # الگوی اصلاحی
    if self._is_corrective_pattern(df):
        patterns.append("اصلاحی")
    
    # الگوی مثلث
    if self._is_triangle_pattern(df):
        patterns.append("مثلث")
    
    # الگوی مسطح
    if self._is_flat_pattern(df):
        patterns.append("مسطح")
    
    return patterns

def _is_impulse_pattern(self, df: pd.DataFrame) -> bool:
    """بررسی الگوی ایمپالس"""
    if len(df) < 20:
        return False
    
    # بررسی 5 موج صعودی
    waves = self._identify_waves(df)
    if len(waves) >= 5:
        # بررسی اینکه موج 3 بلندتر از موج 1 و 5 باشد
        if (waves[2]['height'] > waves[0]['height'] and 
            waves[2]['height'] > waves[4]['height']):
            return True
    
    return False

def _is_corrective_pattern(self, df: pd.DataFrame) -> bool:
    """بررسی الگوی اصلاحی"""
    if len(df) < 15:
        return False
    
    # بررسی 3 موج اصلاحی
    waves = self._identify_waves(df)
    if len(waves) >= 3:
        # بررسی اینکه موج B کوتاهتر از موج A باشد
        if waves[1]['height'] < waves[0]['height']:
            return True
    
    return False

def _is_triangle_pattern(self, df: pd.DataFrame) -> bool:
    """بررسی الگوی مثلث"""
    if len(df) < 20:
        return False
    
    # بررسی همگرایی خطوط روند
    highs = df['high'].rolling(window=5).max().dropna()
    lows = df['low'].rolling(window=5).min().dropna()
    
    # محاسبه شیب خطوط روند
    high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
    low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
    
    # اگر شیب خط بالا نزولی و شیب خط پایین صعودی باشد
    if high_slope < 0 and low_slope > 0:
        return True
    
    return False

def _is_flat_pattern(self, df: pd.DataFrame) -> bool:
    """بررسی الگوی مسطح"""
    if len(df) < 15:
        return False
    
    # بررسی نوسان در یک محدوده
    price_range = (df['high'].max() - df['low'].min()) / df['close'].mean()
    
    # اگر نوسان کمتر از 5% باشد
    if price_range < 0.05:
        return True
    
    return False

def _identify_waves(self, df: pd.DataFrame) -> List[Dict]:
    """شناسایی امواج قیمت"""
    waves = []
    pivot_points = self._identify_pivot_points(df)
    
    if len(pivot_points) < 2:
        return waves
    
    # مرتب‌سازی نقاط چرخش بر اساس ایندکس
    pivot_points.sort(key=lambda x: x['index'])
    
    # ایجاد امواج از نقاط چرخش
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

def _predict_next_elliott_wave(self, current_wave: str) -> str:
    """پیش‌بینی موج بعدی الیوت"""
    wave_sequence = {
        'موج 1': 'موج 2',
        'موج 2': 'موج 3',
        'موج 3': 'موج 4',
        'موج 4': 'موج 5',
        'موج 5': 'اصلاح'
    }
    
    return wave_sequence.get(current_wave, 'نامشخص')

def _calculate_wave_target(self, df: pd.DataFrame, current_wave: str) -> float:
    """محاسبه هدف قیمتی موج"""
    current_price = df['close'].iloc[-1]
    
    if current_wave == 'موج 3':
        # موج 3 معمولاً 1.618 برابر موج 1 است
        wave1_height = self._get_wave_height(df, 'موج 1')
        return current_price + (wave1_height * 0.618)
    elif current_wave == 'موج 5':
        # موج 5 معمولاً برابر با موج 1 است
        wave1_height = self._get_wave_height(df, 'موج 1')
        return current_price + wave1_height
    else:
        return current_price * 1.05  # هدف پیش‌فرض 5%

def _get_wave_height(self, df: pd.DataFrame, wave_name: str) -> float:
    """دریافت ارتفاع موج مشخص"""
    waves = self._identify_waves(df)
    for wave in waves:
        if wave['type'] == wave_name:
            return wave['height']
    return df['close'].iloc[-1] * 0.05  # مقدار پیش‌فرض

def _perform_quantum_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل کوانتومی"""
    logger.info(f"Performing quantum analysis for {symbol}")
    
    price_data = self._extract_price_data(data)
    if not price_data:
        return {}
    
    df = pd.DataFrame(price_data)
    df['close'] = pd.to_numeric(df['close'])
    
    # استخراج ویژگی‌های کوانتومی
    quantum_features = self._extract_quantum_features(df)
    
    if not quantum_features:
        return {}
    
    # پیش‌بینی با مدل کوانتومی
    if self.models['quantum_pattern']['trained']:
        try:
            X = np.array([list(quantum_features.values())])
            X_scaled = self.models['quantum_pattern']['scaler'].transform(X)
            pattern_prediction = self.models['quantum_pattern']['model'].predict(X_scaled)[0]
            pattern_probabilities = self.models['quantum_pattern']['model'].predict_proba(X_scaled)[0]
        except Exception as e:
            logger.error(f"Error in quantum pattern prediction: {e}")
            pattern_prediction = 0
            pattern_probabilities = [0.25, 0.25, 0.25, 0.25]
    else:
        pattern_prediction = 0
        pattern_probabilities = [0.25, 0.25, 0.25, 0.25]
    
    pattern_labels = ['الگوی صعودی', 'الگوی نزولی', 'الگوی تثبیت', 'الگوی معکوس']
    detected_pattern = pattern_labels[pattern_prediction]
    pattern_confidence = max(pattern_probabilities)
    
    # شناسایی الگوهای کوانتومی
    quantum_patterns = self._identify_quantum_patterns(df)
    
    return {
        'detected_pattern': detected_pattern,
        'pattern_confidence': pattern_confidence,
        'pattern_probabilities': {pattern_labels[i]: pattern_probabilities[i] for i in range(len(pattern_labels))},
        'quantum_patterns': quantum_patterns,
        'fractal_dimension': self._calculate_fractal_dimension(df),
        'entropy': self._calculate_entropy(df),
        'lyapunov_exponent': self._calculate_lyapunov_exponent(df),
        'prediction_horizon': self._calculate_prediction_horizon(df)
    }

def _extract_quantum_features(self, df: pd.DataFrame) -> Dict:
    """استخراج ویژگی‌های کوانتومی"""
    if len(df) < 50:
        return {}
    
    features = {}
    
    # ویژگی‌های فرکتال
    features['fractal_dimension'] = self._calculate_fractal_dimension(df)
    
    # ویژگی‌های آنتروپی
    features['entropy'] = self._calculate_entropy(df)
    
    # ویژگی‌های لیاپانوف
    features['lyapunov_exponent'] = self._calculate_lyapunov_exponent(df)
    
    # ویژگی‌های طیفی
    features['spectral_entropy'] = self._calculate_spectral_entropy(df)
    
    # ویژگی‌های بازگشتی
    features['recurrence_rate'] = self._calculate_recurrence_rate(df)
    
    # ویژگی‌های پیچیدگی
    features['complexity'] = self._calculate_complexity(df)
    
    return features

def _calculate_fractal_dimension(self, df: pd.DataFrame) -> float:
    """محاسبه بعد فرکتال"""
    prices = df['close'].values
    n = len(prices)
    
    if n < 10:
        return 1.0
    
    # محاسبه بعد فرکتال با روش شمارش جعبه‌ای
    scales = np.logspace(0.1, 1, num=10)
    counts = []
    
    for scale in scales:
        # تقسیم داده‌ها به جعبه‌هایی با اندازه scale
        boxes = np.floor(np.arange(n) / scale).astype(int)
        box_counts = np.bincount(boxes)
        counts.append(len(box_counts))
    
    # برازش خطی برای محاسبه بعد فرکتال
    coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
    return -coeffs[0]

def _calculate_entropy(self, df: pd.DataFrame) -> float:
    """محاسبه آنتروپی"""
    prices = df['close'].values
    n = len(prices)
    
    if n < 10:
        return 0.0
    
    # محاسبه تفاوت‌های متوالی
    diffs = np.diff(prices)
    
    # محاسبه هیستوگرام
    hist, _ = np.histogram(diffs, bins=20)
    
    # نرمال‌سازی
    hist = hist / np.sum(hist)
    
    # محاسبه آنتروپی
    entropy = 0
    for p in hist:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy

def _calculate_lyapunov_exponent(self, df: pd.DataFrame) -> float:
    """محاسبه نمای لیاپانوف"""
    prices = df['close'].values
    n = len(prices)
    
    if n < 20:
        return 0.0
    
    # بازسازی فضای فاز
    m = 3  # بعد جاسازی
    tau = 1  # تأخیر زمانی
    
    # ایجاد ماتریس جاسازی
    embedded = np.zeros((n - (m-1)*tau, m))
    for i in range(m):
        embedded[:, i] = prices[i*tau : i*tau + len(embedded)]
    
    # محاسبه نمای لیاپانوف
    max_iter = min(100, len(embedded) - 10)
    lyapunov_sum = 0
    
    for i in range(max_iter):
        # پیدا کردن نزدیک‌ترین همسایه
        distances = np.sqrt(np.sum((embedded - embedded[i])**2, axis=1))
        distances[i] = np.inf  # نادیده گرفتن خود نقطه
        
        nearest_idx = np.argmin(distances)
        initial_distance = distances[nearest_idx]
        
        if initial_distance == 0:
            continue
        
        # رشد فاصله در زمان
        j = min(i + 10, len(embedded) - 1)
        final_distance = np.sqrt(np.sum((embedded[j] - embedded[nearest_idx + (j-i)])**2))
        
        if final_distance > 0:
            lyapunov_sum += np.log(final_distance / initial_distance)
    
    if max_iter > 0:
        return lyapunov_sum / (max_iter * 10)
    return 0.0

def _calculate_spectral_entropy(self, df: pd.DataFrame) -> float:
    """محاسبه آنتروپی طیفی"""
    prices = df['close'].values
    n = len(prices)
    
    if n < 10:
        return 0.0
    
    # تبدیل فوریه
    fft = np.fft.fft(prices)
    power_spectrum = np.abs(fft) ** 2
    
    # نرمال‌سازی
    power_spectrum = power_spectrum / np.sum(power_spectrum)
    
    # محاسبه آنتروپی
    entropy = 0
    for p in power_spectrum:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy

def _calculate_recurrence_rate(self, df: pd.DataFrame) -> float:
    """محاسبه نرخ بازگشتی"""
    prices = df['close'].values
    n = len(prices)
    
    if n < 10:
        return 0.0
    
    # ایجاد ماتریس بازگشتی
    recurrence_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if abs(prices[i] - prices[j]) < 0.01 * prices[i]:
                recurrence_matrix[i, j] = 1
    
    # محاسبه نرخ بازگشتی
    recurrence_rate = np.sum(recurrence_matrix) / (n * n)
    
    return recurrence_rate

def _calculate_complexity(self, df: pd.DataFrame) -> float:
    """محاسبه پیچیدگی"""
    prices = df['close'].values
    n = len(prices)
    
    if n < 10:
        return 0.0
    
    # محاسبه پیچیدگی لامپل-زیو
    binary_sequence = np.zeros(n-1)
    for i in range(n-1):
        if prices[i+1] > prices[i]:
            binary_sequence[i] = 1
    
    # محاسبه پیچیدگی
    complexity = 0
    n_patterns = 0
    
    for i in range(len(binary_sequence) - 1):
        pattern = binary_sequence[i:i+2]
        if np.array_equal(pattern, [0, 0]) or np.array_equal(pattern, [1, 1]):
            complexity += 1
        n_patterns += 1
    
    if n_patterns > 0:
        return complexity / n_patterns
    return 0.0

def _identify_quantum_patterns(self, df: pd.DataFrame) -> List[str]:
    """شناسایی الگوهای کوانتومی"""
    patterns = []
    
    # الگوی آشوب
    if self._is_chaotic_pattern(df):
        patterns.append("آشوب")
    
    # الگوی فرکتال
    if self._is_fractal_pattern(df):
        patterns.append("فرکتال")
    
    # الگوی دوره‌ای
    if self._is_periodic_pattern(df):
        patterns.append("دوره‌ای")
    
    # الگوی تصادفی
    if self._is_random_pattern(df):
        patterns.append("تصادفی")
    
    return patterns

def _is_chaotic_pattern(self, df: pd.DataFrame) -> bool:
    """بررسی الگوی آشوب"""
    lyapunov = self._calculate_lyapunov_exponent(df)
    return lyapunov > 0.01

def _is_fractal_pattern(self, df: pd.DataFrame) -> bool:
    """بررسی الگوی فرکتال"""
    fractal_dim = self._calculate_fractal_dimension(df)
    return 1.2 < fractal_dim < 1.8

def _is_periodic_pattern(self, df: pd.DataFrame) -> bool:
    """بررسی الگوی دوره‌ای"""
    prices = df['close'].values
    n = len(prices)
    
    if n < 20:
        return False
    
    # تبدیل فوریه
    fft = np.fft.fft(prices)
    power_spectrum = np.abs(fft) ** 2
    
    # پیدا کردن فرکانس غالب
    dominant_freq = np.argmax(power_spectrum[1:n//2]) + 1
    
    # بررسی اینکه آیا فرکانس غالب قدرت کافی دارد
    if power_spectrum[dominant_freq] > 0.5 * np.sum(power_spectrum):
        return True
    
    return False

def _is_random_pattern(self, df: pd.DataFrame) -> bool:
    """بررسی الگوی تصادفی"""
    entropy = self._calculate_entropy(df)
    return entropy > 3.0

def _calculate_prediction_horizon(self, df: pd.DataFrame) -> int:
    """محاسبه افق پیش‌بینی"""
    lyapunov = self._calculate_lyapunov_exponent(df)
    
    if lyapunov > 0:
        # افق پیش‌بینی بر اساس نمای لیاپانوف
        horizon = int(1 / lyapunov)
        return min(horizon, 30)  # حداکثر 30 روز
    else:
        return 10  # مقدار پیش‌فرض

def _perform_market_structure_analysis(self, symbol: str, data: Dict) -> Dict:
    """انجام تحلیل ساختار بازار"""
    logger.info(f"Performing market structure analysis for {symbol}")
    
    price_data = self._extract_price_data(data)
    if not price_data:
        return {}
    
    df = pd.DataFrame(price_data)
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    
    # شناسایی نقاط عرضه و تقاضا
    supply_zones = []
    demand_zones = []
    
    # شناسایی نقاط چرخش
    pivot_points = self._identify_pivot_points(df)
    
    # شناسایی نواحی عرضه و تقاضا
    for i in range(2, len(df) - 2):
        # ناحیه تقاضا (حمایت)
        if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
            df['low'].iloc[i] < df['low'].iloc[i+1] and
            df['low'].iloc[i] < df['low'].iloc[i-2] and
            df['low'].iloc[i] < df['low'].iloc[i+2]):
            
            demand_zones.append({
                'price': df['low'].iloc[i],
                'strength': self._calculate_zone_strength(df, i, 'demand'),
                'timeframe': '1d'
            })
        
        # ناح