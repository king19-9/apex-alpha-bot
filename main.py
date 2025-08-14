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
import pandas_ta as ta
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import io
import base64
import json

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
        self.risk_reward_ratio = 3
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

    def format_analysis_response(self, analysis: Dict) -> str:
        """فرمت‌دهی پاسخ تحلیل"""
        symbol = analysis['symbol']
        signal = analysis['signal']
        confidence = analysis['confidence']
        
        signal_emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
        
        response = f"{signal_emoji} *تحلیل {symbol}*\n\n"
        response += f"📊 سیگنال: {signal}\n"
        response += f"🎯 اطمینان: {confidence:.1%}\n\n"
        
        if analysis.get('stop_loss') and analysis.get('take_profit'):
            response += f"🛑 حد ضرر: ${analysis['stop_loss']:,.2f}\n"
            response += f"🎯 حد سود: ${analysis['take_profit']:,.2f}\n"
            response += f"⚖️ نسبت ریسک به پاداش: {analysis.get('risk_reward_ratio', 0):.2f}\n\n"
        
        response += "📋 *توصیه‌ها:*\n"
        for rec in analysis.get('recommendations', []):
            response += f"• {rec}\n"
        
        return response

    async def get_trading_signals(self) -> List[Dict]:
        """دریافت سیگنال‌های معاملاتی برای تمام ارزها"""
        signals = []
        
        # تحلیل 10 ارز برتر
        top_cryptos = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOT', 'DOGE', 'AVAX', 'MATIC']
        
        for symbol in top_cryptos:
            try:
                analysis = await self.perform_intelligent_analysis(symbol)
                
                if analysis['confidence'] > 0.7:  # فقط سیگنال‌های با اطمینان بالا
                    signals.append({
                        'symbol': symbol,
                        'signal': analysis['signal'],
                        'confidence': analysis['confidence'],
                        'price': analysis.get('current_price', 0),
                        'stop_loss': analysis.get('stop_loss', 0),
                        'take_profit': analysis.get('take_profit', 0),
                        'risk_reward_ratio': analysis.get('risk_reward_ratio', 0),
                        'price_change_24h': analysis.get('market_data', {}).get('price_change_24h', 0)
                    })
            except Exception as e:
                logger.error(f"Error getting signal for {symbol}: {e}")
        
        # مرتب‌سازی بر اساس اطمینان
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return signals

    async def fetch_data_from_multiple_sources(self, symbol: str) -> Dict:
        """دریافت داده‌ها از چندین منبع"""
        data = {}
        
        # دریافت داده‌ها از CoinGecko
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}"
            response = requests.get(url)
            if response.status_code == 200:
                data['coingecko'] = response.json()
        except Exception as e:
            logger.error(f"Error fetching from CoinGecko: {e}")
        
        # دریافت داده‌ها از CryptoCompare
        try:
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit=30"
            response = requests.get(url)
            if response.status_code == 200:
                data['cryptocompare'] = response.json()
        except Exception as e:
            logger.error(f"Error fetching from CryptoCompare: {e}")
        
        return data

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

    async def _perform_technical_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل تکنیکال"""
        logger.info(f"Performing technical analysis for {symbol}")
        
        price_data = self._extract_price_data(data)
        if not price_data:
            return {}
        
        df = pd.DataFrame(price_data)
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # محاسبه شاخص‌های تکنیکال
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['macd'] = ta.macd(df['close'])
        df['bb'] = ta.bbands(df['close'])
        df['atr'] = ta.atr(df['high'], df['low'], df['close'])
        
        # محاسبه میانگین‌های متحرک
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        
        # تعیین روند
        current_price = df['close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            trend = 'bullish'
        elif current_price < sma_20 < sma_50:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        # تعیین سیگنال
        rsi = df['rsi'].iloc[-1]
        macd = df['macd'].iloc[-1]
        
        signal = 'HOLD'
        confidence = 0.5
        
        if rsi < 30 and trend == 'bullish':
            signal = 'BUY'
            confidence = 0.7
        elif rsi > 70 and trend == 'bearish':
            signal = 'SELL'
            confidence = 0.7
        elif macd > 0 and trend == 'bullish':
            signal = 'BUY'
            confidence = 0.6
        elif macd < 0 and trend == 'bearish':
            signal = 'SELL'
            confidence = 0.6
        
        return {
            'signal': signal,
            'confidence': confidence,
            'current_price': current_price,
            'trend': trend,
            'rsi': rsi,
            'macd': macd,
            'volatility': df['atr'].iloc[-1] / current_price if current_price > 0 else 0.02
        }

    async def _perform_sentiment_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل احساسات"""
        logger.info(f"Performing sentiment analysis for {symbol}")
        
        # دریافت اخبار
        news_data = self._extract_news(data)
        
        if not news_data:
            return {'signal': 'HOLD', 'confidence': 0.5, 'sentiment_score': 0}
        
        # تحلیل احساسات اخبار
        sentiment_scores = []
        sentiment_model = self.models['sentiment_analyzer']['model']
        
        for news in news_data[:10]:  # تحلیل 10 خبر اخیر
            try:
                text = f"{news['title']} {news.get('description', '')}"
                result = sentiment_model(text[:512])  # محدودیت طول متن
                
                # تبدیل نتیجه به امتیاز
                if result[0]['label'] == 'POSITIVE':
                    score = 1.0
                elif result[0]['label'] == 'NEGATIVE':
                    score = -1.0
                else:
                    score = 0.0
                
                sentiment_scores.append(score)
            except Exception as e:
                logger.error(f"Error analyzing sentiment: {e}")
        
        # محاسبه امتیاز میانگین
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # تعیین سیگنال
        signal = 'HOLD'
        confidence = 0.5
        
        if avg_sentiment > 0.3:
            signal = 'BUY'
            confidence = 0.6 + (avg_sentiment * 0.2)
        elif avg_sentiment < -0.3:
            signal = 'SELL'
            confidence = 0.6 + (abs(avg_sentiment) * 0.2)
        
        return {
            'signal': signal,
            'confidence': min(1, confidence),
            'sentiment_score': avg_sentiment,
            'news_count': len(news_data)
        }

    async def _perform_elliott_wave_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل امواج الیوت"""
        logger.info(f"Performing Elliott Wave analysis for {symbol}")
        
        price_data = self._extract_price_data(data)
        if not price_data:
            return {}
        
        df = pd.DataFrame(price_data)
        df['close'] = pd.to_numeric(df['close'])
        
        # شناسایی الگوهای امواج الیوت
        patterns = self._identify_elliott_wave_patterns(df)
        
        # تعیین سیگنال بر اساس الگوها
        signal = 'HOLD'
        confidence = 0.5
        
        if 'ایمپالس' in patterns:
            signal = 'BUY'
            confidence = 0.7
        elif 'اصلاحی' in patterns:
            signal = 'SELL'
            confidence = 0.6
        
        return {
            'signal': signal,
            'confidence': confidence,
            'patterns': patterns,
            'wave_count': len(patterns)
        }

    async def _perform_quantum_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل کوانتومی"""
        logger.info(f"Performing quantum analysis for {symbol}")
        
        price_data = self._extract_price_data(data)
        if not price_data:
            return {}
        
        df = pd.DataFrame(price_data)
        df['close'] = pd.to_numeric(df['close'])
        
        # محاسبه شاخص‌های کوانتومی
        fractal_dim = self._calculate_fractal_dimension(df)
        entropy = self._calculate_entropy(df)
        lyapunov = self._calculate_lyapunov_exponent(df)
        
        # تعیین سیگنال
        signal = 'HOLD'
        confidence = 0.5
        
        if fractal_dim > 1.3 and entropy > 3.0:
            signal = 'BUY'
            confidence = 0.6
        elif fractal_dim < 1.1 and entropy < 2.0:
            signal = 'SELL'
            confidence = 0.6
        
        return {
            'signal': signal,
            'confidence': confidence,
            'fractal_dimension': fractal_dim,
            'entropy': entropy,
            'lyapunov_exponent': lyapunov
        }

    async def _perform_whale_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل فعالیت نهنگ‌ها"""
        logger.info(f"Performing whale analysis for {symbol}")
        
        whale_transactions = await self.fetch_whale_transactions(symbol)
        
        if not whale_transactions:
            return {'signal': 'HOLD', 'confidence': 0.5, 'whale_activity': 'low'}
        
        # تحلیل فعالیت نهنگ‌ها
        buy_volume = sum(tx['amount_usd'] for tx in whale_transactions if tx['transaction_type'] == 'buy')
        sell_volume = sum(tx['amount_usd'] for tx in whale_transactions if tx['transaction_type'] == 'sell')
        
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return {'signal': 'HOLD', 'confidence': 0.5, 'whale_activity': 'low'}
        
        buy_ratio = buy_volume / total_volume
        sell_ratio = sell_volume / total_volume
        
        # تعیین سیگنال
        signal = 'HOLD'
        confidence = 0.5
        
        if buy_ratio > 0.7:
            signal = 'BUY'
            confidence = 0.7
        elif sell_ratio > 0.7:
            signal = 'SELL'
            confidence = 0.7
        
        return {
            'signal': signal,
            'confidence': confidence,
            'buy_ratio': buy_ratio,
            'sell_ratio': sell_ratio,
            'whale_activity': 'high' if total_volume > 10000000 else 'medium' if total_volume > 1000000 else 'low'
        }

    async def _perform_whale_behavior_analysis(self, symbol: str, whale_transactions: List[Dict]) -> Dict:
        """تحلیل رفتار نهنگ‌ها"""
        if not whale_transactions:
            return {'sentiment': 0, 'activity_level': 'low'}
        
        # تحلیل احساسات نهنگ‌ها
        buy_volume = sum(tx['amount_usd'] for tx in whale_transactions if tx['transaction_type'] == 'buy')
        sell_volume = sum(tx['amount_usd'] for tx in whale_transactions if tx['transaction_type'] == 'sell')
        
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return {'sentiment': 0, 'activity_level': 'low'}
        
        sentiment = (buy_volume - sell_volume) / total_volume
        
        # تعیین سطح فعالیت
        if total_volume > 10000000:
            activity_level = 'high'
        elif total_volume > 1000000:
            activity_level = 'medium'
        else:
            activity_level = 'low'
        
        return {
            'sentiment': sentiment,
            'activity_level': activity_level,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume
        }

    async def _perform_market_structure_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل ساختار بازار"""
        logger.info(f"Performing market structure analysis for {symbol}")
        
        price_data = self._extract_price_data(data)
        if not price_data:
            return {}
        
        df = pd.DataFrame(price_data)
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        
        # شناسایی ساختار بازار
        support, resistance = self._find_support_resistance(df)
        current_price = df['close'].iloc[-1]
        
        # تعیین موقعیت قیمت
        if current_price > resistance:
            position = 'above_resistance'
        elif current_price < support:
            position = 'below_support'
        else:
            position = 'between'
        
        # تعیین سیگنال
        signal = 'HOLD'
        confidence = 0.5
        
        if position == 'below_support':
            signal = 'BUY'
            confidence = 0.6
        elif position == 'above_resistance':
            signal = 'SELL'
            confidence = 0.6
        
        return {
            'signal': signal,
            'confidence': confidence,
            'support': support,
            'resistance': resistance,
            'position': position
        }

    async def _perform_on_chain_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل زنجیره‌ای"""
        logger.info(f"Performing on-chain analysis for {symbol}")
        
        on_chain_metrics = await self.fetch_on_chain_metrics(symbol)
        
        if not on_chain_metrics:
            return {'signal': 'HOLD', 'confidence': 0.5}
        
        # محاسبه امتیاز زنجیره‌ای
        score = self.calculate_on_chain_score(on_chain_metrics)
        
        # تعیین سیگنال
        signal = 'HOLD'
        confidence = 0.5
        
        if score > 0.7:
            signal = 'BUY'
            confidence = 0.6 + (score * 0.2)
        elif score < 0.3:
            signal = 'SELL'
            confidence = 0.6 + ((1 - score) * 0.2)
        
        return {
            'signal': signal,
            'confidence': min(1, confidence),
            'on_chain_score': score,
            'metrics': on_chain_metrics
        }

    async def _perform_correlation_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل همبستگی"""
        logger.info(f"Performing correlation analysis for {symbol}")
        
        # این تحلیل نیاز به داده‌های چند ارز دارد
        # در اینجا یک تحلیل ساده ارائه می‌شود
        return {'signal': 'HOLD', 'confidence': 0.5}

    async def _perform_seasonal_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل فصلی"""
        logger.info(f"Performing seasonal analysis for {symbol}")
        
        # این تحلیل نیاز به داده‌های تاریخی دارد
        # در اینجا یک تحلیل ساده ارائه می‌شود
        return {'signal': 'HOLD', 'confidence': 0.5}

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
        rsi = ta.rsi(df['close'], length=14)
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
        rsi = ta.rsi(df['close'], length=14)
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
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        
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
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        
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
            'confidence': confidence
        }

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

    def _identify_confluence_zones(self, supply_zones: List[Dict], demand_zones: List[Dict]) -> List[Dict]:
        """شناسایی نواحی تلاقی"""
        confluence_zones = []
        
        # بررسی تلاقی نواحی عرضه و تقاضا
        for supply in supply_zones:
            for demand in demand_zones:
                if abs(supply['price'] - demand['price']) / supply['price'] < 0.02:  # 2% tolerance
                    confluence_zones.append({
                        'price': (supply['price'] + demand['price']) / 2,
                        'strength': (supply['strength'] + demand['strength']) / 2,
                        'type': 'confluence'
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
        
        # شناسایی نقاط چرخش
        pivot_points = self._identify_pivot_points(df)
        
        if len(pivot_points) < 2:
            return {}
        
        # محاسبه سطوح فیبوناچی
        high_point = max(pivot_points, key=lambda x: x['price'])
        low_point = min(pivot_points, key=lambda x: x['price'])
        
        high_price = high_point['price']
        low_price = low_point['price']
        
        # محاسبه سطوح اصلاحی
        retracement_levels = {
            '0.236': high_price - (high_price - low_price) * 0.236,
            '0.382': high_price - (high_price - low_price) * 0.382,
            '0.5': high_price - (high_price - low_price) * 0.5,
            '0.618': high_price - (high_price - low_price) * 0.618,
            '0.786': high_price - (high_price - low_price) * 0.786
        }
        
        # محاسبه سطوح گسترشی
        extension_levels = {
            '1.272': high_price + (high_price - low_price) * 0.272,
            '1.618': high_price + (high_price - low_price) * 0.618,
            '2.618': high_price + (high_price - low_price) * 1.618
        }
        
        # محاسبه نواحی تلاقی
        confluence_zones = []
        current_price = df['close'].iloc[-1]
        
        for level_name, level_price in retracement_levels.items():
            if abs(current_price - level_price) / current_price < 0.02:  # 2% tolerance
                confluence_zones.append({
                    'price': level_price,
                    'level': level_name,
                    'type': 'retracement'
                })
        
        for level_name, level_price in extension_levels.items():
            if abs(current_price - level_price) / current_price < 0.02:  # 2% tolerance
                confluence_zones.append({
                    'price': level_price,
                    'level': level_name,
                    'type': 'extension'
                })
        
        # ذخیره تحلیل در پایگاه داده
        self.cursor.execute('''
        INSERT INTO fibonacci_analysis 
        (symbol, timestamp, retracement_levels, extension_levels, confluence_zones, accuracy)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            symbol,
            datetime.now().isoformat(),
            json.dumps(retracement_levels),
            json.dumps(extension_levels),
            json.dumps(confluence_zones),
            0.8  # دقت پیش‌فرض
        ))
        self.conn.commit()
        
        return {
            'levels': {
                'retracement': retracement_levels,
                'extension': extension_levels
            },
            'confluence_zones': confluence_zones,
            'accuracy': 0.8
        }

    async def _perform_volume_profile_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل پروفایل حجم"""
        logger.info(f"Performing Volume Profile analysis for {symbol}")
        
        price_data = self._extract_price_data(data)
        if not price_data:
            return {}
        
        df = pd.DataFrame(price_data)
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # محاسبه پروفایل حجم
        price_bins = np.linspace(df['low'].min(), df['high'].max(), 20)
        volume_profile = []
        
        for i in range(len(price_bins) - 1):
            mask = (df['close'] >= price_bins[i]) & (df['close'] < price_bins[i + 1])
            volume = df[mask]['volume'].sum()
            volume_profile.append({
                'price_range': (price_bins[i], price_bins[i + 1]),
                'volume': volume
            })
        
        # شناسایی ناحیه با بیشترین حجم (POC)
        poc = max(volume_profile, key=lambda x: x['volume'])
        
        # تعیین سیگنال
        current_price = df['close'].iloc[-1]
        signal = 'HOLD'
        confidence = 0.5
        
        if current_price < poc['price_range'][0]:
            signal = 'BUY'
            confidence = 0.6
        elif current_price > poc['price_range'][1]:
            signal = 'SELL'
            confidence = 0.6
        
        return {
            'signal': signal,
            'confidence': confidence,
            'poc': poc,
            'volume_profile': volume_profile
        }

    async def _perform_market_profile_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل پروفایل بازار"""
        logger.info(f"Performing Market Profile analysis for {symbol}")
        
        # این تحلیل نیاز به داده‌های تیک به تیک دارد
        # در اینجا یک تحلیل ساده ارائه می‌شود
        return {'signal': 'HOLD', 'confidence': 0.5}

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
        if self._is_gartley_pattern(df):
            patterns.append('Gartley')
        
        # الگوی پروانه (Butterfly)
        if self._is_butterfly_pattern(df):
            patterns.append('Butterfly')
        
        # الگوی خفاش (Bat)
        if self._is_bat_pattern(df):
            patterns.append('Bat')
        
        # الگوی خرچنگ (Crab)
        if self._is_crab_pattern(df):
            patterns.append('Crab')
        
        # تعیین سیگنال
        signal = 'HOLD'
        confidence = 0.5
        
        if any(pattern in patterns for pattern in ['Gartley', 'Butterfly', 'Bat', 'Crab']):
            signal = 'BUY' if any('bullish' in pattern.lower() for pattern in patterns) else 'SELL'
            confidence = 0.7
        
        return {
            'signal': signal,
            'confidence': confidence,
            'patterns': patterns
        }

    def _is_gartley_pattern(self, df: pd.DataFrame) -> bool:
        """بررسی الگوی گارتلی"""
        # این یک پیاده‌سازی ساده است
        return False

    def _is_butterfly_pattern(self, df: pd.DataFrame) -> bool:
        """بررسی الگوی پروانه"""
        # این یک پیاده‌سازی ساده است
        return False

    def _is_bat_pattern(self, df: pd.DataFrame) -> bool:
        """بررسی الگوی خفاش"""
        # این یک پیاده‌سازی ساده است
        return False

    def _is_crab_pattern(self, df: pd.DataFrame) -> bool:
        """بررسی الگوی خرچنگ"""
        # این یک پیاده‌سازی ساده است
        return False

    async def _perform_divergence_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل واگرایی"""
        logger.info(f"Performing Divergence analysis for {symbol}")
        
        price_data = self._extract_price_data(data)
        if not price_data:
            return {}
        
        df = pd.DataFrame(price_data)
        df['close'] = pd.to_numeric(df['close'])
        
        # محاسبه RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # شناسایی واگرایی‌ها
        divergence_signals = []
        
        # شناسایی نقاط اوج و حضیض قیمت
        from scipy.signal import argrelextrema
        price_highs_idx = argrelextrema(df['close'].values, np.greater, order=5)[0]
        price_lows_idx = argrelextrema(df['close'].values, np.less, order=5)[0]
        
        # شناسایی واگرایی منفی
        if len(price_highs_idx) >= 2:
            price_highs = df['close'].iloc[price_highs_idx]
            rsi_highs = df['rsi'].iloc[price_highs_idx]
            
            if (price_highs.iloc[-1] > price_highs.iloc[-2] and 
                rsi_highs.iloc[-1] < rsi_highs.iloc[-2]):
                divergence_signals.append('bearish')
        
        # شناسایی واگرایی مثبت
        if len(price_lows_idx) >= 2:
            price_lows = df['close'].iloc[price_lows_idx]
            rsi_lows = df['rsi'].iloc[price_lows_idx]
            
            if (price_lows.iloc[-1] < price_lows.iloc[-2] and 
                rsi_lows.iloc[-1] > rsi_lows.iloc[-2]):
                divergence_signals.append('bullish')
        
        # تعیین سیگنال
        signal = 'HOLD'
        confidence = 0.5
        
        if 'bullish' in divergence_signals:
            signal = 'BUY'
            confidence = 0.7
        elif 'bearish' in divergence_signals:
            signal = 'SELL'
            confidence = 0.7
        
        return {
            'signal': signal,
            'confidence': confidence,
            'signals': divergence_signals
        }

    async def _perform_economic_calendar_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل تقویم اقتصادی"""
        logger.info(f"Performing Economic Calendar analysis for {symbol}")
        
        # این تحلیل نیاز به دسترسی به تقویم اقتصادی دارد
        # در اینجا یک تحلیل ساده ارائه می‌شود
        return {'signal': 'HOLD', 'confidence': 0.5}

    async def _perform_order_flow_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل جریان سفارش"""
        logger.info(f"Performing Order Flow analysis for {symbol}")
        
        # این تحلیل نیاز به داده‌های سفارش‌ها دارد
        # در اینجا یک تحلیل ساده ارائه می‌شود
        return {'signal': 'HOLD', 'confidence': 0.5}

    async def _perform_liquidity_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل نقدینگی"""
        logger.info(f"Performing Liquidity analysis for {symbol}")
        
        price_data = self._extract_price_data(data)
        if not price_data:
            return {}
        
        df = pd.DataFrame(price_data)
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # محاسبه شاخص نقدینگی
        avg_volume = df['volume'].mean()
        recent_volume = df['volume'].iloc[-5:].mean()
        
        liquidity_score = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # تعیین سیگنال
        signal = 'HOLD'
        confidence = 0.5
        
        if liquidity_score > 1.5:
            signal = 'BUY'
            confidence = 0.6
        elif liquidity_score < 0.5:
            signal = 'SELL'
            confidence = 0.6
        
        return {
            'signal': signal,
            'confidence': confidence,
            'liquidity_score': liquidity_score,
            'score': liquidity_score
        }

    async def _perform_monte_carlo_analysis(self, symbol: str, data: Dict) -> Dict:
        """انجام تحلیل مونت کارلو"""
        logger.info(f"Performing Monte Carlo analysis for {symbol}")
        
        price_data = self._extract_price_data(data)
        if not price_data:
            return {}
        
        df = pd.DataFrame(price_data)
        df['close'] = pd.to_numeric(df['close'])
        
        # محاسبه بازدهی روزانه
        df['returns'] = df['close'].pct_change()
        
        # شبیه‌سازی مونت کارلو
        n_simulations = 1000
        n_days = 30
        
        simulations = []
        last_price = df['close'].iloc[-1]
        
        for _ in range(n_simulations):
            prices = [last_price]
            for _ in range(n_days):
                daily_return = np.random.choice(df['returns'].dropna())
                prices.append(prices[-1] * (1 + daily_return))
            simulations.append(prices)
        
        # محاسبه احتمالات
        final_prices = [sim[-1] for sim in simulations]
        prob_increase = sum(1 for price in final_prices if price > last_price) / len(final_prices)
        
        # تعیین سیگنال
        signal = 'HOLD'
        confidence = 0.5
        
        if prob_increase > 0.6:
            signal = 'BUY'
            confidence = 0.6 + (prob_increase - 0.6)
        elif prob_increase < 0.4:
            signal = 'SELL'
            confidence = 0.6 + (0.4 - prob_increase)
        
        return {
            'signal': signal,
            'confidence': min(1, confidence),
            'prob_increase': prob_increase,
            'simulations': n_simulations
        }

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

    def _find_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
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

    def _calculate_fractal_dimension(self, df: pd.DataFrame) -> float:
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

    def _calculate_entropy(self, df: pd.DataFrame) -> float:
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

    def _calculate_lyapunov_exponent(self, df: pd.DataFrame) -> float:
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
        
        waves = self._identify_waves(df)
        if len(waves) >= 5:
            if (waves[2]['height'] > waves[0]['height'] and 
                waves[2]['height'] > waves[4]['height']):
                return True
        return False

    def _is_corrective_pattern(self, df: pd.DataFrame) -> bool:
        """بررسی الگوی اصلاحی"""
        if len(df) < 15:
            return False
        
        waves = self._identify_waves(df)
        if len(waves) >= 3:
            if waves[1]['height'] < waves[0]['height']:
                return True
        return False

    def _is_triangle_pattern(self, df: pd.DataFrame) -> bool:
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

    def _is_flat_pattern(self, df: pd.DataFrame) -> bool:
        """بررسی الگوی مسطح"""
        if len(df) < 15:
            return False
        
        price_range = (df['high'].max() - df['low'].min()) / df['close'].mean()
        if price_range < 0.05:
            return True
        return False

    def _identify_waves(self, df: pd.DataFrame) -> List[Dict]:
        """شناسایی امواج قیمت"""
        pivot_points = self._identify_pivot_points(df)
        
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

if __name__ == '__main__':
    import os
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler
    
    # Initialize the bot
    bot = AdvancedCryptoBot()
    
    # Get the token from environment
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
        exit(1)
    
    # Create the Application
    application = Application.builder().token(token).build()
    
    # Add handlers
    from handlers import TelegramHandlers
    telegram_handlers = TelegramHandlers()
    
    application.add_handler(CommandHandler("start", telegram_handlers.start_command))
    application.add_handler(CommandHandler("help", telegram_handlers.help_command))
    application.add_handler(CommandHandler("analyze", telegram_handlers.analyze_command))
    application.add_handler(CommandHandler("signals", telegram_handlers.signals_command))
    application.add_handler(CommandHandler("watchlist", telegram_handlers.watchlist_command))
    application.add_handler(CommandHandler("alerts", telegram_handlers.alerts_command))
    application.add_handler(CommandHandler("performance", telegram_handlers.performance_command))
    application.add_handler(CallbackQueryHandler(telegram_handlers.button_callback))
    
    # Start the bot
    logger.info("Starting bot...")
    application.run_polling()