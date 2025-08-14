import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)

class DatabaseManager:
    """مدیریت پایگاه داده"""
    
    def __init__(self, db_path: str = 'data/crypto_bot.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """مقداردهی اولیه پایگاه داده"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # ایجاد جدول تحلیل‌ها
                cursor.execute('''
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
                
                # ایجاد جدول فعالیت نهنگ‌ها
                cursor.execute('''
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
                
                # ایجاد جدول اخبار
                cursor.execute('''
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
                
                # ایجاد جدول تحلیل‌های ویکاف
                cursor.execute('''
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
                
                # ایجاد جدول تحلیل‌های فیبوناچی
                cursor.execute('''
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
                
                # ایجاد جدول کاربران
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER UNIQUE,
                    username TEXT,
                    first_seen DATETIME,
                    last_seen DATETIME,
                    preferences TEXT
                )
                ''')
                
                # ایجاد جدول واچ‌لیست
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    symbol TEXT NOT NULL,
                    added_at DATETIME,
                    UNIQUE(user_id, symbol)
                )
                ''')
                
                # ایجاد جدول هشدارها
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    target_price REAL,
                    triggered BOOLEAN DEFAULT FALSE,
                    created_at DATETIME,
                    triggered_at DATETIME
                )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def save_analysis_performance(self, data: Dict):
        """ذخیره عملکرد تحلیل"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO analysis_performance 
                (symbol, method, timestamp, signal, confidence, actual_outcome, success, profit_loss, market_conditions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['symbol'],
                    data['method'],
                    data['timestamp'],
                    data['signal'],
                    data['confidence'],
                    data.get('actual_outcome'),
                    data.get('success'),
                    data.get('profit_loss'),
                    json.dumps(data.get('market_conditions', {}))
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving analysis performance: {e}")
    
    def save_whale_activity(self, data: Dict):
        """ذخیره فعالیت نهنگ‌ها"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO whale_activity 
                (symbol, transaction_hash, timestamp, amount, amount_usd, transaction_type, exchange, wallet_address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['symbol'],
                    data['transaction_hash'],
                    data['timestamp'],
                    data['amount'],
                    data['amount_usd'],
                    data['transaction_type'],
                    data.get('exchange'),
                    data.get('wallet_address')
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving whale activity: {e}")
    
    def save_news_impact(self, data: Dict):
        """ذخیره تأثیر اخبار"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO news_impact 
                (symbol, news_title, news_source, timestamp, sentiment_score, impact_score, price_change_before, price_change_after, volatility_change)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['symbol'],
                    data['news_title'],
                    data['news_source'],
                    data['timestamp'],
                    data['sentiment_score'],
                    data['impact_score'],
                    data.get('price_change_before'),
                    data.get('price_change_after'),
                    data.get('volatility_change')
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving news impact: {e}")
    
    def save_wyckoff_analysis(self, data: Dict):
        """ذخیره تحلیل ویکاف"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO wyckoff_analysis 
                (symbol, timestamp, phase, event, confidence, price_action)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    data['symbol'],
                    data['timestamp'],
                    data['phase'],
                    data['event'],
                    data['confidence'],
                    data['price_action']
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving wyckoff analysis: {e}")
    
    def save_fibonacci_analysis(self, data: Dict):
        """ذخیره تحلیل فیبوناچی"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO fibonacci_analysis 
                (symbol, timestamp, retracement_levels, extension_levels, confluence_zones, accuracy)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    data['symbol'],
                    data['timestamp'],
                    json.dumps(data['retracement_levels']),
                    json.dumps(data['extension_levels']),
                    json.dumps(data['confluence_zones']),
                    data['accuracy']
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving fibonacci analysis: {e}")
    
    def get_user(self, user_id: int) -> Optional[Dict]:
        """دریافت اطلاعات کاربر"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
                result = cursor.fetchone()
                
                if result:
                    return {
                        'id': result[0],
                        'user_id': result[1],
                        'username': result[2],
                        'first_seen': result[3],
                        'last_seen': result[4],
                        'preferences': json.loads(result[5]) if result[5] else {}
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def save_user(self, user_data: Dict):
        """ذخیره اطلاعات کاربر"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO users 
                (user_id, username, first_seen, last_seen, preferences)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    user_data['user_id'],
                    user_data['username'],
                    user_data.get('first_seen', datetime.now()),
                    user_data.get('last_seen', datetime.now()),
                    json.dumps(user_data.get('preferences', {}))
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving user: {e}")
    
    def get_watchlist(self, user_id: int) -> List[str]:
        """دریافت واچ‌لیست کاربر"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT symbol FROM watchlist WHERE user_id = ?', (user_id,))
                results = cursor.fetchall()
                return [result[0] for result in results]
        except Exception as e:
            logger.error(f"Error getting watchlist: {e}")
            return []
    
    def add_to_watchlist(self, user_id: int, symbol: str):
        """افزودن به واچ‌لیست"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR IGNORE INTO watchlist (user_id, symbol, added_at)
                VALUES (?, ?, ?)
                ''', (user_id, symbol, datetime.now()))
                conn.commit()
        except Exception as e:
            logger.error(f"Error adding to watchlist: {e}")
    
    def remove_from_watchlist(self, user_id: int, symbol: str):
        """حذف از واچ‌لیست"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM watchlist WHERE user_id = ? AND symbol = ?', (user_id, symbol))
                conn.commit()
        except Exception as e:
            logger.error(f"Error removing from watchlist: {e}")
    
    def get_alerts(self, user_id: int) -> List[Dict]:
        """دریافت هشدارهای کاربر"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT * FROM alerts WHERE user_id = ? AND triggered = FALSE
                ORDER BY created_at DESC
                ''', (user_id,))
                results = cursor.fetchall()
                
                alerts = []
                for result in results:
                    alerts.append({
                        'id': result[0],
                        'user_id': result[1],
                        'symbol': result[2],
                        'alert_type': result[3],
                        'target_price': result[4],
                        'triggered': result[5],
                        'created_at': result[6],
                        'triggered_at': result[7]
                    })
                return alerts
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    def add_alert(self, alert_data: Dict):
        """افزودن هشدار"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO alerts 
                (user_id, symbol, alert_type, target_price, created_at)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    alert_data['user_id'],
                    alert_data['symbol'],
                    alert_data['alert_type'],
                    alert_data['target_price'],
                    datetime.now()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error adding alert: {e}")
    
    def trigger_alert(self, alert_id: int):
        """فعال کردن هشدار"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                UPDATE alerts SET triggered = TRUE, triggered_at = ? WHERE id = ?
                ''', (datetime.now(), alert_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
    
    def get_method_performance(self, symbol: str, method: str) -> Dict:
        """دریافت عملکرد روش تحلیلی"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT signal, confidence, success, profit_loss 
                FROM analysis_performance 
                WHERE symbol = ? AND method = ?
                ORDER BY timestamp DESC
                LIMIT 100
                ''', (symbol, method))
                
                results = cursor.fetchall()
                
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
        except Exception as e:
            logger.error(f"Error getting method performance: {e}")
            return {
                'success_rate': 0.5,
                'avg_confidence': 0.5,
                'avg_profit_loss': 0,
                'total_trades': 0,
                'winning_trades': 0
            }
    
    def get_news_impact(self, symbol: str) -> List[Dict]:
        """دریافت تأثیر اخبار"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT news_title, news_source, timestamp, sentiment_score, impact_score, 
                       price_change_before, price_change_after, volatility_change
                FROM news_impact 
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 20
                ''', (symbol,))
                
                results = cursor.fetchall()
                
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
        except Exception as e:
            logger.error(f"Error getting news impact: {e}")
            return []