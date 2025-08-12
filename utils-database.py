import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = 'trading_bot.db'):
        """مقداردهی اولیه پایگاه داده"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
        logger.info("Database initialized")
    
    def create_tables(self):
        """ایجاد جداول پایگاه داده"""
        cursor = self.conn.cursor()
        
        # جدول کاربران
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            language TEXT DEFAULT 'fa',
            preferences TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # جدول تحلیل‌ها
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symbol TEXT,
            analysis_type TEXT,
            result TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # جدول سیگنال‌ها
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symbol TEXT,
            signal_type TEXT,
            signal_value TEXT,
            confidence REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        self.conn.commit()
        logger.info("Database tables created")
    
    def save_analysis(self, user_id: int, symbol: str, analysis_type: str, result: Dict[str, Any]):
        """ذخیره تحلیل در پایگاه داده"""
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO analyses (user_id, symbol, analysis_type, result)
        VALUES (?, ?, ?, ?)
        ''', (user_id, symbol, analysis_type, json.dumps(result)))
        self.conn.commit()
    
    def save_signal(self, user_id: int, symbol: str, signal_type: str, signal_value: str, confidence: float):
        """ذخیره سیگنال در پایگاه داده"""
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO signals (user_id, symbol, signal_type, signal_value, confidence)
        VALUES (?, ?, ?, ?, ?)
        ''', (user_id, symbol, signal_type, signal_value, confidence))
        self.conn.commit()
    
    def get_user_analyses(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """دریافت تحلیل‌های کاربر"""
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT symbol, analysis_type, result, timestamp 
        FROM analyses 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
        ''', (user_id, limit))
        
        analyses = []
        for row in cursor.fetchall():
            analyses.append({
                'symbol': row[0],
                'analysis_type': row[1],
                'result': json.loads(row[2]),
                'timestamp': row[3]
            })
        
        return analyses