# -*- coding: utf-8 -*-
"""
Advanced 24/7 Crypto AI Bot - Final Resilient Version
Handles network errors and geographical blocks gracefully.
"""

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import os
import sys
import time
import asyncio
import logging
import json
import datetime
import re
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import aiohttp
from aiohttp import web
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    Application, CommandHandler, ContextTypes, CallbackQueryHandler,
    ConversationHandler, MessageHandler, filters
)
from telegram.constants import ChatAction, ParseMode
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    Text, ForeignKey
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import redis
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tweepy

# ==============================================================================
# 2. CONFIGURATION (SETTINGS)
# ==============================================================================
@dataclass
class Settings:
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    REDIS_HOST: str = os.getenv('REDIS_HOST')
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', 6379))
    REDIS_PASSWORD: str = os.getenv('REDIS_PASSWORD')
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY")
    TWITTER_BEARER_TOKEN: str = os.getenv("TWITTER_BEARER_TOKEN")
    EXCHANGES: List[str] = field(default_factory=lambda: os.getenv("EXCHANGES", "binance").split(","))
    TIMEFRAMES: List[str] = field(default_factory=lambda: os.getenv("TIMEFRAMES", "1h,4h,1d").split(","))
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "20"))
    WEB_PORT: int = int(os.getenv("WEB_PORT", "8080"))

S = Settings()

# ==============================================================================
# 3. INITIALIZATION (LOGGER, REDIS, DATABASE)
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('crypto_bot.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

redis_available = False
redis_client = None
if S.REDIS_HOST and S.REDIS_PASSWORD:
    try:
        redis_client = redis.Redis(
            host=S.REDIS_HOST, port=S.REDIS_PORT, db=0, password=S.REDIS_PASSWORD,
            decode_responses=True, socket_connect_timeout=5
        )
        redis_client.ping()
        redis_available = True
        logger.info("Redis connected successfully.")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
else:
    logger.warning("Redis credentials not found. Caching will be disabled.")

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, unique=True, nullable=False)
    username = Column(String(50))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    signals = relationship("Signal", back_populates="user")

class Signal(Base):
    __tablename__ = 'signals'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    symbol = Column(String(20))
    signal_type = Column(String(10))
    confidence = Column(Float)
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    source = Column(String(50))
    user = relationship("User", back_populates="signals")

if not S.DATABASE_URL:
    logger.critical("FATAL: DATABASE_URL is not set!")
    sys.exit(1)

try:
    engine = create_engine(S.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    logger.info("Database connection successful and tables created/verified.")
except Exception as e:
    logger.critical(f"Database connection failed: {e}", exc_info=True)
    sys.exit(1)

# ==============================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================
def get_db_session():
    return Session()

def get_or_create_user(session, chat_id, username=None):
    user = session.query(User).filter_by(chat_id=chat_id).first()
    if not user:
        user = User(chat_id=chat_id, username=username)
        session.add(user)
        session.commit()
    elif username and user.username != username:
        user.username = username
        session.commit()
    return user

# ==============================================================================
# 5. CORE ANALYSIS MODULES
# ==============================================================================
class AdvancedTechnicalAnalyzer:
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line

    def full_analysis(self, df: pd.DataFrame) -> dict:
        if df.empty or len(df) < 200:
             return {'trend': 'unknown', 'momentum': 'unknown', 'summary': 'Insufficient data'}
        close_prices = df['close']
        sma_50 = close_prices.rolling(window=50).mean().iloc[-1]
        sma_200 = close_prices.rolling(window=200).mean().iloc[-1]
        trend = 'bullish' if sma_50 > sma_200 else 'bearish' if sma_50 < sma_200 else 'ranging'
        rsi = self.calculate_rsi(close_prices).iloc[-1]
        macd, signal = self.calculate_macd(close_prices)
        macd_val, signal_val = macd.iloc[-1], signal.iloc[-1]
        momentum_score = 0
        if rsi > 70: momentum_score -= 1
        elif rsi < 30: momentum_score += 1
        if macd_val > signal_val: momentum_score += 1
        else: momentum_score -=1
        momentum = 'strong_bullish' if momentum_score > 1 else 'bullish' if momentum_score > 0 else 'strong_bearish' if momentum_score < -1 else 'bearish'
        return {'trend': trend, 'momentum': momentum, 'summary': f"Trend is {trend}, Momentum is {momentum}"}


class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.twitter_client = None
        if S.TWITTER_BEARER_TOKEN:
            try:
                self.twitter_client = tweepy.Client(bearer_token=S.TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)
                logger.info("Twitter client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Twitter client: {e}")
        else:
            logger.warning("TWITTER_BEARER_TOKEN not found. Twitter analysis will be skipped.")

    async def analyze(self, symbol: str) -> dict:
        scores, weights = [], []
        if self.twitter_client:
            twitter_score = await self.analyze_twitter(symbol)
            if twitter_score is not None:
                scores.append(twitter_score); weights.append(0.6)
        if S.NEWS_API_KEY:
            news_score = await self.analyze_news(symbol)
            if news_score is not None:
                scores.append(news_score); weights.append(0.4)
        if not scores:
            return {'sentiment': 'neutral', 'summary': 'No data sources available'}
        final_score = np.average(scores, weights=weights if sum(weights)>0 else None)
        sentiment = 'bullish' if final_score > 0.15 else 'bearish' if final_score < -0.15 else 'neutral'
        return {'sentiment': sentiment, 'summary': f"Sentiment is {sentiment} (score: {final_score:.2f})"}

    async def analyze_twitter(self, symbol: str) -> Optional[float]:
        try:
            query = f"#{symbol} OR ${symbol} -is:retweet lang:en"
            response = await asyncio.to_thread(self.twitter_client.search_recent_tweets, query=query, max_results=50)
            if not response.data: return 0.0
            scores = [self.vader.polarity_scores(tweet.text)['compound'] for tweet in response.data]
            return np.mean(scores)
        except Exception as e:
            logger.error(f"Twitter analysis failed for {symbol}: {e}")
            return None

    async def analyze_news(self, symbol: str) -> Optional[float]:
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={S.NEWS_API_KEY}&language=en&pageSize=20"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=S.REQUEST_TIMEOUT) as response:
                    response.raise_for_status() # Will raise an error for 4xx/5xx responses
                    data = await response.json()
                    articles = data.get('articles', [])
                    if not articles: return 0.0
                    scores = [self.vader.polarity_scores(a.get('title') or '')['compound'] for a in articles]
                    return np.mean(scores)
        # #### THIS IS THE FIX! ####
        except aiohttp.ClientResponseError as e:
            if e.status == 403:
                logger.warning(f"NewsAPI request is forbidden (403), likely due to geo-blocking. Skipping news analysis.")
            else:
                logger.error(f"NewsAPI request failed with status {e.status}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during NewsAPI request: {e}")
            return None


class EnhancedPredictionEngine:
    # ... (کد این کلاس بدون تغییر باقی می‌ماند)
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_path = "./models"
        os.makedirs(self.model_path, exist_ok=True)
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df['returns'] = df['close'].pct_change()
        df['rsi'] = AdvancedTechnicalAnalyzer().calculate_rsi(df['close'])
        macd, macd_signal = AdvancedTechnicalAnalyzer().calculate_macd(df['close'])
        df['macd_diff'] = macd - macd_signal
        df.dropna(inplace=True)
        features = ['returns', 'rsi', 'macd_diff']
        X = df[features]
        y = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        return X, y
    async def train_model(self, symbol: str, df: pd.DataFrame):
        try:
            X, y = self._prepare_features(df)
            if len(X) < 50: return
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            model.fit(X_train_scaled, y_train)
            dump(model, os.path.join(self.model_path, f"{symbol}_model.joblib"))
            dump(scaler, os.path.join(self.model_path, f"{symbol}_scaler.joblib"))
            self.models[symbol] = model; self.scalers[symbol] = scaler
            logger.info(f"Model for {symbol} trained and saved.")
        except Exception as e:
            logger.error(f"Model training failed for {symbol}: {e}")
    async def predict(self, symbol: str, df: pd.DataFrame) -> dict:
        if symbol not in self.models:
            try:
                self.models[symbol] = load(os.path.join(self.model_path, f"{symbol}_model.joblib"))
                self.scalers[symbol] = load(os.path.join(self.model_path, f"{symbol}_scaler.joblib"))
            except FileNotFoundError:
                await self.train_model(symbol, df.copy())
        model = self.models.get(symbol); scaler = self.scalers.get(symbol)
        if not model or not scaler: return {'prediction': 'hold', 'confidence': 0.5}
        X, _ = self._prepare_features(df.copy())
        if X.empty: return {'prediction': 'hold', 'confidence': 0.5}
        last_features = scaler.transform(X.iloc[-1:].values)
        probability = model.predict_proba(last_features)[0][1]
        prediction = 'buy' if probability > 0.55 else 'sell' if probability < 0.45 else 'hold'
        confidence = max(probability, 1 - probability)
        return {'prediction': prediction, 'confidence': confidence}


class SignalGenerator:
    # ... (کد این کلاس بدون تغییر باقی می‌ماند)
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {'ta': 0.4, 'sentiment': 0.3, 'prediction': 0.3}
    def generate(self, analysis: dict) -> dict:
        scores = {}
        ta_summary = analysis['technical_analysis']
        if 'bullish' in ta_summary.get('trend', ''): scores['ta'] = 1.0
        elif 'bearish' in ta_summary.get('trend', ''): scores['ta'] = -1.0
        else: scores['ta'] = 0.0
        if 'bullish' in ta_summary.get('momentum', ''): scores['ta'] += 0.5
        elif 'bearish' in ta_summary.get('momentum', ''): scores['ta'] -= 0.5
        scores['ta'] = np.clip(scores.get('ta', 0), -1, 1)
        sentiment_summary = analysis['sentiment_analysis']
        if sentiment_summary.get('sentiment') == 'bullish': scores['sentiment'] = 1.0
        elif sentiment_summary.get('sentiment') == 'bearish': scores['sentiment'] = -1.0
        else: scores['sentiment'] = 0.0
        prediction_summary = analysis['prediction']
        if prediction_summary.get('prediction') == 'buy': scores['prediction'] = 1.0
        elif prediction_summary.get('prediction') == 'sell': scores['prediction'] = -1.0
        else: scores['prediction'] = 0.0
        final_score = (scores.get('ta', 0) * self.weights['ta'] +
                       scores.get('sentiment', 0) * self.weights['sentiment'] +
                       scores.get('prediction', 0) * self.weights['prediction'])
        signal = 'BUY' if final_score > 0.35 else 'SELL' if final_score < -0.35 else 'HOLD'
        confidence = abs(final_score)
        return {'signal': signal, 'confidence': confidence}

class AdvancedCryptoBot:
    # ... (کد این کلاس بدون تغییر باقی می‌ماند)
    def __init__(self):
        exchange_id = S.EXCHANGES[0] if S.EXCHANGES else 'binance'
        self.exchange = getattr(ccxt, exchange_id)()
        self.ta_analyzer = AdvancedTechnicalAnalyzer()
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        self.prediction_engine = EnhancedPredictionEngine()
        self.signal_generator = SignalGenerator()
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 300) -> Optional[pd.DataFrame]:
        try:
            pair = f"{symbol.upper()}/USDT"
            ohlcv = await self.exchange.fetch_ohlcv(pair, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return None
    async def close_exchange(self):
        if self.exchange: await self.exchange.close()
    async def analyze_symbol(self, symbol: str, user_chat_id: int) -> Optional[dict]:
        df = await self.fetch_ohlcv(symbol, '1h', 500)
        if df is None or df.empty: return None
        tasks = [
            asyncio.to_thread(self.ta_analyzer.full_analysis, df.copy()),
            self.sentiment_analyzer.analyze(symbol),
            self.prediction_engine.predict(symbol, df.copy())
        ]
        results = await asyncio.gather(*tasks)
        analysis = {
            'symbol': symbol.upper(), 'current_price': df['close'].iloc[-1],
            'technical_analysis': results[0], 'sentiment_analysis': results[1],
            'prediction': results[2],
        }
        analysis['signal'] = self.signal_generator.generate(analysis)
        session = get_db_session()
        try:
            user = get_or_create_user(session, user_chat_id)
            db_signal = Signal(
                user_id=user.id, symbol=analysis['symbol'],
                signal_type=analysis['signal']['signal'],
                confidence=analysis['signal']['confidence'],
                price=analysis['current_price'], source='bot_analysis'
            )
            session.add(db_signal); session.commit()
        finally:
            session.close()
        return analysis
    async def generate_chart(self, symbol: str, df: pd.DataFrame, analysis: dict) -> InputFile:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='rgba(80,120,220,0.6)'), row=2, col=1)
        signal = analysis['signal']['signal']
        color = 'green' if signal == 'BUY' else 'red' if signal == 'SELL' else 'orange'
        fig.update_layout(
            title=f'<b>${symbol.upper()}/USDT Analysis | Signal: <span style="color:{color};">{signal}</span></b>',
            yaxis_title='Price (USDT)', xaxis_rangeslider_visible=False,
            template='plotly_dark', height=600, width=900,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        img_bytes = fig.to_image(format="png"); return InputFile(io.BytesIO(img_bytes), filename=f"{symbol}_analysis.png")

# ==============================================================================
# 8. TELEGRAM BOT HANDLERS & MAIN EXECUTION
# ==============================================================================
(START_ROUTES, GETTING_SYMBOL) = range(2)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    session = get_db_session()
    try:
        get_or_create_user(session, user.id, user.username)
    finally:
        session.close()
    keyboard = [[InlineKeyboardButton("🔍 تحلیل یک نماد", callback_data='analyze')]]
    await update.message.reply_html(rf"سلام {user.mention_html()}! به ربات تحلیلگر کریپتو خوش آمدید.", reply_markup=InlineKeyboardMarkup(keyboard))
    return START_ROUTES

async def ask_for_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(text="لطفاً نماد ارز را ارسال کنید (مثال: BTC):")
    return GETTING_SYMBOL

async def analyze_and_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    symbol = update.message.text.strip().upper()
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    msg = await update.message.reply_text(f"در حال تحلیل {symbol}، لطفا کمی صبر کنید...")
    bot_instance: AdvancedCryptoBot = context.application.bot_instance
    analysis = await bot_instance.analyze_symbol(symbol, chat_id)
    if not analysis:
        await msg.edit_text(f"تحلیل {symbol} ناموفق بود. از صحت نام نماد مطمئن شوید.")
        return START_ROUTES
    signal = analysis['signal']; ta = analysis['technical_analysis']; senti = analysis['sentiment_analysis']; pred = analysis['prediction']
    signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal['signal'], "")
    text = (f"<b>📊 تحلیل {analysis['symbol']}/USDT</b>\n<i>قیمت: ${analysis['current_price']:.2f}</i>\n\n"
            f"<b>{signal_emoji} سیگنال: {signal['signal']}</b> (اطمینان: {signal['confidence']:.1%})\n\n<b> جزئیات: </b>\n"
            f"  - 📉 تکنیکال: {ta['summary']}\n  - 🗣️ احساسات: {senti['summary']}\n"
            f"  - 🤖 پیش‌بینی: {pred['prediction']} (اطمینان: {pred['confidence']:.1%})\n")
    df = await bot_instance.fetch_ohlcv(symbol, '1h')
    if df is not None:
        chart = await bot_instance.generate_chart(symbol, df, analysis)
        await context.bot.send_photo(chat_id=chat_id, photo=chart, caption=text, parse_mode=ParseMode.HTML,
                                     reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("تحلیل نماد دیگر", callback_data='analyze')]]))
        await msg.delete()
    else:
        await msg.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("تحلیل نماد دیگر", callback_data='analyze')]]))
    return START_ROUTES

async def main():
    if not S.TELEGRAM_BOT_TOKEN:
        logger.critical("FATAL: TELEGRAM_BOT_TOKEN is not set!"); sys.exit(1)
    bot_instance = AdvancedCryptoBot()
    app = Application.builder().token(S.TELEGRAM_BOT_TOKEN).build()
    app.bot_instance = bot_instance
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start_cmd)],
        states={
            START_ROUTES: [CallbackQueryHandler(ask_for_symbol, pattern='^analyze$')],
            GETTING_SYMBOL: [MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_and_reply)],
        },
        fallbacks=[CommandHandler('start', start_cmd)], allow_reentry=True
    )
    app.add_handler(conv_handler)
    try:
        logger.info("Bot is starting...")
        await app.initialize(); await app.start(); await app.updater.start_polling()
        await asyncio.Event().wait()
    finally:
        await bot_instance.close_exchange(); await app.updater.stop(); await app.stop()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped manually.")
    except Exception as e:
        logger.critical(f"Bot failed to start: {e}", exc_info=True)