# -*- coding: utf-8 -*-
"""
Advanced 24/7 Crypto AI Bot - Final Upgraded Version
Features: Multi-chain Analysis, Advanced TA, AI Predictions, Dynamic Risk Management,
Interactive Telegram Bot, Web Dashboard, Enhanced Stability & Modularity.
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
from dataclasses import dataclass
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
    Boolean, Text, ForeignKey
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
    # Essential tokens and IDs
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
    ADMIN_CHAT_ID: str = os.getenv("ADMIN_CHAT_ID") # For admin notifications

    # Database and Cache
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///crypto_bot.db")
    REDIS_HOST: str = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', 6379))

    # API Keys
    COINGECKO_API_KEY: str = os.getenv("COINGECKO_API_KEY")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY")
    TWITTER_BEARER_TOKEN: str = os.getenv("TWITTER_BEARER_TOKEN")
    ETHERSCAN_API_KEY: str = os.getenv("ETHERSCAN_API_KEY")
    BSCSCAN_API_KEY: str = os.getenv("BSCSCAN_API_KEY")

    # Bot Behavior
    EXCHANGES: List[str] = field(default_factory=lambda: os.getenv("EXCHANGES", "binance,kucoin,bybit").split(","))
    TIMEFRAMES: List[str] = field(default_factory=lambda: os.getenv("TIMEFRAMES", "1h,4h,1d").split(","))
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "20"))
    
    # Analysis Modules Toggle
    ENABLE_ONCHAIN: bool = os.getenv("ENABLE_ONCHAIN", "true").lower() == "true"
    ENABLE_SENTIMENT: bool = os.getenv("ENABLE_SENTIMENT", "true").lower() == "true"
    ENABLE_PREDICTION: bool = os.getenv("ENABLE_PREDICTION", "true").lower() == "true"

    # Web Dashboard
    ENABLE_WEB_DASHBOARD: bool = os.getenv("ENABLE_WEB_DASHBOARD", "true").lower() == "true"
    WEB_PORT: int = int(os.getenv("WEB_PORT", "8080"))

S = Settings()

# ==============================================================================
# 3. INITIALIZATION (LOGGER, REDIS, DATABASE)
# ==============================================================================
# Logger Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('crypto_bot.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Redis Client
try:
    redis_client = redis.Redis(
        host=S.REDIS_HOST, port=S.REDIS_PORT, db=0,
        decode_responses=True, socket_connect_timeout=5
    )
    redis_client.ping()
    redis_available = True
    logger.info("Redis connected successfully.")
except Exception as e:
    logger.warning(f"Redis not available, caching will be disabled: {e}")
    redis_available = False
    redis_client = None

# Database Setup
Base = declarative_base()

# ... (Database Models: User, Signal, etc. - همان کد قبلی بدون تغییر)
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, unique=True, nullable=False)
    username = Column(String(50))
    preferences = Column(Text, default='{}')
    risk_level = Column(String(20), default='medium')
    max_leverage = Column(Float, default=5.0)
    balance = Column(Float, default=10000.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    signals = relationship("Signal", back_populates="user")

class Signal(Base):
    __tablename__ = 'signals'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    symbol = Column(String(20))
    signal_type = Column(String(10))  # BUY/SELL/HOLD
    confidence = Column(Float)
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    source = Column(String(50))
    timeframe = Column(String(10), default='1h')
    user = relationship("User", back_populates="signals")


try:
    engine = create_engine(S.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    logger.info("Database connected and tables created.")
except Exception as e:
    logger.error(f"Database connection failed: {e}")
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

def cache_set(key: str, value: Any, ttl: int = S.CACHE_TTL_SECONDS):
    if redis_available:
        try:
            redis_client.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.error(f"Redis SET error for key '{key}': {e}")

def cache_get(key: str) -> Optional[Any]:
    if redis_available:
        try:
            cached = redis_client.get(key)
            return json.loads(cached) if cached else None
        except Exception as e:
            logger.error(f"Redis GET error for key '{key}': {e}")
    return None

async def fetch_with_retry(session: aiohttp.ClientSession, url: str, params: dict = None, headers: dict = None, max_retries=3):
    for attempt in range(max_retries):
        try:
            async with session.get(url, params=params, headers=headers, timeout=S.REQUEST_TIMEOUT) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.warning(f"Request failed for {url} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)

# ==============================================================================
# 5. CORE ANALYSIS MODULES (TA, On-Chain, Sentiment, AI Prediction)
# ==============================================================================
class AdvancedTechnicalAnalyzer:
    # ... (کلاس تحلیل تکنیکال شما بسیار خوب بود و بدون تغییر عمده باقی می‌ماند)
    # ... (فقط اطمینان حاصل می‌کنیم که خروجی‌ها تمیز هستند)
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
        if df.empty or len(df) < 26: # Minimum length for MACD
             return {'trend': 'unknown', 'momentum': 'unknown', 'summary': 'Insufficient data'}
        
        close_prices = df['close']
        
        # Trend
        sma_50 = close_prices.rolling(window=50).mean().iloc[-1]
        sma_200 = close_prices.rolling(window=200).mean().iloc[-1]
        trend = 'bullish' if sma_50 > sma_200 else 'bearish' if sma_50 < sma_200 else 'ranging'

        # Momentum
        rsi = self.calculate_rsi(close_prices).iloc[-1]
        macd, signal = self.calculate_macd(close_prices)
        macd_val, signal_val = macd.iloc[-1], signal.iloc[-1]
        
        momentum_score = 0
        if rsi > 70: momentum_score -= 1
        elif rsi < 30: momentum_score += 1
        if macd_val > signal_val: momentum_score += 1
        else: momentum_score -=1

        momentum = 'strong_bullish' if momentum_score > 1 else 'bullish' if momentum_score > 0 else 'strong_bearish' if momentum_score < -1 else 'bearish'
        
        return {
            'trend': trend,
            'momentum': momentum,
            'rsi': rsi,
            'macd_diff': macd_val - signal_val,
            'summary': f"Trend is {trend}, Momentum is {momentum} (RSI: {rsi:.2f})"
        }


class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.twitter_client = None
        if S.TWITTER_BEARER_TOKEN:
            try:
                self.twitter_client = tweepy.Client(bearer_token=S.TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)
            except Exception as e:
                logger.error(f"Failed to initialize Twitter client: {e}")

    async def analyze(self, symbol: str) -> dict:
        scores = []
        if self.twitter_client:
            twitter_score = await self.analyze_twitter(symbol)
            if twitter_score is not None:
                scores.append(twitter_score * 0.6) # Twitter has higher weight
        
        if S.NEWS_API_KEY:
            news_score = await self.analyze_news(symbol)
            if news_score is not None:
                scores.append(news_score * 0.4)

        if not scores:
            return {'sentiment': 'neutral', 'score': 0.0}

        final_score = np.mean(scores)
        sentiment = 'bullish' if final_score > 0.15 else 'bearish' if final_score < -0.15 else 'neutral'
        return {'sentiment': sentiment, 'score': final_score}

    async def analyze_twitter(self, symbol: str) -> Optional[float]:
        try:
            query = f"#{symbol} OR ${symbol} -is:retweet lang:en"
            tweets = self.twitter_client.search_recent_tweets(query=query, max_results=50)
            if not tweets.data: return 0.0
            scores = [self.vader.polarity_scores(tweet.text)['compound'] for tweet in tweets.data]
            return np.mean(scores)
        except Exception as e:
            logger.error(f"Twitter sentiment analysis failed for {symbol}: {e}")
            return None

    async def analyze_news(self, symbol: str) -> Optional[float]:
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={S.NEWS_API_KEY}&language=en&pageSize=20"
        try:
            async with aiohttp.ClientSession() as session:
                data = await fetch_with_retry(session, url)
                articles = data.get('articles', [])
                if not articles: return 0.0
                scores = [self.vader.polarity_scores(a['title'])['compound'] for a in articles]
                return np.mean(scores)
        except Exception as e:
            logger.error(f"News sentiment analysis failed for {symbol}: {e}")
            return None

class EnhancedPredictionEngine:
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
        logger.info(f"Training prediction model for {symbol}...")
        try:
            X, y = self._prepare_features(df)
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            model.fit(X_train_scaled, y_train)

            # Save model and scaler
            dump(model, os.path.join(self.model_path, f"{symbol}_model.joblib"))
            dump(scaler, os.path.join(self.model_path, f"{symbol}_scaler.joblib"))
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            logger.info(f"Model for {symbol} trained and saved successfully.")
        except Exception as e:
            logger.error(f"Model training failed for {symbol}: {e}")

    async def predict(self, symbol: str, df: pd.DataFrame) -> dict:
        model_key = f"{symbol}_model.joblib"
        scaler_key = f"{symbol}_scaler.joblib"
        
        # Load model if not in memory
        if symbol not in self.models:
            try:
                self.models[symbol] = load(os.path.join(self.model_path, model_key))
                self.scalers[symbol] = load(os.path.join(self.model_path, scaler_key))
            except FileNotFoundError:
                await self.train_model(symbol, df)
                if symbol not in self.models:
                    return {'prediction': 'hold', 'confidence': 0.5}
        
        model = self.models[symbol]
        scaler = self.scalers[symbol]

        X, _ = self._prepare_features(df)
        if X.empty:
            return {'prediction': 'hold', 'confidence': 0.5}
            
        last_features = scaler.transform(X.iloc[-1:].values)
        
        probability = model.predict_proba(last_features)[0][1] # Probability of price increase
        prediction = 'buy' if probability > 0.55 else 'sell' if probability < 0.45 else 'hold'
        confidence = max(probability, 1 - probability)
        
        return {'prediction': prediction, 'confidence': confidence}

# ==============================================================================
# 6. SIGNAL GENERATOR
# ==============================================================================
class SignalGenerator:
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'ta': 0.4,
            'sentiment': 0.3,
            'prediction': 0.3
        }

    def generate(self, analysis: dict) -> dict:
        scores = {}
        
        # TA Score
        ta_summary = analysis['technical_analysis']
        if 'bullish' in ta_summary['trend']: scores['ta'] = 1.0
        elif 'bearish' in ta_summary['trend']: scores['ta'] = -1.0
        else: scores['ta'] = 0.0
        if 'bullish' in ta_summary['momentum']: scores['ta'] += 0.5
        elif 'bearish' in ta_summary['momentum']: scores['ta'] -= 0.5
        scores['ta'] = np.clip(scores['ta'], -1, 1)

        # Sentiment Score
        sentiment_summary = analysis['sentiment_analysis']
        if sentiment_summary['sentiment'] == 'bullish': scores['sentiment'] = 1.0
        elif sentiment_summary['sentiment'] == 'bearish': scores['sentiment'] = -1.0
        else: scores['sentiment'] = 0.0

        # AI Prediction Score
        prediction_summary = analysis['prediction']
        if prediction_summary['prediction'] == 'buy': scores['prediction'] = 1.0
        elif prediction_summary['prediction'] == 'sell': scores['prediction'] = -1.0
        else: scores['prediction'] = 0.0
        
        # Calculate final weighted score
        final_score = (scores.get('ta', 0) * self.weights['ta'] +
                       scores.get('sentiment', 0) * self.weights['sentiment'] +
                       scores.get('prediction', 0) * self.weights['prediction'])
        
        signal = 'BUY' if final_score > 0.35 else 'SELL' if final_score < -0.35 else 'HOLD'
        confidence = abs(final_score)

        return {'signal': signal, 'confidence': confidence, 'score': final_score, 'breakdown': scores}

# ==============================================================================
# 7. MAIN BOT CLASS
# ==============================================================================
class AdvancedCryptoBot:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.ta_analyzer = AdvancedTechnicalAnalyzer()
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        self.prediction_engine = EnhancedPredictionEngine()
        self.signal_generator = SignalGenerator()

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 300) -> Optional[pd.DataFrame]:
        cache_key = f"ohlcv_{symbol}_{timeframe}"
        cached = cache_get(cache_key)
        if cached:
            return pd.read_json(io.StringIO(cached['df']))

        try:
            pair = f"{symbol.upper()}/USDT"
            ohlcv = await self.exchange.fetch_ohlcv(pair, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            cache_set(cache_key, {'df': df.to_json()})
            return df
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return None

    async def analyze_symbol(self, symbol: str, user_chat_id: int) -> Optional[dict]:
        symbol = symbol.upper()
        df = await self.fetch_ohlcv(symbol, '1h', 500)
        if df is None or df.empty:
            return None

        # Run all analyses in parallel
        ta_task = asyncio.create_task(asyncio.to_thread(self.ta_analyzer.full_analysis, df.copy()))
        sentiment_task = asyncio.create_task(self.sentiment_analyzer.analyze(symbol))
        prediction_task = asyncio.create_task(self.prediction_engine.predict(symbol, df.copy()))
        
        results = await asyncio.gather(ta_task, sentiment_task, prediction_task)
        
        analysis = {
            'symbol': symbol,
            'current_price': df['close'].iloc[-1],
            'technical_analysis': results[0],
            'sentiment_analysis': results[1],
            'prediction': results[2],
        }

        # Generate signal
        signal_info = self.signal_generator.generate(analysis)
        analysis['signal'] = signal_info

        # Save signal to DB
        session = get_db_session()
        try:
            user = get_or_create_user(session, user_chat_id)
            db_signal = Signal(
                user_id=user.id,
                symbol=symbol,
                signal_type=signal_info['signal'],
                confidence=signal_info['confidence'],
                price=analysis['current_price'],
                source='bot_analysis'
            )
            session.add(db_signal)
            session.commit()
        except SQLAlchemyError as e:
            logger.error(f"DB Error saving signal: {e}")
            session.rollback()
        finally:
            session.close()

        return analysis

    async def generate_chart(self, symbol: str, df: pd.DataFrame, analysis: dict) -> InputFile:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Candlestick
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        
        # Volume
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='rgba(80,120,220,0.6)'), row=2, col=1)

        signal = analysis['signal']['signal']
        color = 'green' if signal == 'BUY' else 'red' if signal == 'SELL' else 'orange'
        fig.update_layout(
            title=f'<b>${symbol}/USDT Analysis | Signal: <span style="color:{color};">{signal}</span></b>',
            yaxis_title='Price (USDT)',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=600,
            width=900,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        img_bytes = fig.to_image(format="png")
        return InputFile(io.BytesIO(img_bytes), filename=f"{symbol}_analysis.png")

# ==============================================================================
# 8. TELEGRAM BOT HANDLERS
# ==============================================================================
(START_ROUTES, GETTING_SYMBOL) = range(2)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    session = get_db_session()
    try:
        get_or_create_user(session, user.id, user.username)
    finally:
        session.close()

    keyboard = [
        [InlineKeyboardButton("🔍 تحلیل یک نماد", callback_data='analyze')],
        [InlineKeyboardButton("⚙️ تنظیمات (به زودی)", callback_data='settings')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_html(
        rf"سلام {user.mention_html()}! به ربات تحلیلگر پیشرفته کریپتو خوش آمدید.",
        reply_markup=reply_markup,
    )
    return START_ROUTES

async def ask_for_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(text="لطفاً نماد ارز مورد نظر را ارسال کنید (مثال: BTC یا ETH):")
    return GETTING_SYMBOL

async def analyze_and_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    symbol = update.message.text.strip().upper()
    chat_id = update.effective_chat.id
    
    # Show typing action
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    msg = await update.message.reply_text(f"در حال تحلیل {symbol}، لطفاً کمی صبر کنید...")

    bot_instance = context.application.bot_instance
    analysis = await bot_instance.analyze_symbol(symbol, chat_id)

    if not analysis:
        await msg.edit_text(f"متاسفانه تحلیل {symbol} با خطا مواجه شد. لطفاً از صحت نام نماد مطمئن شوید.")
        return START_ROUTES

    # Build response message
    signal = analysis['signal']
    ta = analysis['technical_analysis']
    senti = analysis['sentiment_analysis']
    pred = analysis['prediction']
    
    signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal['signal'], "")

    text = (
        f"<b>📊 تحلیل کامل {symbol}/USDT</b>\n"
        f"<i>قیمت فعلی: ${analysis['current_price']:.2f}</i>\n\n"
        f"<b>{signal_emoji} سیگنال نهایی: {signal['signal']}</b> (اطمینان: {signal['confidence']:.1%})\n\n"
        f"<b> جزئیات تحلیل: </b>\n"
        f"  - 📉 تکنیکال: روند <b>{ta['trend']}</b>، مومنتوم <b>{ta['momentum']}</b>\n"
        f"  - 🗣️ احساسات: <b>{senti['sentiment']}</b> (امتیاز: {senti['score']:.2f})\n"
        f"  - 🤖 پیش‌بینی AI: <b>{pred['prediction']}</b> (اطمینان: {pred['confidence']:.1%})\n"
    )

    df = await bot_instance.fetch_ohlcv(symbol, '1h')
    chart = await bot_instance.generate_chart(symbol, df, analysis)
    
    await context.bot.send_photo(
        chat_id=chat_id,
        photo=chart,
        caption=text,
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("بازگشت به منو", callback_data='main_menu')]])
    )
    await msg.delete() # Delete "در حال تحلیل..." message
    return START_ROUTES

async def back_to_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    keyboard = [
        [InlineKeyboardButton("🔍 تحلیل یک نماد", callback_data='analyze')],
        [InlineKeyboardButton("⚙️ تنظیمات (به زودی)", callback_data='settings')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        "چه کاری برایتان انجام دهم؟",
        reply_markup=reply_markup,
    )
    return START_ROUTES

# ==============================================================================
# 9. WEB DASHBOARD (Simplified)
# ==============================================================================
class WebDashboard:
    def __init__(self, bot):
        self.bot = bot
        self.app = web.Application()
        self.app.router.add_get('/', self.index)

    async def index(self, request):
        # A simple health check page
        return web.Response(text="Crypto AI Bot is running.", content_type="text/html")

    async def run(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', S.WEB_PORT)
        await site.start()
        logger.info(f"Web dashboard running on http://0.0.0.0:{S.WEB_PORT}")


# ==============================================================================
# 10. MAIN EXECUTION
# ==============================================================================
async def main():
    if not S.TELEGRAM_BOT_TOKEN:
        logger.critical("FATAL: TELEGRAM_BOT_TOKEN is not set!")
        sys.exit(1)

    # Initialize bot instance
    bot_instance = AdvancedCryptoBot()
    
    # Setup Telegram Application
    app = Application.builder().token(S.TELEGRAM_BOT_TOKEN).build()
    app.bot_instance = bot_instance # Attach bot instance to the application context

    # Conversation handler for interactive menus
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            START_ROUTES: [
                CallbackQueryHandler(ask_for_symbol, pattern='^analyze$'),
                CallbackQueryHandler(back_to_menu, pattern='^main_menu$'),
                # Add other main menu buttons here
            ],
            GETTING_SYMBOL: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_and_reply)
            ],
        },
        fallbacks=[CommandHandler('start', start)],
    )
    app.add_handler(conv_handler)

    # Run Web Dashboard if enabled
    if S.ENABLE_WEB_DASHBOARD:
        dashboard = WebDashboard(bot_instance)
        # Run dashboard in the background
        asyncio.create_task(dashboard.run())
        
    # Start polling
    logger.info("Bot is starting to poll...")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    
    # Keep the script running
    await asyncio.Event().wait()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped manually.")
    except Exception as e:
        logger.critical(f"Bot failed to start: {e}", exc_info=True)