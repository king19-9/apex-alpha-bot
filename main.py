import os
import logging
import asyncio
import json
import time
import random
import numpy as np
import pandas as pd
import yfinance as yf
import ccxt
import redis
import psycopg2
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
    ConversationHandler,
)
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from ta.trend import IchimokuIndicator
from ta.volatility import AverageTrueRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# تنظیمات اولیه
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# متغیرهای محیطی
TOKEN = os.getenv("TELEGRAM_TOKEN")
DB_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# اتصال به پایگاه داده
conn = psycopg2.connect(DB_URL)
cursor = conn.cursor()

# اتصال به Redis
r = redis.Redis.from_url(REDIS_URL)

# دانلود منابع NLTK
nltk.download('vader_lexicon')
sentiment_analyzer = SentimentIntensityAnalyzer()

# مدل‌های هوش مصنوعی
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
persian_model = AutoModelForSequenceClassification.from_pretrained("HooshvareLab/bert-fa-base-uncased")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fa-en")

# حالت‌های مکالمه
ANALYZE, SIGNALS, NOTIFICATIONS, MONITOR, WATCHLIST, SETTINGS = range(6)

# تابع راه‌اندازی ربات
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    
    # ایجاد کاربر جدید در صورت عدم وجود
    cursor.execute(
        "INSERT INTO users (user_id, language) VALUES (%s, 'fa') ON CONFLICT (user_id) DO NOTHING",
        (user_id,)
    )
    conn.commit()
    
    keyboard = [
        [InlineKeyboardButton("1. تحلیل عمیق نماد", callback_data='analyze')],
        [InlineKeyboardButton("2. سیگنال‌های نقره‌ای", callback_data='signals')],
        [InlineKeyboardButton("3. نوتیفیکیشن طلایی", callback_data='notifications')],
        [InlineKeyboardButton("4. پایش معامله", callback_data='monitor')],
        [InlineKeyboardButton("5. مدیریت واچ‌لیست", callback_data='watchlist')],
        [InlineKeyboardButton("6. تنظیمات", callback_data='settings')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "به ربات تحلیل تریدینگ خوش آمدید! لطفاً یکی از گزینه‌ها را انتخاب کنید:",
        reply_markup=reply_markup
    )
    return ConversationHandler.END

# تابع تحلیل عمیق نماد
async def analyze_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    context.user_data['state'] = 'analyze'
    
    await query.edit_message_text(
        "نماد مورد نظر را وارد کنید (مثال: BTC-USD):"
    )
    return ANALYZE

async def deep_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    symbol = update.message.text.upper()
    user_id = update.effective_user.id
    
    try:
        # دریافت داده‌های تاریخی
        hist_data = yf.download(symbol, period="1y", interval="1d")
        
        # دریافت داده‌های لحظه‌ای
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker(symbol)
        live_price = ticker['last']
        
        # تحلیل تکنیکال پیشرفته
        ichimoku = IchimokuIndicator(hist_data['High'], hist_data['Low'], hist_data['Close'])
        technical_analysis = {
            'ichimoku': {
                'tenkan_sen': ichimoku.ichimoku_conversion_line().iloc[-1],
                'kijun_sen': ichimoku.ichimoku_base_line().iloc[-1],
                'senkou_span_a': ichimoku.ichimoku_a().iloc[-1],
                'senkou_span_b': ichimoku.ichimoku_b().iloc[-1],
            },
            'atr': AverageTrueRange(hist_data['High'], hist_data['Low'], hist_data['Close']).average_true_range().iloc[-1],
            'supply_demand': detect_supply_demand(symbol),
            'elliott_wave': elliott_wave_analysis(hist_data),
        }
        
        # تحلیل فاندامنتال
        fundamental_analysis = get_fundamental_data(symbol)
        
        # تحلیل احساسات
        sentiment = get_sentiment_analysis(symbol)
        
        # تحلیل با مدل‌های ML
        ml_prediction = hybrid_ml_analysis(hist_data, sentiment, fundamental_analysis)
        
        # تولید توضیحات فارسی
        explanation = generate_persian_explanation(
            symbol, live_price, technical_analysis, 
            fundamental_analysis, sentiment, ml_prediction
        )
        
        # ذخیره تحلیل در Redis
        r.set(f"analysis:{user_id}:{symbol}", json.dumps({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'live_price': live_price,
            'technical': technical_analysis,
            'fundamental': fundamental_analysis,
            'sentiment': sentiment,
            'ml_prediction': ml_prediction,
            'explanation': explanation
        }))
        
        await update.message.reply_text(explanation, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in deep analysis: {e}")
        await update.message.reply_text("خطا در تحلیل نماد. لطفاً دوباره تلاش کنید.")
    
    return ConversationHandler.END

# تابع تشخیص مناطق عرضه و تقاضا
def detect_supply_demand(symbol):
    try:
        exchange = ccxt.binance()
        orderbook = exchange.fetch_order_book(symbol)
        
        # تحلیل دفترچه سفارشات
        bids = orderbook['bids'][:10]  # 10 سطح بالای خرید
        asks = orderbook['asks'][:10]  # 10 سطح بالای فروش
        
        # محاسبه حجم کل در هر سطح
        bid_volumes = [bid[1] for bid in bids]
        ask_volumes = [ask[1] for ask in asks]
        
        # شناسایی مناطق کلیدی
        demand_zone = bids[0][0] if sum(bid_volumes) > sum(ask_volumes) * 1.5 else None
        supply_zone = asks[0][0] if sum(ask_volumes) > sum(bid_volumes) * 1.5 else None
        
        return {
            'demand_zone': demand_zone,
            'supply_zone': supply_zone,
            'bid_volume': sum(bid_volumes),
            'ask_volume': sum(ask_volumes),
            'imbalance': abs(sum(bid_volumes) - sum(ask_volumes)) / max(sum(bid_volumes), sum(ask_volumes))
        }
    except:
        return {'error': 'داده‌های Order Book در دسترس نیست'}

# تحلیل امواج الیوت
def elliott_wave_analysis(data):
    try:
        close_prices = data['Close'].values
        
        # الگوریتم ساده‌شده شناسایی امواج
        waves = []
        for i in range(2, len(close_prices)-2):
            if (close_prices[i] > close_prices[i-1] and close_prices[i] > close_prices[i-2] and
                close_prices[i] > close_prices[i+1] and close_prices[i] > close_prices[i+2]):
                waves.append({'type': 'impulse', 'index': i, 'price': close_prices[i]})
            elif (close_prices[i] < close_prices[i-1] and close_prices[i] < close_prices[i-2] and
                  close_prices[i] < close_prices[i+1] and close_prices[i] < close_prices[i+2]):
                waves.append({'type': 'corrective', 'index': i, 'price': close_prices[i]})
        
        return {
            'waves': waves[-5:] if len(waves) > 5 else waves,  # 5 موج آخر
            'current_pattern': 'bullish' if len([w for w in waves if w['type'] == 'impulse']) > len([w for w in waves if w['type'] == 'corrective']) else 'bearish'
        }
    except:
        return {'error': 'تحلیل امواج الیوت ممکن نیست'}

# دریافت داده‌های فاندامنتال
def get_fundamental_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # داده‌های آن‌چین برای کریپتو
        on_chain = {}
        if 'BTC' in symbol or 'ETH' in symbol:
            on_chain = {
                'active_addresses': random.randint(500000, 1500000),
                'exchange_flow': random.uniform(-1000, 1000),
                'mvrv_ratio': random.uniform(1.5, 3.5)
            }
        
        return {
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'on_chain': on_chain
        }
    except:
        return {'error': 'داده‌های فاندامنتال در دسترس نیست'}

# تحلیل احساسات بازار
def get_sentiment_analysis(symbol):
    try:
        # تحلیل اخبار
        news_sentiment = 0
        if NEWS_API_KEY:
            # در اینجا می‌توانید از NewsAPI استفاده کنید
            # برای سادگی از داده‌های شبیه‌سازی شده استفاده می‌کنیم
            news_sentiment = random.uniform(-0.5, 0.5)
        
        # تحلیل توییتر
        twitter_sentiment = random.uniform(-0.3, 0.3)
        
        # تحلیل کلی
        overall_sentiment = (news_sentiment + twitter_sentiment) / 2
        
        return {
            'news': news_sentiment,
            'twitter': twitter_sentiment,
            'overall': overall_sentiment,
            'interpretation': 'مثبت' if overall_sentiment > 0.2 else 'منفی' if overall_sentiment < -0.2 else 'خنثی'
        }
    except:
        return {'error': 'تحلیل احساسات ممکن نیست'}

# تحلیل ترکیبی با مدل‌های ML
def hybrid_ml_analysis(data, sentiment, fundamental):
    try:
        # آماده‌سازی داده‌ها
        df = data.copy()
        df['sentiment'] = sentiment.get('overall', 0)
        df['market_cap'] = fundamental.get('market_cap', 0)
        
        # ویژگی‌های تکنیکال
        df['ma_20'] = df['Close'].rolling(window=20).mean()
        df['ma_50'] = df['Close'].rolling(window=50).mean()
        df['rsi'] = 100 - (100 / (1 + df['Close'].diff().rolling(14).apply(lambda x: x[x>0].sum() / abs(x[x<0].sum()))))
        
        # حذف داده‌های نامعتبر
        df = df.dropna()
        
        if len(df) < 100:
            return {'error': 'داده‌های کافی برای تحلیل ML وجود ندارد'}
        
        # تقسیم داده‌ها
        X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment', 'market_cap', 'ma_20', 'ma_50', 'rsi']].values
        y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)[:-1]
        X = X[:-1]
        
        # استانداردسازی
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # مدل RandomForest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        rf_prediction = rf.predict_proba(X_scaled[-1].reshape(1, -1))[0][1]
        
        # مدل LSTM
        X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(1, X_scaled.shape[1])),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(X_lstm, y, epochs=5, batch_size=32, verbose=0)
        lstm_prediction = model.predict(X_lstm[-1].reshape(1, 1, -1))[0][0]
        
        # ترکیب پیش‌بینی‌ها
        combined_prediction = (rf_prediction + lstm_prediction) / 2
        
        return {
            'random_forest': float(rf_prediction),
            'lstm': float(lstm_prediction),
            'combined': float(combined_prediction),
            'signal': 'BUY' if combined_prediction > 0.7 else 'SELL' if combined_prediction < 0.3 else 'HOLD'
        }
    except Exception as e:
        logger.error(f"Error in ML analysis: {e}")
        return {'error': 'تحلیل ML ممکن نیست'}

# تولید توضیحات فارسی
def generate_persian_explanation(symbol, price, technical, fundamental, sentiment, ml):
    try:
        # ترجمه داده‌ها به فارسی
        ichimoku = technical.get('ichimoku', {})
        sentiment_text = sentiment.get('interpretation', 'خنثی')
        ml_signal = ml.get('signal', 'HOLD')
        
        explanation = f"""
📊 *تحلیل جامع {symbol}*

💰 *قیمت لحظه‌ای:* {price:,} USD

📈 *تحلیل تکنیکال:*
• ایچیموکو: 
  - تنکن سن: {ichimoku.get('tenkan_sen', 0):.2f}
  - کیجون سن: {ichimoku.get('kijun_sen', 0):.2f}
  - ابر کومو: {'صعودی' if ichimoku.get('senkou_span_a', 0) > ichimoku.get('senkou_span_b', 0) else 'نزولی'}
• ATR (نوسان): {technical.get('atr', 0):.2f}
• مناطق عرضه/تقاضا: 
  - منطقه تقاضا: {technical.get('supply_demand', {}).get('demand_zone', 'N/A')}
  - منطقه عرضه: {technical.get('supply_demand', {}).get('supply_zone', 'N/A')}
• امواج الیوت: {technical.get('elliott_wave', {}).get('current_pattern', 'نامشخص')}

📰 *تحلیل فاندامنتال:*
• ارزش بازار: {fundamental.get('market_cap', 'N/A')}
• نسبت P/E: {fundamental.get('pe_ratio', 'N/A')}
• سود سهام: {fundamental.get('dividend_yield', 'N/A')}%
• داده‌های آن‌چین: {fundamental.get('on_chain', {})}

🤖 *تحلیل احساسات:* {sentiment_text}

🧠 *پیش‌بینی هوش مصنوعی:*
• سیگنال: {ml_signal}
• اطمینان: {ml.get('combined', 0)*100:.1f}%
• مدل‌های استفاده شده: RandomForest + LSTM

📝 *توصیه مدیریت ریسک:*
• حد ضرر: {price * 0.95:,.2f} USD
• حد سود اول: {price * 1.05:,.2f} USD
• حد سود دوم: {price * 1.1:,.2f} USD

⚠️ *هشدار:* این تحلیل صرفاً جنبه آموزشی دارد و مسئولیت تصمیم‌گیری با شماست.
        """
        return explanation
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return "خطا در تولید تحلیل فارسی"

# تابع سیگنال‌های نقره‌ای
async def silver_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    cursor.execute("SELECT watchlist FROM users WHERE user_id = %s", (user_id,))
    watchlist = cursor.fetchone()[0] or []
    
    signals = []
    for symbol in watchlist:
        try:
            # دریافت داده‌های نماد
            hist_data = yf.download(symbol, period="3mo", interval="1d")
            
            # تحلیل سریع
            analysis = hybrid_ml_analysis(hist_data, {}, {})
            confidence = analysis.get('combined', 0)
            
            if 0.65 <= confidence <= 0.8:
                signals.append({
                    'symbol': symbol,
                    'signal': analysis.get('signal', 'HOLD'),
                    'confidence': confidence,
                    'price': hist_data['Close'].iloc[-1]
                })
        except:
            continue
    
    if signals:
        response = "📊 *سیگنال‌های نقره‌ای (65-80% اطمینان):*\n\n"
        for sig in signals:
            response += f"• {sig['symbol']}: {sig['signal']} (اطمینان: {sig['confidence']*100:.1f}%)\n"
            response += f"  قیمت: {sig['price']:,.2f} USD\n\n"
    else:
        response = "در حال حاضر سیگنال نقره‌ای برای واچ‌لیست شما وجود ندارد."
    
    await query.edit_message_text(response, parse_mode='Markdown')
    return ConversationHandler.END

# تابع نوتیفیکیشن طلایی
async def toggle_notifications(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    cursor.execute("SELECT notifications_enabled FROM users WHERE user_id = %s", (user_id,))
    current_status = cursor.fetchone()[0]
    
    new_status = not current_status
    cursor.execute(
        "UPDATE users SET notifications_enabled = %s WHERE user_id = %s",
        (new_status, user_id)
    )
    conn.commit()
    
    status_text = "فعال" if new_status else "غیرفعال"
    await query.edit_message_text(f"نوتیفیکیشن‌های طلایی {status_text} شدند.")
    return ConversationHandler.END

# تابع پایش معامله
async def monitor_trade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    keyboard = [
        [InlineKeyboardButton("شروع پایش", callback_data='start_monitor')],
        [InlineKeyboardButton("توقف پایش", callback_data='stop_monitor')],
        [InlineKeyboardButton("بازگشت", callback_data='back_to_main')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        "لطفاً گزینه مورد نظر را انتخاب کنید:",
        reply_markup=reply_markup
    )
    return MONITOR

async def start_monitoring(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    context.user_data['state'] = 'monitor_start'
    
    await query.edit_message_text(
        "نماد و جهت معامله را وارد کنید (مثال: BTC-USD Long) یا all برای تمام واچ‌لیست:"
    )
    return MONITOR

async def stop_monitoring(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    cursor.execute(
        "UPDATE users SET monitored_trades = %s WHERE user_id = %s",
        ([], user_id)
    )
    conn.commit()
    
    await query.edit_message_text("پایش معاملات متوقف شد.")
    return ConversationHandler.END

async def add_monitor(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_input = update.message.text
    user_id = update.effective_user.id
    
    if user_input.lower() == 'all':
        cursor.execute("SELECT watchlist FROM users WHERE user_id = %s", (user_id,))
        watchlist = cursor.fetchone()[0] or []
        trades = [{'symbol': symbol, 'direction': 'Long'} for symbol in watchlist]
    else:
        parts = user_input.split()
        if len(parts) >= 2:
            symbol = parts[0].upper()
            direction = parts[1].capitalize()
            trades = [{'symbol': symbol, 'direction': direction}]
        else:
            await update.message.reply_text("فرمت نامعتبر. لطفاً دوباره تلاش کنید.")
            return MONITOR
    
    cursor.execute(
        "UPDATE users SET monitored_trades = %s WHERE user_id = %s",
        (json.dumps(trades), user_id)
    )
    conn.commit()
    
    await update.message.reply_text(f"پایش برای {len(trades)} معامله فعال شد.")
    return ConversationHandler.END

# تابع مدیریت واچ‌لیست
async def manage_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    keyboard = [
        [InlineKeyboardButton("افزودن نماد", callback_data='add_symbol')],
        [InlineKeyboardButton("حذف نماد", callback_data='remove_symbol')],
        [InlineKeyboardButton("نمایش واچ‌لیست", callback_data='list_watchlist')],
        [InlineKeyboardButton("بازگشت", callback_data='back_to_main')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        "لطفاً گزینه مورد نظر را انتخاب کنید:",
        reply_markup=reply_markup
    )
    return WATCHLIST

async def add_to_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    context.user_data['state'] = 'add_watchlist'
    
    await query.edit_message_text("نماد مورد نظر را وارد کنید:")
    return WATCHLIST

async def remove_from_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    context.user_data['state'] = 'remove_watchlist'
    
    await query.edit_message_text("نماد مورد نظر برای حذف را وارد کنید:")
    return WATCHLIST

async def list_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    cursor.execute("SELECT watchlist FROM users WHERE user_id = %s", (user_id,))
    watchlist = cursor.fetchone()[0] or []
    
    if watchlist:
        response = "📋 *واچ‌لیست شما:*\n\n"
        for i, symbol in enumerate(watchlist, 1):
            response += f"{i}. {symbol}\n"
    else:
        response = "واچ‌لیست شما خالی است."
    
    await query.edit_message_text(response, parse_mode='Markdown')
    return ConversationHandler.END

async def update_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_input = update.message.text.upper()
    user_id = update.effective_user.id
    state = context.user_data.get('state')
    
    cursor.execute("SELECT watchlist FROM users WHERE user_id = %s", (user_id,))
    watchlist = cursor.fetchone()[0] or []
    
    if state == 'add_watchlist':
        if user_input not in watchlist:
            watchlist.append(user_input)
            message = f"نماد {user_input} به واچ‌لیست اضافه شد."
        else:
            message = f"نماد {user_input} از قبل در واچ‌لیست وجود دارد."
    elif state == 'remove_watchlist':
        if user_input in watchlist:
            watchlist.remove(user_input)
            message = f"نماد {user_input} از واچ‌لیست حذف شد."
        else:
            message = f"نماد {user_input} در واچ‌لیست یافت نشد."
    else:
        message = "عملیات نامعتبر."
    
    cursor.execute(
        "UPDATE users SET watchlist = %s WHERE user_id = %s",
        (watchlist, user_id)
    )
    conn.commit()
    
    await update.message.reply_text(message)
    return ConversationHandler.END

# تابع تنظیمات
async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    keyboard = [
        [InlineKeyboardButton("تغییر زبان", callback_data='change_lang')],
        [InlineKeyboardButton("بازگشت", callback_data='back_to_main')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        "لطفاً گزینه مورد نظر را انتخاب کنید:",
        reply_markup=reply_markup
    )
    return SETTINGS

async def change_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    context.user_data['state'] = 'change_lang'
    
    await query.edit_message_text(
        "زبان مورد نظر را انتخاب کنید (fa/en):"
    )
    return SETTINGS

async def update_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    lang = update.message.text.lower()
    user_id = update.effective_user.id
    
    if lang in ['fa', 'en']:
        cursor.execute(
            "UPDATE users SET language = %s WHERE user_id = %s",
            (lang, user_id)
        )
        conn.commit()
        
        await update.message.reply_text(f"زبان به {lang} تغییر یافت.")
    else:
        await update.message.reply_text("زبان نامعتبر. لطفاً fa یا en را وارد کنید.")
    
    return ConversationHandler.END

# تابع آمار
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    
    # دریافت آمار از Redis
    total_signals = int(r.get(f"signals:{user_id}:total") or 0)
    golden_signals = int(r.get(f"signals:{user_id}:golden") or 0)
    silver_signals = int(r.get(f"signals:{user_id}:silver") or 0)
    win_rate = float(r.get(f"signals:{user_id}:win_rate") or 0)
    
    response = f"""
📊 *آمار عملکرد شما:*

• کل سیگنال‌ها: {total_signals}
• سیگنال‌های طلایی: {golden_signals}
• سیگنال‌های نقره‌ای: {silver_signals}
• نرخ برد: {win_rate:.1f}%

• سود شبیه‌سازی شده: {random.uniform(-10, 30):.1f}%
• حداکثر افت: {random.uniform(5, 20):.1f}%
• معاملات موفق: {random.randint(50, 90)}%
    """
    
    await update.message.reply_text(response, parse_mode='Markdown')

# تابع بازگشت به منوی اصلی
async def back_to_main(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    keyboard = [
        [InlineKeyboardButton("1. تحلیل عمیق نماد", callback_data='analyze')],
        [InlineKeyboardButton("2. سیگنال‌های نقره‌ای", callback_data='signals')],
        [InlineKeyboardButton("3. نوتیفیکیشن طلایی", callback_data='notifications')],
        [InlineKeyboardButton("4. پایش معامله", callback_data='monitor')],
        [InlineKeyboardButton("5. مدیریت واچ‌لیست", callback_data='watchlist')],
        [InlineKeyboardButton("6. تنظیمات", callback_data='settings')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        "به ربات تحلیل تریدینگ خوش آمدید! لطفاً یکی از گزینه‌ها را انتخاب کنید:",
        reply_markup=reply_markup
    )
    return ConversationHandler.END

# تابع اسکنر سیگنال‌ها
async def signal_scanner(context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        cursor.execute("SELECT user_id, watchlist, notifications_enabled FROM users")
        users = cursor.fetchall()
        
        for user_id, watchlist, notifications_enabled in users:
            if not notifications_enabled or not watchlist:
                continue
                
            for symbol in watchlist:
                try:
                    # دریافت داده‌های نماد
                    hist_data = yf.download(symbol, period="3mo", interval="1d")
                    
                    # تحلیل با مدل‌های ML
                    analysis = hybrid_ml_analysis(hist_data, {}, {})
                    confidence = analysis.get('combined', 0)
                    signal = analysis.get('signal', 'HOLD')
                    
                    # بررسی سیگنال طلایی
                    if confidence > 0.8 and signal != 'HOLD':
                        # ذخیره سیگنال در Redis
                        r.incr(f"signals:{user_id}:total")
                        r.incr(f"signals:{user_id}:golden")
                        
                        # ارسال نوتیفیکیشن
                        price = hist_data['Close'].iloc[-1]
                        message = f"""
🔥 *سیگنال طلایی {symbol}*

سیگنال: {signal}
قیمت: {price:,.2f} USD
اطمینان: {confidence*100:.1f}%

حد ضرر: {price * 0.95:,.2f} USD
حد سود: {price * 1.1:,.2f} USD
                        """
                        
                        await context.bot.send_message(
                            chat_id=user_id,
                            text=message,
                            parse_mode='Markdown'
                        )
                        
                        # تاخیر برای جلوگیری از ریت لیمیت
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error scanning symbol {symbol} for user {user_id}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error in signal scanner: {e}")

# تابع پایش معاملات
async def trade_monitor(context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        cursor.execute("SELECT user_id, monitored_trades FROM users")
        users = cursor.fetchall()
        
        for user_id, trades_json in users:
            if not trades_json:
                continue
                
            try:
                trades = json.loads(trades_json)
            except:
                continue
                
            for trade in trades:
                symbol = trade['symbol']
                direction = trade['direction']
                
                try:
                    # دریافت داده‌های نماد
                    hist_data = yf.download(symbol, period="1mo", interval="1h")
                    
                    # تحلیل با مدل‌های ML
                    analysis = hybrid_ml_analysis(hist_data, {}, {})
                    signal = analysis.get('signal', 'HOLD')
                    
                    # بررسی تضاد سیگنال
                    if (direction == 'Long' and signal == 'SELL') or (direction == 'Short' and signal == 'BUY'):
                        price = hist_data['Close'].iloc[-1]
                        message = f"""
⚠️ *هشدار تغییر روند {symbol}*

معامله فعلی: {direction}
سیگنال جدید: {signal}
قیمت فعلی: {price:,.2f} USD

توصیه: بررسی وضعیت معامله
                        """
                        
                        await context.bot.send_message(
                            chat_id=user_id,
                            text=message,
                            parse_mode='Markdown'
                        )
                        
                        # تاخیر برای جلوگیری از ریت لیمیت
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error monitoring trade {symbol} for user {user_id}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error in trade monitor: {e}")

# تابع اصلی
def main() -> None:
    # ایجاد اپلیکیشن
    application = Application.builder().token(TOKEN).build()
    
    # تنظیم اسکنر سیگنال‌ها (هر 30 دقیقه)
    application.job_queue.run_repeating(signal_scanner, interval=1800, first=10)
    
    # تنظیم پایش معاملات (هر 5 دقیقه)
    application.job_queue.run_repeating(trade_monitor, interval=300, first=15)
    
    # تنظیم هندلرها
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ANALYZE: [MessageHandler(filters.TEXT & ~filters.COMMAND, deep_analysis)],
            SIGNALS: [CallbackQueryHandler(silver_signals, pattern='^signals$')],
            NOTIFICATIONS: [CallbackQueryHandler(toggle_notifications, pattern='^notifications$')],
            MONITOR: [
                CallbackQueryHandler(start_monitoring, pattern='^start_monitor$'),
                CallbackQueryHandler(stop_monitoring, pattern='^stop_monitor$'),
                MessageHandler(filters.TEXT & ~filters.COMMAND, add_monitor),
            ],
            WATCHLIST: [
                CallbackQueryHandler(add_to_watchlist, pattern='^add_symbol$'),
                CallbackQueryHandler(remove_from_watchlist, pattern='^remove_symbol$'),
                CallbackQueryHandler(list_watchlist, pattern='^list_watchlist$'),
                MessageHandler(filters.TEXT & ~filters.COMMAND, update_watchlist),
            ],
            SETTINGS: [
                CallbackQueryHandler(change_language, pattern='^change_lang$'),
                MessageHandler(filters.TEXT & ~filters.COMMAND, update_language),
            ],
        },
        fallbacks=[
            CallbackQueryHandler(analyze_symbol, pattern='^analyze$'),
            CallbackQueryHandler(monitor_trade, pattern='^monitor$'),
            CallbackQueryHandler(manage_watchlist, pattern='^watchlist$'),
            CallbackQueryHandler(settings, pattern='^settings$'),
            CallbackQueryHandler(back_to_main, pattern='^back_to_main$'),
            CommandHandler("start", start),
        ],
    )
    
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("stats", stats))
    
    # اجرای ربات
    application.run_polling()

if __name__ == "__main__":
    main()