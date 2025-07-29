# ... (کدهای import اولیه) ...
# from transformers import pipeline # <<< این خط را حذف یا کامنت کنید

# ...

# --- راه‌اندازی مدل هوش مصنوعی برای تحلیل احساسات ---
# sentiment_pipeline = None # <<< این بخش را به این صورت ساده کنید

# ... (بقیه کدها) ...

def whale_and_news_tracker_thread():
    # ...
    while True:
        try:
            # --- رصد اخبار ---
            symbols_for_news = ['bitcoin', 'ethereum']
            for symbol in symbols_for_news:
                url = f"https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}"
                response = requests.get(url)
                if response.status_code == 200:
                    articles = response.json().get('articles', [])
                    # ما دیگر تحلیل احساسات انجام نمی‌دهیم، فقط اخبار را نمایش می‌دهیم
                    # پس نیازی به ذخیره در دیتابیس هم نیست
                    # این بخش از کد را ساده می‌کنیم
            logging.info("News check completed.")
            
            # ... (بخش شبیه‌سازی رصد نهنگ‌ها) ...

        except Exception as e:
            logging.error(f"Whale/News tracker error: {e}")
        
        time.sleep(15 * 60)

# ...

def main_menu_handler(update: Update, context: CallbackContext):
    # ...
    elif query.data == 'menu_news':
        # این بخش را هم برای نمایش ساده اخبار تغییر می‌دهیم
        query.edit_message_text("در حال دریافت آخرین اخبار...")
        try:
            # فقط اخبار بیت‌کوین را برای سادگی می‌گیریم
            url = f"https://newsapi.org/v2/everything?q=bitcoin&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
            response = requests.get(url)
            articles = response.json().get('articles', [])
            
            if not articles:
                query.edit_message_text("خبری یافت نشد.")
                return
            
            message = "📰 **آخرین اخبار بازار:**\n\n"
            for article in articles:
                message += f"🔹 {article['title']}\n"
            
            query.edit_message_text(message)
        except Exception as e:
            logging.error(f"Error fetching news for menu: {e}")
            query.edit_message_text("خطا در دریافت اخبار.")

    # ... (بقیه کد handler)