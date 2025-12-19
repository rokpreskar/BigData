import json
import time
import pandas as pd
from playwright.sync_api import sync_playwright

def scrape_all_data():
    with sync_playwright() as p:
        # Launch browser (headless=True means it runs in the background)
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        results = {"products": [], "testimonials": [], "reviews": []}

        # --- 1. PRODUCTS (Handling Pagination) ---
        print("Scraping Products...")
        for i in range(1, 4): # Pages 1 to 3
            page.goto(f"https://web-scraping.dev/products?page={i}")
            products = page.query_selector_all(".product")
            for item in products:
                results['products'].append({
                    "name": item.query_selector("h3 a").inner_text().strip(),
                    "description": item.query_selector(".short-description").inner_text().strip(),
                    "price": item.query_selector(".price").inner_text().strip()
                })
            print(f"   Fetched Product Page {i}")

        # --- 2. REVIEWS (Handling 'Load More') ---
        print("Scraping Reviews...")
        page.goto("https://web-scraping.dev/reviews")
        # Click "Load More" 3 times to get a large 2023 dataset
        for _ in range(3):
            load_more = page.query_selector("#page-load-more")
            if load_more:
                load_more.click()
                time.sleep(1.5)
        
        reviews = page.query_selector_all(".review")
        for rev in reviews:
            # Parse date and count yellow star SVGs
            date_val = rev.query_selector("[data-testid='review-date']").inner_text()
            stars = len(rev.query_selector_all("svg path[fill='#ffce31']"))
            results['reviews'].append({
                "date": date_val,
                "text": rev.query_selector("[data-testid='review-text']").inner_text().strip(),
                "rating": stars
            })

        # --- 3. TESTIMONIALS (Handling Infinite Scroll) ---
        print("Scraping Testimonials...")
        page.goto("https://web-scraping.dev/testimonials")
        # Scroll down to trigger dynamic loading
        for _ in range(5):
            page.mouse.wheel(0, 3000)
            time.sleep(1)
        
        testimonials = page.query_selector_all(".testimonial")
        for test in testimonials:
            stars = len(test.query_selector_all("svg path[fill='#ffce31']"))
            results['testimonials'].append({
                "text": test.query_selector(".text").inner_text().strip(),
                "rating": stars
            })

        browser.close()

        # Save to data folder
        with open('data/scraped_data.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nSUCCESS! Scraped {len(results['products'])} products, {len(results['reviews'])} reviews, and {len(results['testimonials'])} testimonials.")

if __name__ == "__main__":
    scrape_all_data()
