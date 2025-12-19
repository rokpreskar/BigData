"""
Run this script LOCALLY before deploying to pre-compute sentiment analysis.
This saves results to data/sentiment_results.json

Usage: python preprocess_sentiment.py
"""

import json
from transformers import pipeline
from tqdm import tqdm

def load_scraped_data():
    with open("data/scraped_data.json", "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_reviews(reviews):
    """Analyze all reviews and return results."""
    print("Loading sentiment analysis model...")
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,
    )
    
    results = []
    print(f"Analyzing {len(reviews)} reviews...")
    
    # Process in batches with progress bar
    batch_size = 16
    for i in tqdm(range(0, len(reviews), batch_size)):
        batch = reviews[i:i + batch_size]
        texts = [r.get("text", "") for r in batch]
        
        # Run sentiment analysis
        sentiments = sentiment_pipe(texts, truncation=True, max_length=512)
        
        # Combine original review data with sentiment
        for review, sentiment in zip(batch, sentiments):
            results.append({
                **review,  # Keep all original fields
                "sentiment_label": sentiment["label"],
                "sentiment_score": float(sentiment["score"])
            })
    
    return results

def main():
    print("=" * 60)
    print("Pre-computing Sentiment Analysis for Reviews")
    print("=" * 60)
    
    # Load data
    data = load_scraped_data()
    reviews = data.get("reviews", [])
    
    if not reviews:
        print("❌ No reviews found in scraped_data.json")
        return
    
    print(f"✓ Found {len(reviews)} reviews")
    
    # Analyze
    analyzed_reviews = analyze_reviews(reviews)
    
    # Save results
    output = {
        "products": data.get("products", []),
        "testimonials": data.get("testimonials", []),
        "reviews": analyzed_reviews
    }
    
    with open("data/sentiment_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("✅ Sentiment analysis complete!")
    print(f"📁 Results saved to: data/sentiment_results.json")
    print("=" * 60)
    
    # Show summary
    positive = sum(1 for r in analyzed_reviews if r["sentiment_label"] == "POSITIVE")
    negative = sum(1 for r in analyzed_reviews if r["sentiment_label"] == "NEGATIVE")
    print(f"\nSummary:")
    print(f"  Positive: {positive} ({positive/len(analyzed_reviews)*100:.1f}%)")
    print(f"  Negative: {negative} ({negative/len(analyzed_reviews)*100:.1f}%)")

if __name__ == "__main__":
    main()