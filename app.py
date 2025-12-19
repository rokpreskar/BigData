import os
import json
import re
import streamlit as st
import pandas as pd
import altair as alt
from transformers import pipeline
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import gc

# ---- Render / low-memory friendliness ----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
# Prevent HF from downloading to default cache (which may be readonly on Render)
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
os.environ["HF_HOME"] = "/tmp/hf_home"

st.set_page_config(
    page_title="Brand Reputation Dashboard 2023",
    page_icon="📊",
    layout="wide",
)

# ---------- Data loading ----------
@st.cache_data
def load_scraped_data():
    try:
        with open("data/scraped_data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("⚠️ 'data/scraped_data.json' not found! Please run scraper.py first.")
        return None


# ---------- Sentiment pipeline (optimized for low memory) ----------
@st.cache_resource
def get_sentiment_pipeline():
    """
    Use a smaller, more efficient model than DistilBERT.
    distilbert-base-uncased-finetuned-sst-2-english ≈ 250MB
    cardiffnlp/twitter-roberta-base-sentiment-latest ≈ 500MB (too big)
    
    Best option: use quantization or a tiny model
    """
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,  # CPU only
        framework="pt",  # PyTorch (smaller than TF)
    )


def analyze_in_small_batches(texts, batch_size=3, max_length=64):
    """
    Process texts in very small batches with aggressive truncation.
    Returns results and cleans up intermediate objects.
    """
    sentiment_pipe = get_sentiment_pipeline()
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Process batch
        batch_results = sentiment_pipe(
            batch,
            truncation=True,
            max_length=max_length,  # Aggressive truncation
            padding=False,  # No padding to save memory
        )
        results.extend(batch_results)
        
        # Force garbage collection after each batch
        del batch_results
        gc.collect()
    
    return results


# ---------- App ----------
data = load_scraped_data()

st.sidebar.title("🔍 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("Select a Section:", ["Products", "Testimonials", "Reviews"])

if not data:
    st.stop()

# ---------------------------
# Products
# ---------------------------
if page == "Products":
    st.header("📦 Product Catalog")
    st.write("Overview of all scraped products and their pricing.")

    df_products = pd.DataFrame(data.get("products", []))
    if df_products.empty:
        st.warning("No products found in scraped data.")
        st.stop()

    if "price" in df_products.columns:
        df_products["price"] = pd.to_numeric(df_products["price"], errors="coerce")

    col1, col2 = st.columns(2)
    col1.metric("Total Products", len(df_products))
    col2.metric("Avg. Price", f"${df_products['price'].mean():.2f}" if df_products["price"].notna().any() else "N/A")

    st.dataframe(df_products, use_container_width=True)
    
    # Clean up
    del df_products
    gc.collect()

# ---------------------------
# Testimonials
# ---------------------------
elif page == "Testimonials":
    st.header("💬 Customer Testimonials")
    st.write("Raw feedback collected from the infinite scroll section.")

    df_testimonials = pd.DataFrame(data.get("testimonials", []))
    if df_testimonials.empty:
        st.warning("No testimonials found in scraped data.")
        st.stop()

    st.dataframe(df_testimonials, use_container_width=True)

    if "rating" in df_testimonials.columns:
        st.subheader("Rating Distribution")
        st.bar_chart(df_testimonials["rating"].value_counts())
    else:
        st.info("No 'rating' column found for testimonials.")
    
    # Clean up
    del df_testimonials
    gc.collect()

# ---------------------------
# Reviews (filter + sentiment + viz + wordcloud)
# ---------------------------
else:
    st.header("📈 2023 Reviews Analysis")
    st.write("Filter reviews by specific months within the year 2023.")

    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    selected_month_name = st.select_slider("Select a month in 2023:", options=months)

    df_reviews = pd.DataFrame(data.get("reviews", []))
    if df_reviews.empty:
        st.warning("No reviews found in scraped data.")
        st.stop()

    if "date" not in df_reviews.columns:
        st.error("Reviews data has no 'date' column, cannot filter by month.")
        st.stop()

    df_reviews["date"] = pd.to_datetime(df_reviews["date"], errors="coerce")
    month_index = months.index(selected_month_name) + 1

    filtered_df = df_reviews[
        (df_reviews["date"].dt.month == month_index) &
        (df_reviews["date"].dt.year == 2023)
    ].copy()
    
    # Clean up original dataframe immediately
    del df_reviews
    gc.collect()

    if filtered_df.empty:
        st.warning(f"No reviews were scraped for {selected_month_name} 2023.")
        st.stop()

    st.success(f"Found {len(filtered_df)} reviews for {selected_month_name} 2023.")

    # rating metric (optional)
    if "rating" in filtered_df.columns:
        filtered_df["rating"] = pd.to_numeric(filtered_df["rating"], errors="coerce")

    show_df = filtered_df.copy()
    show_df["date"] = show_df["date"].dt.strftime("%Y-%m-%d")

    if "rating" in filtered_df.columns and filtered_df["rating"].notna().any():
        st.metric(f"Average Rating for {selected_month_name}", f"{filtered_df['rating'].mean():.1f} / 5 ⭐")
    else:
        st.metric(f"Average Rating for {selected_month_name}", "N/A")

    if "text" not in filtered_df.columns:
        st.error("Reviews data has no 'text' column, cannot run sentiment/word cloud.")
        st.dataframe(show_df, use_container_width=True)
        st.stop()

    # ---- Controls (Render-safe) ----
    st.markdown("---")
    st.subheader("Controls (Memory-Optimized)")
    
    st.info("⚠️ Sentiment analysis uses a transformer model (~250MB). Processing limited to avoid memory issues on Render free tier.")
    
    c1, c2 = st.columns(2)
    run_sent = c1.button("Run Sentiment Analysis")
    run_wc = c2.button("Generate Word Cloud")

    # ---- Sentiment Analysis ----
    if run_sent:
        texts = filtered_df["text"].fillna("").astype(str).str.strip().tolist()

        # Very strict limit for Render (512MB total, model is ~250MB)
        MAX_REVIEWS = 100
        if len(texts) > MAX_REVIEWS:
            st.warning(f"⚠️ Processing first {MAX_REVIEWS} reviews (out of {len(texts)}) to stay within memory limits.")
            texts = texts[:MAX_REVIEWS]
            show_df = show_df.head(MAX_REVIEWS).copy()

        with st.spinner(f"Running sentiment analysis on {len(texts)} reviews (using DistilBERT)..."):
            try:
                # Process in very small batches with aggressive truncation
                results = analyze_in_small_batches(texts, batch_size=3, max_length=64)
                
                show_df["sentiment"] = [
                    "Positive" if r["label"] == "POSITIVE" else "Negative" 
                    for r in results
                ]
                show_df["sentiment_score"] = [float(r["score"]) for r in results]

                st.success("✅ Sentiment analysis complete!")
                
                st.subheader("Sentiment Analysis Results")
                m1, m2, m3 = st.columns(3)
                m1.metric("Positive", int((show_df["sentiment"] == "Positive").sum()))
                m2.metric("Negative", int((show_df["sentiment"] == "Negative").sum()))
                m3.metric("Overall Avg Confidence", f"{show_df['sentiment_score'].mean():.2f}")

                # --- Visualization (counts + avg confidence tooltip) ---
                st.subheader("Positive vs Negative Reviews")

                summary = (
                    show_df.groupby("sentiment", as_index=False)
                    .agg(count=("sentiment", "size"), avg_conf=("sentiment_score", "mean"))
                )

                summary = summary.set_index("sentiment").reindex(["Positive", "Negative"]).reset_index()
                summary["count"] = summary["count"].fillna(0).astype(int)
                summary["avg_conf"] = summary["avg_conf"].fillna(0.0)

                chart = alt.Chart(summary).mark_bar().encode(
                    x=alt.X("sentiment:N", title="Sentiment"),
                    y=alt.Y("count:Q", title="Number of Reviews"),
                    tooltip=[
                        alt.Tooltip("sentiment:N", title="Sentiment"),
                        alt.Tooltip("count:Q", title="Count"),
                        alt.Tooltip("avg_conf:Q", title="Avg confidence", format=".2f"),
                    ],
                )

                text_layer = alt.Chart(summary).mark_text(dy=-10).encode(
                    x="sentiment:N",
                    y="count:Q",
                    text=alt.Text("avg_conf:Q", format=".2f"),
                )

                st.altair_chart(chart + text_layer, use_container_width=True)

                st.subheader("Reviews Table (with Sentiment)")
                st.dataframe(show_df, use_container_width=True)
                
                # Clean up
                del results, texts, summary
                gc.collect()
                
            except Exception as e:
                st.error(f"❌ Error during sentiment analysis: {str(e)}")
                st.info("Try reducing the number of reviews or check Render logs for memory issues.")

    # ---- Word Cloud ----
    if run_wc:
        st.subheader("Word Cloud (Selected Month Reviews)")

        # Limit text for word cloud too
        wc_df = show_df.head(200) if len(show_df) > 200 else show_df
        
        text_blob = " ".join(wc_df["text"].fillna("").astype(str)).lower()
        text_blob = re.sub(r"http\S+|www\S+", "", text_blob)
        text_blob = re.sub(r"[^\w\s]", " ", text_blob, flags=re.UNICODE)
        text_blob = re.sub(r"[\d_]+", " ", text_blob)
        text_blob = re.sub(r"\s+", " ", text_blob).strip()

        if len(text_blob) < 20:
            st.info("Not enough text to generate a word cloud.")
        else:
            try:
                wc = WordCloud(
                    width=600,  # Reduced from 700
                    height=300,  # Reduced from 350
                    max_words=60,  # Reduced from 100
                    background_color="white",
                    stopwords=set(STOPWORDS),
                    collocations=False,
                ).generate(text_blob)

                fig, ax = plt.subplots(figsize=(9, 4))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
                
                # Clean up immediately
                plt.close(fig)
                del wc, fig, ax, text_blob, wc_df
                gc.collect()
                
            except Exception as e:
                st.error(f"❌ Error generating word cloud: {str(e)}")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Data Mining Homework #3 Rok Preskar")


# Run locally:
# python -m streamlit run app.py
#
# Render start command:
# streamlit run app.py --server.port $PORT --server.address 0.0.0.0