import os
import json
import re
import streamlit as st
import pandas as pd
import altair as alt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# ---- Render / low-memory friendliness ----
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

st.set_page_config(
    page_title="Brand Reputation Dashboard 2023",
    page_icon="📊",
    layout="wide",
)

# ---------- Data loading ----------
@st.cache_data
def load_sentiment_data():
    """Load pre-computed sentiment results."""
    try:
        with open("data/sentiment_results.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("⚠️ 'data/sentiment_results.json' not found!")
        st.info("""
        **Steps to fix:**
        1. Run `python preprocess_sentiment.py` on your local machine
        2. Commit the generated `data/sentiment_results.json` file
        3. Push to GitHub and redeploy on Render
        """)
        return None


# ---------- App ----------
data = load_sentiment_data()

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
        st.warning("No products found in data.")
        st.stop()

    if "price" in df_products.columns:
        df_products["price"] = pd.to_numeric(df_products["price"], errors="coerce")

    col1, col2 = st.columns(2)
    col1.metric("Total Products", len(df_products))
    col2.metric("Avg. Price", f"${df_products['price'].mean():.2f}" if df_products["price"].notna().any() else "N/A")

    st.dataframe(df_products, use_container_width=True)

# ---------------------------
# Testimonials
# ---------------------------
elif page == "Testimonials":
    st.header("💬 Customer Testimonials")
    st.write("Raw feedback collected from the infinite scroll section.")

    df_testimonials = pd.DataFrame(data.get("testimonials", []))
    if df_testimonials.empty:
        st.warning("No testimonials found in data.")
        st.stop()

    st.dataframe(df_testimonials, use_container_width=True)

    if "rating" in df_testimonials.columns:
        st.subheader("Rating Distribution")
        st.bar_chart(df_testimonials["rating"].value_counts())
    else:
        st.info("No 'rating' column found for testimonials.")

# ---------------------------
# Reviews (with pre-computed sentiment)
# ---------------------------
else:
    st.header("📈 2023 Reviews Analysis")
    st.write("Filter reviews by specific months within the year 2023.")
    st.info("💡 Sentiment analysis was pre-computed offline using DistilBERT transformers. No heavy models loaded here!")

    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    selected_month_name = st.select_slider("Select a month in 2023:", options=months)

    df_reviews = pd.DataFrame(data.get("reviews", []))
    if df_reviews.empty:
        st.warning("No reviews found in data.")
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
        st.error("Reviews data has no 'text' column.")
        st.dataframe(show_df, use_container_width=True)
        st.stop()

    # Check if sentiment data exists
    if "sentiment_label" not in show_df.columns:
        st.warning("⚠️ Sentiment data not found. Please run `preprocess_sentiment.py` locally first.")
        st.dataframe(show_df, use_container_width=True)
        st.stop()

    # ---- Display Pre-computed Sentiment ----
    st.markdown("---")
    st.subheader("Sentiment Analysis Results")
    
    # Map to friendly names
    show_df["sentiment"] = show_df["sentiment_label"].map({
        "POSITIVE": "Positive",
        "NEGATIVE": "Negative"
    })
    show_df["sentiment_score"] = show_df["sentiment_score"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Positive", int((show_df["sentiment"] == "Positive").sum()))
    m2.metric("Negative", int((show_df["sentiment"] == "Negative").sum()))
    m3.metric("Overall Avg Confidence", f"{show_df['sentiment_score'].mean():.2f}")

    # --- Visualization ---
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
    
    # Select columns to display
    display_cols = ["date", "text", "sentiment", "sentiment_score"]
    if "rating" in show_df.columns:
        display_cols.insert(2, "rating")
    
    st.dataframe(show_df[display_cols], use_container_width=True)

    # ---- Word Cloud ----
    st.markdown("---")
    if st.button("Generate Word Cloud"):
        st.subheader("Word Cloud (Selected Month Reviews)")

        text_blob = " ".join(show_df["text"].fillna("").astype(str)).lower()
        text_blob = re.sub(r"http\S+|www\S+", "", text_blob)
        text_blob = re.sub(r"[^\w\s]", " ", text_blob, flags=re.UNICODE)
        text_blob = re.sub(r"[\d_]+", " ", text_blob)
        text_blob = re.sub(r"\s+", " ", text_blob).strip()

        if len(text_blob) < 20:
            st.info("Not enough text to generate a word cloud.")
        else:
            wc = WordCloud(
                width=800,
                height=400,
                max_words=100,
                background_color="white",
                stopwords=set(STOPWORDS),
                collocations=False,
            ).generate(text_blob)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Data Mining Homework #3 Rok Preskar")
st.sidebar.caption("Sentiment analysis pre-computed with DistilBERT")


# Run locally:
# python -m streamlit run app.py
#
# Render start command:
# streamlit run app.py --server.port $PORT --server.address 0.0.0.0