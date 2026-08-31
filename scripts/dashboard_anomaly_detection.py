"""
Streamlit Dashboard for Anomaly & Trend Detection
Analyzes classified reviews to detect spikes and anomalies in customer issues
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.ensemble import IsolationForest

# Page config
st.set_page_config(page_title="Anomaly & Trend Detection", layout="wide", initial_sidebar_state="expanded")

# Title
st.title("📈 Anomaly & Trend Detection Dashboard")
st.markdown("Real-time spike detection for customer issue categories")
st.markdown("---")

BASE_DIR = Path(__file__).resolve().parent.parent

# Load classified reviews
@st.cache_data
def load_reviews():
    try:
        return pd.read_excel(BASE_DIR / "outputs" / "reviews_with_issue_classification.xlsx")
    except:
        st.error("Could not load classified reviews")
        return None

# Process anomalies
@st.cache_data
def detect_anomalies(df):
    """Detect anomalies using z-score and Isolation Forest"""
    
    if 'review_text' not in df.columns:
        # Try alternate column names
        if 'review' in df.columns:
            df['review_text'] = df['review']
    
    # Create a date column if it doesn't exist
    if 'review_date' not in df.columns:
        df['review_date'] = pd.date_range(start='2025-01-01', periods=len(df), freq='H')
    else:
        df['review_date'] = pd.to_datetime(df['review_date'])
    
    # Weekly aggregation
    df['week'] = df['review_date'].dt.to_period('W').apply(lambda p: p.start_time)
    
    # Group by issue category and week
    weekly_data = df.groupby(['predicted_issue_category', 'week']).size().reset_index(name='review_count')
    
    # Calculate statistics per category
    results = []
    Z_THRESHOLD = 2.0
    ROLLING_WINDOW = 6
    
    for category in weekly_data['predicted_issue_category'].unique():
        cat_data = weekly_data[weekly_data['predicted_issue_category'] == category].sort_values('week').reset_index(drop=True)
        
        if len(cat_data) < ROLLING_WINDOW:
            continue
        
        # Z-score calculation
        cat_data['rolling_mean'] = cat_data['review_count'].rolling(window=ROLLING_WINDOW).mean()
        cat_data['rolling_std'] = cat_data['review_count'].rolling(window=ROLLING_WINDOW).std()
        cat_data['z_score'] = (cat_data['review_count'] - cat_data['rolling_mean']) / (cat_data['rolling_std'] + 1e-6)
        cat_data['zscore_flag'] = cat_data['z_score'].abs() > Z_THRESHOLD
        
        # Isolation Forest
        if len(cat_data) > 1:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            cat_data['iforest_anomaly'] = iso_forest.fit_predict(cat_data[['review_count']].values) == -1
        else:
            cat_data['iforest_anomaly'] = False
        
        cat_data['issue_category'] = category
        results.append(cat_data)
    
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

# Load data
df = load_reviews()

if df is not None:
    st.success(f"✅ Loaded {len(df)} classified reviews")
    
    # Detect anomalies
    with st.spinner("Detecting anomalies and trends..."):
        anomaly_df = detect_anomalies(df)
    
    if len(anomaly_df) > 0:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        zscore_flags = anomaly_df['zscore_flag'].sum()
        iforest_flags = anomaly_df['iforest_anomaly'].sum()
        both_flags = ((anomaly_df['zscore_flag']) & (anomaly_df['iforest_anomaly'])).sum()
        total_weeks = len(anomaly_df)
        
        with col1:
            st.metric("Total Weeks Analyzed", total_weeks)
        with col2:
            st.metric("Z-Score Anomalies", zscore_flags)
        with col3:
            st.metric("Isolation Forest Flags", iforest_flags)
        with col4:
            st.metric("High-Confidence Spikes", both_flags)
        
        st.markdown("---")
        
        # Main visualization
        st.subheader("📊 Weekly Review Trends by Category")
        
        # Get top categories by volume
        top_categories = anomaly_df.groupby('issue_category')['review_count'].sum().nlargest(5).index.tolist()
        
        # Filter data for visualization
        viz_data = anomaly_df[anomaly_df['issue_category'].isin(top_categories)]
        
        fig = go.Figure()
        
        for category in top_categories:
            cat_df = viz_data[viz_data['issue_category'] == category].sort_values('week')
            
            # Add line trace
            fig.add_trace(go.Scatter(
                x=cat_df['week'],
                y=cat_df['review_count'],
                mode='lines+markers',
                name=category,
                hovertemplate='<b>%{fullData.name}</b><br>Week: %{x|%Y-%m-%d}<br>Count: %{y}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Weekly Issue Trends (Top 5 Categories)",
            xaxis_title="Week",
            yaxis_title="Review Count",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Anomalies detected
        st.subheader("🔴 Detected Anomalies & Spikes")
        
        # Filter anomalies
        anomalies = anomaly_df[(anomaly_df['zscore_flag']) | (anomaly_df['iforest_anomaly'])].copy()
        
        if len(anomalies) > 0:
            # Highlight high-confidence spikes
            anomalies['confidence'] = (
                (anomalies['zscore_flag'].astype(int) + anomalies['iforest_anomaly'].astype(int)) / 2 * 100
            )
            anomalies = anomalies.sort_values('confidence', ascending=False)
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                view_type = st.radio("View:", ["All Anomalies", "High-Confidence Only"])
            
            if view_type == "High-Confidence Only":
                display_anomalies = anomalies[anomalies['confidence'] == 100.0]
            else:
                display_anomalies = anomalies
            
            # Display table
            display_cols = ['issue_category', 'week', 'review_count', 'z_score', 'rolling_mean', 'confidence']
            available_cols = [col for col in display_cols if col in display_anomalies.columns]
            
            st.dataframe(
                display_anomalies[available_cols].head(20),
                use_container_width=True,
                height=500
            )
            
            st.markdown("---")
            
            # Z-score distribution
            st.subheader("Z-Score Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    anomaly_df.dropna(subset=['z_score']),
                    x='z_score',
                    nbins=30,
                    title='Z-Score Distribution',
                    labels={'z_score': 'Z-Score'},
                    color_discrete_sequence=['#636EFA']
                )
                fig.add_vline(x=2.0, line_dash="dash", line_color="red", annotation_text="Threshold (2σ)")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    anomaly_df.dropna(subset=['z_score']),
                    x='issue_category',
                    y='z_score',
                    title='Z-Score by Issue Category',
                    labels={'z_score': 'Z-Score', 'issue_category': 'Issue Category'}
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Category performance
            st.subheader("📋 Category-wise Anomaly Summary")
            
            category_summary = anomaly_df.groupby('issue_category').agg({
                'review_count': ['sum', 'mean', 'std'],
                'zscore_flag': 'sum',
                'iforest_anomaly': 'sum'
            }).round(2)
            
            category_summary.columns = ['Total Reviews', 'Avg Weekly', 'Std Dev', 'Z-Score Flags', 'IF Flags']
            category_summary = category_summary.sort_values('Z-Score Flags', ascending=False)
            
            st.dataframe(category_summary, use_container_width=True)
            
        else:
            st.info("No anomalies detected in this dataset")
    
    else:
        st.warning("No anomalies could be computed - insufficient data")
        
else:
    st.error("Failed to load data")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
    Anomaly detection uses Z-score (statistical) and Isolation Forest (ML) methods
    <br>Data: Classified reviews from all restaurants and platforms
</div>
""", unsafe_allow_html=True)
