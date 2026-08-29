# ==============================================================================
# app.py — Streamlit Dashboard
# ==============================================================================
# Professional UI for the Market Research Insight Application.
# Features:
#   - Input form with validation logic
#   - Real-time progress tracking
#   - Expandable retrieved news viewer
#   - Clean, modern layout for analyst reports
# ==============================================================================

import streamlit as st
import time
from src.pipeline import run_pipeline
from src.logger import get_logger

logger = get_logger("StreamlitUI")


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Research Insight AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A; /* Dark Blue */
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .insight-card {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-box {
        background-color: #EFF6FF;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .error-box {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        padding: 15px;
        border-radius: 8px;
        color: #991B1B;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        height: 50px;
        font-size: 18px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)


# ── Title ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 Market Research Insight AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Autonomous Commodity Intelligence & RAG Analyst System</div>', unsafe_allow_html=True)

# ── Sidebar Inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameter Input")
    
    metal = st.text_input("Metal / Commodity", placeholder="e.g. Price of Steel", help="Enter the metal name")
    country = st.text_input("Country / Region", placeholder="e.g. India", help="Target market region")
    
    col1, col2 = st.columns(2)
    with col1:
        price = st.text_input("Current Price", placeholder="55000")
    with col2:
        change = st.text_input("Weekly % Change", placeholder="-2.5")
    
    st.markdown("---")
    generate_btn = st.button("Generate Insight", type="primary")
    
    st.markdown("### System Status")
    status_placeholder = st.empty()
    status_placeholder.info("Ready")


# ── Main Logic ──────────────────────────────────────────────────────────────
if generate_btn:
    # Reset UI
    result_container = st.container()
    
    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_ui_progress(msg, value):
        progress_bar.progress(value)
        status_text.text(f"⏳ {msg}")
        status_placeholder.text(f"Processing: {msg}")

    # Run Pipeline
    with st.spinner("Analyzing market dynamics..."):
        results = run_pipeline(
            metal, country, price, change, 
            progress_callback=update_ui_progress
        )
    
    # Process Results
    if results["status"] == "error":
        progress_bar.empty()
        status_text.empty()
        status_placeholder.error("Failed")
        st.markdown(f"""
        <div class="error-box">
            <h3>❌ Stage Failed</h3>
            <p><strong>Reason:</strong> {results.get('error_message')}</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Success State
        progress_bar.empty()
        status_text.empty()
        status_placeholder.success("Completed")
        
        # Display Stats
        stats = results.get("stats", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Articles Analyzed", stats.get("raw_articles", 0))
        c2.metric("Cleaned Articles", stats.get("cleaned_articles", 0))
        c3.metric("Supply Signals", stats.get("supply_signals", 0))
        c4.metric("Demand Signals", stats.get("demand_signals", 0))
        
        st.divider()
        
        # Display Insights
        st.subheader("📝 Executive Summary")
        st.markdown(f'<div class="insight-card" style="border-left-color: #10B981;">{results["summary"]}</div>', unsafe_allow_html=True)
        
        col_sup, col_dem = st.columns(2)
        
        with col_sup:
            st.subheader("🏭 Supply Dynamics")
            st.markdown(f'<div class="insight-card">{results["supply_insight"]}</div>', unsafe_allow_html=True)
            
        with col_dem:
            st.subheader("🛒 Demand Dynamics")
            st.markdown(f'<div class="insight-card" style="border-left-color: #F59E0B;">{results["demand_insight"]}</div>', unsafe_allow_html=True)
            
        # Retrieval Evidence
        with st.expander("📚 Retrieved Context & Evidence (RAG Output)"):
            retrieval = results.get("retrieval_results", {})
            
            st.markdown("#### Top Supply Signals")
            for item in retrieval.get("SUPPLY", [])[:3]:
                st.markdown(f"- **{item['source']}** ({item['date']}): {item['text'][:200]}... *(Score: {item['score']:.2f})*")
                
            st.markdown("---")
            
            st.markdown("#### Top Demand Signals")
            for item in retrieval.get("DEMAND", [])[:3]:
                st.markdown(f"- **{item['source']}** ({item['date']}): {item['text'][:200]}... *(Score: {item['score']:.2f})*")

else:
    st.info("👈 Enter market details in the sidebar and click 'Generate Insight' to start.")
