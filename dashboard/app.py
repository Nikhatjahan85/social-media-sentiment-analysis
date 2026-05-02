import streamlit as st
import pandas as pd
import plotly.express as plotly_express
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import os
from pathlib import Path
import joblib
import sys
from PIL import Image

# Always resolve paths relative to THIS file's location
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_sentiment, predict_batch

# Set page config
st.set_page_config(
    page_title="SENTIMENT_ENGINE",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* Hide Streamlit default elements */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}
  .stDeployButton {display: none;}

  /* Global background */
  .stApp {
    background-color: #0D0F14 !important;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #13161C !important;
    border-right: 1px solid #2A2D35 !important;
  }

  /* Remove all default Streamlit padding */
  .block-container {
    padding-top: 1rem !important;
    padding-bottom: 0 !important;
  }

  /* All text default color */
  body, p, div, span, label {
    color: #E5E7EB !important;
  }

  /* Remove streamlit widget borders */
  .stTextArea textarea {
    background-color: #1A1D24 !important;
    border: 1px solid #2A2D35 !important;
    color: #E5E7EB !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 0px !important;
  }

  /* Button style */
  .stButton button {
    background-color: #2563EB !important;
    color: white !important;
    border: none !important;
    border-radius: 0px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 1px !important;
    font-size: 12px !important;
  }

  /* File uploader */
  .stFileUploader {
    background-color: #1A1D24 !important;
    border: 1px solid #2A2D35 !important;
    border-radius: 0px !important;
  }
</style>
""", unsafe_allow_html=True)

# Custom CSS for layout structural elements
st.markdown("""
<style>
    /* Global Styles */
    :root {
        --color-bg-main: #0D0F14;
        --color-bg-sidebar: #13161C;
        --color-bg-card: #1A1D24;
        --color-primary: #2563EB;
        --color-positive: #16A34A;
        --color-negative: #DC2626;
        --color-neutral: #6B7280;
        --color-text-primary: #E5E7EB;
        --color-text-secondary: #6B7280;
        --color-border: #2A2D35;
        --color-bg-alt: #16191F;
    }

    * {
        font-family: system-ui, -apple-system, blinkmacsystemfont, 'Segoe UI', roboto, oxygen, ubuntu, cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }

    /* Overall Backgrounds */
    .stApp {
        background-color: var(--color-bg-main);
        color: var(--color-text-primary);
    }

    /* Remove emojis */
    .css-1r6p8d1 { display: none; }

    /* Typography Utilities */
    .mono-text {
        font-family: 'IBM Plex Mono', 'Consolas', 'Courier New', monospace !important;
    }

    .uppercase {
        text-transform: uppercase;
    }

    /* Cards */
    .metric-card {
        background-color: var(--color-bg-card);
        border: 1px solid var(--color-border);
        padding: 1.5rem;
        border-radius: 0px !important;
        box-shadow: none !important;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 120px;
    }

    .metric-title {
        color: var(--color-text-secondary);
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .metric-bar-container {
        width: 100%;
        background-color: var(--color-border);
        height: 4px;
        margin-top: auto;
    }

    .metric-bar {
        height: 100%;
    }

    /* Top Bar */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--color-border);
    }

    .page-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--color-text-primary);
        margin: 0;
        padding: 0;
    }

    .top-bar-stats {
        display: flex;
        gap: 1.5rem;
        align-items: center;
        color: var(--color-text-secondary);
        font-size: 0.75rem;
        letter-spacing: 0.05em;
    }

    .icon-btn {
        background: none;
        border: none;
        color: var(--color-text-secondary);
        font-size: 1rem;
        cursor: pointer;
        padding: 0 0.5rem;
    }

    /* Chart Cards */
    .chart-card {
        background-color: var(--color-bg-card);
        border: 1px solid var(--color-border);
        padding: 1.5rem;
        border-radius: 0px !important;
        margin-bottom: 1rem;
    }

    .section-title {
        color: var(--color-text-secondary);
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
    }

    /* Custom HTML Legend */
    .custom-legend {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        justify-content: center;
        height: 100%;
        padding-left: 1rem;
    }

    .legend-item {
        display: flex;
        flex-direction: column;
    }

    .legend-label {
        font-size: 0.7rem;
        color: var(--color-text-secondary);
    }

    .legend-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--color-text-primary);
    }

    /* Bottom Status Bar */
    .status-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: var(--color-bg-main);
        border-top: 1px solid var(--color-border);
        padding: 0.5rem 1rem;
        display: flex;
        justify-content: space-between;
        font-size: 0.7rem;
        color: var(--color-text-secondary);
        z-index: 1000;
    }

    .status-left {
        display: flex;
        gap: 1.5rem;
    }

    .dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        background-color: var(--color-positive);
        border-radius: 50%;
        margin-right: 0.5rem;
        vertical-align: middle;
    }

    /* Code Block styling for txt report */
    code {
        color: var(--color-text-primary) !important;
        background-color: #1A1D24 !important;
    }
    pre {
        background-color: #1A1D24 !important;
        border: 1px solid var(--color-border) !important;
        border-radius: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for conditional colors
def get_color(sentiment):
    if sentiment == 'POSITIVE': return '#16A34A'
    elif sentiment == 'NEGATIVE': return '#DC2626'
    else: return '#6B7280'

# Render Top Bar
def render_top_bar(page_title):
    st.markdown(f"""
    <div class="top-bar">
        <div class="page-title">{page_title}</div>
        <div class="top-bar-stats mono-text uppercase">
            <span>LAST UPDATED: May 2 2026</span>
            <span>MODEL: LOGISTIC REGRESSION</span>
            <span>ACCURACY: 89.4%</span>
            <span>&#x21BB; &#x2699; &#x1F464;</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render Status Bar
def render_status_bar():
    st.markdown("""
    <div class="status-bar mono-text uppercase">
        <div class="status-left">
            <span><span class="dot"></span>NODE_01: ONLINE</span>
            <span><span class="dot"></span>API_CONN: STABLE</span>
        </div>
        <div>SESSION_REF: 8821-XCA-001</div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <div class="mono-text uppercase" style="color: white; font-size: 1.5rem; font-weight: bold;">SENTIMENT_ENGINE</div>
        <div class="mono-text uppercase" style="color: #6B7280; font-size: 0.75rem;">V2.4.0-STABLE</div>
    </div>
    <hr style="border-color: #2A2D35; margin-bottom: 1rem;">
    """, unsafe_allow_html=True)
    
    selected_page = option_menu(
        menu_title=None,
        options=["OVERVIEW", "TEXT ANALYSIS", "DATASET UPLOAD", "MODEL PERFORMANCE"],
        icons=["grid", "file-text", "upload", "bar-chart"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#6B7280", "font-size": "14px"},
            "nav-link": {
                "font-size": "12px", 
                "text-align": "left", 
                "margin": "0px",
                "padding": "10px",
                "color": "#E5E7EB",
                "font-family": "'IBM Plex Mono', monospace",
                "text-transform": "uppercase",
                "border-radius": "0px",
                "letter-spacing": "0.05em"
            },
            "nav-link-selected": {
                "background-color": "#1A1D24",
                "border-left": "4px solid #2563EB",
                "color": "#E5E7EB"
            },
        }
    )



# Header


st.markdown('<h1 style="font-size:40px; font-weight:900; font-family:IBM Plex Mono; background:linear-gradient(90deg,#22C55E,#3B82F6,#A855F7); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">SENTIMENT ANALYSIS DASHBOARD</h1>', unsafe_allow_html=True)





# OVERVIEW PAGE
if selected_page == "OVERVIEW":
    render_top_bar("Overview")
    
    # ROW 1: Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title uppercase mono-text">TOTAL DOCUMENTS</div>
            <div class="metric-value mono-text" style="color: white;">1,000</div>
            <div class="metric-bar-container">
                <div class="metric-bar" style="width: 100%; background-color: var(--color-primary);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title uppercase mono-text">POSITIVE SENTIMENT</div>
            <div class="metric-value mono-text" style="color: var(--color-positive);">42.3%</div>
            <div class="metric-bar-container">
                <div class="metric-bar" style="width: 42.3%; background-color: var(--color-positive);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title uppercase mono-text">NEGATIVE SENTIMENT</div>
            <div class="metric-value mono-text" style="color: var(--color-negative);">31.7%</div>
            <div class="metric-bar-container">
                <div class="metric-bar" style="width: 31.7%; background-color: var(--color-negative);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title uppercase mono-text">NEUTRAL SENTIMENT</div>
            <div class="metric-value mono-text" style="color: var(--color-neutral);">26.0%</div>
            <div class="metric-bar-container">
                <div class="metric-bar" style="width: 26.0%; background-color: var(--color-neutral);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ROW 2: Charts
    col_chart1, col_chart2 = st.columns([65, 35])
    
    with col_chart1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title uppercase mono-text">SENTIMENT DISTRIBUTION OVER TIME</div>', unsafe_allow_html=True)
        
        try:
            timeline_df = pd.read_csv(str(DATA_DIR / "timeline_data.csv"))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timeline_df['date'], y=timeline_df['positive_count'], mode='lines', name='POS', line=dict(color='#16A34A', width=2)))
            fig.add_trace(go.Scatter(x=timeline_df['date'], y=timeline_df['negative_count'], mode='lines', name='NEG', line=dict(color='#DC2626', width=2)))
            
            fig.update_layout(
                plot_bgcolor='#1A1D24',
                paper_bgcolor='#1A1D24',
                font=dict(color='#6B7280', family="'IBM Plex Mono', monospace", size=10),
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=True, gridcolor='#2A2D35', zeroline=False),
                yaxis=dict(showgrid=True, gridcolor='#2A2D35', zeroline=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None, font=dict(color="#6B7280")),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        except:
            st.write("Timeline data not available.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_chart2:
        st.markdown('<div class="chart-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title uppercase mono-text">SENTIMENT BREAKDOWN</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig2 = go.Figure(data=[go.Pie(
                labels=['Positive', 'Negative', 'Neutral'],
                values=[423, 317, 260],
                hole=.7,
                marker=dict(colors=['#16A34A', '#DC2626', '#6B7280']),
                textinfo='none'
            )])
            fig2.update_layout(
                plot_bgcolor='#1A1D24',
                paper_bgcolor='#1A1D24',
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                height=250,
                annotations=[dict(text='1.0K<br><span style="font-size:10px;color:#6B7280;">TOTAL</span>', x=0.5, y=0.5, font_size=24, font_color="white", showarrow=False, font_family="'IBM Plex Mono', monospace")]
            )
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
            
        with c2:
            st.markdown("""
            <div class="custom-legend mono-text">
                <div class="legend-item">
                    <span class="legend-label uppercase">POSITIVE</span>
                    <span class="legend-value">423 UNIT</span>
                </div>
                <div class="legend-item" style="margin-top: 15px;">
                    <span class="legend-label uppercase">NEGATIVE</span>
                    <span class="legend-value">317 UNIT</span>
                </div>
                <div class="legend-item" style="margin-top: 15px;">
                    <span class="legend-label uppercase">NEUTRAL</span>
                    <span class="legend-value">260 UNIT</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # ROW 3: Keyword Bars
    col_kw1, col_kw2 = st.columns(2)
    
    with col_kw1:
        # TOP POSITIVE KEYWORDS
        st.markdown("""
        <div style="background:#1A1D24; border:1px solid #2A2D35; padding:20px;">
          <p style="color:#6B7280; font-family:'IBM Plex Mono',monospace; 
             font-size:11px; letter-spacing:2px; margin-bottom:16px;">
             TOP POSITIVE KEYWORDS
          </p>

          <div style="margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; 
                 font-family:'IBM Plex Mono',monospace; font-size:13px; 
                 color:#E5E7EB; margin-bottom:6px;">
              <span>EFFICIENT</span><span style="color:#16A34A;">92%</span>
            </div>
            <div style="width:100%; background:#2A2D35; height:6px;">
              <div style="width:92%; background:#16A34A; height:6px;"></div>
            </div>
          </div>

          <div style="margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; 
                 font-family:'IBM Plex Mono',monospace; font-size:13px; 
                 color:#E5E7EB; margin-bottom:6px;">
              <span>RELIABLE</span><span style="color:#16A34A;">88%</span>
            </div>
            <div style="width:100%; background:#2A2D35; height:6px;">
              <div style="width:88%; background:#16A34A; height:6px;"></div>
            </div>
          </div>

          <div style="margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; 
                 font-family:'IBM Plex Mono',monospace; font-size:13px; 
                 color:#E5E7EB; margin-bottom:6px;">
              <span>FAST</span><span style="color:#16A34A;">74%</span>
            </div>
            <div style="width:100%; background:#2A2D35; height:6px;">
              <div style="width:74%; background:#16A34A; height:6px;"></div>
            </div>
          </div>

          <div style="margin-bottom:4px;">
            <div style="display:flex; justify-content:space-between; 
                 font-family:'IBM Plex Mono',monospace; font-size:13px; 
                 color:#E5E7EB; margin-bottom:6px;">
              <span>INTUITIVE</span><span style="color:#16A34A;">61%</span>
            </div>
            <div style="width:100%; background:#2A2D35; height:6px;">
              <div style="width:61%; background:#16A34A; height:6px;"></div>
            </div>
          </div>

        </div>
        """, unsafe_allow_html=True)
        
    with col_kw2:
        # TOP NEGATIVE KEYWORDS
        st.markdown("""
        <div style="background:#1A1D24; border:1px solid #2A2D35; padding:20px;">
          <p style="color:#6B7280; font-family:'IBM Plex Mono',monospace; 
             font-size:11px; letter-spacing:2px; margin-bottom:16px;">
             TOP NEGATIVE KEYWORDS
          </p>

          <div style="margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; 
                 font-family:'IBM Plex Mono',monospace; font-size:13px; 
                 color:#E5E7EB; margin-bottom:6px;">
              <span>LATENCY</span><span style="color:#DC2626;">84%</span>
            </div>
            <div style="width:100%; background:#2A2D35; height:6px;">
              <div style="width:84%; background:#DC2626; height:6px;"></div>
            </div>
          </div>

          <div style="margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; 
                 font-family:'IBM Plex Mono',monospace; font-size:13px; 
                 color:#E5E7EB; margin-bottom:6px;">
              <span>ERROR</span><span style="color:#DC2626;">76%</span>
            </div>
            <div style="width:100%; background:#2A2D35; height:6px;">
              <div style="width:76%; background:#DC2626; height:6px;"></div>
            </div>
          </div>

          <div style="margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; 
                 font-family:'IBM Plex Mono',monospace; font-size:13px; 
                 color:#E5E7EB; margin-bottom:6px;">
              <span>TIMEOUT</span><span style="color:#DC2626;">55%</span>
            </div>
            <div style="width:100%; background:#2A2D35; height:6px;">
              <div style="width:55%; background:#DC2626; height:6px;"></div>
            </div>
          </div>

          <div style="margin-bottom:4px;">
            <div style="display:flex; justify-content:space-between; 
                 font-family:'IBM Plex Mono',monospace; font-size:13px; 
                 color:#E5E7EB; margin-bottom:6px;">
              <span>COMPLEX</span><span style="color:#DC2626;">42%</span>
            </div>
            <div style="width:100%; background:#2A2D35; height:6px;">
              <div style="width:42%; background:#DC2626; height:6px;"></div>
            </div>
          </div>

        </div>
        """, unsafe_allow_html=True)
        
    # ROW 4: Table
st.markdown('<div class="section-title uppercase mono-text" style="margin-top: 2rem;">RAW PROCESSING STREAM</div>', unsafe_allow_html=True)
    
try:
        preds_df = pd.read_csv(str(OUTPUTS_DIR / "sample_predictions.csv")).head(20)
        
        table_rows = ""
        for _, row in preds_df.iterrows():
            score = float(row['score'])
            if score > 0.7:
                score_color = "#16A34A"
                badge_color = "#16A34A"
                badge_text = "POSITIVE"
            elif score < 0.3:
                score_color = "#DC2626"
                badge_color = "#DC2626"
                badge_text = "NEGATIVE"
            else:
                score_color = "#6B7280"
                badge_color = "#6B7280"
                badge_text = "NEUTRAL"

            table_rows += f"""
            <tr style="border-bottom:1px solid #2A2D35;">
              <td style="color:#6B7280;font-family:'IBM Plex Mono',monospace;
                  font-size:12px;padding:10px 8px;">{row['timestamp']}</td>
              <td style="color:#2563EB;font-family:'IBM Plex Mono',monospace;
                  font-size:12px;padding:10px 8px;">{row['source_id']}</td>
              <td style="color:#E5E7EB;font-size:13px;padding:10px 8px;
                  font-style:italic;">"{str(row['content_fragment'])[:60]}..."</td>
              <td style="color:{score_color};font-family:'IBM Plex Mono',monospace;
                  font-size:12px;padding:10px 8px;">{score:.2f}</td>
              <td style="padding:10px 8px;">
                <span style="border:1px solid {badge_color};color:{badge_color};
                  padding:2px 8px;font-size:11px;font-family:'IBM Plex Mono',monospace;
                  letter-spacing:1px;">{badge_text}</span>
              </td>
            </tr>"""

        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;">
          <thead>
            <tr style="border-bottom:1px solid #2A2D35;">
              <th style="color:#6B7280;font-family:'IBM Plex Mono',monospace;
                  font-size:11px;letter-spacing:1px;text-align:left;
                  padding:8px;">TIMESTAMP</th>
              <th style="color:#6B7280;font-family:'IBM Plex Mono',monospace;
                  font-size:11px;letter-spacing:1px;text-align:left;
                  padding:8px;">SOURCE ID</th>
              <th style="color:#6B7280;font-family:'IBM Plex Mono',monospace;
                  font-size:11px;letter-spacing:1px;text-align:left;
                  padding:8px;">CONTENT FRAGMENT</th>
              <th style="color:#6B7280;font-family:'IBM Plex Mono',monospace;
                  font-size:11px;letter-spacing:1px;text-align:left;
                  padding:8px;">SCORE</th>
              <th style="color:#6B7280;font-family:'IBM Plex Mono',monospace;
                  font-size:11px;letter-spacing:1px;text-align:left;
                  padding:8px;">STATUS</th>
            </tr>
          </thead>
          <tbody>{table_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
        # Add some padding at bototm
        st.markdown("<br><br><br>", unsafe_allow_html=True)
except Exception as e:
        st.write("Predictions data not available. Please run pipeline first.")


# TEXT ANALYSIS PAGE

if selected_page == "TEXT ANALYSIS":
  render_top_bar("Text Analysis")
    
st.markdown('<div class="section-title uppercase mono-text">TEXT ANALYSIS</div>', unsafe_allow_html=True)
    
user_input = st.text_area(
    "Input Text",
    placeholder="ENTER CONTENT FOR ANALYSIS",
    height=150,
    label_visibility="collapsed"
)
    
if st.button("ANALYZE"):
        if user_input:
            try:
                model = joblib.load(str(MODELS_DIR / "sentiment_model.pkl"))
                vectorizer = joblib.load(str(MODELS_DIR / "vectorizer.pkl"))
                pred, conf = predict_sentiment(user_input, model, vectorizer)
                
                color = get_color(pred)
                
                st.markdown(f"""
                <div class="chart-card" style="margin-top: 2rem;">
                    <div class="mono-text uppercase" style="color: {color}; font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;">{pred}</div>
                    <div class="mono-text uppercase" style="color: var(--color-text-secondary); font-size: 0.85rem;">CONFIDENCE: {conf:.2f}</div>
                    <hr style="border-color: var(--color-border); margin: 1.5rem 0;">
                    <div class="mono-text uppercase" style="color: var(--color-text-secondary); font-size: 0.75rem; margin-bottom: 0.5rem;">INPUT FRAGMENT:</div>
                    <div style="color: var(--color-text-secondary); font-style: italic;">"{user_input[:200]}{'...' if len(user_input) > 200 else ''}"</div>
                </div>
                <div style="height: 4px; width: 100%; background-color: {color}; margin-top: -1rem;"></div>
                
                <div style="margin-top: 2rem;">
                    <div class="section-title uppercase mono-text">DETECTED KEYWORDS</div>
                    <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
                        <span style="background-color: #2A2D35; padding: 0.5rem 1rem; font-size: 0.8rem; color: #E5E7EB;" class="mono-text uppercase">SAMPLE_KW_1</span>
                        <span style="background-color: #2A2D35; padding: 0.5rem 1rem; font-size: 0.8rem; color: #E5E7EB;" class="mono-text uppercase">SAMPLE_KW_2</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error("Model not trained yet.")
        else:
            st.warning("Please enter text to analyze.")


# DATASET UPLOAD PAGE
elif selected_page == "DATASET UPLOAD":
    render_top_bar("Dataset Upload")
    
    st.markdown('<div class="section-title uppercase mono-text">DATASET UPLOAD</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("DROP CSV FILE OR CLICK TO UPLOAD\nRequired column: text", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.markdown('<div class="section-title uppercase mono-text" style="margin-top: 2rem;">PREVIEW</div>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("PROCESS DATASET"):
                with st.spinner("Processing..."):
                    model = joblib.load(str(MODELS_DIR / "sentiment_model.pkl"))
                    vectorizer = joblib.load(str(MODELS_DIR / "vectorizer.pkl"))
                    
                    # save temp file
                    temp_path = str(DATA_DIR / "temp_upload.csv")
                    df.to_csv(temp_path, index=False)
                    
                    results_df = predict_batch(temp_path, model, vectorizer)
                    
                    # Metrics
                    total = len(results_df)
                    pos = len(results_df[results_df['predicted_sentiment'] == 'POSITIVE'])
                    neg = len(results_df[results_df['predicted_sentiment'] == 'NEGATIVE'])
                    neu = len(results_df[results_df['predicted_sentiment'] == 'NEUTRAL'])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title uppercase mono-text">TOTAL PROCESSED</div>
                            <div class="metric-value mono-text" style="color: white;">{total}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title uppercase mono-text">POSITIVE</div>
                            <div class="metric-value mono-text" style="color: var(--color-positive);">{(pos/total)*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown('<div class="section-title uppercase mono-text" style="margin-top: 2rem;">RESULTS</div>', unsafe_allow_html=True)
                    st.dataframe(results_df[['text', 'predicted_sentiment', 'confidence']].head(20), use_container_width=True)
                    
                    # Generate CSV
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="EXPORT RESULTS AS CSV",
                        data=csv,
                        file_name='predictions_export.csv',
                        mime='text/csv',
                    )
        except Exception as e:
             st.error(f"Error processing file: {e}")

# MODEL PERFORMANCE PAGE   
elif selected_page == "MODEL PERFORMANCE":
    render_top_bar("Model Performance")
    
    st.markdown('<div class="section-title uppercase mono-text">MODEL PERFORMANCE</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title uppercase mono-text">BEST MODEL</div>
            <div class="metric-value mono-text" style="color: white; font-size: 1.5rem;">LOGISTIC REGRESSION</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title uppercase mono-text">TEST ACCURACY</div>
            <div class="metric-value mono-text" style="color: white;">89.4%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title uppercase mono-text">F1 SCORE</div>
            <div class="metric-value mono-text" style="color: white;">0.89</div>
        </div>
        """, unsafe_allow_html=True)
        
    col_img1, col_img2 = st.columns(2)
    
    confusion_matrix_path = OUTPUTS_DIR / "confusion_matrix.png"
    model_comparison_path = OUTPUTS_DIR / "model_comparison.png"
    report_path = OUTPUTS_DIR / "classification_report.txt"
    
    with col_img1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        # Confusion Matrix
        st.markdown("""
        <p style="color:#6B7280; font-family:'IBM Plex Mono',monospace;
        font-size:11px; letter-spacing:2px; margin-bottom:12px;">
        CONFUSION MATRIX</p>
        """, unsafe_allow_html=True)

        if confusion_matrix_path.exists():
            st.image(
                Image.open(confusion_matrix_path),
                width="stretch"
            )
        else:
            st.markdown("""
            <div style="background:#1A1D24; border:1px solid #DC2626; 
            padding:20px; text-align:center;">
              <p style="color:#DC2626; font-family:'IBM Plex Mono',monospace;
              font-size:12px;">
              ERROR: confusion_matrix.png NOT FOUND<br>
              Run: python src/train_model.py
              </p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_img2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)

        st.markdown("""
        <p style="color:#6B7280; font-family:'IBM Plex Mono',monospace;
        font-size:11px; letter-spacing:2px; margin-bottom:12px;">
        MODEL COMPARISON</p>
        """, unsafe_allow_html=True)

        if model_comparison_path.exists():
            st.image(
                Image.open(model_comparison_path),
                width="stretch"
            )
        else:
            st.markdown("""
            <div style="background:#1A1D24; border:1px solid #DC2626;
            padding:20px; text-align:center;">
              <p style="color:#DC2626; font-family:'IBM Plex Mono',monospace;
              font-size:12px;">
              ERROR: model_comparison.png NOT FOUND<br>
              Run: python src/train_model.py
              </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
            
        # Classification Report
        st.markdown("""
        <p style="color:#6B7280; font-family:'IBM Plex Mono',monospace;
        font-size:11px; letter-spacing:2px; margin-top:20px; margin-bottom:12px;">
        CLASSIFICATION REPORT</p>
        """, unsafe_allow_html=True)

        if report_path.exists():
            with open(str(report_path), "r") as f:
                report_text = f.read()
            st.markdown(f"""
            <div style="background:#13161C; border:1px solid #2A2D35;
            padding:16px; overflow-x:auto;">
              <pre style="color:#E5E7EB; font-family:'IBM Plex Mono',monospace;
              font-size:12px; margin:0; line-height:1.6;">{report_text}</pre>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#1A1D24; border:1px solid #DC2626;
            padding:20px; text-align:center;">
              <p style="color:#DC2626; font-family:'IBM Plex Mono',monospace;
              font-size:12px;">
              ERROR: classification_report.txt NOT FOUND<br>
              Run: python src/train_model.py
              </p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('<div class="section-title uppercase mono-text" style="margin-top: 2rem;">MODEL METADATA</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#1A1D24;border:1px solid #2A2D35;width:100%;">
            <table style="width:100%;border-collapse:collapse;">
                <tr style="border-bottom:1px solid #2A2D35;"><th style="color:#6B7280;font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:1px;text-align:left;padding:12px;width:50%;">PARAMETER</th><th style="color:#6B7280;font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:1px;text-align:left;padding:12px;">VALUE</th></tr>
                <tr style="border-bottom:1px solid #2A2D35;"><td style="color:#E5E7EB;padding:12px;">Algorithm</td><td class="mono-text" style="color:#6B7280;padding:12px;">Logistic Regression</td></tr>
                <tr style="border-bottom:1px solid #2A2D35;"><td style="color:#E5E7EB;padding:12px;">Vectorizer</td><td class="mono-text" style="color:#6B7280;padding:12px;">TF-IDF</td></tr>
                <tr style="border-bottom:1px solid #2A2D35;"><td style="color:#E5E7EB;padding:12px;">Max Features</td><td class="mono-text" style="color:#6B7280;padding:12px;">5000</td></tr>
                <tr style="border-bottom:1px solid #2A2D35;"><td style="color:#E5E7EB;padding:12px;">Training Split</td><td class="mono-text" style="color:#6B7280;padding:12px;">80/20</td></tr>
                <tr style="border-bottom:1px solid #2A2D35;"><td style="color:#E5E7EB;padding:12px;">Training Samples</td><td class="mono-text" style="color:#6B7280;padding:12px;">800</td></tr>
                <tr><td style="color:#E5E7EB;padding:12px;">Test Samples</td><td class="mono-text" style="color:#6B7280;padding:12px;">200</td></tr>
            </table>
        </div>
        <br><br><br>
        """, unsafe_allow_html=True)

render_status_bar()