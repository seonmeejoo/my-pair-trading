import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. 페이지 설정 (Wide Mode & Title)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pro Quant Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS로 UI 조금 더 예쁘게 다듬기
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Pro Quant Pair Trading Dashboard")
st.markdown("### Cointegration & Rolling Z-Score Strategy")

# ---------------------------------------------------------
# 2. 사이드바 설정
# ---------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Strategy Settings")
    
    st.subheader("Time Horizon")
    window_size = st.slider("Rolling Window (Days)", 20, 120, 60, help="이동평균을 계산할 과거 기간입니다.")
    
    st.subheader("Signal Threshold")
    z_threshold = st.slider("Z-Score Threshold", 1.5, 3.0, 2.0, step=0.1, help="진입 신호를 발생시킬 표준편차 임계값입니다.")
    
    st.subheader("Stat Filter")
    p_cutoff = st.slider("Max P-value", 0.01, 0.20, 0.10, help="공적분 검정 통과 기준 (낮을수록 엄격)")
    
    st.divider()
    run_btn = st.button("RUN ANALYSIS 🚀", type="primary", use_container_width=True)
    st.caption("Data Source: Yahoo Finance")

# ---------------------------------------------------------
# 3. 데이터 로딩 (캐싱 & Yahoo Finance)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_stock_data():
    manual_tickers = {
        '005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '035420.KS': 'NAVER', '035720.KS': '카카오',
        '373220.KS': 'LG에너지솔루션', '006400.KS': '삼성SDI', '051910.KS': 'LG화학', '005490.KS': 'POSCO홀딩스',
        '005380.KS': '현대차', '000270.KS': '기아', '003490.KS': '대한항공', '011200.KS': 'HMM',
        '105560.KS': 'KB금융', '055550.KS': '신한지주', '086790.KS': '하나금융지주', '323410.KS': '카카오뱅크',
        '207940.KS': '삼성바이오로직스', '068270.KS': '셀트리온', '000100.KS': '유한양행', '128940.KS': '한미약품',
        '015760.KS': '한국전력', '033780.KS': 'KT&G', '097950.KS': 'CJ제일제당', '032640.KS': 'LG유플러스',
        '259960.KS': '크래프톤', '009150.KS': '삼성전기', '018260.KS': '삼성SDS', '010130.KS': '고려아연',
        '012330.KS': '현대모비스', '096770.KS': 'SK이노베이션', '011070.KS': 'LG이노텍', '003550.KS': 'LG',
        '032830.KS': '삼성생명', '000810.KS': '삼성화재', '017670.KS': 'SK텔레콤', '030200.KS': 'KT'
    }
    
    tickers_list = list(manual_tickers.keys())
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        df_all = yf.download(tickers_list, start=start_date, end=end_date, progress=False)['Close']
        df_all = df_all.rename(columns=manual_tickers)
        df_all = df_all.fillna(method='ffill').dropna(axis=1)
        return df_all
    except Exception as e:
        return pd.DataFrame()

# ---------------------------------------------------------
# 4. 분석 로직 (최적화)
# ---------------------------------------------------------
@st.cache_data
def analyze_data(df_prices, window, threshold, p_cutoff):
    pairs = []
    cols = df_prices.columns
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            stock_a = cols[i]
            stock_b = cols[j]
            
            try:
                score, pvalue, _ = coint(df_prices[stock_a], df_prices[stock_b])
                
                if pvalue < p_cutoff:
                    log_a = np.log(df_prices[stock_a])
                    log_b = np.log(df_prices[stock_b])
                    spread = log_a - log_b
                    
                    rolling_mean = spread.rolling(window=window).mean()
                    rolling_std = spread.rolling(window=window).std()
                    rolling_z = (spread - rolling_mean) / rolling_std
                    
                    current_z = rolling_z.iloc[-1]
                    corr = df_prices[stock_a].corr(df_prices[stock_b])

                    if not np.isnan(current_z):
                        status = "Watch"
                        if current_z < -threshold: status = "Buy A / Sell B"
                        elif current_z > threshold: status = "Sell A / Buy B"
                        
                        pairs.append({
                            'Stock A': stock_a, 'Stock B': stock_b,
                            'Corr': corr, 'P-value': pvalue,
                            'Z-Score': current_z, 'Status': status,
                            'Spread': spread, 'Mean': rolling_mean, 'Std': rolling_std
                        })
            except: continue
    return pd.DataFrame(pairs)

# ---------------------------------------------------------
# 5. 인터랙티브 차트 (Plotly)
# ---------------------------------------------------------
def plot_interactive(row, df_prices, window, threshold):
    sa, sb = row['Stock A'], row['Stock B']
    
    # 데이터 준비
    spread = row['Spread']
    z_score = (spread - row['Mean']) / row['Std']
    
    pa = df_prices[sa]
    pb = df_prices[sb]
    pa_norm = (pa / pa.iloc[0]) * 100
    pb_norm = (pb / pb.iloc[0]) * 100

    # Subplots 생성 (2행 1열)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1,
                        subplot_titles=(f"Price Comparison (Base=100)", f"Rolling Z-Score (Window={window})"),
                        row_heights=[0.6, 0.4])

    # [상단] 주가 비교
    fig.add_trace(go.Scatter(x=pa_norm.index, y=pa_norm, name=sa, line=dict(color='#1f77b4')), row=1, col=1)
    fig.add_trace(go.Scatter(x=pb_norm.index, y=pb_norm, name=sb, line=dict(color='#ff7f0e')), row=1, col=1)

    # [하단] Z-Score
    fig.add_trace(go.Scatter(x=z_score.index, y=z_score, name='Z-Score', line=dict(color='#9467bd')), row=2, col=1)
    
    # 임계값 라인
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Sell Threshold", row=2, col=1)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="blue", annotation_text="Buy Threshold", row=2, col=1)
    fig.add_hline(y=0, line_color="black", line_width=0.5, row=2, col=1)

    # 레이아웃 설정
    fig.update_layout(
        height=600, 
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# ---------------------------------------------------------
# 6. 메인 실행 (Dashboard Layout)
# ---------------------------------------------------------
if run_btn:
    with st.spinner('Fetching Data & Crunching Numbers... 🤖'):
        df_prices = load_stock_data()
        
        if df_prices.empty:
            st.error("Data Load Failed. Please try again.")
        else:
            results = analyze_data(df_prices, window_size, z_threshold, p_cutoff)
            
            # --- 1. KPI Metrics Section ---
            if not results.empty:
                action_items = results[results['Status'] != 'Watch']
                best_opp = results.loc[results['Z-Score'].abs().idxmax()] if not results.empty else None
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Pairs Analyzed", f"{len(df_prices.columns)*(len(df_prices.columns)-1)//2}")
                col2.metric("Cointegrated Pairs", f"{len(results)}", delta="Candidates")
                col3.metric("Actionable Opportunities", f"{len(action_items)}", delta="Signal Fired", delta_color="normal")
                
                if best_opp is not None:
                    col4.metric("Top Opportunity", f"{best_opp['Z-Score']:.2f} σ", f"{best_opp['Stock A']} - {best_opp['Stock B']}")
                
                st.markdown("---")

                # --- 2. Actionable Items Section ---
                if not action_items.empty:
                    st.subheader("🔥 Action Required (Trading Signals)")
                    
                    # 탭 대신 Expander로 깔끔하게 정리
                    for idx, row in action_items.sort_values(by='Z-Score', key=abs, ascending=False).iterrows():
                        color = "red" if row['Z-Score'] > 0 else "blue"
                        with st.expander(f"**:{color}[{row['Status']}]** | {row['Stock A']} vs {row['Stock B']} (Z: {row['Z-Score']:.2f})", expanded=True):
                            st.plotly_chart(plot_interactive(row, df_prices, window_size, z_threshold), use_container_width=True)
                else:
                    st.success("Currently no pairs exceed the Z-Score threshold. Market is efficient! 🧘")

                # --- 3. Watchlist Table Section ---
                st.markdown("---")
                st.subheader("📋 Full Watchlist (Cointegrated Pairs)")
                
                # 데이터프레임 스타일링 (색상 입히기)
                display_df = results[['Stock A', 'Stock B', 'Z-Score', 'P-value', 'Corr', 'Status']].sort_values(by='P-value')
                
                st.dataframe(
                    display_df.style.background_gradient(subset=['Z-Score'], cmap='RdBu_r', vmin=-3, vmax=3)
                                    .format({'Z-Score': '{:.2f}', 'P-value': '{:.4f}', 'Corr': '{:.2f}'}),
                    use_container_width=True,
                    height=400
                )

            else:
                st.warning("No cointegrated pairs found with current settings. Try relaxing the P-value or Window.")

else:
    # 초기 화면 (Empty State)
    st.info("👈 Please adjust settings in the sidebar and click 'RUN ANALYSIS' to start.")
    
    # 예시 이미지나 설명 추가 가능
    st.markdown("""
    ### How to use this dashboard:
    1. **Set Parameters:** Choose your lookback window and Z-score threshold.
    2. **Run Analysis:** The algorithm checks for cointegration among Top Korean Stocks.
    3. **Trade:** Look for 'Action Required' signals where the spread diverges significantly.
    """)
