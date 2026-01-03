import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pair Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# UI 커스텀 스타일
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("Pair Trading Dashboard")
st.markdown("### Cointegration & Rolling Z-Score Strategy")

# ---------------------------------------------------------
# 2. 사이드바
# ---------------------------------------------------------
with st.sidebar:
    st.header("Strategy Settings")
    
    st.subheader("Time Horizon")
    window_size = st.slider("Rolling Window (Days)", 20, 120, 60)
    
    st.subheader("Signal Threshold")
    z_threshold = st.slider("Z-Score Threshold", 1.5, 3.0, 2.0, step=0.1)
    
    st.subheader("Stat Filter")
    p_cutoff = st.slider("Max P-value", 0.01, 0.20, 0.10)
    
    st.divider()
    run_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)
    st.caption("Data Source: Yahoo Finance")

# ---------------------------------------------------------
# 3. 데이터 로딩 (Yahoo Finance)
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
# 4. 분석 로직
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
    
    spread = row['Spread']
    z_score = (spread - row['Mean']) / row['Std']
    
    pa = df_prices[sa]
    pb = df_prices[sb]
    pa_norm = (pa / pa.iloc[0]) * 100
    pb_norm = (pb / pb.iloc[0]) * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1,
                        subplot_titles=(f"Price Comparison (Base=100)", f"Rolling Z-Score (Window={window})"),
                        row_heights=[0.6, 0.4])

    # 상단: 주가 비교
    fig.add_trace(go.Scatter(x=pa_norm.index, y=pa_norm, name=sa, line=dict(color='#1f77b4')), row=1, col=1)
    fig.add_trace(go.Scatter(x=pb_norm.index, y=pb_norm, name=sb, line=dict(color='#ff7f0e')), row=1, col=1)

    # 하단: Z-Score
    fig.add_trace(go.Scatter(x=z_score.index, y=z_score, name='Z-Score', line=dict(color='#9467bd')), row=2, col=1)
    
    # 임계값 라인
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Sell Threshold", row=2, col=1)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="blue", annotation_text="Buy Threshold", row=2, col=1)
    fig.add_hline(y=0, line_color="black", line_width=0.5, row=2, col=1)

    fig.update_layout(height=600, hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    return fig

# ---------------------------------------------------------
# 6. 메인 실행
# ---------------------------------------------------------
if run_btn:
    with st.spinner('Fetching Data & Crunching Numbers... 🤖'):
        df_prices = load_stock_data()
        
        if df_prices.empty:
            st.error("Data Load Failed (Yahoo Finance). Please try again.")
        else:
            results = analyze_data(df_prices, window_size, z_threshold, p_cutoff)
            
            if not results.empty:
                action_items = results[results['Status'] != 'Watch']
                best_opp = results.loc[results['Z-Score'].abs().idxmax()]
                
                # --- KPI Metrics (상단 요약) ---
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Pairs Analyzed", f"{len(df_prices.columns)*(len(df_prices.columns)-1)//2}")
                col2.metric("Cointegrated", f"{len(results)} Pairs")
                col3.metric("Action Signals", f"{len(action_items)}", delta="Trading Opps", delta_color="normal")
                col4.metric("Top Opportunity", f"{best_opp['Z-Score']:.2f} σ", f"{best_opp['Stock A']} - {best_opp['Stock B']}")
                
                st.markdown("---")

                # --- 3개의 탭 구조 ---
                tab1, tab2, tab3 = st.tabs(["🔥 Action Required", "👀 Watchlist", "📋 Full List"])

                # Tab 1: 즉시 진입 (Action)
                with tab1:
                    if not action_items.empty:
                        st.success(f"현재 {len(action_items)}개의 종목이 진입 임계값(Threshold {z_threshold})을 초과했습니다.")
                        
                        # Z-Score 절대값 순 정렬
                        for idx, row in action_items.sort_values(by='Z-Score', key=abs, ascending=False).iterrows():
                            color = "red" if row['Z-Score'] > 0 else "blue"
                            with st.expander(f"**:{color}[{row['Status']}]** | {row['Stock A']} vs {row['Stock B']} (Z: {row['Z-Score']:.2f})", expanded=True):
                                st.plotly_chart(plot_interactive(row, df_prices, window_size, z_threshold), use_container_width=True)
                    else:
                        st.info("현재 진입 조건(Threshold)을 만족하는 페어가 없습니다. (Market is efficient)")

                # Tab 2: 관심 종목 (Watchlist)
                with tab2:
                    watch_df = results[results['Status'] == 'Watch'].sort_values(by='P-value')
                    
                    if not watch_df.empty:
                        col_a, col_b = st.columns([1, 2])
                        
                        with col_a:
                            st.markdown("##### Select Pair to Analyze")
                            # 선택 상자
                            options = watch_df.apply(lambda x: f"{x['Stock A']} - {x['Stock B']} (P: {x['P-value']:.4f})", axis=1)
                            selected_option = st.selectbox("Choose a pair:", options.index, format_func=lambda x: options[x])
                            
                            st.info("P-value가 낮을수록 공적분(장기적 연관성)이 강한 페어입니다.")
                        
                        with col_b:
                            if selected_option is not None:
                                row = watch_df.loc[selected_option]
                                st.plotly_chart(plot_interactive(row, df_prices, window_size, z_threshold), use_container_width=True)
                    else:
                        st.warning("관심 종목(Watch) 리스트가 비어있습니다.")

                # Tab 3: 전체 리스트 (Table)
                with tab3:
                    st.markdown("##### All Cointegrated Pairs Table")
                    display_df = results[['Stock A', 'Stock B', 'Z-Score', 'P-value', 'Corr', 'Status']].sort_values(by='P-value')
                    
                    st.dataframe(
                        display_df.style.background_gradient(subset=['Z-Score'], cmap='RdBu_r', vmin=-3, vmax=3)
                                        .format({'Z-Score': '{:.2f}', 'P-value': '{:.4f}', 'Corr': '{:.2f}'}),
                        use_container_width=True,
                        height=600
                    )

            else:
                st.warning("No pairs found with current settings. Try increasing Max P-value.")
else:
    st.info("👈 사이드바에서 설정을 확인하고 [RUN ANALYSIS] 버튼을 눌러주세요.")
