import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import time
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 0. 한글 폰트 설정 (Streamlit Cloud 대응)
# ---------------------------------------------------------
def set_korean_font():
    # 리눅스(Streamlit Cloud) vs 윈도우(Local) 자동 구분
    if os.name == 'posix':
        plt.rcParams['font.family'] = 'NanumGothic'
    else:
        plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# ---------------------------------------------------------
# 1. 페이지 및 사이드바 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Quant Pair Trading", layout="wide")
st.title("📈 실전 퀀트 페어 트레이딩 (Rolling Z-Score)")

with st.sidebar:
    st.header("⚙️ 파라미터 설정")
    window_size = st.slider("이동평균 기간 (Window)", 30, 120, 60)
    z_threshold = st.slider("진입 임계값 (Z-Score)", 1.5, 3.0, 2.0)
    run_btn = st.button("🚀 분석 시작", type="primary")

# ---------------------------------------------------------
# 2. 데이터 로딩 함수 (캐싱 적용)
# ---------------------------------------------------------
@st.cache_data(ttl=3600) # 1시간 동안 데이터 저장
def load_data():
    manual_tickers = {
        # [반도체/IT]
        '005930': '삼성전자', '000660': 'SK하이닉스', '042700': '한미반도체', '403870': 'HPSP',
        '000990': 'DB하이텍', '011070': 'LG이노텍', '009150': '삼성전기', '035420': 'NAVER',
        '035720': '카카오', '018260': '삼성SDS', '259960': '크래프톤', '377300': '카카오페이',

        # [2차전지/화학]
        '373220': 'LG에너지솔루션', '006400': '삼성SDI', '051910': 'LG화학', '096770': 'SK이노베이션',
        '003670': '포스코퓨처엠', '247540': '에코프로비엠', '086520': '에코프로', '066970': '엘앤에프',
        '005490': 'POSCO홀딩스', '010130': '고려아연', '051900': 'LG생활건강', '090430': '아모레퍼시픽',
        '010950': 'S-Oil', '009830': '한화솔루션', '011780': '금호석유', '278280': '천보',

        # [자동차/운송/기계/조선]
        '005380': '현대차', '000270': '기아', '012330': '현대모비스', '086280': '현대글로비스',
        '003490': '대한항공', '011200': 'HMM', '028670': '팬오션', '010120': 'LS ELECTRIC',
        '034020': '두산에너빌리티', '329180': 'HD현대중공업', '009540': 'HD한국조선해양', '042660': '한화오션',
        '012450': '한화에어로스페이스', '047810': '한국항공우주', '079550': 'LIG넥스원', '267250': 'HD현대일렉트릭',

        # [금융/지주]
        '105560': 'KB금융', '055550': '신한지주', '086790': '하나금융지주', '316140': '우리금융지주',
        '323410': '카카오뱅크', '024110': '기업은행', '071050': '한국금융지주', '000810': '삼성화재',
        '003550': 'LG', '000830': '삼성물산', '034730': 'SK', '000150': '두산',

        # [바이오/헬스케어]
        '207940': '삼성바이오로직스', '068270': '셀트리온', '000100': '유한양행', '128940': '한미약품',
        '196170': '알테오젠', '028300': 'HLB', '214150': '클래시스', '145020': '휴젤',
        '326030': 'SK바이오팜', '302440': 'SK바이오사이언스',

        # [유틸리티/기타]
        '015760': '한국전력', '017670': 'SK텔레콤', '030200': 'KT', '032640': 'LG유플러스',
        '033780': 'KT&G', '352820': '하이브', '035900': 'JYP Ent.', '041510': '에스엠',
        '097950': 'CJ제일제당', '021240': '코웨이', '004370': '농심', '007310': '오뚜기'
    }
    
    df_target = pd.DataFrame(list(manual_tickers.items()), columns=['Code', 'Name'])
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    price_data = {}
    
    # Progress Bar 설정
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(df_target)
    
    for idx, row in df_target.iterrows():
        try:
            df = fdr.DataReader(row['Code'], start_date, end_date)
            if len(df) > 150:
                price_data[row['Name']] = df['Close']
            time.sleep(0.01) # 차단 방지용 딜레이
        except:
            continue
            
        # 진행률 업데이트
        percent = int((idx + 1) / total * 100)
        status_text.text(f"데이터 수집 중: {row['Name']} ({percent}%)")
        progress_bar.progress(percent)
    
    status_text.empty()
    progress_bar.empty()
    
    return pd.DataFrame(price_data).dropna(axis=1)

# ---------------------------------------------------------
# 3. 분석 로직 (캐싱 적용)
# ---------------------------------------------------------
@st.cache_data
def analyze_data(df_prices, window, threshold):
    pairs = []
    cols = df_prices.columns
    total_checks = len(cols) * (len(cols) - 1) // 2
    
    status_text = st.empty()
    status_text.info(f"총 {total_checks}개 조합에 대해 공적분 및 Rolling Z-Score 분석 중...")
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            stock_a = cols[i]
            stock_b = cols[j]

            try:
                # 1. 공적분 검정
                score, pvalue, _ = coint(df_prices[stock_a], df_prices[stock_b])
                
                if pvalue < 0.05:
                    corr = df_prices[stock_a].corr(df_prices[stock_b])
                    
                    # 2. Rolling Z-Score Calculation
                    log_a = np.log(df_prices[stock_a])
                    log_b = np.log(df_prices[stock_b])
                    spread = log_a - log_b
                    
                    rolling_mean = spread.rolling(window=window).mean()
                    rolling_std = spread.rolling(window=window).std()
                    rolling_z = (spread - rolling_mean) / rolling_std
                    
                    current_z = rolling_z.iloc[-1]
                    
                    status = "Watch"
                    signal = "대기"
                    
                    if current_z < -threshold: 
                        status = "Action"
                        signal = f"매수: {stock_a} / 매도: {stock_b}"
                    elif current_z > threshold: 
                        status = "Action"
                        signal = f"매도: {stock_a} / 매수: {stock_b}"

                    if not np.isnan(current_z):
                        pairs.append({
                            'Stock A': stock_a, 'Stock B': stock_b,
                            'Corr': corr, 'P-value': pvalue,
                            'Z-Score': current_z, 'Signal': signal, 'Status': status,
                            'Spread_Series': spread,
                            'Rolling_Mean': rolling_mean,
                            'Rolling_Std': rolling_std
                        })
            except: continue
            
    status_text.empty()
    return pd.DataFrame(pairs)

# ---------------------------------------------------------
# 4. 차트 그리기 함수 (Figure 반환)
# ---------------------------------------------------------
def create_chart(row, df_prices, window, threshold):
    sa, sb = row['Stock A'], row['Stock B']
    spread = row['Spread_Series']
    mean = row['Rolling_Mean']
    std = row['Rolling_Std']
    z_series = (spread - mean) / std
    
    # Figure 객체 생성 (중요: plt.show() 대신 fig를 반환해야 함)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    # [상단] 주가 흐름
    pa = df_prices[sa]
    pb = df_prices[sb]
    pa_norm = (pa / pa.iloc[0]) * 100
    pb_norm = (pb / pb.iloc[0]) * 100
    
    ax1.set_title(f"[{sa} vs {sb}] 주가 흐름 (Corr: {row['Corr']:.2f}, P-val: {row['P-value']:.4f})", fontweight='bold')
    ax1.plot(pa_norm.index, pa_norm, label=sa, color='tab:blue', lw=1.5)
    ax1.plot(pb_norm.index, pb_norm, label=sb, color='tab:orange', lw=1.5)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("수익률 (Base=100)")
    
    # [하단] Rolling Z-Score
    current_z = row['Z-Score']
    ax2.set_title(f"Rolling Z-Score (Window={window}일) | 현재: {current_z:.2f}")
    ax2.plot(z_series.index, z_series, color='purple', lw=1)
    
    ax2.axhline(threshold, color='red', ls='--', label='매도 진입')
    ax2.axhline(-threshold, color='blue', ls='--', label='매수 진입')
    ax2.axhline(0, color='black', alpha=0.5)
    
    ax2.fill_between(z_series.index, threshold, z_series, where=(z_series >= threshold), facecolor='red', alpha=0.3)
    ax2.fill_between(z_series.index, -threshold, z_series, where=(z_series <= -threshold), facecolor='blue', alpha=0.3)
    
    ax2.set_ylim(-4, 4)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# 5. 메인 실행 로직
# ---------------------------------------------------------
if run_btn:
    st.write("### 1. 데이터 수집 시작")
    df_prices = load_data()
    st.success(f"데이터 수집 완료! 총 {len(df_prices.columns)}개 종목 분석 가능")
    
    st.write("### 2. 알고리즘 분석 시작 (Cointegration & Rolling Z-Score)")
    df_result = analyze_data(df_prices, window_size, z_threshold)
    
    if not df_result.empty:
        st.success(f"분석 완료! 총 {len(df_result)}개의 유의미한 페어를 발견했습니다.")
        
        # 탭으로 결과 구분
        tab1, tab2, tab3 = st.tabs(["🔥 즉시 진입 (Action)", "👀 관심 종목 (Watch)", "📋 전체 리스트"])
        
        # Tab 1: Action
        with tab1:
            df_action = df_result[df_result['Status'] == 'Action'].sort_values(by='Z-Score', key=abs, ascending=False)
            if not df_action.empty:
                st.markdown(f"**총 {len(df_action)}개의 진입 기회가 포착되었습니다.**")
                for idx, row in df_action.iterrows():
                    with st.expander(f"{row['Stock A']} & {row['Stock B']} | {row['Signal']} (Z: {row['Z-Score']:.2f})", expanded=True):
                        # 차트 그리기
                        fig = create_chart(row, df_prices, window_size, z_threshold)
                        st.pyplot(fig)
            else:
                st.info("현재 진입 조건(Threshold 초과)을 만족하는 종목이 없습니다.")

        # Tab 2: Watch
        with tab2:
            df_watch = df_result[df_result['Status'] == 'Watch'].sort_values(by='P-value')
            st.markdown(f"**총 {len(df_watch)}개의 관망 페어 (P-value 낮은 순)**")
            
            # 선택 박스로 차트 보기
            options = df_watch.apply(lambda x: f"{x['Stock A']} - {x['Stock B']} (P: {x['P-value']:.4f})", axis=1)
            selected_option = st.selectbox("차트를 확인할 페어를 선택하세요:", options.index, format_func=lambda x: options[x])
            
            if selected_option is not None:
                row = df_watch.loc[selected_option]
                fig = create_chart(row, df_prices, window_size, z_threshold)
                st.pyplot(fig)

        # Tab 3: Dataframe
        with tab3:
            st.dataframe(df_result[['Stock A', 'Stock B', 'Signal', 'Z-Score', 'P-value', 'Corr', 'Status']])
            
    else:
        st.warning("분석 결과, 공적분 조건(P-value < 0.05)을 만족하는 페어가 없습니다.")

else:
    st.info("👈 왼쪽 사이드바에서 '분석 시작' 버튼을 눌러주세요.")
