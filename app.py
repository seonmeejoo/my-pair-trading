!pip install pykrx

import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import koreanize_matplotlib
import warnings
import time

warnings.filterwarnings('ignore')

# ==========================================
# 1. 설정: "실전용" 우량주 Top 100 리스트 로딩
# ==========================================
print("1. 종목 리스트를 로딩합니다 (실전 우량주 100선)...")

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
print(f"-> 분석 대상: 총 {len(df_target)}개 핵심 우량주")

# ==========================================
# 2. 주가 데이터 수집
# ==========================================
name_to_code = df_target.set_index('Name')['Code'].to_dict()
end_date = datetime.now()
start_date = end_date - timedelta(days=365) # 1년치 데이터

print(f"2. 주가 데이터를 수집합니다... (약 1분 소요)")
price_data = {}
count = 0

for idx, row in df_target.iterrows():
    try:
        df = fdr.DataReader(row['Code'], start_date, end_date)
        if len(df) > 150:
            price_data[row['Name']] = df['Close']
        time.sleep(0.02)
        count += 1
        if count % 20 == 0: print(f"   ...{count}개 완료")
    except: continue

df_prices = pd.DataFrame(price_data).dropna(axis=1)
print(f"-> 수집 완료 (유효 종목: {len(df_prices.columns)}개)")

# ==========================================
# 3. [핵심] 실전형 페어 트레이딩 분석 로직 (Cointegration First)
# ==========================================
print("3. Rolling Window(60일) 기반 분석을 시작합니다...")
print("   (기준 변경: 상관계수 무시 -> 공적분(Cointegration) 우선 필터링)")

pairs = []
cols = df_prices.columns
window_size = 60 

# 진행률 표시를 위해
total_checks = len(cols) * (len(cols) - 1) // 2
print(f"   (총 {total_checks}개 조합 검증 중...)")

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        stock_a = cols[i]
        stock_b = cols[j]

        try:
            # 1. [변경됨] 공적분 검정(Cointegration)을 가장 먼저 수행!
            # P-value가 0.05 미만이면 '장기적 균형 관계'가 있다고 판단
            score, pvalue, _ = coint(df_prices[stock_a], df_prices[stock_b])
            
            if pvalue < 0.05:
                # 2. 상관계수는 필터링 조건이 아니라 '참조용'으로 계산
                corr = df_prices[stock_a].corr(df_prices[stock_b])
                
                # ========================================================
                # 🚀 Rolling Z-Score 계산 (Look-ahead Bias 제거)
                # ========================================================
                
                # (1) 로그 가격
                log_a = np.log(df_prices[stock_a])
                log_b = np.log(df_prices[stock_b])
                
                # (2) 로그 스프레드
                spread = log_a - log_b
                
                # (3) 이동평균(Rolling)
                rolling_mean = spread.rolling(window=window_size).mean()
                rolling_std = spread.rolling(window=window_size).std()
                
                # (4) Z-Score
                rolling_z = (spread - rolling_mean) / rolling_std
                
                current_z = rolling_z.iloc[-1]

                status = "Watch"
                signal = "대기"
                
                if current_z < -2.0: 
                    status = "Action"
                    signal = f"🔥매수: {stock_a} / 매도: {stock_b}"
                elif current_z > 2.0: 
                    status = "Action"
                    signal = f"🔥매도: {stock_a} / 매수: {stock_b}"

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

# ==========================================
# 4. 결과 시각화
# ==========================================
df_result = pd.DataFrame(pairs)

def plot_advanced_pair(row):
    sa, sb = row['Stock A'], row['Stock B']
    
    spread = row['Spread_Series']
    mean = row['Rolling_Mean']
    std = row['Rolling_Std']
    z_series = (spread - mean) / std
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    # [상단] 주가 흐름
    pa = df_prices[sa]
    pb = df_prices[sb]
    pa_norm = (pa / pa.iloc[0]) * 100
    pb_norm = (pb / pb.iloc[0]) * 100
    
    ax1.set_title(f"[{sa} vs {sb}] 주가 흐름 (Corr: {row['Corr']:.2f}, P-val: {row['P-value']:.4f})", fontsize=14, fontweight='bold')
    ax1.plot(pa_norm.index, pa_norm, label=sa, color='tab:blue', lw=1.5)
    ax1.plot(pb_norm.index, pb_norm, label=sb, color='tab:orange', lw=1.5)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("수익률 (Base=100)")
    
    # [하단] Rolling Z-Score
    current_z = row['Z-Score']
    ax2.set_title(f"Rolling Z-Score (Window=60일) | 현재: {current_z:.2f}", fontsize=12)
    ax2.plot(z_series.index, z_series, color='purple', lw=1)
    
    ax2.axhline(2.0, color='red', ls='--', label='매도 진입')
    ax2.axhline(-2.0, color='blue', ls='--', label='매수 진입')
    ax2.axhline(0, color='black', alpha=0.5)
    
    ax2.fill_between(z_series.index, 2.0, z_series, where=(z_series >= 2.0), facecolor='red', alpha=0.3)
    ax2.fill_between(z_series.index, -2.0, z_series, where=(z_series <= -2.0), facecolor='blue', alpha=0.3)
    
    ax2.set_ylim(-4, 4)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if not df_result.empty:
    print("\n" + "="*60)
    print(f"🎉 분석 완료: 총 {len(df_result)}개의 Cointegration 페어 발견")
    print("="*60)
    
    # Action
    df_action = df_result[df_result['Status'] == 'Action'].sort_values(by='Z-Score', key=abs, ascending=False)
    
    if not df_action.empty:
        print(f"\n🚀 [즉시 진입 추천] {len(df_action)}개 페어")
        for idx, row in df_action.iterrows():
            print(f"\n[{idx+1}] {row['Stock A']} & {row['Stock B']}")
            print(f"   👉 {row['Signal']}")
            print(f"   📊 Z-Score: {row['Z-Score']:.2f} | P-val: {row['P-value']:.4f}")
            plot_advanced_pair(row)
    else:
        print("\n🚀 [즉시 진입 추천] 현재 진입 조건(Rolling Z > 2.0)을 만족하는 종목이 없습니다.")

    # Watch
    df_watch = df_result[df_result['Status'] == 'Watch'].sort_values(by='P-value', ascending=True) # P-value 낮은 순
    
    if not df_watch.empty:
        print(f"\n👀 [관심 종목] {len(df_watch)}개 페어 (P-value 낮은 순)")
        print("-" * 75)
        print(f"{'Stock A':<10} {'Stock B':<10} {'P-val':<8} {'Z-Score':<8} {'Signal'}")
        print("-" * 75)
        for idx, row in df_watch.head(5).iterrows():
            print(f"{row['Stock A']:<10} {row['Stock B']:<10} {row['P-value']:.4f}   {row['Z-Score']:<8.2f} {row['Signal']}")
            
        print("\n(관심 종목 1위 상세 차트)")
        plot_advanced_pair(df_watch.iloc[0])
            
else:
    print("공적분(P-value < 0.05)을 만족하는 페어가 없습니다.")
