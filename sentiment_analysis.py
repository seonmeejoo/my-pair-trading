import os
import json
import xml.etree.ElementTree as ET
import requests
import yfinance as yf
from openai import OpenAI
from datetime import datetime

# ─────────────────────────────────────────────
# 설정 (여기만 바꾸면 어떤 종목이든 분석 가능)
# ─────────────────────────────────────────────
TICKER = "AAPL"          # 분석할 종목 티커 (예: AAPL, MSFT, TSLA, 005930.KS)
TICKER_NAME = "Apple"    # 종목 이름 (뉴스 검색용)
MAX_NEWS = 8             # 분석할 최대 뉴스 개수

# OpenAI 클라이언트 초기화 (환경변수에서 API 키 자동 로드)
client = OpenAI()


def fetch_news(ticker: str, company_name: str, max_items: int = 8) -> list[dict]:
    """Yahoo Finance RSS 피드에서 최신 뉴스 수집"""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    news_list = []
    for item in root.iter("item"):
        if len(news_list) >= max_items:
            break
        title = item.findtext("title", "")
        description = item.findtext("description", "")[:500]
        pub_date = item.findtext("pubDate", "")
        link = item.findtext("link", "")
        news_list.append({
            "title": title,
            "summary": description,
            "published": pub_date,
            "link": link
        })

    return news_list


def analyze_sentiment_with_llm(news_list: list[dict], ticker: str, company_name: str) -> list[dict]:
    """GPT를 이용해 각 뉴스의 감성 점수 분석"""

    results = []

    for i, news in enumerate(news_list):
        print(f"  [{i+1}/{len(news_list)}] 분석 중: {news['title'][:60]}...")

        prompt = f"""
당신은 전문 퀀트 애널리스트입니다. 다음 뉴스가 {company_name}({ticker}) 주가에 미치는 단기적(1~5일) 영향을 분석하세요.

[뉴스 제목]
{news['title']}

[뉴스 요약]
{news['summary']}

다음 JSON 형식으로만 응답하세요:
{{
  "sentiment_score": <-1.0에서 +1.0 사이의 실수. -1=매우 부정, 0=중립, +1=매우 긍정>,
  "confidence": <0.0에서 1.0 사이. 판단의 확신도>,
  "reason": <한 문장으로 판단 근거>,
  "key_factor": <"earnings"|"macro"|"product"|"legal"|"management"|"other" 중 하나>
}}
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )

            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1].replace("json", "").strip()
            analysis = json.loads(raw)

            results.append({
                **news,
                "sentiment_score": float(analysis.get("sentiment_score", 0)),
                "confidence": float(analysis.get("confidence", 0.5)),
                "reason": analysis.get("reason", ""),
                "key_factor": analysis.get("key_factor", "other")
            })

        except Exception as e:
            print(f"    ⚠️  분석 실패: {e}")
            results.append({**news, "sentiment_score": 0.0, "confidence": 0.0, "reason": "분석 실패", "key_factor": "other"})

    return results


def generate_trading_signal(analyzed_news: list[dict]) -> dict:
    """분석된 뉴스들을 종합하여 최종 트레이딩 시그널 생성"""

    if not analyzed_news:
        return {"signal": "HOLD", "score": 0.0, "reason": "뉴스 없음"}

    total_weight = sum(n["confidence"] for n in analyzed_news)
    if total_weight == 0:
        weighted_score = 0.0
    else:
        weighted_score = sum(n["sentiment_score"] * n["confidence"] for n in analyzed_news) / total_weight

    if weighted_score >= 0.3:
        signal = "BUY 📈"
        signal_color = "🟢"
    elif weighted_score <= -0.3:
        signal = "SELL 📉"
        signal_color = "🔴"
    else:
        signal = "HOLD ⏸️"
        signal_color = "🟡"

    factor_counts = {}
    for n in analyzed_news:
        f = n.get("key_factor", "other")
        factor_counts[f] = factor_counts.get(f, 0) + 1
    top_factor = max(factor_counts, key=factor_counts.get)

    return {
        "signal": signal,
        "signal_color": signal_color,
        "weighted_score": round(weighted_score, 3),
        "news_count": len(analyzed_news),
        "top_factor": top_factor,
        "bullish_count": sum(1 for n in analyzed_news if n["sentiment_score"] > 0.1),
        "bearish_count": sum(1 for n in analyzed_news if n["sentiment_score"] < -0.1),
        "neutral_count": sum(1 for n in analyzed_news if -0.1 <= n["sentiment_score"] <= 0.1),
    }


def get_current_price(ticker: str) -> dict:
    """yfinance로 현재 주가 정보 가져오기"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info
        hist = stock.history(period="2d")

        current_price = hist['Close'].iloc[-1] if len(hist) > 0 else 0
        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0

        return {
            "current_price": round(current_price, 2),
            "change_pct": round(change_pct, 2),
            "currency": "USD"
        }
    except Exception as e:
        return {"current_price": 0, "change_pct": 0, "currency": "USD"}


def print_report(ticker: str, company_name: str, price_info: dict, signal: dict, analyzed_news: list[dict]):
    """최종 리포트 출력"""

    print("\n" + "="*65)
    print(f"  🤖 LLM 센티먼트 트레이딩 시그널 리포트")
    print(f"  생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65)

    change_icon = "▲" if price_info['change_pct'] >= 0 else "▼"
    print(f"\n📊 종목: {company_name} ({ticker})")
    print(f"   현재가: ${price_info['current_price']:,.2f}  {change_icon} {abs(price_info['change_pct']):.2f}%")

    print(f"\n{signal['signal_color']} 최종 시그널: {signal['signal']}")
    print(f"   종합 감성 점수: {signal['weighted_score']:+.3f}  (범위: -1.0 ~ +1.0)")
    print(f"   분석 뉴스 수: {signal['news_count']}개  |  긍정: {signal['bullish_count']}  부정: {signal['bearish_count']}  중립: {signal['neutral_count']}")
    print(f"   주요 드라이버: {signal['top_factor'].upper()}")

    print(f"\n📰 뉴스별 감성 분석:")
    print("-"*65)
    for i, news in enumerate(analyzed_news):
        score = news['sentiment_score']
        icon = "🟢" if score > 0.1 else ("🔴" if score < -0.1 else "🟡")
        print(f"  {icon} [{score:+.2f}] {news['title'][:55]}...")
        print(f"       → {news['reason']}")

    print("\n" + "="*65)
    print("  ⚠️  이 시그널은 교육/연구 목적입니다. 실제 투자 결정에")
    print("     단독으로 사용하지 마세요. 항상 추가 검증이 필요합니다.")
    print("="*65 + "\n")


def main():
    print(f"\n🚀 {TICKER_NAME}({TICKER}) 뉴스 센티먼트 분석 시작...")

    print(f"\n[1/4] 뉴스 수집 중 (최대 {MAX_NEWS}개)...")
    news_list = fetch_news(TICKER, TICKER_NAME, MAX_NEWS)
    print(f"  → {len(news_list)}개 뉴스 수집 완료")

    if not news_list:
        print("  ⚠️  뉴스를 찾을 수 없습니다.")
        return

    print(f"\n[2/4] LLM 감성 분석 중...")
    analyzed_news = analyze_sentiment_with_llm(news_list, TICKER, TICKER_NAME)

    print(f"\n[3/4] 트레이딩 시그널 생성 중...")
    signal = generate_trading_signal(analyzed_news)

    print(f"\n[4/4] 현재 주가 조회 중...")
    price_info = get_current_price(TICKER)

    print_report(TICKER, TICKER_NAME, price_info, signal, analyzed_news)


if __name__ == "__main__":
    main()
