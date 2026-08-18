import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re
from datetime import datetime

# =========================
# 페이지 기본 설정 (모바일 최적화)
# =========================
st.set_page_config(page_title="QUANT SIGNAL", page_icon="⬛", layout="centered")

st.markdown("""
<style>
    @import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.8/dist/web/static/pretendard.css");
    html, body, [class*="css"] { font-family: 'Pretendard', sans-serif !important; background-color: #FAFAFA; }
    .block-container { padding-top: 2rem !important; padding-left: 0.8rem !important; padding-right: 0.8rem !important; max-width: 100vw !important; overflow-x: hidden !important; }
    .main-title { font-size: 2.2rem; font-weight: 800; color: #000000; letter-spacing: -0.03em; margin-bottom: 0rem; text-transform: uppercase; }
    .sub-title { font-size: 0.95rem; font-weight: 400; color: #666666; letter-spacing: -0.01em; margin-bottom: 1.5rem; }
    .stButton > button { background-color: #000000 !important; color: #ffffff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; padding: 0.6rem 1.5rem !important; width: 100% !important; box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; border-bottom: 1px solid #EAEAEA; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: transparent; border-radius: 0px; padding: 5px 0px; font-size: 1rem; font-weight: 600; color: #999999; }
    .stTabs [aria-selected="true"] { color: #000000 !important; border-bottom: 2.5px solid #000000 !important; }
    .table-container { width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; border-radius: 10px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05); background-color: #ffffff; border: 1px solid #F0F0F0; margin-bottom: 2rem; }
    .custom-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; text-align: center; }
    .custom-table th { background-color: #F8F9FA; color: #555555; font-weight: 700; padding: 8px 6px; border-bottom: 2px solid #EAEAEA; white-space: nowrap; }
    .custom-table td { padding: 10px 6px; border-bottom: 1px solid #F5F5F5; color: #111111; white-space: nowrap; }
    .custom-table tbody tr:hover { background-color: #FAFAFA; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Quant Signal</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Technical Dashboard for Nasdaq</p>', unsafe_allow_html=True)

# =========================
# 핵심 로직 (원본 코드와 100% 동일 + 검토 카운터 탑재)
# =========================
def is_valid_ticker(ticker):
    return bool(re.match(r'^[A-Za-z0-9.-]+$', str(ticker)))

def is_golden_alignment(ma5, ma10, ma20, ma50, idx):
    try:
        return ma5.iloc[idx] > ma10.iloc[idx] > ma20.iloc[idx] > ma50.iloc[idx]
    except IndexError:
        return False

def find_signals(data_df, nasdaq_tickers, offset_days=0, is_backtest=False):
    passed_tickers = []
    scanned_count = 0  # 검토된 종목 수 카운트

    for ticker in nasdaq_tickers:
        try:
            if isinstance(data_df.columns, pd.MultiIndex):
                if ticker not in data_df.columns.levels[0]:
                    continue
                hist = data_df[ticker].dropna(subset=['Close'])
            else:
                if ticker not in data_df:
                    continue
                hist = data_df[ticker].dropna(subset=['Close'])

            if len(hist) > 0:
                scanned_count += 1  # 실제 데이터가 존재하여 검토된 경우에만 카운트 증가

            full_high = hist['High'] 

            if offset_days > 0:
                hist = hist.iloc[:-offset_days]

            if len(hist) < 60:
                continue

            close = hist['Close']

            ma5 = close.rolling(window=5).mean()
            ma10 = close.rolling(window=10).mean()
            ma20 = close.rolling(window=20).mean()
            ma50 = close.rolling(window=50).mean()

            x_day = None
            score = 0
            near_ma = None

            for back in range(3, 11):
                i = -back
                price = close.iloc[i]
                for ma, pts in [(ma20, 3), (ma10, 2), (ma5, 1)]:
                    ma_val = ma.iloc[i]
                    if ma_val * 0.995 <= price <= ma_val * 1.005:
                        x_day = i
                        score = pts
                        near_ma = ma
                        break
                if x_day is not None:
                    break

            if x_day is None:
                continue

            cond1 = (is_golden_alignment(ma5, ma10, ma20, ma50, x_day-1) or
                     is_golden_alignment(ma5, ma10, ma20, ma50, x_day-2))

            cond2 = (is_golden_alignment(ma5, ma10, ma20, ma50, x_day+1) or
                     is_golden_alignment(ma5, ma10, ma20, ma50, x_day+2))

            cond3 = ((close.iloc[x_day-1] > near_ma.iloc[x_day-1]) or
                     (close.iloc[x_day-2] > near_ma.iloc[x_day-2]))

            cond4 = ((close.iloc[x_day+1] > near_ma.iloc[x_day+1]) or
                     (close.iloc[x_day+2] > near_ma.iloc[x_day+2]))

            cond5 = (near_ma.iloc[-1] < close.iloc[-1]) and (close.iloc[x_day+1] < close.iloc[-1])

            if cond1 and cond2 and cond3 and cond4 and cond5:
                signal_date = hist.index[x_day].strftime('%Y-%m-%d')
                signal_close = close.iloc[x_day]

                item = {
                    'Ticker': ticker,
                    'Close': round(close.iloc[-1], 2),
                    '눌림목 점수': score,
                    '시그널 발생일자': signal_date
                }

                if is_backtest:
                    signal_timestamp = hist.index[x_day]
                    future_highs = full_high.loc[signal_timestamp:]
                    if len(future_highs) > 0:
                        max_high = future_highs.max()
                        max_return = ((max_high / signal_close) - 1) * 100
                    else:
                        max_high = signal_close
                        max_return = 0.0

                    item['발생일 종가'] = round(signal_close, 2)
                    item['이후 최고가'] = round(max_high, 2)
                    item['최대 수익률(%)'] = round(max_return, 2)

                passed_tickers.append(item)

        except Exception as e:
            continue

    return passed_tickers, scanned_count

# =========================
# 서버 캐시 및 분석 실행 함수
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def run_dashboard_analysis():
    file = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTgA3hs0AXyyXwkg3U6j902doYiv9U8BrwejaodapPou48w7j2jX56Zlwh4RKJFmV6wBV4TJ21U3_cF/pub?output=csv"
    tickers_df = pd.read_csv(file)
    nasdaq_tickers = [t for t in tickers_df['Symbol'].tolist()[:1000] if is_valid_ticker(t)]

    # 야후 다운로드 실행
    data = yf.download(nasdaq_tickers, period="6mo", group_by="ticker", threads=True, progress=False)

    # API 수신 상태 진단 (멀티인덱스 여부 확인)
    if isinstance(data.columns, pd.MultiIndex):
        downloaded_tickers = len(data.columns.levels[0])
    else:
        downloaded_tickers = len(data.columns)

    # 1. 현재 기준 분석
    current_passed, scanned_live = find_signals(data, nasdaq_tickers, offset_days=0, is_backtest=False)
    results = []
    sector_cache = {}

    if len(current_passed) > 0:
        for item in current_passed:
            ticker = item['Ticker']
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                sector = info.get('sector', 'N/A')
                sector_cache[ticker] = sector
                psr = info.get('priceToSalesTrailing12Months', 0)
                rev_growth_rate = info.get('revenueGrowth', 0)

                psg = 0
                if psr is not None and rev_growth_rate is not None and rev_growth_rate > 0:
                    psg = psr / (rev_growth_rate * 100)

                if not psr or not psg:
                    try:
                        revenues = stock.financials.loc['Total Revenue']
                        if len(revenues) >= 2:
                            rev_curr = revenues.iloc[0]
                            rev_prev = revenues.iloc[1]
                            if rev_prev != 0 and not pd.isna(rev_prev):
                                growth_current = (rev_curr - rev_prev) / abs(rev_prev) * 100
                                market_cap = info.get('marketCap')
                                if market_cap and growth_current != 0:
                                    psr = market_cap / rev_curr
                                    psg = psr / growth_current
                    except:
                        pass

                psr = 0 if pd.isna(psr) or psr is None else psr
                psg = 0 if pd.isna(psg) or psg is None else psg

                results.append({
                    '시그널 발생일자': item['시그널 발생일자'],
                    'Ticker': ticker,
                    'Sector': sector,
                    '눌림목 점수': item['눌림목 점수'],
                    'PSR': round(psr, 2),
                    'PSG': round(psg, 2)
                })
            except Exception:
                sector_cache[ticker] = 'N/A'
                continue

    curr_df = pd.DataFrame(results)
    if not curr_df.empty:
        curr_df = curr_df.sort_values(by=['시그널 발생일자', '눌림목 점수'], ascending=[False, False])

    # 2. 1개월 전 백테스팅 수행
    backtest_passed, scanned_bt = find_signals(data, nasdaq_tickers, offset_days=21, is_backtest=True)
    bt_results = []

    for item in backtest_passed:
        ticker = item['Ticker']
        if ticker in sector_cache and sector_cache[ticker] != 'N/A':
            sector = sector_cache[ticker]
        else:
            try:
                sector = yf.Ticker(ticker).info.get('sector', 'N/A')
                sector_cache[ticker] = sector
            except:
                sector = 'N/A'

        bt_results.append({
            '시그널 발생일자': item['시그널 발생일자'],
            'Ticker': ticker,
            'Sector': sector,
            '눌림목 점수': item['눌림목 점수'],
            '최대 수익률(%)': item['최대 수익률(%)']
        })

    bt_df = pd.DataFrame(bt_results)
    if not bt_df.empty:
        bt_df = bt_df.sort_values(by=['최대 수익률(%)', '시그널 발생일자'], ascending=[False, False])
        bt_df = bt_df[['시그널 발생일자', 'Ticker', 'Sector', '눌림목 점수', '최대 수익률(%)']]

    return curr_df, bt_df, datetime.now(), downloaded_tickers, scanned_live

# =========================
# 화면 렌더링 및 디버깅 상태 표시
# =========================
with st.spinner('Scanning Market Data...'):
    curr_df, bt_df, last_update, dl_count, scanned_count = run_dashboard_analysis()

# 💡 화면 상단에 야후 API 수신 상태 및 검토된 종목 수 실시간 노출
st.caption(f"🔄 Last Updated: **{last_update.strftime('%m.%d %H:%M:%S')}** | 📥 API 수신: **{dl_count}개** | 🔍 검토된 종목: **{scanned_count}개**")

if st.button("새로고침 (Refresh)", use_container_width=True):
    time_diff = (datetime.now() - last_update).total_seconds()
    if time_diff >= 3600 or (curr_df.empty and bt_df.empty):
        run_dashboard_analysis.clear()
        st.rerun()
    else:
        remain_min = int((3600 - time_diff) // 60)
        st.warning(f"⚠️ 데이터 새로고침한지 1시간이 지나지 않았습니다. (약 {remain_min}분 후 가능)")

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["LIVE SIGNALS", "1-MONTH TEST"])

with tab1:
    if curr_df.empty:
        st.info("현재 조건에 부합하는 종목이 없습니다. (상단의 API 수신 개수와 검토된 종목 수를 확인해 보세요)")
    else:
        display_df = curr_df.copy()
        display_df.rename(columns={'시그널 발생일자': '발생일', 'Ticker': '종목', 'Sector': '섹터', '눌림목 점수': '점수'}, inplace=True)
        display_df['PSR'] = display_df['PSR'].apply(lambda x: f"{x:.2f}")
        display_df['PSG'] = display_df['PSG'].apply(lambda x: f"{x:.2f}")
        st.markdown(f'<div class="table-container">{display_df.to_html(index=False, classes="custom-table")}</div>', unsafe_allow_html=True)

with tab2:
    if bt_df.empty:
        st.info("1달 전 조건에 부합했던 종목이 없습니다.")
    else:
        display_bt = bt_df.copy()
        display_bt.rename(columns={'시그널 발생일자': '발생일', 'Ticker': '종목', 'Sector': '섹터', '눌림목 점수': '점수', '최대 수익률(%)': '수익률'}, inplace=True)
        display_bt['수익률'] = display_bt['수익률'].apply(lambda x: f"{x:,.2f}%")
        st.markdown(f'<div class="table-container">{display_bt.to_html(index=False, classes="custom-table")}</div>', unsafe_allow_html=True)
