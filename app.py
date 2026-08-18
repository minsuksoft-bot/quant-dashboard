import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re
from datetime import datetime

# =========================
# 1. 스트림릿 페이지 기본 설정 & 커스텀 CSS (하이엔드 감성)
# =========================
st.set_page_config(page_title="QUANT SIGNAL", page_icon="⬛", layout="wide")

# 입생로랑 & 토스 믹스 럭셔리 커스텀 CSS 적용
st.markdown("""
<style>
    /* 트렌디한 폰트 프리텐다드(Pretendard) 적용 */
    @import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.8/dist/web/static/pretendard.css");
    
    html, body, [class*="css"] {
        font-family: 'Pretendard', sans-serif !important;
        background-color: #FAFAFA; /* 매우 연한 회색 배경 (토스 스타일) */
    }

    /* 메인 타이틀 (보그/YSL 스타일의 볼드 & 미니멀) */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #000000;
        letter-spacing: -0.03em;
        margin-bottom: 0rem;
        text-transform: uppercase;
    }
    .sub-title {
        font-size: 1.1rem;
        font-weight: 400;
        color: #666666;
        letter-spacing: -0.01em;
        margin-bottom: 2rem;
    }

    /* 버튼 스타일 (YSL 스타일: 블랙 배경, 화이트 텍스트, 호버 시 유려한 움직임) */
    .stButton > button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important; /* 약간의 라운딩 처리 */
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    .stButton > button:hover {
        background-color: #333333 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15) !important;
    }

    /* 탭 스타일 (미니멀 & 깔끔한 밑줄) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid #EAEAEA;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0px;
        padding: 10px 0px;
        font-size: 1.1rem;
        font-weight: 600;
        color: #999999;
    }
    .stTabs [aria-selected="true"] {
        color: #000000 !important;
        border-bottom: 2.5px solid #000000 !important;
    }

    /* 데이터프레임 컨테이너에 부드러운 그림자 부여 (토스 스타일) */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        box-shadow: 0px 8px 24px rgba(0, 0, 0, 0.04);
        background-color: #ffffff;
        padding: 1rem;
        border: 1px solid #F0F0F0;
    }
    
    /* 경고/알림 박스 라운딩 처리 */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Quant Signal</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Technical & Fundamental Dashboard for Nasdaq</p>', unsafe_allow_html=True)

# =========================
# 세션 상태(Session State) 초기화
# =========================
if 'last_update' not in st.session_state:
    st.session_state['last_update'] = None
if 'current_df' not in st.session_state:
    st.session_state['current_df'] = pd.DataFrame()
if 'backtest_df' not in st.session_state:
    st.session_state['backtest_df'] = pd.DataFrame()

# =========================
# 2. 핵심 로직 함수
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
    for ticker in nasdaq_tickers:
        try:
            if ticker not in data_df: continue
            hist = data_df[ticker].dropna(subset=['Close'])
            full_high = hist['High']
            
            if offset_days > 0: hist = hist.iloc[:-offset_days]
            if len(hist) < 60: continue

            close = hist['Close']
            ma5, ma10, ma20, ma50 = (close.rolling(window=w).mean() for w in [5, 10, 20, 50])

            x_day, score, near_ma = None, 0, None
            for back in range(3, 11):
                i = -back
                price = close.iloc[i]
                for ma, pts in [(ma20, 3), (ma10, 2), (ma5, 1)]:
                    ma_val = ma.iloc[i]
                    if ma_val * 0.995 <= price <= ma_val * 1.005:
                        x_day, score, near_ma = i, pts, ma
                        break
                if x_day is not None: break

            if x_day is None: continue

            cond1 = is_golden_alignment(ma5, ma10, ma20, ma50, x_day-1) or is_golden_alignment(ma5, ma10, ma20, ma50, x_day-2)
            cond2 = is_golden_alignment(ma5, ma10, ma20, ma50, x_day+1) or is_golden_alignment(ma5, ma10, ma20, ma50, x_day+2)
            cond3 = (close.iloc[x_day-1] > near_ma.iloc[x_day-1]) or (close.iloc[x_day-2] > near_ma.iloc[x_day-2])
            cond4 = (close.iloc[x_day+1] > near_ma.iloc[x_day+1]) or (close.iloc[x_day+2] > near_ma.iloc[x_day+2])
            cond5 = (near_ma.iloc[-1] < close.iloc[-1]) and (close.iloc[x_day+1] < close.iloc[-1])

            if cond1 and cond2 and cond3 and cond4 and cond5:
                signal_close = close.iloc[x_day]
                item = {
                    'Ticker': ticker,
                    'Close': round(close.iloc[-1], 2),
                    '눌림목 점수': score,
                    '시그널 발생일자': hist.index[x_day].strftime('%Y-%m-%d')
                }
                if is_backtest:
                    future_highs = full_high.loc[hist.index[x_day]:]
                    max_high = future_highs.max() if len(future_highs) > 0 else signal_close
                    max_return = ((max_high / signal_close) - 1) * 100 if len(future_highs) > 0 else 0.0
                    item.update({'발생일 종가': round(signal_close, 2), '이후 최고가': round(max_high, 2), '최대 수익률(%)': round(max_return, 2)})
                passed_tickers.append(item)
        except Exception: continue
    return passed_tickers

def run_analysis():
    file_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTgA3hs0AXyyXwkg3U6j902doYiv9U8BrwejaodapPou48w7j2jX56Zlwh4RKJFmV6wBV4TJ21U3_cF/pub?output=csv"
    tickers_df = pd.read_csv(file_url)
    nasdaq_tickers = [t for t in tickers_df['Symbol'].tolist()[:1000] if is_valid_ticker(t)]
    
    data = yf.download(nasdaq_tickers, period="6mo", group_by="ticker", threads=True, progress=False)
    
    current_passed = find_signals(data, nasdaq_tickers, offset_days=0, is_backtest=False)
    results = []
    for item in current_passed:
        try:
            stock = yf.Ticker(item['Ticker'])
            info = stock.info
            psr = info.get('priceToSalesTrailing12Months', 0)
            rev_growth = info.get('revenueGrowth', 0)
            psg = (psr / (rev_growth * 100)) if (psr and rev_growth and rev_growth > 0) else 0
            
            if not psr or not psg:
                try:
                    revenues = stock.financials.loc['Total Revenue']
                    if len(revenues) >= 2 and revenues.iloc[1] != 0 and not pd.isna(revenues.iloc[1]):
                        g_curr = (revenues.iloc[0] - revenues.iloc[1]) / abs(revenues.iloc[1]) * 100
                        m_cap = info.get('marketCap')
                        if m_cap and g_curr != 0: psr, psg = m_cap / revenues.iloc[0], psr / g_curr
                except: pass
            
            results.append({
                '시그널 발생일자': item['시그널 발생일자'], 'Ticker': item['Ticker'],
                'Sector': info.get('sector', 'N/A'), 'Close': item['Close'],
                '눌림목 점수': item['눌림목 점수'],
                'PSR': round(psr if pd.notna(psr) else 0, 3), 'PSG': round(psg if pd.notna(psg) else 0, 3)
            })
        except Exception: continue
                
    curr_df = pd.DataFrame(results).sort_values(by=['시그널 발생일자', '눌림목 점수'], ascending=[False, False]) if results else pd.DataFrame()
    
    bt_passed = find_signals(data, nasdaq_tickers, offset_days=21, is_backtest=True)
    bt_df = pd.DataFrame(bt_passed).sort_values(by=['시그널 발생일자', '최대 수익률(%)'], ascending=[False, False])[['시그널 발생일자', 'Ticker', '눌림목 점수', '발생일 종가', '이후 최고가', '최대 수익률(%)']] if bt_passed else pd.DataFrame()
        
    return curr_df, bt_df

# =========================
# 3. 데이터 업데이트 및 UI 제어
# =========================
def attempt_refresh():
    now = datetime.now()
    if st.session_state['last_update'] is not None:
        time_diff = (now - st.session_state['last_update']).total_seconds()
        if time_diff < 3600:
            st.warning(f"데이터 갱신은 1시간에 1번만 가능합니다. ({int((3600 - time_diff) // 60)}분 후 가능)")
            return

    with st.spinner('Scanning Market Data...'):
        curr_df, bt_df = run_analysis()
        st.session_state.update({'current_df': curr_df, 'backtest_df': bt_df, 'last_update': now})

if st.session_state['last_update'] is None:
    attempt_refresh()

col1, col2 = st.columns([4, 1])
with col1:
    if st.session_state['last_update']:
        st.caption(f"Last Updated: **{st.session_state['last_update'].strftime('%Y.%m.%d %H:%M:%S')}**")
with col2:
    if st.button("새로고침 (Refresh)", use_container_width=True): attempt_refresh()

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["LIVE SIGNALS", "1-MONTH BACKTEST"])

with tab1:
    st.markdown("##### 현재 시점 기술적 분석 통과 종목")
    if st.session_state['current_df'].empty: st.info("조건에 부합하는 종목이 없습니다.")
    else: st.dataframe(st.session_state['current_df'], use_container_width=True, hide_index=True)

with tab2:
    st.markdown("##### 21거래일 전 시그널 발생 종목의 현재 수익률 추적")
    if st.session_state['backtest_df'].empty: st.info("1달 전 조건에 부합했던 종목이 없습니다.")
    else: st.dataframe(st.session_state['backtest_df'], use_container_width=True, hide_index=True)
