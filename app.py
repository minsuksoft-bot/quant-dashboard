import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re
from datetime import datetime

# =========================
# 1. 페이지 설정 (모바일 최적화를 위해 layout="centered"로 변경)
# =========================
st.set_page_config(page_title="QUANT SIGNAL", page_icon="⬛", layout="centered")

st.markdown("""
<style>
    @import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.8/dist/web/static/pretendard.css");
    
    html, body, [class*="css"] {
        font-family: 'Pretendard', sans-serif !important;
        background-color: #FAFAFA;
    }
    
    /* 모바일 꽉 찬 화면을 위한 여백 최소화 */
    .block-container {
        padding-top: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100vw !important;
        overflow-x: hidden !important;
    }

    .main-title {
        font-size: 2.2rem; font-weight: 800; color: #000000;
        letter-spacing: -0.03em; margin-bottom: 0rem; text-transform: uppercase;
    }
    .sub-title {
        font-size: 0.95rem; font-weight: 400; color: #666666;
        letter-spacing: -0.01em; margin-bottom: 1.5rem;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background-color: #000000 !important; color: #ffffff !important;
        border: none !important; border-radius: 8px !important; 
        font-weight: 600 !important; padding: 0.6rem 1.5rem !important;
        width: 100% !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; border-bottom: 1px solid #EAEAEA; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: transparent; border-radius: 0px;
        padding: 5px 0px; font-size: 1rem; font-weight: 600; color: #999999;
    }
    .stTabs [aria-selected="true"] { color: #000000 !important; border-bottom: 2.5px solid #000000 !important; }
    
    /* 데이터프레임 스타일 */
    [data-testid="stDataFrame"] {
        border-radius: 12px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
        background-color: #ffffff; padding: 0.5rem; border: 1px solid #F0F0F0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Quant Signal</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Technical Dashboard for Nasdaq</p>', unsafe_allow_html=True)

# =========================
# 2. 핵심 로직 및 서버 캐싱
# =========================
def is_valid_ticker(ticker): return bool(re.match(r'^[A-Za-z0-9.-]+$', str(ticker)))

def is_golden_alignment(ma5, ma10, ma20, ma50, idx):
    try: return ma5.iloc[idx] > ma10.iloc[idx] > ma20.iloc[idx] > ma50.iloc[idx]
    except IndexError: return False

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
                # 모바일 공간 절약을 위해 연도(YYYY) 제외하고 MM.DD 로 포맷
                item = {
                    'Ticker': ticker, 'Close': round(close.iloc[-1], 2),
                    '눌림목 점수': score, '시그널 발생일자': hist.index[x_day].strftime('%m.%d')
                }
                if is_backtest:
                    future_highs = full_high.loc[hist.index[x_day]:]
                    max_high = future_highs.max() if len(future_highs) > 0 else signal_close
                    max_return = ((max_high / signal_close) - 1) * 100 if len(future_highs) > 0 else 0.0
                    item.update({'발생일 종가': round(signal_close, 2), '이후 최고가': round(max_high, 2), '최대 수익률(%)': round(max_return, 2)})
                passed_tickers.append(item)
        except Exception: continue
    return passed_tickers

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data_and_analyze():
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
                'Close': item['Close'], '눌림목 점수': item['눌림목 점수'],
                'PSR': round(psr if pd.notna(psr) else 0, 2), 'PSG': round(psg if pd.notna(psg) else 0, 2)
            })
        except Exception: continue
                
    curr_df = pd.DataFrame(results).sort_values(by=['시그널 발생일자', '눌림목 점수'], ascending=[False, False]) if results else pd.DataFrame()
    
    bt_passed = find_signals(data, nasdaq_tickers, offset_days=21, is_backtest=True)
    # 요청하신 대로 1개월 백테스트는 [최대 수익률] 최우선 내림차순 정렬
    bt_df = pd.DataFrame(bt_passed).sort_values(by=['최대 수익률(%)', '시그널 발생일자'], ascending=[False, False])[['시그널 발생일자', 'Ticker', '눌림목 점수', '발생일 종가', '이후 최고가', '최대 수익률(%)']] if bt_passed else pd.DataFrame()
        
    return curr_df, bt_df, datetime.now()

# =========================
# 3. 데이터 로드 및 UI 렌더링
# =========================
with st.spinner('Scanning Market Data...'):
    curr_df, bt_df, last_update = fetch_data_and_analyze()

st.caption(f"🔄 Last Updated: **{last_update.strftime('%m.%d %H:%M:%S')}**")

if st.button("새로고침 (Refresh)"):
    time_diff = (datetime.now() - last_update).total_seconds()
    if time_diff < 3600:
        st.warning(f"1시간 쿨타임이 적용 중입니다. ({int((3600 - time_diff) // 60)}분 후 가능)")
    else:
        fetch_data_and_analyze.clear()
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["LIVE SIGNALS", "1-MONTH TEST"])

# 모바일 가독성을 위해 표(Column) 설정 최적화
with tab1:
    if curr_df.empty: st.info("조건에 부합하는 종목이 없습니다.")
    else:
        st.dataframe(
            curr_df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "시그널 발생일자": st.column_config.TextColumn("발생일", width="small"),
                "Ticker": st.column_config.TextColumn("종목", width="small"),
                "Close": st.column_config.NumberColumn("현재가", format="$%.2f", width="small"),
                "눌림목 점수": st.column_config.NumberColumn("점수", width="small"),
                "PSR": st.column_config.NumberColumn("PSR", width="small"),
                "PSG": st.column_config.NumberColumn("PSG", width="small")
            }
        )

with tab2:
    if bt_df.empty: st.info("1달 전 조건에 부합했던 종목이 없습니다.")
    else:
        st.dataframe(
            bt_df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "시그널 발생일자": st.column_config.TextColumn("발생일", width="small"),
                "Ticker": st.column_config.TextColumn("종목", width="small"),
                "눌림목 점수": st.column_config.NumberColumn("점수", width="small"),
                "발생일 종가": st.column_config.NumberColumn("당시가", format="$%.2f", width="small"),
                "이후 최고가": st.column_config.NumberColumn("최고가", format="$%.2f", width="small"),
                "최대 수익률(%)": st.column_config.NumberColumn("최대수익률", format="%.2f%%", width="medium")
            }
        )
